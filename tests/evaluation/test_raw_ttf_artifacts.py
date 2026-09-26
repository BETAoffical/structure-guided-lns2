from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments._common import json_fingerprint
from lns2_selector.evaluation.raw_ttf_artifacts import (
    ARTIFACT_FILES, validate_raw_ttf_identity, validate_raw_ttf_outcome,
)
from scripts import run_sa_onpolicy as run
from scripts import run_sa_raw_ttf as cli


def fixture():
    job = dict(job_id='j', pair_id='p', replica=0, comparison_arm='raw_fast', solver_seed=7,
               budget_seconds=120., case=dict(task_id='t', map_id='m'), expected_initial='i'*64,
               model=dict(policy_sha256='b'*64), arm='trained_actor', phase='test',
               plan=dict(config=dict(stream_seed=1)))
    row = {k: job[k] for k in ('job_id', 'pair_id', 'replica', 'comparison_arm', 'solver_seed', 'budget_seconds')}
    row.update(schema='lns2.sa_raw_ttf.episode.v1', status='ok', binding='binding', task_id='t',
               map_id='m', initial_fingerprint='i'*64, policy_sha256='b'*64,
               rng_stream_id=json_fingerprint([1,'test','p',0]),
               feasible=False, success_within_budget=False, delivered_within_budget=False,
               files={n:'a'*64 for n in ARTIFACT_FILES}, initial_conflicts=1, final_conflicts=1,
               decisions=0, legal_noops=0, pp_calls=0, generated=0, soc=2, makespan=1, wait_steps=0)
    row.update({k:0. for k in ('reset_seconds','selection_seconds','step_wall_seconds','native_pp_seconds',
                'trace_seconds','bookkeeping_seconds','search_end_seconds','finalization_seconds',
                'delivery_seconds','setup_seconds','cold_worker_seconds')})
    return job,row


class RawTTFArtifactTests(unittest.TestCase):
    def test_reader_binds_full_job_identity_and_requires_all_files(self):
        job,row=fixture()
        mutations = [{k:v} for k,v in (
            ('task_id','wrong'),('solver_seed',8),('solver_seed',True),('replica',False),
            ('rng_stream_id','wrong'),('files',{}),('files',[]),('status','error'),
            ('schema','old'),('policy_sha256','c'*64),('feasible','false'))]
        mutations += [{'files':{**row['files'],'extra.json':'a'*64}},
                      {'files':dict(row['files'], **{'initial.json':'invalid'})}]
        for changes in mutations:
            with self.subTest(changes=changes), patch.object(run,'read_json',return_value=run.sealed(dict(row,**changes))), \
                 patch.object(run,'sha256_file',return_value='a'*64):
                with self.assertRaises(ValueError):cli.read_result({'binding':'binding'},Path('.'),job)
        with patch.object(run,'read_json',return_value=run.sealed(row)), patch.object(run,'sha256_file',return_value='a'*64), \
             patch.object(run,'contained_file',return_value=Path('fixture')):
            self.assertEqual(cli.read_result({'binding':'binding'},Path('.'),job),run.sealed(row))

    def test_non_object_identity_rejected(self):
        job,_=fixture()
        for row in (None,[],1,'row'):
            with self.subTest(row=row),self.assertRaises(ValueError):
                validate_raw_ttf_identity(row,job,'binding')
            with self.subTest(reader=row),patch.object(run,'read_json',return_value=row),self.assertRaises(ValueError):
                cli.read_result({'binding':'binding'},Path('.'),job)

    def test_outcome_counts_and_pp_total_derive_from_states_and_trace(self):
        _,row=fixture()
        state=dict(num_of_colliding_pairs=1)
        validate_raw_ttf_outcome(row,state,state,0.)
        for changes in ({'initial_conflicts':999},{'final_conflicts':999},
                        {'native_pp_seconds':999.},{'decisions':True},{'final_conflicts':1.0}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                validate_raw_ttf_outcome(dict(row,**changes),state,state,0.)
        for value in (float('nan'),float('inf'),-1.,True,'0',10**10000):
            with self.subTest(type=type(value).__name__),self.assertRaises(ValueError):
                validate_raw_ttf_outcome(dict(row,native_pp_seconds=value),state,state,0.)

    def test_trace_sum_roundoff_and_no_input_mutation(self):
        job,row=fixture()
        row['native_pp_seconds']=.1+.2
        saved=deepcopy(row)
        validate_raw_ttf_identity(row,job,'binding')
        validate_raw_ttf_outcome(row,{'num_of_colliding_pairs':1},{'num_of_colliding_pairs':1},.3)
        self.assertEqual(row,saved)
