"""Independently verify saved prefix-feedback controls, constraints and rollback."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments._common import read_json, write_json, write_jsonl
from experiments.full_neighborhood_recovery import check_paths, choose_feedback
from experiments.local_path_compatibility import ROOT, contained, sha256_file
from experiments.local_path_search import at, pair_events
from experiments.native_path_compatibility import check_seal, seal
from experiments.prefix_budget_feedback import evaluate_gate, rank_relations, result_signature, scientific_signature
from experiments.prefix_feedback_collection import check_result, load, output_file


def target_pair_attribution(base, attempt):
    """Inspect returned prefix paths, never the rolled-back final path array."""
    option = attempt['option']
    if not option:
        return None
    blocker, victim = option['blocker'], option['victim']
    found = {r['agent']: r['search'] for r in attempt['result']['records']}
    original = {r['agent']: r['search'] for r in base['records']}
    info = dict(blocker=blocker, victim=victim,
                branch_status=attempt['result']['status'])
    if blocker not in found or found[blocker]['status'] != 'path':
        return dict(info, classification='blocker_path_unavailable')
    if victim not in found or found[victim]['status'] != 'path':
        return dict(info, classification='victim_path_unavailable')
    events = pair_events(found[blocker]['path'], found[victim]['path'])
    baseline_events = pair_events(original[blocker]['path'], original[victim]['path'])
    return dict(info, classification='pair_persists' if events else 'pair_removed',
                baseline_events=len(baseline_events), attempt_events=len(events))


def verify(output, partial=False):
    m=load(output)
    cases={c['case_id']:c for c in m['cases']}
    rows=[]
    diagnostics=[]
    counts=Counter()
    journal=output/'collection_manifest.jsonl'
    manifest={r['job_id']:r for r in (json.loads(line) for line in journal.read_text().splitlines())}
    for job in m['jobs']:
        path=output_file(output,job)
        if not path.exists():
            if partial:
                continue
            raise ValueError('missing registered condition')
        row=read_json(path)
        check_result(m,job,row)
        if row['status']!='ok' or row['budget_exhausted']:
            raise ValueError('invalid or censored condition')
        if manifest[job['job_id']]['sha256']!=sha256_file(path):
            raise ValueError('collector manifest SHA mismatch')
        case=cases[job['case_id']]
        state=read_json(contained(case['state_file']))
        transition=read_json(contained(case['historical_transition_file']))
        order=transition['metrics']['repair_order']
        reference=m['references'][job['job_id']]
        old=read_json(contained(reference['baseline']))
        if scientific_signature(row['base'])!=scientific_signature(old['base']):
            raise ValueError('ordinary result changed')
        counts['baseline_equivalent']+=1
        if reference['control']:
            if result_signature(row)!=result_signature(read_json(contained(reference['control']))):
                raise ValueError('control result changed')
            counts['frozen_control_equivalent']+=1
        if check_paths(state,row['final']['paths'],order)!=row['path_check'] or not row['native_paths_verified']:
            raise ValueError('final path integrity mismatch')
        if not row['recovered'] and row['final']!=row['base']:
            raise ValueError('non-recovery changed baseline')
        if row['recovered'] and row['final']['conflicts']>=row['base']['conflicts']:
            raise ValueError('claimed recovery without strict decrease')
        counts['final_paths']+=1
        feedback=row['feedback']
        if feedback:
            if len(feedback['evidence'])>m['config']['max_feedback_blockers']:
                raise ValueError('excess feedback relations')
            if job['method']=='prefix_resources':
                ranked,external=rank_relations(row['base'],order)
                if feedback['ranked_relations']!=ranked or feedback['external_blockers']!=external:
                    raise ValueError('prefix feedback uses unexpected information')
                actual=[{k:r[k] for k in ('blocker','victim','first_time','blocker_pair_degree')}
                        for r in feedback['evidence']]
                if actual!=ranked[:m['config']['max_feedback_blockers']]:
                    raise ValueError('incorrect relation selection')
            options=choose_feedback(feedback['candidates'],'directed_resources',m['config']['max_attempts'])
            if [a['option'] for a in row['attempts']]!=options[:len(row['attempts'])]:
                raise ValueError('resource sampling mismatch')
        if len(row['attempts'])>m['config']['max_attempts']:
            raise ValueError('excess retries')
        if len(row['attempt_ledgers'])!=len(row['attempts']):
            raise ValueError('attempt attribution count mismatch')
        for index,attempt in enumerate(row['attempts']):
            option,branch=attempt['option'],attempt['result']
            if option:
                blocker,victim=option['blocker'],option['victim']
                if blocker not in order or victim not in order or order.index(blocker)>=order.index(victim):
                    raise ValueError('feedback changes external or unplanned agent')
                length=order.index(blocker)
                if length!=option['prefix_length'] or option['cap']!=len(row['base']['records'][length]['search']['path'])-1:
                    raise ValueError('backtrack or baseline cost cap mismatch')
                if branch['records'][:length]!=row['base']['records'][:length]:
                    raise ValueError('reused prefix changed')
                found=next((r['search'] for r in branch['records'] if r['agent']==blocker),None)
                if found and found['status']=='path':
                    p=found['path']; kind,t,u,v=option['constraint']
                    if len(p)-1>option['cap']:
                        raise ValueError('predecessor exceeded cost cap')
                    violated=at(p,t)==u if kind=='vertex' else at(p,t-1)==u and at(p,t)==v
                    if violated:
                        raise ValueError('native path violates feedback constraint')
                    counts['constrained_paths']+=1
            if branch['status']=='accepted':
                check_paths(state,branch['paths'],order)
            pair_info=target_pair_attribution(row['base'],attempt)
            if pair_info is not None:
                diagnostics.append(dict(job=job,attempt_index=index,role=case['role'],
                    map_id=case['map_id'],target_pair=pair_info,
                    common_prefix_change=row['attempt_ledgers'][index]['common_prefix_change']))
            counts['attempts']+=1
        rows.append(row)
    if len(manifest)!=len(rows):
        raise ValueError('collector manifest has unexpected conditions')
    gate=evaluate_gate(rows,m['cases'],m['config']['pp_seeds'])
    if not partial:
        report=read_json(output/'report.json')
        check_seal(report)
        if report['gate']!=gate or report['fingerprint']!=m['content_sha256'] or report['timing_allowed']:
            raise ValueError('report or timing gate mismatch')
        expected_hashes={output_file(output,r['job']).relative_to(ROOT).as_posix():
                         manifest[r['job']['job_id']]['sha256'] for r in rows}
        if report['result_hashes']!=expected_hashes or not report['complete']:
            raise ValueError('report source binding mismatch')
    diagnostics_path=output/('partial_target_pair_diagnostics.jsonl' if partial else 'target_pair_diagnostics.jsonl')
    write_jsonl(diagnostics_path,diagnostics)
    pair_summary={method:dict(Counter(d['target_pair']['classification'] for d in diagnostics
                                    if d['job']['method']==method))
                  for method in ('directed_resources','prefix_resources')}
    failure_summary={}
    for method in pair_summary:
        group=[d for d in diagnostics if d['job']['method']==method]
        failure_summary[method]=dict(
            pair_and_branch_status=dict(Counter(d['target_pair']['classification']+'/'+
                d['target_pair']['branch_status'] for d in group)),
            internal_removed_external_added=sum(
                bool(d['common_prefix_change'].get('internal_pairs',{}).get('removed')) and
                bool(d['common_prefix_change'].get('external_pairs',{}).get('added')) for d in group),
            interpretation='common-prefix changes are not full-outcome or external-agent necessity proofs')
    result=seal(dict(schema='lns2.prefix_feedback_verification.v1',passed=True,partial=partial,
        fingerprint=m['content_sha256'],counts=dict(counts),conditions=len(rows),gate=gate,
        target_pair_summary=pair_summary,target_pair_diagnostics_sha256=sha256_file(diagnostics_path),
        failure_summary=failure_summary,
        report_sha256=None if partial else sha256_file(output/'report.json'),
        collection_manifest_sha256=sha256_file(journal),
        verifier_sha256=sha256_file(Path(__file__)),timing_allowed=False))
    write_json(output/('partial_verification.json' if partial else 'verification.json'),result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='build/initlns-prefix-budget-feedback-v1')
    parser.add_argument('--partial',action='store_true')
    args=parser.parse_args()
    print(verify(contained(args.output),args.partial))
