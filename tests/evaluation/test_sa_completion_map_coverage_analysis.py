from copy import deepcopy
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import analyze_sa_completion_map_coverage as analysis


class MapCoverageAnalysisTests(unittest.TestCase):
    def test_help_does_not_verify_or_write(self):
        with patch('sys.argv', ['analysis', '--help']), \
                patch.object(analysis.cli, 'verify', side_effect=AssertionError('unexpected I/O')), \
                contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as caught:
            analysis.main()
        self.assertEqual(caught.exception.code, 0)

    def reconstructed(self, success):
        initial = dict(agents=[dict(id=10, path=[0, 1]), dict(id=30, path=[1, 0])],
                       conflict_edges=[[10, 30]], num_of_colliding_pairs=1,
                       low_level=dict(generated=0))
        final = deepcopy(initial)
        final['low_level']['generated'] = 2
        if success:
            final['agents'][0]['path'] = [0, 2, 3, 1]
            final['conflict_edges'] = []
            final['num_of_colliding_pairs'] = 0
        metrics = dict(neighborhood=[10], conflicts_before=1,
                       conflicts_after=final['num_of_colliding_pairs'],
                       replan_success=success, pp_failure_reason='' if success else 'rejected',
                       acceptance_evaluated=False, native_replan_seconds=0.01)
        event = dict(decision=0, metrics=metrics, delta={}, probabilities={'c': 1.0},
                     anchor_id='c', selected_id='c', action=dict(agents=[10], random_seed=7),
                     selection_draw=0.3, temperature=1.0, out_of_range_fraction=0.0,
                     pool=[dict(candidate_id='c', agents=[10])])
        job = dict(job_id='j', pair_id='p', replica=0, comparison_arm='parent')
        row = dict(decisions=1, success=success, final_conflicts=final['num_of_colliding_pairs'], map_id='m')
        with patch.object(analysis.cli.old.previous, 'read_result', return_value=row), \
                patch.object(analysis.cli.old.previous, 'folder', return_value=Path('unused')), \
                patch.object(analysis.cli.run, 'read_json', side_effect=[initial, final]), \
                patch.object(analysis.cli.run, 'trace_read', return_value=[event]), \
                patch.object(analysis, 'apply_state_delta', return_value=final):
            return analysis.trace_worker(job)

    def test_noncontinuous_agent_success_reconstruction(self):
        result = self.reconstructed(True)
        self.assertEqual(result['final_conflicts'], 0)
        self.assertEqual(result['all']['conflict_decreased'], 1)
        self.assertEqual(result['changed_path_revisits'], 0)
        self.assertEqual(result['final_events']['pairs'], 0)

    def test_unchanged_paths_not_a_changed_path_revisit(self):
        result = self.reconstructed(False)
        self.assertEqual(result['all']['unchanged_paths'], 1)
        self.assertEqual(result['changed_path_revisits'], 0)
        self.assertEqual(result['final_events']['pairs'], 1)
        self.assertEqual(result['final_events']['swap_events'], 1)

    def test_existing_analysis_rejects_changed_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / 'report.json').write_text('{}', encoding='utf-8')
            csv = out / 'comparison.csv'
            csv.write_text('original\n', encoding='utf-8')
            body = dict(binding='b', source_report_sha256=analysis.cli.run.sha256_file(out / 'report.json'),
                        csv_sha256=analysis.cli.run.sha256_file(csv),
                        diagnostic_source_sha256=analysis.cli.run.sha256_file(Path(analysis.__file__)),
                        paired_positions=48, reviewed_episodes=9)
            analysis.cli.run.once(out / 'diagnostic.json', analysis.cli.run.sealed(body))
            with patch.object(analysis.cli, 'audited_rows', return_value=iter(range(144))):
                self.assertTrue(analysis.verify_existing(dict(binding='b'), out)['verified'])
            csv.write_text('changed\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed CSV'):
                analysis.verify_existing(dict(binding='b'), out)


if __name__ == '__main__':
    unittest.main()
