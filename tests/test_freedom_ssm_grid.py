import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from freedom_local_ssm_plan import build_plan, build_grid_plan, build_item_grid_plan, TAUS, LAMBDAS, II_WEIGHTS, II_KS
from run_night import summarize


class FreedomSSMGridTest(unittest.TestCase):
    def test_sports_local_only_sweep_leaves_tau_lambda_and_ii_fixed(self):
        jobs = build_plan()
        self.assertEqual([j['overrides']['local_weight'] for j in jobs], [.1, .01, .001])
        fixed = [{k:v for k,v in j['overrides'].items() if k!='local_weight'} for j in jobs]
        self.assertEqual(fixed, [fixed[0]] * 3)
        self.assertEqual((fixed[0]['tau'], fixed[0]['lambda'], fixed[0]['ii_weight'], fixed[0]['knn_k']),
                         (.2, .01, .0005, 10))
        self.assertEqual(fixed[0]['seed'], 999)
        run = subprocess.run([sys.executable, str(ROOT / 'experiments/run_night.py'),
                              '--suite', 'freedom_local_ssm', '--dataset', 'sports',
                              '--data-path', '/tmp/local-dry-run', '--dry-run'],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        manifest = json.loads(run.stdout)
        self.assertEqual(manifest['dataset'], 'sports')
        self.assertEqual(manifest['jobs'], jobs)

    def test_full_cartesian_grid_and_dry_run(self):
        jobs = build_grid_plan()
        self.assertEqual(len(jobs), 20)
        self.assertEqual(len({j['name'] for j in jobs}), 20)
        self.assertEqual({(j['overrides']['tau'], j['overrides']['lambda']) for j in jobs},
                         {(tau, weight) for tau in TAUS for weight in LAMBDAS})
        for job in jobs:
            self.assertEqual(job['model'], 'FreedomLocalSSM')
            self.assertEqual(job['overrides']['local_weight'], .01)
            self.assertEqual(job['overrides']['alpha'], .5)
            self.assertEqual(job['overrides']['seed'], 999)
        run = subprocess.run([sys.executable, str(ROOT / 'experiments/run_night.py'),
                              '--suite', 'freedom_local_ssm_grid', '--data-path', '/tmp/grid-dry-run',
                              '--dry-run'], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['jobs'], jobs)

    def test_item_grid_locks_selected_settings_and_dispatches_all_pairs(self):
        jobs = build_item_grid_plan()
        self.assertEqual(len(jobs), 16)
        self.assertEqual(len({j['name'] for j in jobs}), 16)
        self.assertEqual({(j['overrides']['ii_weight'], j['overrides']['knn_k']) for j in jobs},
                         {(weight, k) for weight in II_WEIGHTS for k in II_KS})
        for job in jobs:
            config = job['overrides']
            self.assertEqual((config['tau'], config['lambda'], config['local_weight'], config['alpha']),
                             (.2, .01, .01, .5))
            self.assertEqual(config['seed'], 999)
            self.assertEqual(config['ii_image_weight'], .1)
            self.assertTrue(config['knn_keep_self'])
        run = subprocess.run([sys.executable, str(ROOT / 'experiments/run_night.py'),
                              '--suite', 'freedom_local_ssm_item_grid', '--data-path', '/tmp/grid-dry-run',
                              '--dry-run'], capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(json.loads(run.stdout)['jobs'], jobs)

    def test_ranking_uses_only_validation_and_tracks_partial_completion(self):
        jobs = build_item_grid_plan()[:3]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            for job, valid, test in zip(jobs, (.1, .2), (.9, .01)):
                target = output / job['name']; target.mkdir()
                (target / 'status.json').write_text(json.dumps(dict(state='complete')))
                (target / 'result.json').write_text(json.dumps([dict(
                    best_epoch=4, valid={'recall@20': valid}, test={'recall@20': test})]))
            summarize(output, jobs)
            best = json.loads((output / 'best_validation.json').read_text())
            self.assertEqual(best['best']['name'], jobs[1]['name'])
            self.assertEqual(best['overrides'], jobs[1]['overrides'])
            self.assertEqual(best['completed'], 2)
            self.assertTrue(best['provisional'])
            self.assertTrue((output / 'validation_ranking.csv').exists())
            rows = json.loads((output / 'summary.json').read_text())
            self.assertEqual(rows[0]['config_tau'], jobs[0]['overrides']['tau'])
            self.assertEqual(rows[0]['config_lambda'], jobs[0]['overrides']['lambda'])
            self.assertEqual(rows[0]['config_ii_weight'], jobs[0]['overrides']['ii_weight'])
            self.assertEqual(rows[0]['config_knn_k'], jobs[0]['overrides']['knn_k'])
            self.assertEqual(rows[2]['state'], 'pending')


if __name__ == '__main__':
    unittest.main()
