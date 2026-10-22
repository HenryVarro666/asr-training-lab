import json
from pathlib import Path
import tempfile
import unittest

from asr_lab.benchmark import summarize_run, compare_runs


class BenchmarkContracts(unittest.TestCase):
    def make_run(self, directory, name, seconds, world_size):
        run = Path(directory) / name
        run.mkdir()
        (run / 'run.json').write_text(json.dumps(dict(effective_batch_size=32, world_size=world_size)))
        events = [dict(phase='baseline', step=0, elapsed_seconds=5)]
        events += [dict(phase='train', step=i, loss=1, grad_norm=1,
            data_seconds=320, step_seconds=100 if i <= 20 else seconds,
            cuda_peak_memory_gb=7) for i in range(1, 71)]
        (run / 'metrics.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in events))
        return run

    def test_warmup_excluded_and_fixed_global_scaling_computed(self):
        with tempfile.TemporaryDirectory() as root:
            one = self.make_run(root, 'one', 4, 1)
            two = self.make_run(root, 'two', 2, 2)
            single = summarize_run(one, 20, 50)
            self.assertEqual(single['measured_steps'], 50)
            self.assertEqual(single['step_seconds_p50'], 4)
            self.assertEqual(single['audio_seconds_per_wall_second'], 80)
            comparison = compare_runs(single, summarize_run(two, 20, 50))
            self.assertEqual(comparison['throughput_speedup'], 2)
            self.assertEqual(comparison['scaling_efficiency'], 1)

    def test_incomplete_or_duplicate_steps_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            run = self.make_run(root, 'one', 4, 1)
            with self.assertRaisesRegex(ValueError, '50'):
                summarize_run(run, 21, 50)
            with (run / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(phase='train', step=30)) + '\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                summarize_run(run, 20, 50)

    def test_comparison_rejects_different_global_batch(self):
        with self.assertRaisesRegex(ValueError, 'batch'):
            compare_runs(dict(effective_batch_size=32), dict(effective_batch_size=64))
