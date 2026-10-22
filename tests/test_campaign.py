import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

_SPEC = importlib.util.spec_from_file_location('campaign', Path(__file__).parents[1] / 'scripts/run_qwen_campaign.py')
campaign = importlib.util.module_from_spec(_SPEC)
if _SPEC.loader:
    _SPEC.loader.exec_module(campaign)


class CampaignContracts(unittest.TestCase):
    def test_dual_selection_ignores_fast_single_and_unsafe_dual(self):
        rows = [dict(name='single', world_size=1, valid=True, throughput=999, peak_gib=30, min_free_gib=50),
                dict(name='dual2', world_size=2, valid=True, throughput=10, peak_gib=45, min_free_gib=20),
                dict(name='dual4', world_size=2, valid=True, throughput=12, peak_gib=55, min_free_gib=15),
                dict(name='unsafe', world_size=2, valid=True, throughput=99, peak_gib=61, min_free_gib=5)]
        self.assertEqual(campaign.choose_dual(rows)['name'], 'dual4')
        self.assertTrue(campaign.allow_batch8(rows[2]))
        self.assertFalse(campaign.allow_batch8(rows[3]))

    def test_unknown_failure_is_never_classified_as_recoverable_oom(self):
        self.assertFalse(campaign.is_oom('ValueError: CUDA visible devices invalid'))
        self.assertTrue(campaign.is_oom('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate'))
        self.assertFalse(campaign.is_oom('nan loss followed by CUDA out of memory'))
        self.assertFalse(campaign.is_oom('{"loss": NaN} then CUDA out of memory'))

    def test_generation_evidence_rejects_subset_or_truncated_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = root/'dev.jsonl'
            manifest.write_text(json.dumps(dict(id='a',text='HELLO'))+'\n')
            output = root/'predictions.jsonl'
            output.write_text(json.dumps(dict(id='a', reference='HELLO', hypothesis='HELLO'))+'\n')
            metrics = dict(wer=0,reference_words=1,examples=1,selected_ids=['a'],
                           manifest_sha256=campaign.file_hash(manifest),generation_truncated_count=0,
                           model_revision='pin')
            path=output.with_suffix('.metrics.json')
            path.write_text(json.dumps(metrics))
            self.assertEqual(campaign.validate_evaluation(path,manifest,'pin')['wer'],0)
            metrics['generation_truncated_count']=1
            path.write_text(json.dumps(metrics))
            with self.assertRaisesRegex(ValueError,'truncat'):
                campaign.validate_evaluation(path,manifest,'pin')

    def test_command_uses_current_python_module_and_fixed_global_batch(self):
        command=campaign.train_command('python-env',Path('config.yaml'),Path('run'),2,benchmark=True)
        self.assertEqual(command[:3],['python-env','-m','torch.distributed.run'])
        self.assertIn('--nproc_per_node=2',command)
        self.assertEqual(command[-6:],['--max-steps','40','--limit-train','2048','--limit-eval','8'])

    def test_memory_headroom_accepts_monitor_float_serialized_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            (root/'monitor').mkdir()
            (root/'monitor/resources.csv').write_text('gpu_index,gpu_memory_total_mib,gpu_memory_used_mib\n0.0,98304,61440\n1.0,98304,40960\n0.0,98304,65536\n')
            self.assertEqual(campaign.memory_headroom(root),32)

    def test_owned_process_group_cleanup_reaps_only_its_child(self):
        import subprocess
        import sys
        process = subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)
        try:
            campaign.stop_owned(process)
            self.assertIsNotNone(process.poll())
            self.assertFalse(campaign.group_alive(process.pid))
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_project_lock_prevents_a_second_campaign(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            first=campaign.Campaign(SimpleNamespace(project_root=root,campaign_dir=root/'first',max_wall_seconds=100))
            try:
                with self.assertRaises(BlockingIOError):
                    campaign.Campaign(SimpleNamespace(project_root=root,campaign_dir=root/'second',max_wall_seconds=100))
            finally:
                first.lock.close()
                first.global_lock.close()

    def test_main_storage_projection_counts_only_future_peak_checkpoint_writes(self):
        self.assertEqual(campaign.checkpoint_projection_gib(2349000000, 1),28)
        self.assertEqual(campaign.checkpoint_projection_gib(2349000000, 3),84)
