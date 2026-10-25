"""CPU-only lifecycle/resource contract tests; nvidia-smi is the external boundary."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch


class ProbeSupervisionContracts(unittest.TestCase):
    def setUp(self):
        from asr_lab import probe_supervision
        self.module=probe_supervision

    def query(self,command,**kwargs):
        self.assertGreater(kwargs['timeout'],0)
        if '--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total' in command:
            output='0, GPU-zero, 0, 1, 80000\n1, GPU-one, 0, 1, 80000\n'
        else:output='GPU-zero, 999, PersonaPlex\n'
        return subprocess.CompletedProcess(command,0,output,'')

    def test_selected_gpu_rejects_foreign_personaplex_even_with_free_memory(self):
        with patch.object(self.module.subprocess,'run',side_effect=self.query):
            with self.assertRaisesRegex(RuntimeError,'compute applications'):
                self.module.require_gpu_idle(0,deadline=time.monotonic()+1)
            receipt=self.module.require_gpu_idle(1,deadline=time.monotonic()+1)
        self.assertEqual(receipt['uuid'],'GPU-one');self.assertEqual(receipt['physical_gpu'],1)

    def test_gpu_uuid_drift_and_visibility_mismatch_prevent_cuda_migration(self):
        with patch.object(self.module.subprocess,'run',side_effect=self.query):
            with self.assertRaisesRegex(ValueError,'UUID'):
                self.module.require_gpu_idle(1,deadline=time.monotonic()+1,expected_uuid='changed')
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'}):
            with self.assertRaisesRegex(ValueError,'visibility'):
                self.module.worker_gpu_preflight({'physical_gpu':0,'gpu_uuid':'GPU-zero'})

    def test_only_own_pid_can_be_ignored_by_selected_idle_gate(self):
        def query(command,**kwargs):
            result=self.query(command,**kwargs)
            if 'compute-apps' in command[1]:result.stdout=f'GPU-one, {os.getpid()}, current worker\n'
            return result
        with patch.object(self.module.subprocess,'run',side_effect=query):
            with self.assertRaises(RuntimeError):self.module.require_gpu_idle(1,deadline=time.monotonic()+1)
            self.assertEqual(self.module.require_gpu_idle(1,ignore_own=True,deadline=time.monotonic()+1)['ignored_own_pid'],os.getpid())

    def test_monitor_records_both_gpus_phase_elapsed_and_terminal_without_torch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);phase=root/'campaign.json';phase.write_text(json.dumps({'stage':'source_sha','status':'running'}))
            with patch.object(self.module.subprocess,'run',side_effect=self.query):
                snapshot=self.module.resource_snapshot(root,time.monotonic()+1)
            self.assertEqual([gpu['uuid'] for gpu in snapshot['gpus']],['GPU-zero','GPU-one'])
            self.assertEqual(snapshot['compute_apps'][0]['process_name'],'PersonaPlex')
            self.module.write_observation(root,snapshot,started=time.monotonic()-.1,phase='source_sha',campaign_status='budget_stopped',terminal=True)
            status=json.loads((root/'preflight-monitor/status.json').read_text())
            self.assertTrue(status['terminal']);self.assertGreater(status['elapsed_seconds'],0)
            self.assertEqual(status['phase'],'source_sha');self.assertEqual(status['campaign_status'],'budget_stopped')
            self.assertEqual(len((root/'preflight-monitor/metrics.jsonl').read_text().splitlines()),1)
        # A fresh Python interpreter importing the actual sampler must remain torch-free.
        result=subprocess.run([sys.executable,'-c','import sys; import asr_lab.probe_supervision; assert "torch" not in sys.modules'],capture_output=True,text=True,timeout=5)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_cleanup_kills_only_owned_session_within_remaining_cap(self):
        foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True)
        owned=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)'],start_new_session=True)
        try:
            time.sleep(.05);started=time.monotonic()
            self.module.stop_owned(owned,started+.4)
            self.assertLess(time.monotonic()-started,.5);self.assertIsNotNone(owned.poll());self.assertIsNone(foreign.poll())
        finally:
            for process in (foreign,owned):
                if process.poll() is None:os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=2)

    def test_cleanup_observation_failure_still_kills_and_reaps_owned_sampler(self):
        owned=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(10)'],start_new_session=True)
        try:
            time.sleep(.05)
            with patch.object(self.module,'group_alive',side_effect=PermissionError('ps denied')):
                with self.assertRaises(PermissionError):self.module.stop_owned(owned,time.monotonic()+.5)
            self.assertIsNotNone(owned.poll())
        finally:
            if owned.poll() is None:os.killpg(owned.pid,signal.SIGKILL)
            owned.wait(timeout=2)
