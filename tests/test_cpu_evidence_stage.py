"""CPU-only supervisor contracts with host/process boundaries mocked."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace


class CpuEvidenceStageContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec=importlib.util.spec_from_file_location('cpu_evidence_runner',Path(__file__).parents[1]/'scripts/run_cpu_evidence_stage.py')
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def args(self,root,stage='diagnostic'):
        return SimpleNamespace(analysis_root=root/'analysis',project_root=root,stage=stage,seconds=10,command=['python','check.py'])

    def test_cpu_environment_and_caps(self):
        env=self.runner.cpu_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'')
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):self.assertEqual(env[name],'4')
        self.assertEqual(env['PYTHONDONTWRITEBYTECODE'],'1')
        with tempfile.TemporaryDirectory() as temp:
            args=self.args(Path(temp));args.seconds=121
            with self.assertRaises(ValueError):self.runner.validate(args)
            args.seconds=10;args.command=['bash','check.sh']
            with self.assertRaises(ValueError):self.runner.validate(args)

    def test_old_stage_failure_and_cumulative_budget_block_launch(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root,'recompute');prior=args.analysis_root/'diagnostic';prior.mkdir(parents=True)
            receipt=prior/'stage.json';receipt.write_text(json.dumps({'stage':'diagnostic','status':'failed','exit_code':1,'elapsed_seconds':1}))
            before=receipt.read_bytes()
            with self.assertRaises(ValueError):self.runner.validate(args)
            self.assertEqual(receipt.read_bytes(),before)
            receipt.write_text(json.dumps({'stage':'diagnostic','status':'complete','exit_code':0,'elapsed_seconds':1495}))
            with self.assertRaises(ValueError):self.runner.validate(args)

    def test_rss_analysis_and_project_guards(self):
        for change in ({'rss_bytes':8*1024**3+1},{'analysis_bytes':128*1024**2+1},{'disk_free_bytes':500*1024**3-1},{'estimated_project_bytes':300*1024**3+1}):
            row=dict(rss_bytes=0,analysis_bytes=0,disk_free_bytes=600*1024**3,estimated_project_bytes=0);row.update(change)
            with self.assertRaises(RuntimeError):self.runner.check_resources(row)

    def test_deadline_failure_is_durable_before_owned_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root);observed=[]
            def expired(*_):raise self.runner.BudgetExpired('test deadline')
            def cleanup(process,deadline):observed.append(json.loads((args.analysis_root/'diagnostic/stage.json').read_text())['status'])
            process=SimpleNamespace(pid=12345,returncode=None,poll=lambda:None)
            with patch.object(self.runner,'project_bytes',return_value=0),patch.object(self.runner,'snapshot',return_value=dict(rss_bytes=0,analysis_bytes=0,disk_free_bytes=600*1024**3,estimated_project_bytes=0)), \
                 patch.object(self.runner.subprocess,'Popen',return_value=process),patch.object(self.runner,'wait_command',side_effect=expired), \
                 patch.object(self.runner,'stop_owned',side_effect=cleanup):
                with self.assertRaises(self.runner.BudgetExpired):self.runner.run_stage(args)
            self.assertEqual(observed,['budget_stopped'])
            state=json.loads((args.analysis_root/'diagnostic/stage.json').read_text())
            self.assertEqual(state['status'],'budget_stopped');self.assertEqual(state['attestation']['supervisor_model_calls'],0)

    def test_fresh_directory_race_loser_never_writes_foreign_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root);directory=args.analysis_root/'diagnostic';original=Path.mkdir
            def raced(path,*a,**kw):
                if path==directory:
                    original(path,*a,**kw);(path/'stage.json').write_text('foreign receipt');raise FileExistsError('race lost')
                return original(path,*a,**kw)
            row=dict(rss_bytes=0,analysis_bytes=0,disk_free_bytes=600*1024**3,estimated_project_bytes=0)
            with patch.object(Path,'mkdir',raced),patch.object(self.runner,'snapshot',return_value=row),patch.object(self.runner.subprocess,'Popen') as launch:
                with self.assertRaises(FileExistsError):self.runner.run_stage(args)
            launch.assert_not_called();self.assertEqual((directory/'stage.json').read_text(),'foreign receipt')
            self.assertEqual([path.name for path in directory.iterdir()],['stage.json'])

    def test_cleanup_refuses_a_foreign_group_identity(self):
        process=SimpleNamespace(pid=12345,poll=lambda:None)
        with patch.object(self.runner.os,'getpgid',return_value=98765),patch.object(self.runner.os,'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'refusing signal'):self.runner.stop_owned(process,1000000000000)
        kill.assert_not_called()
