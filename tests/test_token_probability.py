import unittest
import torch
from asr_lab.token_probability import (teacher_inputs,aligned_teacher_logits,check_generation_rules,
    row_statistics,verify_choices,persist_distributions,estimated_bytes)


class ProbabilityContracts(unittest.TestCase):
    def test_shift_first_last_and_single_eos_without_retokenization(self):
        original=dict(input_ids=torch.tensor([[10,11,12]]),attention_mask=torch.ones(1,3,dtype=torch.long),input_features=torch.zeros(1,2,3),feature_attention_mask=torch.ones(1,3))
        for targets in ([20,21,99],[99]):
            extended=teacher_inputs(original,targets)
            self.assertEqual(extended['input_ids'].tolist(),[[10,11,12]+targets[:-1]])
            self.assertIs(extended['input_features'],original['input_features'])
            logits=torch.arange(extended['input_ids'].shape[1]*4,dtype=torch.float32).reshape(1,-1,4)
            result=aligned_teacher_logits(logits,3,targets)
            torch.testing.assert_close(result,logits[0,2:2+len(targets)])
            self.assertEqual(result.untyped_storage().nbytes(),result.numel()*4)
            with self.assertRaises(ValueError):aligned_teacher_logits(logits[:,:-1],3,targets)
    def test_only_two_observation_flags_may_differ(self):
        old=dict(effective={'output_logits':None,'output_scores':False,'do_sample':False},special_tokens={'eos':[2,3]},logits_processor_types=[])
        new=dict(old,effective=dict(old['effective'],output_logits=True,output_scores=True))
        check_generation_rules(new,old)
        with self.assertRaises(ValueError):check_generation_rules(dict(new,effective=dict(new['effective'],do_sample=True)),old)
        with self.assertRaises(ValueError):check_generation_rules(dict(new,logits_processor_types=['custom']),old)
    def test_probability_eos_union_rank_ties_entropy_and_top10_ties(self):
        row=torch.tensor([2.,2.,0.,-1000.]+[1.]*10)
        result=row_statistics(row,0,[0,1,3],[0,1,1],lambda ids:[str(i) for i in ids])
        self.assertEqual(result['target']['rank'],1);self.assertEqual(result['target']['tie_count'],2)
        self.assertEqual([x['id'] for x in result['top10']],[0,1,4,5,6,7,8,9,10,11])
        self.assertAlmostEqual(result['eos']['total_prob'],sum(x['prob'] for x in result['eos']['individual']),places=7)
        self.assertEqual(result['eos']['ids'],[0,1])
        small=next(x for x in result['candidates'] if x['id']==3)
        self.assertEqual(small['prob'],0.);self.assertLess(small['logprob'],-900.)
        self.assertGreater(result['entropy'],0.)
    def test_actual_processed_argmax_and_missing_or_nonfinite_fail(self):
        raw=torch.tensor([[3.,3.,1.]])
        verify_choices(raw,raw,[0])
        with self.assertRaises(ValueError):verify_choices(raw,raw,[1])
        for bad in (float('nan'),float('inf'),float('-inf')):
            with self.assertRaises(FloatingPointError):row_statistics(torch.tensor([bad,1.]),1,[0],[1],lambda ids:['x']*len(ids))
    def test_equal_roles_share_storage_and_budget_counts_one_payload(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            path=Path(temp)/'matrix.pt';raw=torch.randn(3,7)
            receipt=persist_distributions(path,raw,raw.clone())
            saved=torch.load(path,weights_only=True)
            self.assertTrue(receipt['raw_scores_equal'])
            self.assertEqual(saved['raw_logits'].data_ptr(),saved['scores'].data_ptr())
            self.assertEqual(estimated_bytes(7,[3],gen_lengths=[3]),(3+3)*7*4)


class ProbabilitySupervisorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('probability_runner',Path(__file__).parents[1]/'scripts/run_token_probability.py')
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def test_three_workers_fixed_budget_and_fail_stop_without_retry(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        for problem in (None,'worker','validation','hash','resource'):
            with self.subTest(problem=problem),TemporaryDirectory() as temp:
                root=Path(temp);campaign=self.runner.ProbabilityCampaign.__new__(self.runner.ProbabilityCampaign)
                campaign.directory=root;campaign.root=root;campaign.shared={};campaign.deadline=10**20
                campaign.initial_project_bytes=0;campaign.vocab=20;campaign.stage_deadline=None
                campaign.reference={'shared_identity':{},'models':{stage:{'path':'/fixture/'+stage} for stage in self.runner.ORDER}}
                campaign.args=SimpleNamespace(trajectory_reference=root/'reference.json');campaign.state={}
                campaign.prepare=lambda:None;campaign.remaining=lambda:1000.;campaign.write_state=lambda:None
                calls=[]
                def execute(stage,command,run,gpu):
                    self.assertEqual(gpu,'1');self.assertEqual(command[1:3],['-m','asr_lab.token_probability'])
                    calls.append(stage)
                    if problem=='worker' and stage=='step050':raise RuntimeError('worker failed')
                campaign.execute=execute
                def verify(stage,*args):
                    self.assertIsNotNone(campaign.stage_deadline)
                    if problem=='hash':raise self.runner.BudgetExpired('hash deadline')
                    return {'files':{}}
                def validate(run,stage,reference,remaining,physical_gpu=1):
                    self.assertIsNotNone(campaign.stage_deadline)
                    if problem=='validation' and stage=='step050':raise ValueError('instrumented_trajectory_drift')
                    if problem=='resource' and stage=='step050':raise RuntimeError('storage limit')
                with patch.object(self.runner,'verify_model',side_effect=verify),patch.object(self.runner,'verify_stats'), \
                     patch.object(self.runner,'require_gpu_idle',return_value={'uuid':'fixture-GPU1'}),patch.object(self.runner,'validate_worker',side_effect=validate), \
                     patch.object(self.runner,'compare_saved',return_value={}):
                    if problem:
                        with self.assertRaises((RuntimeError,ValueError)):campaign.run()
                    else:
                        campaign.run();self.assertEqual((campaign.state['actual_generate_calls'],campaign.state['actual_teacher_forward_calls']),(3,9))
                self.assertEqual(calls,list(self.runner.ORDER) if problem is None else ([] if problem=='hash' else ['base','step050']))
                self.assertIsNone(campaign.stage_deadline)

    def test_expired_validation_and_summary_cannot_start_reading(self):
        from unittest.mock import patch
        from pathlib import Path
        def expired():raise self.runner.BudgetExpired('expired')
        with patch.object(self.runner,'read_json') as reader:
            for action in (lambda:self.runner.validate_worker(Path('/missing'),'base',{},expired),lambda:self.runner.compare_saved(Path('/missing'),expired)):
                with self.assertRaises(self.runner.BudgetExpired):action()
            reader.assert_not_called()

    def test_summary_rechecks_deadline_per_row(self):
        from pathlib import Path
        from unittest.mock import patch
        count=0
        def remaining():
            nonlocal count
            count+=1
            if count==4:raise self.runner.BudgetExpired('summary deadline')
        row={'prefix':{},'target_id':0,'candidates':[],'target':{'logprob':0.,'rank':1},'eos':{'total_prob':1.,'total_logprob':0.}}
        def read(path):
            if path.name=='raw-tokens.json':return {'continuation_token_ids':[[0]]}
            return {'rows':[row,row],'total_target_nll':0.,'mean_target_nll':0.}
        with patch.object(self.runner,'read_json',side_effect=read):
            with self.assertRaises(self.runner.BudgetExpired):self.runner.compare_saved(Path('/fixture'),remaining)
        self.assertEqual(count,4)

    def test_validation_detects_deadline_crossed_during_artifact_hash(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        required={'inputs.pt','inputs.json','gpu-preflight.json','metadata.json','generation/raw-tokens.json','generation/raw-tokens.pt',
            'generation/effective-generation.json','generation/distributions.pt','generation/distribution-manifest.json','generation/derived.json','generation/decoded.json','own-trajectory-comparison.json','prefix-forks.json'}
        for trajectory in self.runner.ORDER:
            required.update('teacher/'+trajectory+'/'+name for name in ('inputs.pt','inputs.json','distributions.pt','distribution-manifest.json','forward-receipt.json','derived.json'))
        with TemporaryDirectory() as temp:
            run=Path(temp)
            for name in required:
                path=run/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'x')
            result={'status':'complete','stage':'base','generate_calls':1,'teacher_forward_calls':3,'optimizer_updates':0,
                'artifacts':{name:{'bytes':1,'sha256':'hash'} for name in required}}
            metadata={'generate_calls':1,'teacher_forward_calls':3,'logits_collected':True}
            expired=False
            def remaining():
                if expired:raise self.runner.BudgetExpired('hash crossed stage deadline')
            def hash_file(path):
                nonlocal expired
                expired=True
                return 'hash'
            with patch.object(self.runner,'read_json',side_effect=[result,{'status':'complete'},metadata]),patch.object(self.runner,'file_hash',side_effect=hash_file) as hashes:
                with self.assertRaises(self.runner.BudgetExpired):self.runner.validate_worker(run,'base',{},remaining)
                self.assertEqual(hashes.call_count,1)

class ProbabilityLifecycleRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import importlib.util
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('probability_lifecycle_runner',Path(__file__).parents[1]/'scripts/run_token_probability.py')
        cls.runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(cls.runner)

    def args(self,root,budget=1):
        from types import SimpleNamespace
        return SimpleNamespace(project_root=root,campaign_dir=root/'fresh',run_dir=root/'fresh',max_wall_seconds=budget,
            trajectory_reference=root/'reference.json',upstream_source=root/'source.json',physical_gpu=1)

    def test_preworker_failure_report_has_zero_observed_and_no_archive_claim(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as temp:
            campaign=self.runner.ProbabilityCampaign.__new__(self.runner.ProbabilityCampaign)
            campaign.directory=Path(temp);campaign.state={'status':'failed','stage':'initializing','commands':[]}
            campaign.report();text=(campaign.directory/'REPORT.md').read_text()
            self.assertIn('Observed generate calls: 0',text)
            self.assertIn('Archived generation matrices: 0',text)
            self.assertNotIn('FP32 continuation rows are archived',text)

    def test_partial_report_uses_invocations_and_validated_manifest_files(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        from asr_lab.training_utils import atomic_json
        from asr_lab.training_utils import sha256_file
        with TemporaryDirectory() as temp:
            root=Path(temp);worker=root/'base/worker';generation=worker/'generation';generation.mkdir(parents=True)
            payload=generation/'distributions.pt';payload.write_bytes(b'fake tensor receipt fixture')
            atomic_json(worker/'metadata.json',{'generate_calls':1,'teacher_forward_calls':2})
            atomic_json(generation/'distribution-manifest.json',{'path':str(payload),'sha256':sha256_file(payload),'bytes':payload.stat().st_size})
            campaign=self.runner.ProbabilityCampaign.__new__(self.runner.ProbabilityCampaign)
            campaign.directory=root;campaign.state={'status':'failed','stage':'base','commands':[{'name':'base'}]}
            campaign.report();text=(root/'REPORT.md').read_text()
            self.assertIn('Observed generate calls: 1',text);self.assertIn('Observed teacher forward calls: 2',text)
            self.assertIn('Archived generation matrices: 1',text);self.assertIn('Archived teacher matrices: 0',text)
            payload.write_bytes(b'corrupt');campaign.report()
            self.assertIn('Archived generation matrices: 0',(root/'REPORT.md').read_text())

    def test_existing_directory_rejected_without_any_mutation(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root);args.run_dir.mkdir();(args.run_dir/'old').write_text('preserve')
            before={str(p.relative_to(root)):(p.stat().st_mtime_ns,p.read_bytes() if p.is_file() else None) for p in root.rglob('*')}
            with self.assertRaises(FileExistsError):self.runner.ProbabilityCampaign(args)
            after={str(p.relative_to(root)):(p.stat().st_mtime_ns,p.read_bytes() if p.is_file() else None) for p in root.rglob('*')}
            self.assertEqual(before,after)

    def test_slow_initial_tree_walk_is_bounded_and_receipted_without_worker(self):
        import time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root,.4);started=time.monotonic()
            # The read-only hardware boundary is stubbed; the real timer, lifecycle and locks run.
            with patch.object(self.runner,'tree_size',side_effect=lambda *_:(time.sleep(2) or 0)), \
                 patch.object(self.runner,'resource_snapshot',return_value={'gpus':[],'compute_apps':[],'disk_free_bytes':1},create=True), \
                 patch.object(self.runner.ProbabilityCampaign,'run') as run, \
                 patch('sys.argv',['run_token_probability.py','--run-dir',str(args.run_dir),'--project-root',str(root),'--max-wall-seconds','.4']):
                with self.assertRaises(self.runner.BudgetExpired):self.runner.main()
            self.assertLess(time.monotonic()-started,.65);run.assert_not_called()
            state=self.runner.read_json(args.run_dir/'campaign.json')
            self.assertEqual(state['status'],'budget_stopped');self.assertEqual(state['actual_generate_calls'],0)
            self.assertIn('cleanup_elapsed_seconds',state)
            monitor=self.runner.read_json(args.run_dir/'preflight-monitor/status.json')
            self.assertTrue(monitor['terminal']);self.assertEqual(monitor['campaign_status'],'budget_stopped')
            self.assertEqual(state['owned_processes_remaining'],[])

    def test_monitor_failure_is_preserved_in_terminal_receipt(self):
        import subprocess
        import sys
        import time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        real_popen=subprocess.Popen
        def exiting_sampler(*args,**kwargs):
            return real_popen([sys.executable,'-c','raise SystemExit(3)'],start_new_session=True)
        def finish(campaign):
            time.sleep(.1);campaign.state['status']='complete'
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root,1)
            with patch.object(self.runner,'resource_snapshot',return_value={'gpus':[],'compute_apps':[],'disk_free_bytes':1}), \
                 patch.object(self.runner,'tree_size',return_value=0),patch.object(self.runner,'check_space'), \
                 patch.object(self.runner.subprocess,'Popen',side_effect=exiting_sampler), \
                 patch.object(self.runner.ProbabilityCampaign,'run',finish):
                with self.assertRaisesRegex(RuntimeError,'monitor'):self.runner.run_campaign(args)
            state=self.runner.read_json(args.run_dir/'campaign.json');terminal=self.runner.read_json(args.run_dir/'preflight-monitor/status.json')
            self.assertEqual(state['status'],'failed');self.assertEqual(terminal['status'],'failed')
            self.assertTrue(terminal['terminal']);self.assertFalse(terminal['continuous_observation'])
            self.assertTrue(terminal['error'])

    def test_global_lock_failure_does_not_release_foreign_lock(self):
        import fcntl
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root,1);path=root/'campaign.lock'
            with path.open('w') as owner:
                owner.write('foreign owner');owner.flush();fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
                before=(path.stat().st_mtime_ns,path.read_bytes())
                with patch.object(self.runner,'resource_snapshot',return_value={}):
                    with self.assertRaises(BlockingIOError):self.runner.run_campaign(args)
                with path.open('a+') as contender:
                    with self.assertRaises(BlockingIOError):fcntl.flock(contender,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(before,(path.stat().st_mtime_ns,path.read_bytes()))
            self.assertEqual(self.runner.read_json(args.run_dir/'campaign.json')['status'],'failed')

    def test_worker_without_metadata_has_unknown_calls(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            campaign=self.runner.ProbabilityCampaign.__new__(self.runner.ProbabilityCampaign);campaign.directory=Path(temp)
            campaign.state={'status':'failed','stage':'base','commands':[{'name':'base','worker_started':True}]}
            campaign.report()
            self.assertIsNone(campaign.state['actual_generate_calls'])
            self.assertEqual(campaign.state['observed_evidence']['unknown_invocation_workers'],['base'])

    def test_foreign_compute_race_after_source_hash_prevents_worker_start(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            root=Path(temp);source=root/'source';source.write_bytes(b'identity')
            campaign=self.runner.ProbabilityCampaign.__new__(self.runner.ProbabilityCampaign)
            campaign.root=root;campaign.directory=root;campaign.state={'source_sha256':{str(source):'sha'},'gpu_uuid':'GPU-zero','commands':[]}
            campaign.remaining=lambda:100.;campaign.write_state=lambda:None
            events=[]
            def hashed(path):events.append('hash');return 'sha'
            def busy(*args,**kwargs):events.append('idle_gate');raise RuntimeError('foreign compute applications')
            with patch.object(self.runner,'file_hash',side_effect=hashed),patch.object(self.runner,'check_space'), \
                 patch.object(self.runner,'require_gpu_idle',side_effect=busy),patch.object(self.runner.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(RuntimeError,'foreign compute'):campaign.execute('base',['unused'],root/'worker','0')
            self.assertEqual(events,['hash','idle_gate']);launch.assert_not_called();self.assertFalse((root/'worker').exists())

    def test_existing_broken_run_symlink_is_preserved(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root);args.run_dir.symlink_to(root/'missing-target',target_is_directory=True)
            with self.assertRaises(FileExistsError):self.runner.ProbabilityCampaign(args)
            self.assertTrue(args.run_dir.is_symlink());self.assertFalse((root/'missing-target').exists())

    def test_real_readonly_sampler_starts_before_tree_and_samples_each_ten_seconds(self):
        import os
        import time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            root=Path(temp);binary=root/'bin';binary.mkdir();fake=binary/'nvidia-smi'
            fake.write_text('#!/bin/sh\ncase "$1" in\n --query-gpu=*) printf "0, GPU-zero, 0, 1, 80000\\n1, GPU-one, 0, 1, 80000\\n" ;;\n *) : ;;\nesac\n');fake.chmod(0o755)
            args=self.args(root,15)
            def tree(path):
                self.assertTrue((args.run_dir/'preflight-monitor/metrics.jsonl').exists())
                return 0
            def run(campaign):
                campaign.state['stage']='source_sha';campaign.write_state();time.sleep(10.15)
                campaign.remaining();campaign.state.update(status='complete',stage='test_lifecycle_only')
            with patch.dict(os.environ,{'PATH':str(binary)+os.pathsep+os.environ['PATH']}), \
                 patch.object(self.runner,'tree_size',side_effect=tree),patch.object(self.runner,'check_space'), \
                 patch.object(self.runner.ProbabilityCampaign,'run',run):
                self.runner.run_campaign(args)
            rows=[__import__('json').loads(line) for line in (args.run_dir/'preflight-monitor/metrics.jsonl').read_text().splitlines()]
            self.assertGreaterEqual(len(rows),3);self.assertEqual(rows[0]['phase'],'initial_resource_observation')
            middle=[row for row in rows[1:-1] if row['phase']=='source_sha'];self.assertTrue(middle)
            self.assertGreaterEqual(middle[0]['elapsed_seconds'],10)
            self.assertLess(middle[0]['elapsed_seconds'],11)
            self.assertTrue(rows[-1]['terminal']);self.assertEqual(rows[-1]['campaign_status'],'complete')
            self.assertEqual([gpu['uuid'] for gpu in rows[-1]['gpus']],['GPU-zero','GPU-one'])
            state=self.runner.read_json(args.run_dir/'campaign.json');self.assertEqual(state['owned_processes_remaining'],[])
            self.assertGreaterEqual(state['elapsed_seconds'],rows[-1]['elapsed_seconds'])

class SelectedProbabilityRuntime(unittest.TestCase):
    def test_gpu_selection_changes_only_physical_identity_field(self):
        from asr_lab.token_probability import input_gate
        inputs={'input_ids':[[1]],'attention_mask':[[1]],'prompt_padded_width':1,'effective_prompt_tokens':[1],'tensors':{}}
        reference={'common_input':inputs,'common_runtime_metadata':{'runtime':{'physical_gpu':'1','eval_dtype':'torch.bfloat16','attention':'sdpa'}}}
        metadata={'runtime':{'physical_gpu':'0','eval_dtype':'torch.bfloat16','attention':'sdpa'}}
        input_gate(inputs,metadata,reference,physical_gpu=0)
        with self.assertRaises(ValueError):input_gate(inputs,metadata,reference)
        with self.assertRaises(ValueError):input_gate(inputs,{'runtime':dict(metadata['runtime'],attention='eager')},reference,physical_gpu=0)
        self.assertEqual(reference['common_runtime_metadata']['runtime']['physical_gpu'],'1')

class ProbabilityAbsoluteBoundaryRegressions(unittest.TestCase):
    setUpClass=classmethod(ProbabilityLifecycleRegressions.setUpClass.__func__)
    args=ProbabilityLifecycleRegressions.args
    def test_finalize_error_rewrites_failure_instead_of_leaving_complete(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        def finish(campaign):
            campaign.state.update(status='models_complete_pending_cleanup',stage='models_complete');campaign.write_state()
        def failed_finalize(campaign):
            for child in campaign.owned:self.runner.stop_owned(child,campaign.deadline)
            campaign.owned=[]
            for handle in campaign.locks:handle.close()
            campaign.locks=[]
            raise OSError('final receipt disk error')
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root,1)
            with patch.object(self.runner,'resource_snapshot',return_value={}),patch.object(self.runner,'tree_size',return_value=0), \
                 patch.object(self.runner,'check_space'),patch.object(self.runner.ProbabilityCampaign,'run',finish), \
                 patch.object(self.runner.ProbabilityCampaign,'finalize',failed_finalize):
                with self.assertRaisesRegex(OSError,'final receipt'):self.runner.run_campaign(args)
            state=self.runner.read_json(args.run_dir/'campaign.json')
            self.assertEqual(state['status'],'failed');self.assertIn('final receipt disk error',state['error'])

    def test_script_absolute_cap_interrupts_slow_dependency_import(self):
        import os
        import subprocess
        import sys
        import time
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            root=Path(temp);hook=root/'sitecustomize.py';marker=root/'import-entered';run=root/'fresh'
            hook.write_text('import importlib.abc, importlib.util, pathlib, sys, time\n'
                'class Slow(importlib.abc.MetaPathFinder, importlib.abc.Loader):\n'
                ' def find_spec(self, fullname, path=None, target=None):\n'
                '  return importlib.util.spec_from_loader(fullname,self) if fullname=="run_qwen_campaign" else None\n'
                ' def create_module(self, spec): return None\n'
                f' def exec_module(self, module): pathlib.Path({str(marker)!r}).write_text("entered"); time.sleep(5)\n'
                'sys.meta_path.insert(0,Slow())\n')
            started=time.monotonic()
            result=subprocess.run([sys.executable,str(Path(self.runner.__file__)),'--run-dir',str(run),'--project-root',str(root),'--max-wall-seconds','.3'],
                env=dict(os.environ,PYTHONPATH=str(root)+os.pathsep+str(Path(self.runner.__file__).parents[1]/'src'),CUDA_VISIBLE_DEVICES=''),capture_output=True,text=True,timeout=7)
            elapsed=time.monotonic()-started
            self.assertTrue(marker.exists());self.assertNotEqual(result.returncode,0);self.assertLess(elapsed,1,result.stderr)
            state=self.runner.read_json(run/'campaign.json');self.assertEqual(state['status'],'budget_stopped')
            self.assertEqual(state['actual_generate_calls'],0);self.assertEqual(state['stage'],'importing_dependencies')
            self.assertTrue(self.runner.read_json(run/'preflight-monitor/status.json')['terminal'])

    def test_failed_complete_commit_leaves_durable_pending_when_failure_write_also_fails(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        original=self.runner.atomic_json;failed=False
        def disk(path,value):
            nonlocal failed
            if Path(path).name=='campaign.json' and value['status']=='complete':
                failed=True
                self.assertEqual(self.runner.read_json(path)['status'],'models_complete_pending_cleanup')
                raise OSError('complete commit denied')
            if failed and Path(path).name=='campaign.json':raise OSError('disk remains unavailable')
            original(path,value)
        def finish(campaign):
            campaign.state.update(status='models_complete_pending_cleanup',stage='models_complete');campaign.write_state()
        with TemporaryDirectory() as temp:
            root=Path(temp);args=self.args(root,1)
            with patch.object(self.runner,'resource_snapshot',return_value={}),patch.object(self.runner,'tree_size',return_value=0), \
                 patch.object(self.runner,'check_space'),patch.object(self.runner.ProbabilityCampaign,'run',finish),patch.object(self.runner,'atomic_json',side_effect=disk):
                with self.assertRaisesRegex(OSError,'complete commit denied'):self.runner.run_campaign(args)
            self.assertTrue(failed)
            self.assertEqual(self.runner.read_json(args.run_dir/'campaign.json')['status'],'models_complete_pending_cleanup')

    def test_initial_resource_deadline_keeps_budget_status_without_starting_worker(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            args=self.args(Path(temp),1)
            with patch.object(self.runner,'resource_snapshot',side_effect=self.runner.BudgetExpired('initial query cap')), \
                 patch.object(self.runner.subprocess,'Popen') as launch:
                with self.assertRaises(self.runner.BudgetExpired):self.runner.run_campaign(args)
            launch.assert_not_called()
            self.assertEqual(self.runner.read_json(args.run_dir/'campaign.json')['status'],'budget_stopped')
