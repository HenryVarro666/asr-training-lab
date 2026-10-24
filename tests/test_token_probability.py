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
                def validate(run,stage,reference,remaining):
                    self.assertIsNotNone(campaign.stage_deadline)
                    if problem=='validation' and stage=='step050':raise ValueError('instrumented_trajectory_drift')
                    if problem=='resource' and stage=='step050':raise RuntimeError('storage limit')
                with patch.object(self.runner,'verify_model',side_effect=verify),patch.object(self.runner,'verify_stats'), \
                     patch.object(self.runner,'require_gpu1_idle',return_value={}),patch.object(self.runner,'validate_worker',side_effect=validate), \
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
