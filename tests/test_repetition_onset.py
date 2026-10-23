import unittest
from asr_lab.repetition_metrics import longest_periodic_run,repetition_observation,summarize_onset


class RepetitionContracts(unittest.TestCase):
    def test_arbitrary_periods_not_only_one_named_word(self):
        result=longest_periodic_run(['alpha','beta','gamma']*25,8)
        self.assertEqual(result['period'],3);self.assertEqual(result['repeat_count'],25)
        self.assertEqual(result['coverage'],1.)
    def test_short_genuine_repetition_is_not_strong(self):
        row=repetition_observation([7,8]*3,[151645],'very very good')
        self.assertFalse(row['strong_periodic_repetition']);self.assertFalse(row['budget_exhausted'])
    def test_eos_and_following_padding_are_excluded(self):
        row=repetition_observation([3,4,151645]+[151643]*509,[151645,151643],'plain text')
        self.assertEqual(row['analysis_token_count'],2)
        self.assertFalse(row['strong_periodic_repetition']);self.assertFalse(row['budget_exhausted'])
    def test_early_stop_strong_repeat_and_exhaustion_are_separate(self):
        early=repetition_observation([7,8]*25+[151645],[151645],'word pair '*25)
        self.assertTrue(early['strong_periodic_repetition']);self.assertFalse(early['budget_exhausted']);self.assertFalse(early['budget_loop'])
        loop=repetition_observation([11,12]*256,[151645,151643],'any unit '*100)
        self.assertTrue(loop['budget_loop'])
        long_nonrepeat=repetition_observation(list(range(512)),[151645,151643],' '.join('w'+str(i) for i in range(512)))
        self.assertTrue(long_nonrepeat['budget_exhausted']);self.assertFalse(long_nonrepeat['strong_periodic_repetition'])
    def test_coverage_and_twenty_period_thresholds(self):
        below=repetition_observation([1,2]*19+[151645],[151645],'plain text')
        self.assertFalse(below['strong_periodic_repetition'])
        diluted=repetition_observation(list(range(100,200))+[1,2]*20+[151645],[151645],'plain text')
        self.assertFalse(diluted['strong_periodic_repetition'])
    def test_sampled_earliest_does_not_assume_monotonicity_or_causal_onset(self):
        rows=[dict(stage=stage,step=step,strong_periodic_repetition=flag,budget_loop=flag) for stage,step,flag in
            [('base',0,False),('step025',25,True),('step050',50,False),('step100',100,True)]]
        result=summarize_onset(rows)
        self.assertEqual(result['strong_periodic_repetition']['earliest_observed_stage'],'step025')
        self.assertEqual(result['strong_periodic_repetition']['previous_sampled_negative_step'],0)
        self.assertTrue(result['strong_periodic_repetition']['observed_positive_then_negative'])
        self.assertFalse(result['exact_first_update_identified']);self.assertFalse(result['training_cause_identified'])


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import run_repetition_onset
        cls.runner=run_repetition_onset
    def metadata(self):
        return dict(runtime={'physical_gpu':'1'},generation_kwargs=self.runner.frozen.GENERATION_KWARGS,
            audio=[{'id':self.runner.frozen.TARGET}],model_class='official',processor_class='official',
            inference_mode=True,model_training=False,optimizer_updates=0,backward_calls=0,train_called=False,
            logits_collected=False,seed_reset=False)
    def write(self,path,value):
        import json
        path.write_text(json.dumps(value))
    def test_pregeneration_guard_pass_calls_original_once_same_args_and_failure_calls_zero(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        for drift in (False,True):
            with TemporaryDirectory() as temp:
                root=Path(temp);prior=root/'prior';output=root/'out';prior.mkdir();output.mkdir()
                for path in (prior,output):
                    self.write(path/'inputs.json',{'input':'same'});self.write(path/'metadata.json',self.metadata())
                if drift:self.write(output/'inputs.json',{'input':'different'})
                calls=[];sentinel=object();args=(object(),object(),output);kwargs={'opaque':object()}
                def original(*a,**kw):calls.append((a,kw));return sentinel
                identity={'onset':{'compare_dirs':[str(prior)]}}
                with patch.object(self.runner.frozen,'generate_and_persist_raw',original):
                    with self.runner.guarded_generator(identity,output) as receipt:
                        if drift:
                            with self.assertRaises(ValueError):self.runner.frozen.generate_and_persist_raw(*args,**kwargs)
                        else:self.assertIs(self.runner.frozen.generate_and_persist_raw(*args,**kwargs),sentinel)
                        inside_hash=self.runner.file_hash(output/'input-guard.json')
                    self.assertEqual(self.runner.file_hash(output/'input-guard.json'),inside_hash)
                    self.assertIs(self.runner.frozen.generate_and_persist_raw,original)
                self.assertEqual(len(calls),0 if drift else 1)
                self.assertEqual(receipt['actual_generate_calls'],len(calls))
                if calls:self.assertEqual(calls[0],(args,kwargs))
    def test_hash_checks_deadline_between_chunks(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            path=Path(temp)/'weight';path.write_bytes(b'x'*(3*1024**2));checks=[]
            def remaining():
                checks.append(True)
                if len(checks)==3:raise self.runner.BudgetExpired('hash deadline')
            with self.assertRaises(self.runner.BudgetExpired):self.runner.bounded_sha256(path,remaining)
            self.assertEqual(len(checks),3)
    def test_single_capture_base_provenance_and_later_stat_drift(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            root=Path(temp);model=root/'7278e1e70fe206f11671096ffdd38061171dd6e5';model.mkdir()
            weight=model/'model.safetensors';weight.write_bytes(b'fake CPU weight fixture')
            expected=dict(bytes=weight.stat().st_size,sha256=None,expected_origin='pending_single_capture_at_stage',historical_expected_sha256=None)
            definition=dict(path=str(model),revision=model.name,step=0,files={weight.name:expected})
            receipt=self.runner.verify_model('base',definition,lambda:100.,root/'verification.json')
            record=receipt['files'][weight.name]
            self.assertTrue(record['base_first_capture']);self.assertIsNone(record['expected_sha256'])
            self.assertEqual(record['actual_sha256'],self.runner.file_hash(weight))
            self.runner.verify_stats(definition,receipt);weight.write_bytes(b'changed')
            with self.assertRaises(ValueError):self.runner.verify_stats(definition,receipt)
    def test_four_calls_continue_observations_but_failure_or_drift_stops_following_models(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        from unittest.mock import patch
        for problem in (None,'inputs','runtime','rules','worker_failure','hash_timeout'):
            with self.subTest(problem=problem),TemporaryDirectory() as temp:
                root=Path(temp);prior=root/'prior';prior.mkdir()
                inputs={'tensors':'same'};metadata=self.metadata();rules=[dict(effective={'do_sample':False},special_tokens={'eos':[1,2]},logits_processor_types=['official'])]
                for name,value in [('inputs.json',inputs),('metadata.json',metadata),('effective-generation.json',rules)]:self.write(prior/name,value)
                reference_path=root/'reference.json';self.write(reference_path,{})
                campaign=self.runner.OnsetCampaign.__new__(self.runner.OnsetCampaign)
                campaign.directory=root;campaign.prior=prior;campaign.shared={};campaign.deadline=10**20
                campaign.reference={'shared_identity':{},'models':{stage:dict(path='/fixture/'+stage,step=step) for stage,step in zip(self.runner.ORDER,self.runner.STEPS)}}
                campaign.args=SimpleNamespace(model_reference=reference_path);campaign.state={'observations':[]}
                campaign.prepare=lambda:None;campaign.remaining=lambda:1000.;campaign.write_state=lambda:None
                calls=[]
                def execute(stage,command,run,gpu):
                    calls.append(stage)
                    if problem=='worker_failure' and stage=='step050':raise RuntimeError('worker failed')
                    run.mkdir(parents=True)
                    row=dict(inputs);meta=dict(metadata);rule=rules
                    if stage=='step025':
                        if problem=='inputs':row={'tensors':'different'}
                        elif problem=='runtime':meta=dict(meta,runtime={'physical_gpu':'0'})
                        elif problem=='rules':rule=[dict(rules[0],special_tokens={'eos':[1]})]
                    for name,value in [('inputs.json',row),('metadata.json',meta),('effective-generation.json',rule),('result.json',{'effective_eos_ids':[151645,151643]}),('input-guard.json',{'status':'input_gate_passed','actual_generate_calls':1})]:self.write(run/name,value)
                campaign.execute=execute
                def verify(stage,*a,**kw):
                    self.assertIsNotNone(campaign.stage_deadline)
                    if problem=='hash_timeout':raise self.runner.BudgetExpired('hash budget')
                    return {'files':{}}
                target=dict(token_ids=[7,8]*256,parser_input='any unit '*100,parser_output=['English','any unit'],counts={},repetition_fix_observation={})
                with patch.object(self.runner,'verify_model',side_effect=verify),patch.object(self.runner,'verify_stats'), \
                     patch.object(self.runner.frozen,'require_gpu1_idle',return_value={}),patch.object(self.runner,'validate_worker',return_value=[target]):
                    if problem:
                        with self.assertRaises((RuntimeError,ValueError,self.runner.BudgetExpired)):campaign.run()
                    else:campaign.run();self.assertEqual(campaign.state['actual_generate_calls'],4)
                expected=list(self.runner.ORDER) if problem is None else ([] if problem=='hash_timeout' else list(self.runner.ORDER[:3 if problem=='worker_failure' else 2]))
                self.assertEqual(calls,expected)

    def test_symlink_target_drift_is_saved_and_rejected(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            root=Path(temp);first=root/'blob1';second=root/'blob2'
            first.write_bytes(b'same content');second.write_bytes(b'same content')
            link=root/'model.safetensors';link.symlink_to(first)
            before=self.runner.file_stat(link)
            self.assertTrue(before['is_symlink']);self.assertEqual(before['symlink_target'],str(first))
            self.assertIn('logical_path',before);self.assertIn('symlink_lstat',before)
            link.unlink();link.symlink_to(second)
            with self.assertRaises(ValueError):self.runner.verify_stats({'path':str(root)},{'files':{link.name:{'before':before}}})
    def test_historical_checkpoint_sha_mismatch_rejected_before_content_read(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            root=Path(temp);weight=root/'model.safetensors';weight.write_bytes(b'fixture')
            marker=root/'prefix-complete.json'
            self.write(marker,{'status':'complete','step':25,'files':{weight.name:{'sha256':'historical'}}})
            model=dict(path=str(root),step=25,revision='7278e1e70fe206f11671096ffdd38061171dd6e5',historical_anchor={'prefix_complete':{'sha256':self.runner.file_hash(marker)}},
                files={weight.name:dict(bytes=weight.stat().st_size,sha256='different',expected_origin='historical_prefix_complete',historical_expected_sha256='historical')})
            with patch.object(self.runner,'bounded_sha256') as reader:
                with self.assertRaisesRegex(ValueError,'historical'):self.runner.verify_model('step025',model,lambda:100.,root/'verification.json')
                reader.assert_not_called()


class RepetitionTieContracts(unittest.TestCase):
    def test_period_tie_and_partial_period_are_deterministic(self):
        row=longest_periodic_run(['a','b']*20+['a'],16)
        self.assertEqual((row['period'],row['start'],row['repeat_count']),(2,0,20))
        self.assertEqual(row['following_partial_period'],['a'])
        self.assertFalse(row['partial_period_counted_as_complete'])
        tied=longest_periodic_run(['x']*40,16)
        self.assertEqual(tied['period'],1)
    def test_longest_run_agrees_with_brute_force_for_small_sequences(self):
        from itertools import product
        for length in range(8):
            for sequence in product((0,1),repeat=length):
                candidates=[]
                for period in range(1,min(4,length//2)+1):
                    for start in range(length):
                        unit=sequence[start:start+period];count=1
                        if len(unit)!=period:continue
                        while sequence[start+count*period:start+(count+1)*period]==unit:count+=1
                        if count>=2:candidates.append((count*period,period,start))
                expected=min(candidates,key=lambda item:(-item[0],item[1],item[2])) if candidates else (0,None,None)
                actual=longest_periodic_run(sequence,4)
                self.assertEqual((actual['span'],actual['period'],actual['start']),expected)
