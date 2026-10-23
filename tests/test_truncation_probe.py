import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from asr_lab.qwen_truncation_probe import (select_probe_rows,token_evidence,capture_generation,
    generate_and_persist_raw,decode_records,GENERATION_KWARGS)


class ProbeContracts(unittest.TestCase):
    def reference(self):
        rows=[dict(manifest_row=dict(id=f'x{i:02d}')) for i in range(16)]
        rows[11]['manifest_row']['id']='652-129742-0006'
        return dict(rows=rows,ordered_ids=[x['manifest_row']['id'] for x in rows],target_id='652-129742-0006',target_index_in_batch=11)
    def test_exact_three_calls33_examples_and_order(self):
        ref=self.reference()
        first=select_probe_rows(ref,'batch16-first');repeat=select_probe_rows(ref,'batch16-repeat');single=select_probe_rows(ref,'target-single')
        self.assertEqual(first,repeat);self.assertEqual(len(first)+len(repeat)+len(single),33)
        self.assertEqual(single[0]['id'],ref['target_id'])
        ref['ordered_ids'].reverse()
        with self.assertRaises(ValueError):select_probe_rows(ref,'batch16-first')
    def test_eos_and_padding_retained_and_legacy_rule_is_per_row(self):
        ids=[7,151645,9,151643,151643]
        result=token_evidence(ids,[151645,151643],151643,{151645,151643})
        self.assertEqual(result['token_ids'],ids)
        self.assertEqual(result['effective_eos_positions'],[1,3,4]);self.assertEqual(result['pad_id_positions'],[3,4])
        self.assertFalse(result['legacy_generation_truncated'])
        self.assertTrue(token_evidence([7]*512,[151645,151643],151643,set())['legacy_generation_truncated'])
    def test_capture_returns_original_processors_and_never_inserts_one(self):
        returned=object();model=SimpleNamespace(thinker=SimpleNamespace())
        from transformers import GenerationConfig
        config=GenerationConfig(do_sample=False,eos_token_id=[4,5],pad_token_id=5)
        model.thinker._get_logits_processor=lambda *a,**kw:returned
        original=model.thinker._get_logits_processor;records=[]
        with capture_generation(model,records):
            self.assertIs(model.thinker._get_logits_processor(generation_config=config),returned)
        self.assertIs(model.thinker._get_logits_processor,original)
        self.assertEqual(records[0]['effective']['eos_token_id'],[4,5])
    def test_exact_kwargs_and_tokens_exist_before_decoder_failure(self):
        class Model:
            def generate(self,**kw):
                self.kw=kw
                return SimpleNamespace(sequences=torch.tensor([[10,11,151645,151643]]))
        with TemporaryDirectory() as temp:
            model=Model();inputs=dict(input_ids=torch.tensor([[10,11]]),attention_mask=torch.ones(1,2,dtype=torch.long))
            raw=generate_and_persist_raw(model,inputs,Path(temp))
            self.assertEqual({k:v for k,v in model.kw.items() if k not in inputs},GENERATION_KWARGS)
            self.assertTrue((Path(temp)/'raw-tokens.json').is_file());self.assertTrue((Path(temp)/'raw-tokens.pt').is_file())
            self.assertEqual(raw['continuation_token_ids'],[[151645,151643]])
            class Broken:
                def batch_decode(self,*a,**kw):raise ValueError('decoder failed')
            with self.assertRaises(ValueError):decode_records(Broken(),raw,[dict(id='a',text='X')],lambda *a,**kw:None,lambda s:s,[151645],151643)
            self.assertTrue((Path(temp)/'raw-tokens.json').is_file())
    def test_two_decode_modes_and_actual_parser_input(self):
        class Processor:
            tokenizer=SimpleNamespace(all_special_ids=[5],convert_ids_to_tokens=lambda ids:['piece']*len(ids))
            def batch_decode(self,ids,**kw):
                self.calls=getattr(self,'calls',[])+[kw]
                return ['AA AA AA' if kw['skip_special_tokens'] else 'AA AA AA<EOS>']
        processor=Processor();seen=[]
        def parser(raw,**kw):seen.append((raw,kw));return ('English','AA')
        raw=dict(continuation_token_ids=[[2,5,5]])
        result=decode_records(processor,raw,[dict(id='a',text='AA')],parser,lambda s:s.lower(),[5],5)
        self.assertEqual(seen,[('AA AA AA',dict(user_language='English'))])
        self.assertEqual(result[0]['decoded_keep_special'],'AA AA AA<EOS>')
        self.assertEqual(result[0]['parser_output'],['English','AA'])
        self.assertEqual([x['skip_special_tokens'] for x in processor.calls],[False,True])
        self.assertTrue(all(x['clean_up_tokenization_spaces'] is False for x in processor.calls))


class SupervisorContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import run_truncation_probe
        cls.runner=run_truncation_probe
    def test_identity_file_drift_is_rejected(self):
        from asr_lab.qwen_truncation_probe import verify_identity
        from asr_lab.training_utils import sha256_file
        with TemporaryDirectory() as temp:
            path=Path(temp)/'source';path.write_text('first')
            identity=dict(files={str(path):sha256_file(path)},packages={})
            verify_identity(identity);path.write_text('changed')
            with self.assertRaises(ValueError):verify_identity(identity)
    def test_worker_deadline_uses_minimum_with_total(self):
        from unittest.mock import patch
        campaign=self.runner.ProbeCampaign.__new__(self.runner.ProbeCampaign)
        campaign.stage_deadline=11.;campaign.next_space_check=100.
        with patch.object(self.runner.Campaign,'remaining',return_value=20.),patch.object(self.runner.time,'monotonic',return_value=10.):
            self.assertEqual(campaign.remaining(),1.)
        with patch.object(self.runner.Campaign,'remaining',return_value=20.),patch.object(self.runner.time,'monotonic',return_value=12.):
            with self.assertRaises(self.runner.BudgetExpired):campaign.remaining()
    def test_worker_failure_or_incomplete_evidence_stops_remaining_stages(self):
        from unittest.mock import patch
        for error in (RuntimeError('OOM'),self.runner.BudgetExpired('timeout'),None):
            with TemporaryDirectory() as temp:
                campaign=self.runner.ProbeCampaign.__new__(self.runner.ProbeCampaign)
                campaign.root=Path(temp);campaign.directory=Path(temp);campaign.args=None
                campaign.deadline=10**20;campaign.state={};campaign.write_state=lambda:None;campaign.remaining=lambda:1000.
                calls=[]
                def execute(stage,*a,**kw):
                    calls.append(stage)
                    if error:raise error
                    # No mandatory outputs, despite a hypothetically successful process return.
                campaign.execute=execute
                with patch.object(self.runner,'check_space'),patch.object(self.runner,'prepare_identity',return_value={'files':{}}),patch.object(self.runner,'require_gpu1_idle',return_value={}):
                    with self.assertRaises((RuntimeError,self.runner.BudgetExpired,FileNotFoundError,ValueError)):campaign.run()
                self.assertEqual(calls,['batch16-first'])
    def test_partial_results_cannot_be_claimed_complete(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):self.runner.compare_workers(Path(temp))

    def test_expected_truncation_does_not_schedule_retry_or_fourth_call(self):
        from unittest.mock import patch
        with TemporaryDirectory() as temp:
            campaign=self.runner.ProbeCampaign.__new__(self.runner.ProbeCampaign)
            campaign.root=Path(temp);campaign.directory=Path(temp);campaign.args=None
            campaign.deadline=10**20;campaign.state={};campaign.write_state=lambda:None;campaign.remaining=lambda:1000.
            calls=[];campaign.execute=lambda stage,*a,**kw:calls.append(stage)
            summary=dict(status='three_preset_probes_complete',total_generate_calls=3,total_consumed_examples=33)
            with patch.object(self.runner,'check_space'),patch.object(self.runner,'prepare_identity',return_value={'files':{}}),patch.object(self.runner,'require_gpu1_idle',return_value={}), \
                 patch.object(self.runner,'validate_worker',return_value=[{'legacy_generation_truncated':True}]), \
                 patch.object(self.runner,'compare_workers',return_value=summary),patch.object(self.runner,'validate_repeat_identity',return_value={}):
                campaign.run()
            self.assertEqual(calls,['batch16-first','batch16-repeat','target-single'])
            self.assertEqual(campaign.state['status'],'complete')

    def test_gpu1_busy_guard_preserves_gpu0_and_rejects_other_gpu1_apps(self):
        from unittest.mock import patch
        from asr_lab.qwen_truncation_probe import require_gpu1_idle
        devices=SimpleNamespace(stdout='0, GPU-zero\n1, GPU-one\n')
        plex_only=SimpleNamespace(stdout='GPU-zero, 123, PersonaPlex\n')
        busy=SimpleNamespace(stdout='GPU-zero, 123, PersonaPlex\nGPU-one, 456, video-worker\n')
        with patch('asr_lab.qwen_truncation_probe.subprocess.run',side_effect=[devices,plex_only]):
            self.assertEqual(require_gpu1_idle()['uuid'],'GPU-one')
        with patch('asr_lab.qwen_truncation_probe.subprocess.run',side_effect=[devices,busy]):
            with self.assertRaisesRegex(RuntimeError,'456'):require_gpu1_idle()

    def test_identity_includes_chat_template_and_both_upstream_receipt_shapes(self):
        from asr_lab.training_utils import sha256_file
        import importlib.metadata
        with TemporaryDirectory() as temp:
            root=Path(temp).resolve();checkpoint=root/'checkpoint-100';checkpoint.mkdir()
            audio=root/'input.flac';audio.write_bytes(b'CPU fixture; not decoded by identity preflight')
            ids=[f'6000-0000-{i:04d}' for i in range(11)]+['652-129742-0006']+[f'7000-0000-{i:04d}' for i in range(4)]
            rows=[dict(id=identifier,audio_filepath=str(audio),text='TEXT',duration=1.,sha256='0'*64) for identifier in ids]
            manifest=root/'manifest.jsonl'
            manifest.write_text(''.join(json.dumps(dict(id=f'{i:04d}'))+'\n' for i in range(2192))+''.join(json.dumps(row)+'\n' for row in rows))
            reference=root/'reference.json';reference.write_text(json.dumps(dict(rows=[dict(manifest_row=row) for row in rows],ordered_ids=ids,target_id='652-129742-0006',target_index_in_batch=11,checkpoint_path=str(checkpoint),manifest_sha256=sha256_file(manifest),original_batch_index_zero_based=137)))
            model=checkpoint/'model.safetensors';model.write_bytes(b'header fixture only')
            template=checkpoint/'chat_template.jinja';template.write_text('{{prompt}}')
            marker=dict(status='complete',step=100,original_total_steps=892,model_only=False,resumable=True,files={path.name:dict(bytes=path.stat().st_size,sha256=sha256_file(path)) for path in (model,template)})
            (checkpoint/'prefix-complete.json').write_text(json.dumps(marker))
            (checkpoint/'checkpoint.json').write_text(json.dumps(dict(revision='7278e1e70fe206f11671096ffdd38061171dd6e5',global_step=100,source_sha256={},packages={'transformers':importlib.metadata.version('transformers')},package_source_sha256={})))
            spelling=root/'spelling.json';spelling.write_text('{}')
            normalizer=root/'normalizer.json';normalizer.write_text(json.dumps(dict(normalizer=dict(spelling_sha256=sha256_file(spelling),transformers=importlib.metadata.version('transformers')))))
            source=root/'upstream.py';source.write_text('# fixture')
            record=dict(path=str(source),sha256=sha256_file(source))
            generation=root/'generation.json';generation.write_text(json.dumps(dict(files=[record])))
            parser=root/'parser.json';parser.write_text(json.dumps(record))
            args=SimpleNamespace(reference_batch=reference,checkpoint=checkpoint,manifest=manifest,spelling=spelling,normalizer_receipt=normalizer,upstream_generation=generation,upstream_parser=parser)
            identity=self.runner.prepare_identity(args)
            self.assertIn(str(template),identity['files'])
            template.write_text('x'*template.stat().st_size)
            with self.assertRaisesRegex(ValueError,'identity changed'):self.runner.prepare_identity(args)
            template.unlink()
            with self.assertRaisesRegex(ValueError,'missing'):self.runner.prepare_identity(args)


class RawFailureEvidenceContracts(unittest.TestCase):
    def test_prefix_and_shape_failures_keep_complete_returned_sequences(self):
        for sequences in (torch.tensor([[99,88,7]]),torch.tensor([1,2,3])):
            with TemporaryDirectory() as temp:
                output=Path(temp)
                model=SimpleNamespace(generate=lambda **kw:SimpleNamespace(sequences=sequences))
                with self.assertRaises(ValueError):generate_and_persist_raw(model,dict(input_ids=torch.tensor([[10,11]])),output)
                saved=torch.load(output/'raw-tokens.pt',weights_only=True)
                torch.testing.assert_close(saved['sequences'],sequences)
                self.assertEqual(json.loads((output/'raw-tokens.json').read_text())['full_sequences'],sequences.tolist())


class RepeatIdentityGateContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import run_truncation_probe
        cls.runner=run_truncation_probe
    def test_input_or_effective_rule_drift_stops_before_single_tokens_and_raw_metadata_do_not(self):
        from unittest.mock import patch
        for drift in ('inputs','effective','special_tokens','logits_processor_types','raw','tokens'):
            with self.subTest(drift=drift),TemporaryDirectory() as temp:
                directory=Path(temp);calls=[]
                campaign=self.runner.ProbeCampaign.__new__(self.runner.ProbeCampaign)
                campaign.root=directory;campaign.directory=directory;campaign.args=None;campaign.deadline=10**20
                campaign.state={};campaign.write_state=lambda:None;campaign.remaining=lambda:1000.
                def execute(stage,*a,**kw):
                    calls.append(stage);run=directory/stage;run.mkdir()
                    inputs={'tensors':{'input_ids':{'sha256':'same'}}}
                    generation=[dict(effective={'do_sample':False},special_tokens={'_eos_token_tensor':[1,2]},logits_processor_types=['original'],raw={'temperature':1.})]
                    tokens={'full_sequences':[[1]]};decoded=[dict(parser_input='a',parser_output=['English','a'])]
                    if stage=='batch16-repeat':
                        if drift=='inputs':inputs['tensors']['input_ids']['sha256']='different'
                        elif drift=='effective':generation[0]['effective']['do_sample']=True
                        elif drift=='special_tokens':generation[0]['special_tokens']['_eos_token_tensor']=[1]
                        elif drift=='logits_processor_types':generation[0]['logits_processor_types']=['different']
                        elif drift=='raw':generation[0]['raw']['temperature']=2.
                        elif drift=='tokens':tokens['full_sequences']=[[2]];decoded[0]['parser_output']=['English','b']
                    for name,value in [('inputs.json',inputs),('effective-generation.json',generation),('raw-tokens.json',tokens),('decoded.json',decoded)]:
                        (run/name).write_text(json.dumps(value))
                campaign.execute=execute
                with patch.object(self.runner,'check_space'),patch.object(self.runner,'prepare_identity',return_value={'files':{}}), \
                     patch.object(self.runner,'require_gpu1_idle',return_value={}),patch.object(self.runner,'validate_worker'), \
                     patch.object(self.runner,'compare_workers',return_value={}):
                    if drift in ('inputs','effective','special_tokens','logits_processor_types'):
                        with self.assertRaisesRegex(ValueError,'drift'):campaign.run()
                        self.assertEqual(calls,['batch16-first','batch16-repeat'])
                    else:
                        campaign.run();self.assertEqual(calls,['batch16-first','batch16-repeat','target-single'])
