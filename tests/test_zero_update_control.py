import json
from pathlib import Path
import tempfile
import unittest

import torch
from transformers import GenerationConfig, Trainer

from asr_lab.qwen_roundtrip_control import (
    canonical_tokens, fingerprint_model, effective_generation, compare_evaluations,
    GreedyTrace, freeze_receipt, file_hash, align_special_tokens,
)


class FingerprintContracts(unittest.TestCase):
    def test_all_parameter_aliases_and_only_persistent_buffers_are_recorded(self):
        model=torch.nn.Module()
        model.a=torch.nn.Parameter(torch.tensor([1.,2.]))
        model.b=model.a
        model.register_buffer('integer',torch.tensor([7],dtype=torch.int64))
        model.register_buffer('cache',torch.tensor([5.]),persistent=False)
        result=fingerprint_model(model)
        self.assertEqual(set(result['tensors']),{'a','b','integer'})
        self.assertEqual(result['tensors']['integer']['dtype'],'torch.int64')
        self.assertIn(['a','b'],result['aliases'])
        self.assertNotIn('cache',result['tensors'])
        old=result['tensors']['a']['sha256']
        with torch.no_grad(): model.a[0]=3
        self.assertNotEqual(old,fingerprint_model(model)['tensors']['a']['sha256'])

    def test_eval_projection_preserves_integer_dtype_and_compares_bf16_values(self):
        model=torch.nn.Module()
        model.register_buffer('count',torch.tensor([42],dtype=torch.int32))
        model.a=torch.nn.Parameter(torch.tensor([1.001],dtype=torch.float32))
        projected=fingerprint_model(model,torch.bfloat16)
        model.to(torch.bfloat16)
        actual=fingerprint_model(model)
        self.assertEqual(projected,actual)
        self.assertEqual(actual['tensors']['count']['dtype'],'torch.int32')

    def test_nonfinite_model_state_is_rejected(self):
        model=torch.nn.Linear(1,1)
        with torch.no_grad(): model.weight.fill_(float('nan'))
        with self.assertRaises(FloatingPointError): fingerprint_model(model)


class DecodeContracts(unittest.TestCase):
    def test_eos_is_retained_and_only_following_batch_padding_is_removed(self):
        self.assertEqual(canonical_tokens([9,151645,0,0],[151645,151643]),[9,151645])
        self.assertEqual(canonical_tokens([9,151643],[151645,151643]),[9,151643])
        self.assertEqual(canonical_tokens([9,8],[151645,151643]),[9,8])

    def test_greedy_ignores_only_inactive_sampling_and_still_compares_active_rules(self):
        first=GenerationConfig(do_sample=False,temperature=.000001,eos_token_id=[4,5])
        second=GenerationConfig(do_sample=False,temperature=1,eos_token_id=[4,5])
        self.assertEqual(effective_generation(first),effective_generation(second))
        second.eos_token_id=[4]
        self.assertNotEqual(effective_generation(first),effective_generation(second))

    def test_trace_does_not_change_scores_and_keeps_margin(self):
        trace=GreedyTrace()
        scores=torch.tensor([[1.,3.,2.],[4.,0.,1.]])
        result=trace(torch.tensor([[1],[2]]),scores)
        self.assertIs(result,scores)
        self.assertEqual(trace.steps[0][0]['top2_token_ids'],[1,2])
        self.assertEqual(trace.steps[0][0]['margin'],1)

    def test_token_mismatch_never_passes_even_with_other_matching_evidence(self):
        base=dict(model={'x':1},inputs=[1],generation=[1],loading_info={},predictions=[dict(id='a',token_ids=[1,2])])
        cand=dict(base,predictions=[dict(id='a',token_ids=[1,3])])
        result=compare_evaluations(base,cand)
        self.assertFalse(result['passed'])
        self.assertTrue(result['token_only_difference'])
        self.assertEqual(result['first_token_difference']['position'],1)

    def test_actual_trainer_token_alignment_handles_bos_pad_and_eos_lists(self):
        from types import SimpleNamespace
        processor=SimpleNamespace(eos_token_id=2,bos_token_id=None,pad_token_id=2)
        model=SimpleNamespace(config=SimpleNamespace(eos_token_id=1,bos_token_id=1,pad_token_id=0),
                              generation_config=GenerationConfig(eos_token_id=[1,2],bos_token_id=1,pad_token_id=0))
        trainer=SimpleNamespace(model=model,processing_class=processor)
        align_special_tokens(trainer)
        self.assertEqual(model.config.eos_token_id,2)
        self.assertEqual(model.generation_config.eos_token_id,[2,1,2])
        self.assertIsNone(model.config.bos_token_id)
        self.assertEqual(model.config.pad_token_id,2)


class ReceiptContracts(unittest.TestCase):
    def test_frozen_four_source_hashes_and_manifest_are_required(self):
        from asr_lab.qwen_data import MODEL_ID,MODEL_REVISION
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); repo=root/'repo'; (repo/'src/asr_lab').mkdir(parents=True)
            hashes={}
            for name in ('qwen_train.py','qwen_data.py','qwen_evaluate.py','benchmark.py'):
                path=repo/'src/asr_lab'/name;path.write_text(name)
                hashes['/frozen/repo/src/asr_lab/'+name]=file_hash(path)
            manifest=root/'dev-clean-fixed128.jsonl';manifest.write_text('{}\n')
            receipt=dict(source_sha256=hashes,manifests={'/frozen/dev-clean-fixed128.jsonl':file_hash(manifest)},
                         model_id=MODEL_ID,model_revision=MODEL_REVISION)
            path=root/'receipt.json';path.write_text(json.dumps(receipt))
            frozen=freeze_receipt(path,manifest,repo)
            self.assertEqual(len(frozen['source_sha256']),4)
            (repo/'src/asr_lab/qwen_data.py').write_text('changed')
            with self.assertRaisesRegex(ValueError,'source'):freeze_receipt(path,manifest,repo)


class StateChainContracts(unittest.TestCase):
    def evidence(self):
        return dict(model={'value':'same'},post_generation_model={'value':'same'},inputs=[],generation=[],
                    loading_info={},runtime={},predictions=[dict(id='a',token_ids=[1,2])])

    def test_prepare_projection_must_match_both_evaluation_models(self):
        base=self.evidence(); reload=self.evidence()
        self.assertTrue(compare_evaluations(base,reload,{'value':'same'})['passed'])
        result=compare_evaluations(base,reload,{'value':'different'})
        self.assertFalse(result['passed'])
        self.assertIn('prepared_bf16_projection_differs_from_base',result['differences'])
        self.assertIn('prepared_bf16_projection_differs_from_reload',result['differences'])

    def test_persistent_state_mutation_cannot_pass_even_when_both_sides_mutate_the_same(self):
        base=self.evidence(); reload=self.evidence()
        base['post_generation_model']={'value':'changed'}
        reload['post_generation_model']={'value':'changed'}
        result=compare_evaluations(base,reload,{'value':'same'})
        self.assertFalse(result['passed'])
        self.assertIn('base_persistent_state_changed_during_generation',result['differences'])
        self.assertIn('reload_persistent_state_changed_during_generation',result['differences'])
