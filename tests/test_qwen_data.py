import importlib.util
from pathlib import Path
import tempfile
import unittest

from asr_lab.qwen_data import LANGUAGE_PREFIX, build_causal_labels, prompt_text, select_records

HAS_TORCH = importlib.util.find_spec('torch') is not None


class SelectionTests(unittest.TestCase):
    def test_fixed_subset_covers_speakers_and_is_reproducible(self):
        rows = [dict(id=f'{speaker}-{index}',speaker_id=str(speaker)) for speaker in range(4) for index in range(10)]
        subset = select_records(rows,8,42)
        self.assertEqual(subset,select_records(list(reversed(rows)),8,42))
        self.assertEqual({row['speaker_id'] for row in subset},{'0','1','2','3'})
        self.assertEqual(len({row['id'] for row in subset}),8)
    def test_official_forced_language_prompt(self):
        class Processor:
            def apply_chat_template(self,messages,**kwargs):
                self_messages=messages
                assert messages[1]['content'][0]['type']=='audio'
                assert kwargs==dict(add_generation_prompt=True,tokenize=False)
                return 'CHAT_AUDIO_PREFIX'
        self.assertEqual(prompt_text(Processor(),True),'CHAT_AUDIO_PREFIX'+LANGUAGE_PREFIX)


@unittest.skipUnless(HAS_TORCH,'requires torch runtime')
class CausalMaskTests(unittest.TestCase):
    def test_prompt_and_padding_mask_preserves_eos_equals_pad(self):
        import torch
        # token9 is both EOS and padding. The real EOS remains supervised.
        ids=torch.tensor([[1,2,3,4,9],[1,2,5,9,9]])
        mask=torch.tensor([[1,1,1,1,1],[1,1,1,1,0]])
        prefix=torch.tensor([[1,2,0],[1,2,5]])
        prefix_mask=torch.tensor([[1,1,0],[1,1,1]])
        labels=build_causal_labels(ids,mask,prefix,prefix_mask)
        self.assertEqual(labels.tolist(),[[-100,-100,3,4,9],[-100,-100,-100,9,-100]])
        # Labels are not shifted; QwenForCausalLM performs its own causal shift.
        self.assertEqual(labels[0,2].item(),ids[0,2].item())
    def test_left_padding_is_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError,'RIGHT padding'):
            build_causal_labels(torch.tensor([[0,1,2,3]]),torch.tensor([[0,1,1,1]]),torch.tensor([[1,2]]),torch.tensor([[1,1]]))
    def test_changed_audio_prefix_is_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError,'exact tokenized'):
            build_causal_labels(torch.tensor([[1,2,3]]),torch.ones(1,3,dtype=torch.long),torch.tensor([[1,8]]),torch.ones(1,2,dtype=torch.long))
    def test_targetless_example_is_rejected(self):
        import torch
        with self.assertRaisesRegex(ValueError,'No supervised'):
            build_causal_labels(torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.long),torch.tensor([[1,2]]),torch.ones(1,2,dtype=torch.long))


if __name__=='__main__':
    unittest.main()
