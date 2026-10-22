import tempfile
import unittest
from pathlib import Path

import torch

from asr_lab.training_utils import mask_labels, validate_resume, update_groups, normalized_wer


class TrainingContracts(unittest.TestCase):
    def test_padding_mask_preserves_real_eos(self):
        ids = torch.tensor([[1, 7, 2, 2], [1, 8, 9, 2]])
        attention = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        got = mask_labels(ids, attention, decoder_start_token_id=1)
        self.assertEqual(got.tolist(), [[7, 2, -100], [8, 9, 2]])

    def test_decoder_start_removed_only_when_present_in_every_row(self):
        got = mask_labels(torch.tensor([[1, 2], [5, 2]]), torch.ones(2, 2), 1)
        self.assertEqual(got.tolist(), [[1, 2], [5, 2]])

    def test_resume_rejects_ddp_world_and_batch_change(self):
        base = dict(world_size=2, batch_size=4, gradient_accumulation_steps=4,
                    seed=42, train_manifest_sha256='abc', model_id='whisper', revision='sha')
        validate_resume(base, base)
        for key in ('world_size', 'batch_size', 'gradient_accumulation_steps', 'train_manifest_sha256'):
            changed = dict(base, **{key: 'changed'})
            with self.assertRaisesRegex(ValueError, key):
                validate_resume(base, changed)

    def test_partial_accumulation_group_and_resume_boundary(self):
        self.assertEqual(list(update_groups(10, 4, 0)), [(0, 4), (4, 4), (8, 2)])
        self.assertEqual(list(update_groups(10, 4, 4)), [(4, 4), (8, 2)])
        with self.assertRaises(ValueError):
            list(update_groups(10, 4, 3))

    def test_normalized_wer_keeps_raw_text_outside_normalizer(self):
        normalizer = lambda s: s.lower().replace(',', '').replace('.', '')
        references = ['HELLO, WORLD.']
        result = normalized_wer(references, ['Hello world'], normalizer)
        self.assertEqual(result['wer'], 0.0)
        self.assertEqual(result['reference_words'], 2)
        self.assertEqual(references, ['HELLO, WORLD.'])


class RecoveryContracts(unittest.TestCase):
    def test_checkpoint_rng_and_optimizer_restore_matches_uninterrupted_updates(self):
        import copy
        from asr_lab.train import rng_state, restore_rng
        torch.manual_seed(97)
        model = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 1))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / 10)
        x = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 12

        def step(net, opt, sched):
            opt.zero_grad()
            loss = net(x).square().mean()
            loss.backward()
            opt.step()
            sched.step()

        for _ in range(2):
            step(model, optimizer, scheduler)
        checkpoint = copy.deepcopy(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                        scheduler=scheduler.state_dict(), rng=rng_state()))
        for _ in range(2):
            step(model, optimizer, scheduler)
        resumed = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(.4), torch.nn.Linear(4, 1))
        resumed.load_state_dict(checkpoint['model'])
        opt = torch.optim.AdamW(resumed.parameters(), lr=.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1 - step / 10)
        opt.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        restore_rng(checkpoint['rng'])
        for _ in range(2):
            step(resumed, opt, sched)
        for expected, actual in zip(model.parameters(), resumed.parameters()):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
        self.assertEqual(scheduler.state_dict(), sched.state_dict())

    def test_distributed_sampler_resume_reconstructs_rank_order_without_overlap(self):
        from torch.utils.data import DistributedSampler
        dataset = list(range(33))
        orders = []
        for rank in range(2):
            sampler = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42, drop_last=True)
            sampler.set_epoch(3)
            uninterrupted = list(sampler)
            recreated = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42, drop_last=True)
            recreated.set_epoch(3)
            self.assertEqual(list(recreated)[8:], uninterrupted[8:])
            orders.append(set(uninterrupted))
        self.assertFalse(orders[0] & orders[1])

    def test_eval_subset_is_seeded_and_independent_of_manifest_order(self):
        import json
        from types import SimpleNamespace
        from asr_lab.data import ManifestDataset
        class Tokenizer:
            def __call__(self, text):
                return SimpleNamespace(input_ids=[1, 4, 2])
            def convert_tokens_to_ids(self, token):
                return 1
        records = [dict(id=f'utterance-{i:03}', text='A TEST', duration=2, sample_rate=16000) for i in range(30)]
        with tempfile.TemporaryDirectory() as root:
            paths = [Path(root) / 'forward.jsonl', Path(root) / 'reverse.jsonl']
            for path, rows in zip(paths, (records, reversed(records))):
                path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            forward = ManifestDataset(paths[0], Tokenizer(), limit=8)
            reverse = ManifestDataset(paths[1], Tokenizer(), limit=8)
            self.assertEqual(forward.report()['selected_ids'], reverse.report()['selected_ids'])
            self.assertNotEqual(forward.report()['selected_ids'], [r['id'] for r in records[:8]])


if __name__ == '__main__':
    unittest.main()
