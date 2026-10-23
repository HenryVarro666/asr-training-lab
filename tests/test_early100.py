import random
from types import SimpleNamespace
import unittest
import numpy as np
import torch

from asr_lab.qwen_early100 import (
    save_policy, validate_schedule, validate_step, validate_config,
    capture_rng, restore_rng, rng_digest, PrefixCollator,
)


class EarlyScheduleContracts(unittest.TestCase):
    def test_original892_and45_warmup_are_required_not100(self):
        args=SimpleNamespace(max_steps=-1,get_warmup_steps=lambda total:45)
        validate_schedule(args,SimpleNamespace(max_steps=892,global_step=0))
        for total,warmup in [(100,5),(892,5)]:
            bad=SimpleNamespace(max_steps=-1,get_warmup_steps=lambda total:warmup)
            with self.assertRaises(ValueError):validate_schedule(bad,SimpleNamespace(max_steps=total,global_step=0))
        with self.assertRaises(ValueError):validate_schedule(SimpleNamespace(max_steps=100,get_warmup_steps=lambda n:45),SimpleNamespace(max_steps=892,global_step=0))

    def test_export25_50_and_full100_stop_policy(self):
        self.assertEqual(save_policy(24),dict(save=False,model_only=False,stop=False))
        self.assertEqual(save_policy(25),dict(save=True,model_only=True,stop=False))
        self.assertEqual(save_policy(50),dict(save=True,model_only=True,stop=False))
        self.assertEqual(save_policy(100),dict(save=True,model_only=False,stop=True))
        with self.assertRaises(ValueError):save_policy(101)

    def test_used_lr_and_consumption_mismatch_fail(self):
        row=dict(step=1,learning_rate=0.,examples=32,data_seconds=381.285,loss=.81,grad_norm=40.)
        self.assertEqual(validate_step(row,row,0.)['lr_abs_difference'],0.)
        with self.assertRaisesRegex(ValueError,'LR'):validate_step(row,row,2.2222222222222224e-7)
        for key,value in [('examples',31),('data_seconds',380.)]:
            changed=dict(row,**{key:value})
            with self.assertRaises(ValueError):validate_step(changed,row,0.)

    def test_only_wall_budget_may_change_in_original_config(self):
        original=dict(seed=42,batch_size=8,gradient_accumulation_steps=2,epochs=1,warmup_ratio=.05,max_wall_seconds=42040)
        validate_config(original,dict(original,max_wall_seconds=1200))
        for changed in [dict(original,batch_size=4),dict(original,max_steps=100),dict(original,limit_train=3200)]:
            with self.assertRaises(ValueError):validate_config(original,changed)

    def test_rng_can_be_restored_without_reseeding(self):
        before=capture_rng()
        fingerprint=rng_digest(before)
        random.random();np.random.random();torch.rand(7)
        self.assertNotEqual(fingerprint,rng_digest(capture_rng()))
        restore_rng(before)
        self.assertEqual(fingerprint,rng_digest(capture_rng()))

    def test_private_id_observation_does_not_change_collator_tensors(self):
        class Delegate:
            def __init__(self,processor):pass
            def __call__(self,rows):return {'input_ids':torch.tensor([[1,2]]),'labels':torch.tensor([[-100,2]])}
        collator=PrefixCollator(None,delegate_class=Delegate)
        batch=collator([dict(id='a')])
        self.assertEqual(batch.pop('_early100_ids'),['a'])
        torch.testing.assert_close(batch['input_ids'],torch.tensor([[1,2]]))
        torch.testing.assert_close(batch['labels'],torch.tensor([[-100,2]]))


class RunnerContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import run_early100_control
        cls.runner=run_early100_control

    def test_stage_deadline_is_minimum_not_reset_total_budget(self):
        self.assertEqual(self.runner.bounded_deadline(100.,1000.,1200.),1000.)
        self.assertEqual(self.runner.bounded_deadline(100.,1000.,600.),700.)
        with self.assertRaises(self.runner.BudgetExpired):self.runner.bounded_deadline(1000.,1000.,600.)

    def test_original_epoch_complete_is_not_prefix_complete(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        with TemporaryDirectory() as temp:
            path=Path(temp)
            (path/'prefix-status.json').write_text(json.dumps(dict(status='complete',actual_step=100,original_total_steps=892,completed_epoch=True)))
            (path/'trajectory-validation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'bounded-prefix'):self.runner.check_terminal(path)

    def test_paired_eval_order_and_decode_must_match(self):
        row=dict(evaluation_batch_size=16,seed=42,normalizer='Whisper EnglishTextNormalizer',
            normalizer_revision='973afd24965f72e36ca33b3055d56a652f456b4d',generation_max_new_tokens=512,
            selected_ids=['a','b'],reference_words=12)
        self.runner.validate_decode(row,row)
        with self.assertRaises(ValueError):self.runner.validate_decode(dict(row,selected_ids=['b','a']),row)
        with self.assertRaises(ValueError):self.runner.validate_decode(dict(row,evaluation_batch_size=2),row)

    def test_checkpoint_mode_and_full_resume_requirements(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        from safetensors.torch import save_file
        from asr_lab.qwen_early100 import verify_checkpoint
        with TemporaryDirectory() as temp:
            path=Path(temp)
            for name in ('config.json','preprocessor_config.json'):(path/name).write_text('{}')
            (path/'trainer_state.json').write_text(json.dumps(dict(global_step=25,max_steps=892)))
            (path/'checkpoint.json').write_text(json.dumps(dict(global_step=25)))
            save_file({'weight':torch.zeros(2,dtype=torch.float32)},path/'model.safetensors')
            self.assertTrue(verify_checkpoint(path,25,True)['model_only'])
            with self.assertRaisesRegex(ValueError,'Missing full recovery'):verify_checkpoint(path,25,False)
            torch.save({},path/'optimizer.pt')
            with self.assertRaisesRegex(ValueError,'unexpectedly'):verify_checkpoint(path,25,True)
            (path/'optimizer.pt').unlink()
            save_file({'weight':torch.zeros(2,dtype=torch.bfloat16)},path/'model.safetensors')
            with self.assertRaisesRegex(ValueError,'FP32'):verify_checkpoint(path,25,True)


def _rank_save_failure_worker(rank,rendezvous,result):
    """Exercise our real coordination wrapper with a fake native I/O failure, CPU/Gloo only."""
    import json
    from pathlib import Path
    from unittest.mock import patch
    import torch.distributed as dist
    from asr_lab.qwen_early100 import PrefixTrainer,OriginalTrainer
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        trainer=PrefixTrainer.__new__(PrefixTrainer)
        trainer.args=SimpleNamespace(load_best_model_at_end=False,metric_for_best_model=None)
        trainer.state=SimpleNamespace(best_global_step=None,global_step=25)
        trainer.is_deepspeed_enabled=False;trainer.is_fsdp_enabled=False
        def native(*args,**kwargs):
            if rank==0:raise OSError('simulated rank0 write failure')
        with patch.object(OriginalTrainer,'_save_checkpoint',native):
            try:trainer._save_checkpoint(None,None)
            except RuntimeError as error:
                Path(result+str(rank)).write_text(json.dumps(dict(error=str(error))))
            else:raise AssertionError('Rank did not receive coordinated failure')
    finally:dist.destroy_process_group()


class DistributedSaveContract(unittest.TestCase):
    def test_rank0_io_failure_reaches_both_ranks_without_hang(self):
        import json
        import multiprocessing
        from pathlib import Path
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as temp:
            directory=Path(temp)
            ctx=multiprocessing.get_context('spawn')
            processes=[ctx.Process(target=_rank_save_failure_worker,args=(rank,str(directory/'rendezvous'),str(directory/'result'))) for rank in (0,1)]
            try:
                for process in processes:process.start()
                for process in processes:process.join(30)
                self.assertFalse(any(process.is_alive() for process in processes),'Save failure synchronization hung')
                self.assertEqual([process.exitcode for process in processes],[0,0])
                for rank in (0,1):
                    self.assertIn('simulated rank0 write failure',json.loads((directory/f'result{rank}').read_text())['error'])
            finally:
                for process in processes:
                    if process.is_alive():process.terminate();process.join(5)


class CallbackBoundaryContract(unittest.TestCase):
    def test_callback_exports25_and_stops100_without_changing_eval_flag(self):
        from unittest.mock import patch
        from asr_lab.qwen_early100 import PrefixCallback,OriginalCallback
        for step in (25,50,100):
            callback=PrefixCallback.__new__(PrefixCallback)
            callback.trainer=SimpleNamespace(prefix_consumed=[['x']*8,['y']*8])
            callback.last_step=dict(examples=32,data_seconds=10.)
            callback.reference={step:dict(step=step,learning_rate=0.,examples=32,data_seconds=10.,loss=1.,grad_norm=1.)}
            callback.used_lr=0.;callback.config=dict(project_root='/unused');callback.checkpoint_bytes=1
            args=SimpleNamespace(save_only_model=False)
            state=SimpleNamespace(global_step=step)
            control=SimpleNamespace(should_save=False,should_training_stop=False,should_evaluate=False)
            scheduler=SimpleNamespace(last_epoch=step,get_last_lr=lambda:[0.])
            with patch.object(OriginalCallback,'on_step_end',lambda self,a,s,c,**kw:c), \
                 patch('asr_lab.qwen_early100.gather',side_effect=lambda value:[value,value]), \
                 patch('asr_lab.qwen_early100.frozen.rank_zero_action'):
                callback.on_step_end(args,state,control,lr_scheduler=scheduler)
            self.assertTrue(control.should_save)
            self.assertEqual(args.save_only_model,step!=100)
            self.assertEqual(control.should_training_stop,step==100)
            self.assertFalse(control.should_evaluate)
