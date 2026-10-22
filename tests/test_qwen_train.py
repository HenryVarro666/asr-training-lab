import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from asr_lab.qwen_train import ExperimentCallback


class MetricContractTests(unittest.TestCase):
    def test_final_trainer_summary_does_not_duplicate_optimizer_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            callback = ExperimentCallback(run, {'parameters': 1}, None, {})
            callback.last_step = dict(step_seconds=2, data_seconds=32, examples=32)
            state = SimpleNamespace(global_step=40)
            callback.on_log(None, state, None, logs={'loss':1.2,'grad_norm':.7})
            callback.on_log(None, state, None, logs={'train_runtime':80,'train_loss':1.1})
            rows = [json.loads(line) for line in (run/'metrics.jsonl').read_text().splitlines()]
            self.assertEqual([row['phase'] for row in rows], ['train','summary'])
            self.assertEqual(sum(row['phase']=='train' for row in rows),1)
            self.assertEqual(rows[0]['examples'],32)
            self.assertNotIn('step_seconds',rows[1])

    def test_nonfinite_metric_fails_before_json_write(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            callback = ExperimentCallback(run, {'parameters':1},None,{})
            with self.assertRaisesRegex(RuntimeError,'Nonfinite metric'):
                callback.on_log(None,SimpleNamespace(global_step=1),None,logs={'loss':float('nan')})
            self.assertFalse((run/'metrics.jsonl').exists())


if __name__=='__main__':
    unittest.main()

class GenerationSerializationTests(unittest.TestCase):
    def test_greedy_generation_config_serializes_and_preserves_token_ids(self):
        from transformers import GenerationConfig
        from asr_lab.qwen_train import sanitize_generation_configs
        bad = GenerationConfig(do_sample=False, temperature=1e-6,
            eos_token_id=[151645,151643], pad_token_id=151643, bos_token_id=151643)
        model = SimpleNamespace(generation_config=bad,
            thinker=SimpleNamespace(generation_config=bad))
        sanitize_generation_configs(model)
        with tempfile.TemporaryDirectory() as temporary:
            model.generation_config.save_pretrained(temporary)
            loaded = GenerationConfig.from_pretrained(temporary)
        self.assertFalse(loaded.do_sample)
        self.assertEqual(loaded.temperature,1.0)
        self.assertEqual(loaded.eos_token_id,[151645,151643])
        self.assertEqual(loaded.pad_token_id,151643)
        self.assertEqual(model.thinker.generation_config.temperature,1.0)


def coordinated_creation_worker(rank, rendezvous, run_path, results):
    import torch.distributed as distributed
    from datetime import timedelta
    from asr_lab.qwen_train import prepare_run_directory
    distributed.init_process_group('gloo', init_method='file://' + rendezvous,
        rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        run = Path(run_path)
        prepare_run_directory(run)
        if rank == 0:
            (run/'run.json').write_text('{"existing":"preserved"}')
        distributed.barrier()
        # Both ranks already passed the only coordinated creation check.
        output = dict(created=(run/'status.json').is_file(), sees_metadata=(run/'run.json').is_file())
        try:
            prepare_run_directory(run)
            output['existing_rejected'] = False
        except RuntimeError as error:
            output['existing_rejected'] = 'Run already exists' in str(error)
        output['metadata_preserved'] = json.loads((run/'run.json').read_text()) == {'existing':'preserved'}
        (Path(results)/f'rank-{rank}.json').write_text(json.dumps(output))
    finally:
        distributed.destroy_process_group()


class DistributedCreationTests(unittest.TestCase):
    def test_two_rank_run_creation_is_coordinated_and_existing_failure_is_broadcast(self):
        import multiprocessing
        context = multiprocessing.get_context('spawn')
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            workers = [context.Process(target=coordinated_creation_worker,
                args=(rank,str(root/'rendezvous'),str(root/'run'),str(root))) for rank in range(2)]
            try:
                for worker in workers:
                    worker.start()
                for worker in workers:
                    worker.join(45)
                    self.assertEqual(worker.exitcode,0)
                for rank in range(2):
                    result=json.loads((root/f'rank-{rank}.json').read_text())
                    self.assertEqual(result,dict(created=True,sees_metadata=True,
                        existing_rejected=True,metadata_preserved=True))
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join(5)
