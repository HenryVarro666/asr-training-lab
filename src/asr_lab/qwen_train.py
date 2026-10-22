"""Pinned Qwen3-ASR full-parameter DDP SFT with FP32 optimizer state.

Outer-forward adaptation derived from Alibaba Qwen3-ASR official SFT,
commit7c6daf77a2421100f5fb066495372c00129d39ff.
Copyright 2026 The Alibaba Qwen team. SPDX-License-Identifier: Apache-2.0
"""
import argparse
from datetime import timedelta
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time

import torch
import torch.distributed as dist
from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments, set_seed
import yaml

from .prepare_data import check_space
from .qwen_data import (MODEL_ID, MODEL_REVISION, OFFICIAL_REVISION, QwenCollator,
                        QwenManifestDataset, local_model_path)
from .training_utils import atomic_json, sha256_file

_STOP = False
_RUN_DIR = None
_RUN_STARTED = False


def request_stop(*_):
    global _STOP
    terminal_failure = False
    if _RUN_DIR is not None:
        try:
            terminal_failure = json.loads((_RUN_DIR / 'status.json').read_text()).get('status') == 'failed'
        except (OSError, ValueError):
            pass
    if _STOP or terminal_failure:
        # A second signal is an elastic-launcher teardown, not a fresh checkpoint request.
        raise SystemExit(143)
    _STOP = True


def rank_zero_action(action):
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get('RANK', 0))
    error = [None]
    if rank == 0:
        try:
            action()
        except Exception as exception:
            error[0] = f'{type(exception).__name__}: {exception}'
    if dist.is_initialized():
        dist.broadcast_object_list(error, src=0)
    if error[0]:
        raise RuntimeError(error[0])


def prepare_run_directory(run, resume=False):
    def prepare():
        run.mkdir(parents=True, exist_ok=True)
        if (run / 'run.json').exists() and not resume:
            raise ValueError('Run already exists; use a new directory or explicit --resume')
        atomic_json(run / 'status.json', dict(status='running', phase='starting', time_unix=time.time()))
    rank_zero_action(prepare)


def sanitize_generation_configs(model):
    for component in (model, model.thinker):
        previous = getattr(component, 'generation_config', None)
        # Fresh defaults avoid official inference-only temperature=1e-6 with do_sample=False.
        tokens = {key: getattr(previous, key, None) for key in
                  ('bos_token_id', 'eos_token_id', 'pad_token_id', 'decoder_start_token_id')}
        component.generation_config = GenerationConfig(**tokens, do_sample=False)
        component.generation_config.validate()


def patch_outer_forward(model):
    if not hasattr(model, 'thinker'):
        raise ValueError('Expected official Qwen3-ASR model.thinker')
    # No **kwargs: Trainer must use standard microbatch mean-loss scaling.
    def forward(self, input_ids=None, attention_mask=None, input_features=None,
                feature_attention_mask=None, labels=None):
        return self.thinker(input_ids=input_ids, attention_mask=attention_mask,
            input_features=input_features, feature_attention_mask=feature_attention_mask,
            labels=labels, use_cache=False)
    model.__class__.forward = forward


class SafeTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        seconds = inputs.pop('_audio_seconds', None)
        examples = inputs.pop('_examples', None)
        if model.training and seconds is not None:
            self.audio_seconds_step = getattr(self, 'audio_seconds_step', 0.0) + float(seconds)
            self.examples_step = getattr(self, 'examples_step', 0.0) + float(examples)
        outputs = model(**inputs)
        finite = torch.isfinite(outputs.loss.detach()).to(dtype=torch.int32)
        if dist.is_initialized():
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
        if not bool(finite.item()):
            raise FloatingPointError('Nonfinite Qwen training/evaluation loss')
        return (outputs.loss, outputs) if return_outputs else outputs.loss


class ExperimentCallback(TrainerCallback):
    def __init__(self, run, metadata, processor, config, initial_wall=0):
        self.run, self.metadata, self.processor, self.config = run, metadata, processor, config
        self.started = time.monotonic()
        self.initial_wall = initial_wall
        self.checkpoint_bytes = metadata['parameters'] * 12 + 1024**3
    def elapsed(self):
        return self.initial_wall + time.monotonic() - self.started
    def on_train_begin(self, args, state, control, **kwargs):
        self.last_step_end = time.monotonic()
        return control
    def on_step_begin(self, args, state, control, **kwargs):
        self.trainer.audio_seconds_step = 0.0
        self.trainer.examples_step = 0.0
        torch.cuda.reset_peak_memory_stats(args.device)
        return control
    def on_step_end(self, args, state, control, **kwargs):
        seconds = torch.tensor(getattr(self.trainer, 'audio_seconds_step', 0.0), device=args.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(seconds, op=dist.ReduceOp.SUM)
        examples = torch.tensor(getattr(self.trainer, 'examples_step', 0.0), device=args.device, dtype=torch.float64)
        torch.cuda.synchronize(args.device)
        elapsed = torch.tensor(time.monotonic() - self.last_step_end, device=args.device, dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(examples, op=dist.ReduceOp.SUM)
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        peaks = torch.tensor([torch.cuda.max_memory_allocated(args.device),
                              torch.cuda.max_memory_reserved(args.device)], device=args.device, dtype=torch.float64) / 1024**3
        if dist.is_initialized():
            dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        if state.global_step == 1:
            optimizer = kwargs.get('optimizer')
            invalid = any(value[key].dtype != torch.float32 for value in optimizer.state.values()
                for key in ('exp_avg','exp_avg_sq') if key in value)
            bad = torch.tensor(int(invalid), device=args.device)
            if dist.is_initialized():
                dist.all_reduce(bad, op=dist.ReduceOp.MAX)
            if bool(bad.item()):
                raise RuntimeError('Adam moments are not FP32')
        elapsed = float(elapsed)
        self.last_step_end = time.monotonic()
        self.last_step = dict(step_seconds=elapsed, data_seconds=float(seconds), examples=int(examples),
            cuda_peak_memory_gb=float(peaks[0]), cuda_peak_reserved_gb=float(peaks[1]),
            audio_seconds_per_wall_second=float(seconds) / max(elapsed, 1e-6))
        stop = _STOP or self.elapsed() >= self.config.get('max_wall_seconds', 43200)
        if dist.is_initialized():
            value = torch.tensor(int(stop), device=args.device)
            dist.all_reduce(value, op=dist.ReduceOp.MAX)
            stop = bool(value.item())
        if stop:
            control.should_training_stop = True
            control.should_save = True
        if control.should_save:
            rank_zero_action(lambda: check_space(Path(self.config.get('project_root', '/mnt/raid/asr-lab')),
                self.config.get('reserve_gib', 500), self.config.get('quota_gib', 300), self.checkpoint_bytes))
        return control
    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        phase = 'dev' if 'eval_loss' in logs else ('train' if 'loss' in logs else 'summary')
        record = dict(logs, step=state.global_step, phase=phase,
                      time_unix=time.time(), elapsed_seconds=self.elapsed())
        if phase == 'train':
            record.update(getattr(self, 'last_step', {}))
        def write():
            for key, value in list(record.items()):
                if isinstance(value, torch.Tensor):
                    record[key] = value.item()
                if isinstance(record[key], float) and not math.isfinite(record[key]):
                    raise FloatingPointError(f'Nonfinite metric: {key}')
            with (self.run / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
            atomic_json(self.run / 'status.json', dict(status='running', **record))
        rank_zero_action(write)
        return control
    def on_save(self, args, state, control, **kwargs):
        def save_metadata():
            checkpoint = self.run / f'checkpoint-{state.global_step}'
            self.processor.save_pretrained(checkpoint)
            atomic_json(checkpoint / 'checkpoint.json', dict(self.metadata,
                global_step=state.global_step, epoch=state.epoch, wall_elapsed_seconds=self.elapsed(),
                resumable_optimizer=(checkpoint / 'optimizer.pt').exists()))
        rank_zero_action(save_metadata)
        return control


def _main():
    global _RUN_DIR, _RUN_STARTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--model-path')
    parser.add_argument('--resume', help='Explicit complete HF checkpoint directory')
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--limit-eval', type=int)
    args = parser.parse_args()
    _RUN_DIR = args.run_dir
    config = yaml.safe_load(Path(args.config).read_text())
    for key, value in (('max_steps', args.max_steps), ('limit_train', args.limit_train), ('eval_limit', args.limit_eval)):
        if value is not None:
            config[key] = value
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend='nccl', init_method='env://',
            timeout=timedelta(minutes=30), device_id=torch.device('cuda', local_rank))
    prepare_run_directory(args.run_dir, bool(args.resume))
    _RUN_STARTED = True
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    set_seed(config.get('seed', 42))
    check_space(Path(config.get('project_root', '/mnt/raid/asr-lab')),
                config.get('reserve_gib', 500), config.get('quota_gib', 300), 35 * 1024**3)
    model_path = local_model_path(args.model_path or config.get('model_id', MODEL_ID), config.get('revision', MODEL_REVISION))
    from qwen_asr import Qwen3ASRModel
    wrapper = Qwen3ASRModel.from_pretrained(model_path, dtype=torch.float32,
        device_map=None, attn_implementation='sdpa', token=False)
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)
    sanitize_generation_configs(model)
    model.config.use_cache = False
    model.thinker.config.use_cache = False
    for parameter in model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.float32:
            raise ValueError('Training parameters must remain FP32')
    train = QwenManifestDataset(config['train_manifest'], config.get('limit_train'), config.get('seed', 42))
    dev = QwenManifestDataset(config['eval_manifest'], config.get('eval_limit', 128), config.get('seed', 42))
    for key in ('id', 'speaker_id', 'sha256'):
        left = {row[key] for row in train.records if key in row}
        right = {row[key] for row in dev.records if key in row}
        if left & right:
            raise ValueError(f'Train/dev leakage on {key}')
    repository = Path(__file__).resolve().parents[2]
    source_hashes = {str(path.relative_to(repository)): sha256_file(path)
                     for path in sorted(Path(__file__).parent.glob('qwen*.py'))}
    for path in (Path(__file__).with_name('training_utils.py'), Path(__file__).with_name('prepare_data.py')):
        source_hashes[str(path.relative_to(repository))] = sha256_file(path)
    git_commit = subprocess.run(['git','-C',str(repository),'rev-parse','HEAD'], capture_output=True,text=True)
    git_status = subprocess.run(['git','-C',str(repository),'status','--porcelain'], capture_output=True,text=True)
    package_hashes = {}
    for label, cls in (('model',model.__class__),('processor',processor.__class__)):
        path = inspect.getsourcefile(cls)
        if path and Path(path).is_file():
            package_hashes[label] = sha256_file(path)
    metadata = dict(source_sha256=source_hashes, package_source_sha256=package_hashes,
        config_file_sha256=sha256_file(args.config),
        git_commit=git_commit.stdout.strip() if git_commit.returncode == 0 else None,
        git_dirty=bool(git_status.stdout.strip()) if git_status.returncode == 0 else None,
        model_id=config.get('model_id', MODEL_ID), revision=config.get('revision', MODEL_REVISION),
        official_code_revision=OFFICIAL_REVISION, model_path=model_path, world_size=world_size,
        effective_batch_size=int(config.get('batch_size',2))*int(config.get('gradient_accumulation_steps',8))*world_size,
        config=config, train_manifest_sha256=sha256_file(config['train_manifest']),
        eval_manifest_sha256=sha256_file(config['eval_manifest']),
        train_ids_sha256=__import__('hashlib').sha256('\n'.join(r['id'] for r in train.records).encode()).hexdigest(),
        eval_ids=[r['id'] for r in dev.records], parameters=sum(p.numel() for p in model.parameters()),
        precision='FP32 weights/gradients/Adam moments; BF16 autocast', attention='sdpa',
        loss_reduction='mean token CE within each microbatch; arithmetic mean over accumulation microbatches and DDP ranks',
        packages={name: importlib.metadata.version(name) for name in ('torch','transformers','accelerate','qwen-asr')})
    initial_wall = 0
    if args.resume:
        checkpoint = Path(args.resume)
        previous = json.loads((checkpoint / 'checkpoint.json').read_text())
        for key in ('model_id','revision','world_size','config','train_manifest_sha256','eval_manifest_sha256','train_ids_sha256'):
            if previous.get(key) != metadata.get(key):
                raise ValueError(f'Resume invariant differs: {key}')
        if not (checkpoint / 'optimizer.pt').exists() or not list(checkpoint.glob('rng_state*.pth')):
            raise ValueError('Checkpoint lacks optimizer or per-rank RNG state')
        initial_wall = previous.get('wall_elapsed_seconds', 0)
    def write_initial_metadata():
        preflight = args.run_dir / 'save_preflight'
        model.generation_config.save_pretrained(preflight / 'outer')
        model.thinker.generation_config.save_pretrained(preflight / 'thinker')
        processor.save_pretrained(preflight / 'processor')
        atomic_json(args.run_dir / 'run.json', metadata)
        atomic_json(args.run_dir / 'status.json', dict(status='running', phase='initializing', time_unix=time.time()))
    rank_zero_action(write_initial_metadata)
    training_args = TrainingArguments(output_dir=str(args.run_dir),
        per_device_train_batch_size=int(config.get('batch_size', 2)),
        gradient_accumulation_steps=int(config.get('gradient_accumulation_steps', 8)),
        per_device_eval_batch_size=int(config.get('eval_batch_size', 2)),
        num_train_epochs=float(config.get('epochs', 1)), max_steps=int(config.get('max_steps', -1)),
        learning_rate=float(config.get('learning_rate', 2e-5)), weight_decay=float(config.get('weight_decay', .01)),
        warmup_ratio=float(config.get('warmup_ratio', .03)), lr_scheduler_type='linear',
        max_grad_norm=float(config.get('max_grad_norm', 1)), optim='adamw_torch_fused',
        bf16=True, fp16=False, tf32=True, gradient_checkpointing=bool(config.get('gradient_checkpointing', True)),
        gradient_checkpointing_kwargs={'use_reentrant': False}, ddp_find_unused_parameters=False,
        remove_unused_columns=False, label_names=['labels'], prediction_loss_only=True,
        eval_strategy='steps', eval_steps=int(config.get('eval_steps', 200)),
        save_strategy='steps', save_steps=int(config.get('save_steps', 200)), save_total_limit=2,
        save_safetensors=True, logging_steps=int(config.get('log_steps', 1)), logging_first_step=True,
        logging_nan_inf_filter=False, report_to=['tensorboard'],
        logging_dir=str(args.run_dir / 'tensorboard'),
        dataloader_num_workers=int(config.get('num_workers', 4)), dataloader_pin_memory=True,
        dataloader_persistent_workers=bool(config.get('num_workers', 4)),
        seed=int(config.get('seed', 42)), data_seed=int(config.get('seed', 42)),
        ddp_timeout=1800, local_rank=local_rank)
    callback = ExperimentCallback(args.run_dir, metadata, processor, config, initial_wall)
    trainer = SafeTrainer(model=model, args=training_args, train_dataset=train, eval_dataset=dev,
        data_collator=QwenCollator(processor), processing_class=processor, callbacks=[callback])
    callback.trainer = trainer
    trainer.model_accepts_loss_kwargs = False
    try:
        result = trainer.train(resume_from_checkpoint=args.resume)
        trainer.save_state()
        terminal_checkpoint = args.run_dir / f'checkpoint-{trainer.state.global_step}'
        def verify_terminal_checkpoint():
            required = ('checkpoint.json', 'optimizer.pt', 'scheduler.pt', 'trainer_state.json')
            if not all((terminal_checkpoint / name).is_file() for name in required):
                raise RuntimeError(f'Terminal checkpoint lacks training state: {terminal_checkpoint}')
            if not list(terminal_checkpoint.glob('*.safetensors')) or not list(terminal_checkpoint.glob('rng_state*.pth')):
                raise RuntimeError(f'Terminal checkpoint lacks weights/RNG: {terminal_checkpoint}')
        rank_zero_action(verify_terminal_checkpoint)
        if rank == 0:
            step_target = int(config.get('max_steps', -1))
            epoch_target = float(config.get('epochs', 1))
            epoch = float(trainer.state.epoch or 0)
            achieved_steps = step_target > 0 and trainer.state.global_step >= step_target
            achieved_epochs = step_target <= 0 and epoch >= epoch_target - 1e-6
            complete = achieved_steps or achieved_epochs
            atomic_json(args.run_dir / 'status.json', dict(status='complete' if complete else 'interrupted',
                completion_reason='max_steps' if achieved_steps else ('epochs' if achieved_epochs else 'wall_or_signal'),
                epoch=epoch, target_epochs=epoch_target,
                step=trainer.state.global_step, elapsed_seconds=callback.elapsed(), metrics=result.metrics,
                latest_checkpoint=str(max(args.run_dir.glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1])))
                    if list(args.run_dir.glob('checkpoint-*')) else None))
    except BaseException as error:
        if rank == 0:
            atomic_json(args.run_dir / 'status.json', dict(status='failed', error=f'{type(error).__name__}: {error}',
                step=trainer.state.global_step, time_unix=time.time()))
        raise


def main():
    try:
        _main()
    except BaseException as error:
        if _RUN_STARTED and int(os.environ.get('RANK', 0)) == 0:
            try:
                previous = json.loads((_RUN_DIR / 'status.json').read_text())
                atomic_json(_RUN_DIR / 'status.json', dict(previous, status='failed',
                    error=f'{type(error).__name__}: {error}', time_unix=time.time()))
            except (OSError, ValueError):
                pass
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
