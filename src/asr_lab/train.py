"""Full Whisper finetuning with a minimal resumable PyTorch DDP loop."""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import WhisperForConditionalGeneration, WhisperProcessor
import yaml

from .data import ManifestDataset, WhisperCollator
from .evaluate import autocast_context, evaluate_model
from .training_utils import atomic_json, sha256_file, update_groups, validate_resume

_STOP = False


def request_stop(*_):
    global _STOP
    _STOP = True


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'])


def rank_zero_action(fn, rank, world_size):
    """Every rank calls; propagate a rank-zero I/O/evaluation failure to its peers."""
    payload = [None]
    if rank == 0:
        try:
            payload[0] = dict(ok=True, result=fn())
        except Exception as error:
            payload[0] = dict(ok=False, error=f'{type(error).__name__}: {error}')
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if not payload[0]['ok']:
        raise RuntimeError(payload[0]['error'])
    return payload[0]['result']


def checkpoint_model(model, processor, optimizer, scheduler, run_dir, step, epoch, next_batch,
                     best_wer, is_best, signature, rank, world_size, limit, wall_elapsed_seconds=0.0):
    states = [None] * world_size
    if world_size > 1:
        dist.all_gather_object(states, rng_state())
    else:
        states[0] = rng_state()

    def save():
        root = run_dir / 'checkpoints'
        root.mkdir(exist_ok=True)
        name = f'step-{step:08d}'
        temporary = root / (name + '.tmp')
        destination = root / name
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir()
        model.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        torch.save(dict(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                        global_step=step, epoch=epoch, next_batch=next_batch, best_wer=best_wer,
                        signature=signature, rng_by_rank=states, wall_elapsed_seconds=wall_elapsed_seconds), temporary / 'training_state.pt')
        atomic_json(temporary / 'checkpoint.json', dict(global_step=step, epoch=epoch,
                    next_batch=next_batch, best_wer=best_wer, signature=signature))
        if destination.exists():
            raise FileExistsError(f'Refusing to overwrite checkpoint: {destination}')
        os.replace(temporary, destination)
        for pointer in ('last', 'best') if is_best else ('last',):
            link = root / (pointer + '.tmp')
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(name, target_is_directory=True)
            os.replace(link, root / pointer)
        protected = {(root / pointer).resolve() for pointer in ('last', 'best') if (root / pointer).exists()}
        checkpoints = sorted((p for p in root.glob('step-*') if p.is_dir() and not p.name.endswith('.tmp')), reverse=True)
        keep = set(protected)
        for path in checkpoints:
            if len(keep) < limit:
                keep.add(path.resolve())
        for path in checkpoints:
            if path.resolve() not in keep:
                shutil.rmtree(path)
        return str(destination)
    return rank_zero_action(save, rank, world_size)


def make_signature(config, world_size, total_steps, warmup_steps):
    return dict(config, world_size=world_size, total_steps=total_steps, warmup_steps=warmup_steps,
                train_manifest_sha256=sha256_file(config['train_manifest']),
                eval_manifest_sha256=sha256_file(config['eval_manifest']))


def main():
    invocation_started = time.monotonic()
    prior_wall_seconds = 0.0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--limit-train', type=int)
    parser.add_argument('--limit-eval', type=int)
    parser.add_argument('--resume', help='Checkpoint directory containing model and training_state.pt')
    args = parser.parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    config['limit_train'] = args.limit_train
    config['limit_eval'] = args.limit_eval if args.limit_eval is not None else config.get('eval_limit', 128)
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError('--max-steps must be positive')
    if not 2 <= int(config.get('checkpoint_limit', 3)) <= 3:
        raise ValueError('checkpoint_limit must be 2 or 3')
    for key in ('batch_size', 'gradient_accumulation_steps', 'epochs', 'eval_steps', 'eval_batch_size'):
        if int(config[key]) < 1:
            raise ValueError(f'{key} must be positive')
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    if config['precision'] == 'bf16' and device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requested bf16 is unsupported on this GPU')
    if world_size > 1:
        dist.init_process_group(backend='nccl' if device.type == 'cuda' else 'gloo', timeout=timedelta(hours=2))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    seed = int(config['seed'])
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True
    run_dir = Path(args.run_dir).resolve()

    def init_run():
        run_dir.mkdir(parents=True, exist_ok=True)
        if (run_dir / 'run.json').exists() and not args.resume:
            raise FileExistsError('Run already exists. Use a new --run-dir or --resume.')
    rank_zero_action(init_run, rank, world_size)
    # Trust only the public base model and local checkpoint, never remote custom code.
    model_source = args.resume or config['model_id']
    processor = WhisperProcessor.from_pretrained(model_source, revision=config['revision'], token=False, language='english', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(
        model_source, revision=config['revision'], token=False, attn_implementation=config.get('attention', 'sdpa'),
    ).to(device)
    resolved_revision = getattr(model.config, '_commit_hash', None) or config['revision']
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False
    if config.get('gradient_checkpointing'):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    train_data = ManifestDataset(config['train_manifest'], processor.tokenizer,
        config['max_audio_seconds'], config['max_label_length'], config['limit_train'])
    eval_data = ManifestDataset(config['eval_manifest'], processor.tokenizer,
        config['max_audio_seconds'], config['max_label_length'], config['limit_eval'])
    if set(r['id'] for r in train_data.records) & set(r['id'] for r in eval_data.records):
        raise ValueError('Training and evaluation utterance IDs overlap')
    collator = WhisperCollator(processor, model.config.decoder_start_token_id)
    sampler = DistributedSampler(train_data, num_replicas=world_size, rank=rank, shuffle=True,
                                 seed=seed, drop_last=True)
    generator = torch.Generator().manual_seed(seed + rank)
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], sampler=sampler,
        drop_last=False, num_workers=config['num_workers'], collate_fn=collator,
        pin_memory=device.type == 'cuda', generator=generator,
        persistent_workers=config['num_workers'] > 0)
    if not len(train_loader):
        raise ValueError('Training subset is smaller than the world size')
    eval_loader = DataLoader(eval_data, batch_size=config['eval_batch_size'], shuffle=False,
        num_workers=config['num_workers'], collate_fn=collator, pin_memory=device.type == 'cuda',
        generator=torch.Generator().manual_seed(seed), persistent_workers=config['num_workers'] > 0) if rank == 0 else None
    accumulation = config['gradient_accumulation_steps']
    epoch_steps = math.ceil(len(train_loader) / accumulation)
    planned_steps = epoch_steps * config['epochs']
    total_steps = min(args.max_steps, planned_steps) if args.max_steps is not None else planned_steps
    warmup_steps = int(config.get('warmup_steps', math.ceil(total_steps * config.get('warmup_ratio', 0.05))))
    signature = make_signature(config, world_size, total_steps, warmup_steps)
    fused_adamw_active = bool(config.get('fused_adamw', False) and device.type == 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                  weight_decay=config['weight_decay'], fused=fused_adamw_active)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step, start_epoch, next_batch = 0, 0, 0
    best_wer = None
    resume_rng = None
    if args.resume:
        state = torch.load(Path(args.resume) / 'training_state.pt', map_location='cpu', weights_only=False)
        validate_resume(state['signature'], signature)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        global_step, start_epoch, next_batch = state['global_step'], state['epoch'], state['next_batch']
        best_wer = state['best_wer']
        prior_wall_seconds = state.get('wall_elapsed_seconds', 0.0)
        resume_rng = state['rng_by_rank'][rank]
        if global_step >= total_steps:
            raise ValueError('Checkpoint has already reached the configured total_steps')
    if world_size > 1:
        wrapped = DDP(model, device_ids=[local_rank] if device.type == 'cuda' else None, broadcast_buffers=False, gradient_as_bucket_view=True)
    else:
        wrapped = model
    writer = SummaryWriter(str(run_dir / 'tensorboard')) if rank == 0 else None

    def log(event):
        event.update(time_unix=time.time())
        if rank == 0:
            with (run_dir / 'metrics.jsonl').open('a') as stream:
                stream.write(json.dumps(event, allow_nan=False) + '\n')
            for key, value in event.items():
                if isinstance(value, (int, float)) and key != 'time_unix':
                    writer.add_scalar(f'{event["phase"]}/{key}', value, event.get('step', global_step))
            writer.flush()
            print(json.dumps(event, allow_nan=False), flush=True)
            atomic_json(run_dir / 'status.json', dict(status='running', phase=event['phase'],
                global_step=global_step, last_update_unix=time.time()))

    def write_metadata():
        try:
            git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
            git_dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True, stderr=subprocess.DEVNULL).strip())
        except (subprocess.CalledProcessError, FileNotFoundError):
            git_hash, git_dirty = None, None
        packages = {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'numpy', 'soundfile', 'jiwer', 'tensorboard')}
        metadata = dict(config=config, signature=signature, resolved_model_revision=resolved_revision,
            git_hash=git_hash, git_dirty=git_dirty, versions=packages, cuda=torch.version.cuda,
            source_sha256={name: sha256_file(Path(__file__).parent / name) for name in
                           ('train.py', 'data.py', 'evaluate.py', 'training_utils.py')},
            world_size=world_size, effective_batch_size=config['batch_size'] * accumulation * world_size,
            train_batches_per_rank=len(train_loader), updates_per_epoch=epoch_steps,
            planned_steps=total_steps, resume_from=args.resume,
            gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            deterministic_note='Fixed seeds and sampler order; CUDA kernels are not guaranteed bitwise deterministic.',
            architecture='unmodified openai/whisper-small; full parameter finetuning',
            objective='DDP average of microbatch mean token cross-entropies; equal microbatch weights per update',
            reported_loss='Global valid-token-weighted cross-entropy for monitoring',
            selected_train_examples=len(train_data), sampler_examples_per_rank=len(sampler),
            sampler_dropped_examples_per_epoch=len(train_data) - len(sampler) * world_size,
            unique_examples_consumed_per_complete_epoch=len(sampler) * world_size,
            cumulative_wall_seconds_before_resume=prior_wall_seconds,
            fused_adamw_active=fused_adamw_active, ddp_gradient_as_bucket_view=world_size > 1)
        if args.resume:
            atomic_json(run_dir / f'resume-{global_step:08d}.json', metadata)
        else:
            atomic_json(run_dir / 'run.json', metadata)
        atomic_json(run_dir / 'train_data_audit.json', train_data.report())
        atomic_json(run_dir / 'eval_data_audit.json', eval_data.report())
        atomic_json(run_dir / 'status.json', dict(status='running', phase='initializing',
            global_step=global_step, last_update_unix=time.time()))
    rank_zero_action(write_metadata, rank, world_size)

    def validation(phase):
        rank_zero_action(lambda: atomic_json(run_dir / 'status.json',
            dict(status='running', phase=phase, global_step=global_step, last_update_unix=time.time())), rank, world_size)
        result = rank_zero_action(lambda: evaluate_model(model, processor, eval_loader, device,
            run_dir / 'predictions' / f'{phase}-step-{global_step:08d}.jsonl',
            config['precision'], config['generation_max_new_tokens']), rank, world_size)
        log(dict(phase=phase, step=global_step, **result))
        return result['wer']

    try:
        # Baseline stays separate from fine-tuned best selection. Never overwrite it on resume.
        if not args.resume:
            validation('baseline')
        for epoch in range(start_epoch, config['epochs']):
            sampler.set_epoch(epoch)
            cursor = next_batch if epoch == start_epoch else 0
            iterator = iter(train_loader)
            for _ in range(cursor):
                next(iterator)
            # Iteration and worker startup must not consume restored model RNG.
            if resume_rng is not None:
                restore_rng(resume_rng)
                resume_rng = None
            wrapped.train()
            for group_start, group_size in update_groups(len(train_loader), accumulation, cursor):
                started = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                totals = torch.zeros(4, device=device, dtype=torch.float64)
                data_wait = 0.0
                lr_used = optimizer.param_groups[0]['lr']
                for micro in range(group_size):
                    loaded_at = time.monotonic()
                    batch = next(iterator)
                    data_wait += time.monotonic() - loaded_at
                    inputs = {key: batch[key].to(device, non_blocking=True)
                              for key in ('input_features', 'attention_mask', 'labels')}
                    sync = micro == group_size - 1
                    with wrapped.no_sync() if world_size > 1 and not sync else nullcontext():
                        with autocast_context(device, config['precision']):
                            loss = wrapped(**inputs).loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Nonfinite loss at step {global_step}, rank {rank}')
                        (loss / group_size).backward()
                    token_count = inputs['labels'].ne(-100).sum()
                    totals[0] += loss.detach().double() * token_count
                    totals[1] += token_count
                    totals[2] += batch['durations'].sum().to(device)
                    totals[3] += len(batch['ids'])
                grad_norm = torch.nn.utils.clip_grad_norm_(wrapped.parameters(), config['max_grad_norm'], error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                global_step += 1
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                if world_size > 1:
                    dist.all_reduce(totals)
                elapsed = time.monotonic() - started
                timing = torch.tensor([elapsed, data_wait], device=device, dtype=torch.float64)
                if world_size > 1:
                    dist.all_reduce(timing, op=dist.ReduceOp.MAX)
                elapsed, data_wait = timing.tolist()
                log(dict(phase='train', step=global_step, epoch=epoch,
                    batch_cursor=group_start + group_size, loss=(totals[0] / totals[1]).item(),
                    grad_norm=float(grad_norm), learning_rate=lr_used, data_seconds=totals[2].item(),
                    examples=int(totals[3].item()), step_seconds=elapsed,
                    data_wait_seconds=data_wait, audio_seconds_per_wall_second=totals[2].item() / elapsed,
                    cuda_peak_memory_gb=torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0.0))
                wall_elapsed = prior_wall_seconds + time.monotonic() - invocation_started
                stop = torch.tensor([int(_STOP), int(wall_elapsed >= config.get('max_wall_seconds', 43200))], device=device)
                if world_size > 1:
                    dist.all_reduce(stop, op=dist.ReduceOp.MAX)
                interrupted, budget_stopped = bool(stop[0].item()), bool(stop[1].item())
                at_end = group_start + group_size == len(train_loader)
                at_limit = global_step >= total_steps
                evaluate_now = (global_step % config['eval_steps'] == 0 or at_end or at_limit) and not (interrupted or budget_stopped)
                is_best = False
                if evaluate_now:
                    wer = validation('dev')
                    is_best = best_wer is None or wer < best_wer
                    best_wer = wer if best_wer is None else min(best_wer, wer)
                    # Include signals or budget exhaustion that occurred during generation.
                    post_eval_stop = torch.tensor([int(_STOP), int(prior_wall_seconds + time.monotonic() - invocation_started >= config.get('max_wall_seconds', 43200))], device=device)
                    if world_size > 1:
                        dist.all_reduce(post_eval_stop, op=dist.ReduceOp.MAX)
                    interrupted = interrupted or bool(post_eval_stop[0].item())
                    budget_stopped = budget_stopped or bool(post_eval_stop[1].item())
                if evaluate_now or global_step % config.get('save_steps', 100) == 0 or interrupted or budget_stopped:
                    checkpoint_model(model, processor, optimizer, scheduler, run_dir, global_step,
                        epoch + 1 if at_end else epoch, 0 if at_end else group_start + group_size,
                        best_wer, is_best, signature, rank, world_size, config['checkpoint_limit'],
                        prior_wall_seconds + time.monotonic() - invocation_started)
                if at_limit or interrupted or budget_stopped:
                    status = 'budget_stopped' if budget_stopped else ('interrupted' if interrupted else 'complete')
                    rank_zero_action(lambda: atomic_json(run_dir / 'status.json',
                        dict(status=status, global_step=global_step, best_dev_wer=best_wer,
                             cumulative_wall_seconds=prior_wall_seconds + time.monotonic() - invocation_started)), rank, world_size)
                    return
            next_batch = 0
    except BaseException as error:
        if rank == 0:
            atomic_json(run_dir / 'status.json', dict(status='failed', global_step=global_step,
                error=f'{type(error).__name__}: {error}', last_update_unix=time.time()))
        raise
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
