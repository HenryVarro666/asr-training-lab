"""Summarize measured training steps and fixed-global-batch DDP scaling."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from .training_utils import atomic_json


def summarize_run(run_dir, warmup_steps=20, measure_steps=50):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'run.json').read_text())
    rows = {}
    with (run_dir / 'metrics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get('phase') != 'train':
                continue
            step = int(row['step'])
            if step in rows:
                raise ValueError(f'Duplicate training step {step} in {run_dir}')
            rows[step] = row
    expected = list(range(warmup_steps + 1, warmup_steps + measure_steps + 1))
    if any(step not in rows for step in expected):
        raise ValueError(f'Expected all {measure_steps} measured steps after {warmup_steps} warmup steps in {run_dir}')
    measured = [rows[step] for step in expected]
    for row in measured:
        for key in ('step_seconds', 'data_seconds', 'loss', 'grad_norm'):
            if not math.isfinite(float(row[key])):
                raise ValueError(f'Nonfinite {key} at step {row["step"]}')
        if row['step_seconds'] <= 0 or row['data_seconds'] <= 0:
            raise ValueError('Timing and audio seconds must be positive')
    timings = [row['step_seconds'] for row in measured]
    audio_seconds = sum(row['data_seconds'] for row in measured)
    wall_seconds = sum(timings)
    return dict(run_dir=str(run_dir.resolve()), world_size=metadata['world_size'],
        effective_batch_size=metadata['effective_batch_size'], warmup_steps=warmup_steps,
        measured_steps=measure_steps, measured_first_step=expected[0], measured_last_step=expected[-1],
        step_seconds_p50=float(np.percentile(timings, 50)), step_seconds_p95=float(np.percentile(timings, 95)),
        total_measured_step_seconds=wall_seconds, total_measured_audio_seconds=audio_seconds,
        audio_seconds_per_wall_second=audio_seconds / wall_seconds,
        cuda_peak_memory_gb=max(row.get('cuda_peak_memory_gb', 0) for row in measured),
        scope='Training updates only; generation evaluation, checkpoint I/O, model loading excluded',
        schedule=metadata.get('signature'))


def compare_runs(single, dual):
    if single['effective_batch_size'] != dual['effective_batch_size']:
        raise ValueError('Cannot compare runs with different global batch sizes')
    if single['world_size'] != 1 or dual['world_size'] != 2:
        raise ValueError('Expected one-GPU and two-GPU runs, in that order')
    speedup = dual['audio_seconds_per_wall_second'] / single['audio_seconds_per_wall_second']
    return dict(one_gpu=single, two_gpu=dual, throughput_speedup=speedup,
                scaling_efficiency=speedup / 2,
                caveat='GPU0 hosts an existing service and PCIe/NODE interconnect may limit scaling; inspect resource traces.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', help='Summarize one run of any GPU count without a scaling comparison')
    parser.add_argument('--one-gpu')
    parser.add_argument('--two-gpu')
    parser.add_argument('--warmup-steps', type=int, default=20)
    parser.add_argument('--measure-steps', type=int, default=50)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.run:
        result = summarize_run(args.run, args.warmup_steps, args.measure_steps)
    elif args.one_gpu and args.two_gpu:
        result = compare_runs(summarize_run(args.one_gpu, args.warmup_steps, args.measure_steps),
                              summarize_run(args.two_gpu, args.warmup_steps, args.measure_steps))
    else:
        parser.error('Supply --run or both --one-gpu and --two-gpu')
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
