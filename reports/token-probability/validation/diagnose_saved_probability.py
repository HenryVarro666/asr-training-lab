#!/usr/bin/env python3
"""Single saved-row CPU diagnosis; never imports a model or reads weights."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True, type=Path)
    p.add_argument('--inventory', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run.resolve()):
        raise FileExistsError('Output must be new and outside immutable original run')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s diagnosis')))
    signal.alarm(120)
    started = time.monotonic()
    import torch
    torch.set_num_threads(4)
    inventory = json.loads(args.inventory.read_text())
    if Path(inventory['run']).resolve() != args.run.resolve():
        raise ValueError('Inventory run differs')
    def read(name, tensor=False):
        raw = (args.run/name).read_bytes()
        expected = inventory['files'][name]
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('Original artifact drift: '+name)
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True) if tensor else json.loads(raw)
    name = 'base/worker/generation/distributions.pt'
    matrix = read(name, True)['raw_logits']
    saved = read('base/worker/generation/derived.json')['rows'][2]
    row = matrix[2]
    if row.dtype != torch.float32 or not bool(torch.isfinite(row).all()):
        raise ValueError('Expected original finite FP32 row')
    target = saved['target']['id']
    eos = saved['eos']['ids']
    # Archived worker used OMP_NUM_THREADS=1; reproduce its reduction first.
    torch.set_num_threads(1)
    old = row-torch.logsumexp(row, dim=0)
    torch.set_num_threads(4)
    centered = row-row.max()
    stable32 = torch.log_softmax(centered, dim=0)
    # Candidate production repair: fused log_softmax in FP64 on stored FP32 values.
    stable64 = torch.log_softmax(row.double(), dim=0)
    # Reference remains independently max-shifted exp/sum/log arithmetic.
    shifted = row.double()-row.double().max()
    denominator = shifted.exp().sum()
    reference = shifted-denominator.log()
    def stats(logp):
        probability = logp.exp()
        return dict(target_logprob=float(logp[target]), target_prob=float(probability[target]),
            probability_sum=float(probability.sum()), entropy=float(-(probability*logp).sum()),
            eos_prob=float(probability[eos].sum()), eos_logprob=float(torch.logsumexp(logp[eos], 0)))
    expected = stats(reference)
    variants = {}
    for label, values in [('original_fp32', old), ('centered_fp32', stable32), ('fused_fp64', stable64)]:
        result = stats(values)
        result['absolute_errors'] = {k: abs(result[k]-expected[k]) for k in expected}
        variants[label] = result
    if variants['original_fp32']['target_prob'] != saved['target']['prob']:
        raise ValueError('Old formula failed to reproduce saved original target probability exactly')
    if variants['original_fp32']['absolute_errors']['target_prob'] <= 1e-6:
        raise ValueError('Original declared failure was not reproduced')
    if variants['fused_fp64']['absolute_errors']['target_prob'] > 1e-6:
        raise ValueError('Candidate repair does not satisfy original probability tolerance')
    output = dict(status='original_failure_reproduced_candidate_repair_validated',
        dataset='base/generation', row_index=2, target_id=target, saved_target=saved['target'],
        matrix_sha256=inventory['files'][name]['sha256'], shape=list(matrix.shape),
        logits_dtype=str(row.dtype), min_logit=float(row.min()), max_logit=float(row.max()),
        original_formula_threads=1, candidate_reference_threads=4,
        max_logit_fp32_spacing=float(torch.nextafter(row.max(), torch.tensor(float('inf')))-row.max()),
        original_absolute_logsumexp=float(torch.logsumexp(row, 0)),
        centered_fp32_logsumexp=float(torch.logsumexp(centered, 0)),
        independent_fp64_logsumexp_shifted=float(denominator.log()), reference=expected, variants=variants,
        original_probability_tolerance=1e-6, tolerances_changed=False,
        model_calls=0, weight_content_reads=0, gpu_calls=0,
        root_cause='FP32 absolute logsumexp rounds after adding the logit offset; subtracting that rounded normalizer propagates error into high-probability events. FP64 fused log_softmax removes the precision loss for the same stored FP32 logits.',
        elapsed_seconds=time.monotonic()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output))
    signal.alarm(0)


if __name__ == '__main__':
    main()
