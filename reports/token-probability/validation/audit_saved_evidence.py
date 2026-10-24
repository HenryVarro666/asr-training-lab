#!/usr/bin/env python3
"""Independent saved-logit CPU audit; no model, generation or weight reads.

Tolerances are frozen before the actual run. Every large PT file is read from
disk once, SHA256 checked on those bytes, then loaded from the same byte buffer.
All-vocabulary statistics use independent FP64 arithmetic in bounded chunks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import signal
import sys
import time

REPO = Path(__file__).resolve().parents[3]
MODEL_ORDER = ('base', 'step050', 'step100')
TRAJECTORY_ORDER = MODEL_ORDER
TARGET = '652-129742-0006'
TOLERANCES = dict(abs_logprob_entropy_margin=5e-5, abs_probability=1e-6,
                  abs_difference_of_two_logprobs=1e-4,
                  nll_sum_error_per_row=5e-5,
                  fp64_normalization=2e-12,
                  rank_ties_argmax_top10_ids='exact')
MAX_AUDIT_SECONDS = 600


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prefix_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


class Budget:
    def __init__(self, seconds=MAX_AUDIT_SECONDS):
        require(0 < seconds <= MAX_AUDIT_SECONDS, 'Audit budget must be <=600 seconds')
        self.started = time.monotonic()
        self.deadline = self.started + seconds

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError('Independent CPU audit exceeded its predeclared wall budget')


class Evidence:
    def __init__(self, torch, budget):
        self.torch, self.budget = torch, budget
        self.records, self.objects, self.disk_reads = {}, {}, Counter()

    def load(self, path, expected_sha=None, expected_bytes=None):
        path = Path(path).resolve()
        require(not path.name.endswith('.safetensors'), 'Weight-content reads are prohibited')
        require(path.suffix != '.bin', 'Model/optimizer binary reads are prohibited')
        require(path.name not in ('optimizer.pt', 'training_state.pt'),
                'Training-state reads are prohibited')
        self.budget.check()
        if path not in self.objects:
            raw = path.read_bytes()
            self.disk_reads[str(path)] += 1
            self.budget.check()
            self.records[path] = dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                                      bytes=len(raw))
            if path.suffix == '.pt':
                value = self.torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            else:
                value = json.loads(raw)
            self.objects[path] = value
        record = self.records[path]
        if expected_sha is not None:
            require(record['sha256'] == expected_sha, f'Artifact SHA mismatch: {path}')
        if expected_bytes is not None:
            require(record['bytes'] == expected_bytes, f'Artifact byte-size mismatch: {path}')
        return self.objects[path]


def full_vocab_statistics(matrix, targets, candidates, eos_ids, budget, chunk_rows=16):
    """Independent FP64 normalization, exhaustive ranking and stable tie order."""
    import torch
    require(matrix.device.type == 'cpu' and matrix.dtype == torch.float32,
            'Persisted full-vocabulary matrix must be CPU FP32')
    require(matrix.ndim == 2 and matrix.shape[0] == len(targets), 'Logit/target row mismatch')
    vocab = matrix.shape[1]
    eos_ids = sorted(set(eos_ids))
    require(vocab >= 10 and len(set(eos_ids)) == 2, 'Missing full vocabulary/distinct EOS')
    require(candidates == sorted(set(candidates)) and set(targets) <= set(candidates)
            and set(eos_ids) <= set(candidates), 'Candidate union/order mismatch')
    require(min(candidates) >= 0 and max(candidates) < vocab, 'Candidate outside vocabulary')
    rows = []
    candidate_index = {token: i for i, token in enumerate(candidates)}
    for start in range(0, len(targets), chunk_rows):
        budget.check()
        original = matrix[start:start+chunk_rows]
        require(bool(torch.isfinite(original).all()), 'Unexplained nonfinite logits/scores')
        x = original.to(torch.float64)
        maximum = x.max(dim=1).values
        shifted = x - maximum[:, None]
        exponentials = shifted.exp()
        denominator = exponentials.sum(dim=1)
        log_denominator = denominator.log()
        probabilities = exponentials / denominator[:, None]
        norm_error = (probabilities.sum(dim=1)-1).abs()
        require(float(norm_error.max()) <= TOLERANCES['fp64_normalization'],
                'Independent FP64 probabilities fail normalization')
        entropy = log_denominator - (probabilities * shifted).sum(dim=1)
        require(bool(torch.isfinite(entropy).all()), 'Nonfinite independently computed entropy')
        # This independently establishes ID-ascending tie order and every rank.
        order = torch.argsort(original, dim=1, descending=True, stable=True)
        sorted_values = original.gather(1, order)
        query = original[:, candidates]
        ascending_negative = (-sorted_values).contiguous()
        left = torch.searchsorted(ascending_negative, (-query).contiguous(), right=False)
        right = torch.searchsorted(ascending_negative, (-query).contiguous(), right=True)
        candidate_logp = query.to(torch.float64)-maximum[:, None]-log_denominator[:, None]
        top_ids = order[:, :10]
        top_logp = original.gather(1, top_ids).to(torch.float64)-maximum[:, None]-log_denominator[:, None]
        margins = sorted_values[:, 0].to(torch.float64)-sorted_values[:, 1].to(torch.float64)
        for local in range(len(original)):
            candidate_rows = []
            for ci, token in enumerate(candidates):
                lp = float(candidate_logp[local, ci])
                candidate_rows.append(dict(id=token, logit=float(original[local, token]),
                    logprob=lp, prob=math.exp(lp),
                    rank=1+int(left[local, ci]),
                    tie_count=int(right[local, ci]-left[local, ci])))
            target = candidate_rows[candidate_index[targets[start+local]]]
            eos = [candidate_rows[candidate_index[token]] for token in eos_ids]
            eos_max = max(e['logprob'] for e in eos)
            eos_logp = eos_max + math.log(sum(math.exp(e['logprob']-eos_max) for e in eos))
            top = []
            for k, token in enumerate(top_ids[local].tolist()):
                value = original[local, token]
                greater = int((original[local] > value).sum())
                tied = int((original[local] == value).sum())
                lp = float(top_logp[local, k])
                top.append(dict(id=token, logit=float(original[local, token]),
                                logprob=lp, prob=math.exp(lp),
                                rank=greater+1, tie_count=tied))
            rows.append(dict(target=target, candidates=candidate_rows, top10=top,
                argmax_id=int(top_ids[local, 0]), top1_tie_count=top[0]['tie_count'],
                margin=float(margins[local]), entropy=float(entropy[local]),
                eos=dict(ids=list(eos_ids), individual=eos,
                         total_prob=sum(e['prob'] for e in eos), total_logprob=eos_logp),
                fp64_normalization_error=float(norm_error[local])))
        budget.check()
    return rows


def close(actual, expected, tolerance, label, maxima):
    require(isinstance(actual, (float, int)) and math.isfinite(actual),
            f'Nonfinite or invalid derived statistic: {label}')
    delta = abs(float(actual)-float(expected))
    maxima[label] = max(maxima.get(label, 0.), delta)
    require(delta <= tolerance, f'Derived statistic mismatch {label}: error={delta}, tolerance={tolerance}')


def compare_token(actual, expected, label, maxima, ranks=True):
    for key in (('id', 'rank', 'tie_count') if ranks else ('id',)):
        require(type(actual[key]) is int and actual[key] == expected[key],
                f'Exact token statistic differs: {label}.{key}')
    require(actual['logit'] == expected['logit'], f'Original FP32 logit differs: {label}')
    close(actual['logprob'], expected['logprob'], TOLERANCES['abs_logprob_entropy_margin'],
          label+'.logprob', maxima)
    close(actual['prob'], expected['prob'], TOLERANCES['abs_probability'],
          label+'.prob', maxima)


def compare_row(actual, expected, maxima):
    compare_token(actual['target'], expected['target'], 'target', maxima)
    require([v['id'] for v in actual['candidates']] ==
            [v['id'] for v in expected['candidates']], 'Unified candidate IDs/order differ')
    for saved, computed in zip(actual['candidates'], expected['candidates']):
        compare_token(saved, computed, 'candidate', maxima)
    require([v['id'] for v in actual['top10']] ==
            [v['id'] for v in expected['top10']], 'Stable top10 IDs/order differ')
    for saved, computed in zip(actual['top10'], expected['top10']):
        compare_token(saved, computed, 'top10', maxima, ranks=False)
    require(actual['top10'][0]['id'] == expected['argmax_id'] and
            actual['top1_tie_count'] == expected['top1_tie_count'], 'Argmax/tie count differs')
    require(actual['eos']['ids'] == expected['eos']['ids'], 'EOS must contain two distinct fixed IDs')
    require(len(actual['eos']['individual']) == 2, 'EOS/PAD double counting')
    for saved, computed in zip(actual['eos']['individual'], expected['eos']['individual']):
        compare_token(saved, computed, 'eos', maxima)
    close(actual['eos']['total_prob'], expected['eos']['total_prob'],
          TOLERANCES['abs_probability'], 'eos.total_prob', maxima)
    close(actual['eos']['total_logprob'], expected['eos']['total_logprob'],
          TOLERANCES['abs_logprob_entropy_margin'], 'eos.total_logprob', maxima)
    close(actual['top1_minus_top2_margin'], expected['margin'],
          TOLERANCES['abs_logprob_entropy_margin'], 'margin', maxima)
    close(actual['entropy'], expected['entropy'],
          TOLERANCES['abs_logprob_entropy_margin'], 'entropy', maxima)


def verify_teacher_inputs(extended, original, targets, mapping):
    """Prove causal alignment from complete saved IDs; do not infer it from logits."""
    import torch
    require(len(targets) >= 1 and original['input_ids'].shape[0] == 1,
            'Expected nonempty singleton target')
    width = original['input_ids'].shape[1]
    target_tensor = torch.tensor([targets[:-1]], dtype=original['input_ids'].dtype)
    expected_ids = torch.cat((original['input_ids'], target_tensor), dim=1)
    expected_mask = torch.cat((original['attention_mask'], torch.ones(
        (1, len(targets)-1), dtype=original['attention_mask'].dtype)), dim=1)
    require(torch.equal(extended['input_ids'], expected_ids), 'TF input is not P || Y[:-1]')
    require(torch.equal(extended['attention_mask'], expected_mask), 'TF extended attention mask differs')
    for key in original:
        if key not in ('input_ids', 'attention_mask'):
            require(key in extended and torch.equal(extended[key], original[key]),
                    f'TF changed original audio/feature tensor: {key}')
    require(mapping['input_positions'] == list(range(width-1, width-1+len(targets))) and
            mapping['target_ids'] == targets, 'TF first/last/target alignment differs')
    require(mapping['input_ids'] == expected_ids.tolist() and
            mapping['attention_mask'] == expected_mask.tolist(), 'TF JSON/PT input mismatch')


def verify_prefixes(rows, prompt, targets):
    require(len(rows) == len(targets), 'Derived row count differs')
    for j, row in enumerate(rows):
        ids = prompt + targets[:j]
        require(row['prefix']['length'] == len(ids) and
                row['prefix']['sha256'] == prefix_hash(ids), f'Wrong causal prefix at target {j}')
        require(row['target']['id'] == targets[j], f'Wrong target at causal position {j}')


def piece_map(model_path, evidence):
    """Reconstruct token pieces from frozen JSONs without loading any tokenizer/model."""
    directory = Path(model_path)
    vocab = evidence.load(directory/'vocab.json')
    mapping = {int(token): piece for piece, token in vocab.items()}
    config = evidence.load(directory/'tokenizer_config.json')
    for token, value in config.get('added_tokens_decoder', {}).items():
        mapping[int(token)] = value['content']
    added = directory/'added_tokens.json'
    if added.is_file():
        for piece, token in evidence.load(added).items():
            mapping[int(token)] = piece
    return mapping


def bytelevel_decode(ids, mapping):
    """Decode this frozen Qwen byte-level vocabulary without a model/tokenizer API."""
    direct = list(range(33, 127))+list(range(161, 173))+list(range(174, 256))
    remaining = [i for i in range(256) if i not in direct]
    inverse = {chr(i): i for i in direct}
    inverse.update({chr(256+j): value for j, value in enumerate(remaining)})
    text = ''.join(mapping[token] for token in ids)
    return bytes(inverse[c] for c in text).decode('utf-8', errors='replace')


def tensor_fingerprint(tensor):
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype), numel=tensor.numel(),
        sha256=hashlib.sha256(tensor.contiguous().view(__import__('torch').uint8).numpy().tobytes()).hexdigest())


def independent_forks(trajectories):
    forks = []
    for i, left in enumerate(MODEL_ORDER):
        for right in MODEL_ORDER[i+1:]:
            a, b = trajectories[left], trajectories[right]
            common = 0
            for x, y in zip(a, b):
                if x != y:
                    break
                common += 1
            forks.append(dict(left=left, right=right, lcp_length=common,
                first_difference_index=None if a == b else common,
                shared_continuation_prefix=a[:common],
                left_next_id=a[common] if common < len(a) else None,
                right_next_id=b[common] if common < len(b) else None))
    return forks


def verify_model_identity(stage, model, outer, evidence):
    verification = evidence.load(outer/'model-verification.json')
    require(verification['status'] == 'passed_with_post_worker_stat_match' and
            set(verification['files']) == set(model['files']), 'Model verification incomplete')
    prior = model['prior_model_verification']
    older = evidence.load(prior['path'], prior['sha256'], prior['bytes'])
    if stage != 'base':
        anchor = model['historical_anchor']['prefix_complete']
        marker = evidence.load(Path(model['path'])/'prefix-complete.json', anchor['sha256'])
        require(marker['status'] == 'complete' and marker['step'] == model['step'],
                'Historical checkpoint marker differs')
    for name, record in verification['files'].items():
        expected = model['files'][name]
        require(record['expected_sha256'] == expected['sha256'] == record['actual_sha256'] ==
                older['files'][name]['actual_sha256'], 'Model SHA lacks prior-stage anchor')
        require(record['before'] == record['after_hash'] == verification['post_worker_stats'][name],
                'Model stat/softlink drift')
        require(record['before']['size'] == expected['bytes'], 'Model byte size differs')
        for key in ('logical_path', 'resolved_path', 'dev', 'inode', 'size', 'mtime_ns',
                    'ctime_ns', 'is_symlink', 'symlink_target', 'symlink_lstat'):
            require(key in record['before'], f'Incomplete model stat: {key}')
        if stage != 'base':
            require(record['actual_sha256'] == marker['files'][name]['sha256'],
                    'Historical prefix SHA mismatch')
    return dict(files=len(verification['files']), current_weight_content_rehashed_by_audit=False,
                prior_onset_anchor_verified=True, stable_saved_before_after_post_stat=True,
                base_boundary='Base SHA first captured in prior onset, reused here; not original-download historical SHA.' if stage=='base' else None)


def derived_check(saved, recomputed, prompt, targets, candidates, mapping, maxima):
    require(saved['candidate_ids'] == candidates, 'Candidate union differs')
    verify_prefixes(saved['rows'], prompt, targets)
    for j, (actual, expected) in enumerate(zip(saved['rows'], recomputed)):
        require(actual['j'] == j and actual['input_position'] == len(prompt)-1+j and
                actual['target_id'] == targets[j], 'Derived shifted row mapping differs')
        try:
            compare_row(actual, expected, maxima)
        except ValueError as error:
            raise ValueError(f'At row {j}: {error}') from error
        for item in [actual['target'], *actual['candidates'], *actual['top10'], *actual['eos']['individual']]:
            require(item['piece'] == mapping.get(item['id']), 'Token piece/vocabulary mismatch')
    total = -sum(row['target']['logprob'] for row in recomputed)
    close(saved['total_target_nll'], total,
          len(targets)*TOLERANCES['nll_sum_error_per_row'], 'total_target_nll', maxima)
    close(saved['mean_target_nll'], total/len(targets),
          TOLERANCES['abs_logprob_entropy_margin'], 'mean_target_nll', maxima)
    return dict(total=total, mean=total/len(targets))


def self_test():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(2)
    matrix = torch.tensor([[2., 2., 1., 0., -1., -2., -3., -4., -5., -6., -1000., -1001.],
                           [0.]*12], dtype=torch.float32)
    rows = full_vocab_statistics(matrix, [0, 11], list(range(12)), [10, 11, 11], Budget(10), 1)
    require(rows[0]['argmax_id'] == 0 and rows[0]['top1_tie_count'] == 2 and
            rows[0]['target']['rank'] == 1, 'Argmax/rank tie contract failed')
    require([t['id'] for t in rows[0]['top10']] == list(range(10)), 'Stable top10 tie order failed')
    require(rows[1]['target']['rank'] == 1 and rows[1]['target']['tie_count'] == 12,
            'Uniform tied rank failed')
    require(abs(rows[1]['entropy']-math.log(12)) < 1e-14, 'Independent entropy failed')
    require(abs(rows[1]['eos']['total_prob']-2/12) < 1e-14,
            'Distinct two-EOS probability must not count PAD as a third event')
    require(rows[0]['eos']['total_prob'] == 0 and math.isfinite(rows[0]['eos']['total_logprob']),
            'Underflow must preserve finite log probability')
    maxima = {}
    compare_token(rows[0]['target'], rows[0]['target'], 'toy', maxima)
    for bad in (float('nan'), float('inf'), float('-inf')):
        corrupted = matrix.clone(); corrupted[0, 0] = bad
        try:
            full_vocab_statistics(corrupted, [0, 11], list(range(12)), [10, 11], Budget(10))
        except ValueError:
            pass
        else:
            raise AssertionError('Nonfinite matrix was accepted')
    require(prefix_hash([1, 2, 3]) == hashlib.sha256(b'[1,2,3]').hexdigest(),
            'Canonical prefix hash changed')
    original = dict(input_ids=torch.tensor([[1, 2, 3]]),
                    attention_mask=torch.ones(1, 3, dtype=torch.int64),
                    input_features=torch.zeros(1, 2))
    for labels in ([4, 5, 9], [9]):
        extended = dict(original, input_ids=torch.tensor([[1, 2, 3]+labels[:-1]]),
                        attention_mask=torch.ones(1, 3+len(labels)-1, dtype=torch.int64))
        mapping = dict(input_ids=extended['input_ids'].tolist(),
                       attention_mask=extended['attention_mask'].tolist(),
                       input_positions=list(range(2, 2+len(labels))), target_ids=labels)
        verify_teacher_inputs(extended, original, labels, mapping)
        incorrect = dict(mapping, input_positions=list(range(3, 3+len(labels))))
        try:
            verify_teacher_inputs(extended, original, labels, incorrect)
        except ValueError:
            pass
        else:
            raise AssertionError('Off-by-one TF alignment accepted')
    print('Independent FP64 CPU toy checks passed; no real run or model was read.')


def audit(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(args.threads)
    budget = Budget(args.seconds)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600-second CPU audit alarm')))
    signal.alarm(args.seconds)
    evidence = Evidence(torch, budget)
    reference = evidence.load(args.reference)
    require(reference['model_order'] == list(MODEL_ORDER) and reference['target_id'] == TARGET,
            'Frozen experiment identity differs')
    trajectories = {s: reference['trajectories'][s]['token_ids'] for s in MODEL_ORDER}
    require([len(trajectories[s]) for s in MODEL_ORDER] == [27, 37, 512], 'Frozen lengths differ')
    prompt = reference['common_input']['input_ids'][0]
    require(len(prompt) == 137, 'Frozen prompt width differs')
    candidates = sorted(set(reference['effective_eos_ids'] + sum(trajectories.values(), [])))
    for name in ('vocab.json', 'tokenizer_config.json'):
        record = reference['models']['base']['files'][name]
        evidence.load(args.tokenizer_dir/name, record['sha256'], record['bytes'])
    mapping = piece_map(args.tokenizer_dir, evidence)
    maxima, summaries, matrices, recomputed = {}, {}, {}, {}
    all_rows = 0
    for stage in MODEL_ORDER:
        budget.check()
        outer = args.run/stage
        worker = outer/'worker'
        result = evidence.load(worker/'result.json')
        require(result['status'] == 'complete' and result['stage'] == stage and
                result['generate_calls'] == 1 and result['teacher_forward_calls'] == 3 and
                result['optimizer_updates'] == 0, 'Incomplete worker/call receipts')
        # Read/hash every artifact exactly once, including JSON and shared-storage PTs.
        for name, record in result['artifacts'].items():
            path = worker/name
            require(path.resolve().is_relative_to(worker.resolve()), 'Artifact escapes worker')
            if path.suffix in ('.json', '.pt'):
                evidence.load(path, record['sha256'], record['bytes'])
            else:
                raw = path.read_bytes()
                require(hashlib.sha256(raw).hexdigest() == record['sha256'] and
                        len(raw) == record['bytes'], 'Non-JSON artifact identity differs')
        original = evidence.load(worker/'inputs.pt')
        require(original['input_ids'].tolist() == [prompt], 'Saved prompt differs')
        for key, value in reference['common_input']['tensors'].items():
            require(tensor_fingerprint(original[key]) == value, 'Frozen input tensor differs: '+key)
        metadata = evidence.load(worker/'metadata.json')
        require(metadata['phase'] == 'complete' and metadata['inference_mode'] is True and
                metadata['model_training'] is False and metadata['logits_collected'] is True and
                metadata['seed_reset'] is False and metadata['train_called'] is False and
                metadata['optimizer_updates'] == metadata['backward_calls'] == 0,
                'Runtime inference/no-update receipts differ')
        summaries[stage] = {}
        for role in ('generation', *MODEL_ORDER):
            folder = worker/'generation' if role == 'generation' else worker/'teacher'/role
            targets = trajectories[stage if role == 'generation' else role]
            payload = evidence.load(folder/'distributions.pt')
            matrix = payload['raw_logits' if role == 'generation' else 'logits']
            manifest = evidence.load(folder/'distribution-manifest.json')
            require(manifest['shape'] == list(matrix.shape) and manifest['dtype'] == str(matrix.dtype),
                    'Manifest tensor shape/dtype differs')
            require(matrix.shape[1] == metadata['vocab_size'], 'Incomplete vocabulary')
            if role == 'generation':
                raw = evidence.load(folder/'raw-tokens.json')
                require(raw['continuation_token_ids'] == [targets] and
                        raw['full_sequences'] == [prompt+targets], 'Instrumented trajectory differs')
                require(torch.equal(matrix, payload['scores']), 'Empty-processor raw/scores differ')
                require(torch.argmax(payload['scores'], dim=1).tolist() == targets,
                        'Processed-score selection differs')
            else:
                verify_teacher_inputs(evidence.load(folder/'inputs.pt'), original, targets,
                                      evidence.load(folder/'inputs.json'))
                receipt = evidence.load(folder/'forward-receipt.json')
                require(receipt['status'] == 'complete' and receipt['use_cache'] is False and
                        receipt['past_key_values'] is None and receipt['cache_position'] is None and
                        receipt['rope_deltas_after_reset'] is None and receipt['past_cache_returned'] is False and
                        receipt['explicit_teacher_forward'] == 1 and receipt['labels'] is None,
                        'Teacher cache/forward receipt differs')
            calculated = full_vocab_statistics(matrix, targets, candidates,
                                               reference['effective_eos_ids'], budget)
            saved = evidence.load(folder/'derived.json')
            summaries[stage][role] = derived_check(saved, calculated, prompt, targets,
                                                  candidates, mapping, maxima)
            matrices[stage, role] = matrix
            recomputed[stage, role] = calculated
            all_rows += len(targets)
        own = evidence.load(worker/'own-trajectory-comparison.json')['rows']
        teacher, generated = matrices[stage, stage], matrices[stage, 'generation']
        require(len(own) == len(trajectories[stage]), 'Missing TF/generation comparison rows')
        for j, row in enumerate(own):
            budget.check()
            delta = (teacher[j].double()-generated[j].double()).abs()
            close(row['max_absolute_logit_difference'], float(delta.max()), 5e-5, 'own.max_delta', maxima)
            close(row['mean_absolute_logit_difference'], float(delta.mean()), 5e-5, 'own.mean_delta', maxima)
            a, b = recomputed[stage, stage][j], recomputed[stage, 'generation'][j]
            for key, value in (
                ('target_logprob_difference', a['target']['logprob']-b['target']['logprob']),
                ('eos_total_logprob_difference', a['eos']['total_logprob']-b['eos']['total_logprob'])):
                close(row[key], value, 1e-4, 'own.'+key, maxima)
            require(row['j'] == j, 'Own comparison row index differs')
            close(row['teacher_margin'], a['margin'], 5e-5, 'own.teacher_margin', maxima)
            close(row['generation_margin'], b['margin'], 5e-5, 'own.generation_margin', maxima)
            require(row['argmax_equal'] == (a['argmax_id'] == b['argmax_id']), 'Own argmax comparison differs')
    require(all_rows == 2304, 'Expected all 2304 distribution rows')
    require(all(v == 1 for v in evidence.disk_reads.values()), 'Evidence read more than once')
    output = dict(status='passed', audited_at=datetime.now(timezone.utc).isoformat(),
        run=str(args.run.resolve()), reference_sha256=evidence.records[args.reference.resolve()]['sha256'],
        arithmetic='Independent FP64 max-shift normalization; exhaustive stable sorting and ranks',
        tolerances=TOLERANCES, total_distribution_rows=all_rows,
        full_vocabulary_matrices=12, model_calls=0, weight_content_reads=0,
        matrix_disk_read_policy='Each PT read/hash/load once from identical bytes',
        max_observed_absolute_errors=maxima, nll=summaries,
        free_trajectory_forks=independent_forks(trajectories),
        artifact_records=[v for p,v in evidence.records.items()],
        elapsed_seconds=time.monotonic()-budget.started,
        boundary='Saved evidence audit; saved model SHA/stat receipts are not a new weight rehash. '
                 'Fixed abnormal prefixes change conditioning; low EOS is not a training cause. '
                 'Finite cached-generation/full-forward differences are observations.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2)+'\n')
    args.output.with_suffix('.md').write_text('# Independent saved-evidence CPU audit\n\n'
        f"Status: **passed**. Audited {all_rows} complete vocabulary rows across 12 matrices.\n\n"
        'Independent FP64 recomputation verifies every target/candidate, exact rank/tie/top10 order, '
        'entropy, margins, distinct EOS sum, causal prefix/teacher row alignment and saved TF/generation '
        'differences against predeclared tolerances. Every PT was read, hashed and loaded once. '
        'No model calls or weight reads.\n\n'+output['boundary']+'\n')
    signal.alarm(0)
    print(json.dumps({k: output[k] for k in ('status','total_distribution_rows','elapsed_seconds')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--run', type=Path, default=REPO/'reports/token-probability/run')
    parser.add_argument('--reference', type=Path, default=REPO/'reports/token-probability/inputs/trajectory-reference.json')
    parser.add_argument('--tokenizer-dir', type=Path)
    parser.add_argument('--output', type=Path, default=REPO/'reports/token-probability/independent-audit.json')
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        require(args.tokenizer_dir is not None, '--tokenizer-dir is required; small frozen vocabulary JSONs only')
        audit(args)


if __name__ == '__main__':
    main()
