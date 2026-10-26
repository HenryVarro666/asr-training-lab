#!/usr/bin/env python3
"""Recompute a separate CPU-only derived overlay from immutable FP32 matrices."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time

ROOT = Path(__file__).resolve().parents[1]
ORDER = ('base', 'step050', 'step100')
ORIGINAL_WORKER_SHA = 'acc34e309468c950b02e930ee7be7d45a0394b092ad0b0a35e07f83016f3a43b'
REFERENCE_SHA = '6753b127d40288a558761c6b8e79964f07af7e96f4e85964a0a8b66c26b03f49'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Refusing overwrite: '+str(path))
    raw = (json.dumps(value, indent=2, allow_nan=False)+'\n').encode()
    path.write_bytes(raw)
    return dict(sha256=digest(raw), bytes=len(raw))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('run', 'inventory', 'reference', 'tokenizer-dir', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.run.resolve()):
        raise ValueError('Output must be new and outside immutable original run')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('600s recomputation')))
    signal.alarm(600)
    started = time.monotonic()
    import torch
    from asr_lab.token_probability import matrix_statistics, difference_rows, trajectory_forks
    torch.set_num_threads(4)
    inventory = json.loads(args.inventory.read_text())
    if Path(inventory['run']).resolve() != args.run.resolve():
        raise ValueError('Frozen inventory run differs')
    source = ROOT/'src/asr_lab/token_probability.py'
    new_source_sha = digest(source.read_bytes())
    reference_raw = args.reference.read_bytes()
    if digest(reference_raw) != REFERENCE_SHA:
        raise ValueError('Frozen reference changed')
    reference = json.loads(reference_raw)
    trajectories = {s: reference['trajectories'][s]['token_ids'] for s in ORDER}
    prompt = reference['common_input']['input_ids'][0]
    eos = sorted(set(reference['effective_eos_ids']))
    candidates = sorted(set(eos+sum(trajectories.values(), [])))
    if [len(trajectories[s]) for s in ORDER] != [27, 37, 512] or len(prompt) != 137:
        raise ValueError('Frozen row mapping differs')
    vocab, tokenizer = {}, {}
    for name in ('vocab.json', 'tokenizer_config.json'):
        raw = (args.tokenizer_dir/name).read_bytes()
        record = reference['models']['base']['files'][name]
        if digest(raw) != record['sha256'] or len(raw) != record['bytes']:
            raise ValueError('Tokenizer JSON drift')
        tokenizer[name] = json.loads(raw)
    vocab.update({identifier: piece for piece, identifier in tokenizer['vocab.json'].items()})
    vocab.update({int(identifier): token['content'] for identifier, token in tokenizer['tokenizer_config.json']['added_tokens_decoder'].items()})
    pieces = lambda ids: [vocab[identifier] for identifier in ids]
    args.output.mkdir(parents=True)
    input_records, corrected, maxima, counts = {}, {}, {}, {}
    all_stats, own_summaries = {}, {}
    rows_done = 0
    def read(name, tensor=False):
        raw = (args.run/name).read_bytes()
        expected = inventory['files'][name]
        if len(raw) != expected['bytes'] or digest(raw) != expected['sha256']:
            raise ValueError('Immutable original changed: '+name)
        input_records[name] = dict(expected)
        return torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True) if tensor else json.loads(raw)
    def save_corrected(name, value):
        original = input_records[name]
        identity = write(args.output/name, value)
        corrected[name] = dict(path=name, **identity,
            original_sha256=original['sha256'], original_bytes=original['bytes'])
    def observe(old, new):
        for label, a, b, tol in (
            ('target.prob', old['target']['prob'], new['target']['prob'], 1e-6),
            ('target.logprob', old['target']['logprob'], new['target']['logprob'], 5e-5),
            ('eos.total_prob', old['eos']['total_prob'], new['eos']['total_prob'], 1e-6),
            ('eos.total_logprob', old['eos']['total_logprob'], new['eos']['total_logprob'], 5e-5),
            ('entropy', old['entropy'], new['entropy'], 5e-5)):
            delta = abs(a-b)
            maxima[label] = max(maxima.get(label, 0.), delta)
            counts[label] = counts.get(label, 0)+int(delta > tol)
        for group in ('candidates', 'top10'):
            if [x['id'] for x in old[group]] != [x['id'] for x in new[group]]:
                raise ValueError('Exact ID order changed')
        if old['target']['rank'] != new['target']['rank'] or old['target']['tie_count'] != new['target']['tie_count']:
            raise ValueError('Exact rank/ties changed')
    for stage in ORDER:
        base = f'{stage}/worker'
        result = read(base+'/result.json')
        if result['status'] != 'complete':
            raise ValueError('Original worker incomplete')
        generation, own_matrix = None, None
        for role in ('generation', *ORDER):
            folder = base+'/generation' if role == 'generation' else base+'/teacher/'+role
            payload = read(folder+'/distributions.pt', True)
            matrix = payload['raw_logits' if role == 'generation' else 'logits']
            targets = trajectories[stage if role == 'generation' else role]
            if role == 'generation':
                if not torch.equal(matrix, payload['scores']):
                    raise ValueError('Original raw/scores differ')
                raw = read(folder+'/raw-tokens.json')
                if raw['continuation_token_ids'] != [targets]:
                    raise ValueError('Original generated tokens differ')
            old = read(folder+'/derived.json')
            stats = matrix_statistics(matrix, targets, prompt, candidates, eos, pieces)
            if role == 'generation':
                stats.update(scores_rows_ref='rows', raw_scores_exactly_equal=True)
                generation = matrix
            elif role == stage:
                own_matrix = matrix
            for a, b in zip(old['rows'], stats['rows']):
                if a['prefix'] != b['prefix'] or a['target_id'] != b['target_id'] or a['input_position'] != b['input_position']:
                    raise ValueError('Causal prefix/row alignment changed')
                observe(a, b)
            if len(old['rows']) != len(stats['rows']):
                raise ValueError('Row coverage changed')
            save_corrected(folder+'/derived.json', stats)
            all_stats[stage, role] = stats
            rows_done += len(targets)
            progress=dict(stage=stage,role=role,completed_rows=rows_done,elapsed_seconds=time.monotonic()-started)
            with (args.output/'progress.jsonl').open('a') as f:
                f.write(json.dumps(progress)+'\n')
            if os.environ.get('CPU_EVIDENCE_STAGE_DIR'):
                progress_path=Path(os.environ['CPU_EVIDENCE_STAGE_DIR'])/'progress.json'
                temporary=progress_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(progress)+'\n');temporary.replace(progress_path)
            del payload, matrix, old
        name = base+'/own-trajectory-comparison.json'
        read(name)
        comparison = difference_rows(own_matrix, generation, all_stats[stage, stage], all_stats[stage, 'generation'])
        save_corrected(name, comparison)
        own_summaries[stage] = dict(argmax_equal_rows=sum(r['argmax_equal'] for r in comparison['rows']),
            total_rows=len(comparison['rows']),
            max_absolute_logit_difference=max(r['max_absolute_logit_difference'] for r in comparison['rows']),
            max_absolute_target_logprob_difference=max(abs(r['target_logprob_difference']) for r in comparison['rows']))
        del own_matrix, generation
    if rows_done != 2304:
        raise ValueError('Expected all2304rows')
    comparisons = {}
    for trajectory in ORDER:
        rows = []
        for j, baseline in enumerate(all_stats['base', trajectory]['rows']):
            models = {}
            for stage in ORDER:
                row = all_stats[stage, trajectory]['rows'][j]
                if row['prefix'] != baseline['prefix'] or row['target_id'] != baseline['target_id']:
                    raise ValueError('Cross-model prefix differs')
                models[stage] = dict(target_logprob=row['target']['logprob'], target_prob=row['target']['prob'],
                    target_rank=row['target']['rank'], eos_total_prob=row['eos']['total_prob'],
                    eos_total_logprob=row['eos']['total_logprob'],
                    target_logprob_delta_vs_base=row['target']['logprob']-baseline['target']['logprob'],
                    eos_total_logprob_delta_vs_base=row['eos']['total_logprob']-baseline['eos']['total_logprob'],
                    candidate_deltas=[dict(id=a['id'],logprob_delta=b['logprob']-a['logprob'],rank=b['rank'],base_rank=a['rank'])
                        for a,b in zip(baseline['candidates'],row['candidates'])])
            rows.append(dict(j=j, prefix=baseline['prefix'], target_id=baseline['target_id'], models=models))
        comparisons[trajectory] = dict(rows=rows, per_model_nll={s: dict(total=all_stats[s, trajectory]['total_target_nll'], mean=all_stats[s, trajectory]['mean_target_nll']) for s in ORDER})
    comparison_identity=write(args.output/'comparisons.json', dict(free_trajectory_forks=trajectory_forks(trajectories),
        fixed_trajectory_comparisons=comparisons,
        interpretation='Corrected saved-matrix CPU analysis; compare identical fixed prefixes. No model or training cause claim.'))
    change_identity=write(args.output/'change-summary.json', dict(total_distribution_rows=rows_done,
        maximum_absolute_old_vs_corrected_changes=maxima, rows_exceeding_old_tolerance_vs_corrected=counts,
        own_trajectory_comparison=own_summaries,
        boundary='Old-vs-corrected changes become numerical error estimates only after the independent overlay audit passes.'))
    manifest = dict(schema_version=1, original_run=str(args.run.resolve()),
        reference_sha256=REFERENCE_SHA,
        original_source_sha256={'src/asr_lab/token_probability.py': ORIGINAL_WORKER_SHA},
        corrected_source_sha256={'src/asr_lab/token_probability.py': new_source_sha,
            'scripts/recompute_saved_probability.py': digest(Path(__file__).read_bytes())},
        corrected_artifacts=corrected, input_artifacts=input_records,
        auxiliary_artifacts={'comparisons.json':dict(path='comparisons.json',**comparison_identity),
            'change-summary.json':dict(path='change-summary.json',**change_identity)},
        arithmetic='torch.log_softmax on FP64-promoted original FP32 logits; raw storage unchanged',
        total_distribution_rows=rows_done, full_vocabulary_matrices=12,
        model_calls=0, weight_content_reads=0, gpu_calls=0,
        elapsed_seconds=time.monotonic()-started)
    write(args.output/'manifest.json', manifest)
    if digest(source.read_bytes()) != new_source_sha:
        raise ValueError('CPU statistics source changed during recomputation')
    print(json.dumps(dict(status='complete_pending_independent_audit', rows=rows_done, elapsed_seconds=time.monotonic()-started)))
    signal.alarm(0)


if __name__ == '__main__':
    main()
