#!/usr/bin/env python3
"""Independent CPU-only audit of four saved singleton generation probes.

No model loading, generation, training, downloads, or weight-content rehashing.
Run only after the operator declares the complete run mirror ready.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import random
import shlex
import sys

REPO = Path(__file__).resolve().parents[3]
ORDER = ('base', 'step025', 'step050', 'step100')
STEPS = (0, 25, 50, 100)
TARGET = '652-129742-0006'
KWARGS = dict(max_new_tokens=512, do_sample=False, num_beams=1, use_cache=True)
COMMON_FIELDS = ('runtime', 'generation_kwargs', 'audio', 'model_class',
                 'processor_class', 'inference_mode', 'model_training',
                 'optimizer_updates', 'backward_calls', 'train_called',
                 'logits_collected', 'seed_reset')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def brute_periods(sequence, max_period):
    """Exhaustive starts and whole repeated blocks; no DP or source helper."""
    values = list(sequence)
    n = len(values)

    def empty(period=None):
        return dict(period=period, start=None, end_exclusive=None,
                    repeat_count=0, span=0, coverage=0., unit=[])

    best = empty()
    per_period = []
    for period in range(1, min(max_period, n // 2) + 1):
        current = empty(period)
        for start in range(n - 2 * period + 1):
            unit = values[start:start + period]
            repeats = 1
            while start + (repeats + 1) * period <= n:
                left = start + repeats * period
                if values[left:left + period] != unit:
                    break
                repeats += 1
            if repeats < 2:
                continue
            span = repeats * period
            if span > current['span']:
                current = dict(period=period, start=start,
                               end_exclusive=start + span,
                               repeat_count=repeats, span=span,
                               coverage=span / n, unit=unit)
        partial = []
        if current['end_exclusive'] is not None:
            for offset in range(period - 1):
                pos = current['end_exclusive'] + offset
                if pos >= n or values[pos] != current['unit'][offset]:
                    break
                partial.append(values[pos])
        current.update(following_partial_period=partial,
                       partial_period_counted_as_complete=False)
        per_period.append(current)
        if current['span'] > best['span']:
            best = dict(current)
    return dict(best, sequence_length=n, by_period=per_period)


def independent_observation(tokens, eos, text):
    first = next((i for i, token in enumerate(tokens) if token in eos), None)
    analysis = tokens if first is None else tokens[:first]
    token_run = brute_periods(analysis, 16)
    word_run = brute_periods(text.split(), 8)

    def strong(run):
        return any(p['repeat_count'] >= 20 and p['coverage'] >= .5
                   for p in run['by_period'])

    token_strong, word_strong = strong(token_run), strong(word_run)
    exhausted = len(tokens) == 512 and first is None
    strong_either = token_strong or word_strong
    if first is not None:
        category = 'eos_terminated'
    elif exhausted:
        category = ('budget_exhausted_with_periodic_repetition' if strong_either
                    else 'budget_exhausted_without_detected_periodicity')
    else:
        category = 'unexpected_stop_without_eos'
    return dict(returned_width=len(tokens), first_effective_eos_position=first,
                analysis_token_count=len(analysis), token_periodic_run=token_run,
                preparse_word_periodic_run=word_run,
                strong_token_repetition=token_strong,
                strong_word_repetition=word_strong,
                strong_periodic_repetition=strong_either,
                budget_exhausted=exhausted, budget_loop=strong_either and exhausted,
                operational_category=category)


def load_metrics_helper(path):
    spec = importlib.util.spec_from_file_location('compared_onset_metrics', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def self_test():
    helper = load_metrics_helper(REPO / 'src/asr_lab/repetition_metrics.py')
    examples = [[], [1], [1, 2, 1, 2, 1], [7] * 40,
                [9] + [1, 2, 3] * 20 + [1], list(range(64))]
    rng = random.Random(20261023)
    examples += [[rng.randrange(4) for _ in range(n)] for n in (2, 5, 19, 64)]
    for sequence in examples:
        for maximum in (1, 8, 16):
            require(brute_periods(sequence, maximum) ==
                    helper.longest_periodic_run(sequence, maximum),
                    'Independent exhaustive periodicity disagrees with helper')
    require(brute_periods([0, 1] * 20 + [0], 2)['following_partial_period'] == [0],
            'Trailing partial period must not count as a complete repetition')
    print('CPU algorithm self-test passed: 30 independent brute/helper comparisons.')


def audit(args):
    # This precedes every third-party import; no torch.cuda API is called.
    os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1',
                      TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1')
    import torch
    import jiwer
    from transformers import AutoTokenizer
    from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
    from qwen_asr.inference.utils import parse_asr_output, detect_and_fix_repetitions

    run = args.run_dir.resolve()
    reference = read(args.model_reference)
    require(reference['stage_order'] == list(ORDER), 'Stage order differs')
    require(reference['target_id'] == TARGET and reference['generation_kwargs'] == KWARGS,
            'Frozen target/kwargs differ')
    campaign = read(run / 'campaign.json')
    require((campaign['status'], campaign['stage']) ==
            ('complete', 'four_preset_singletons_complete'), 'Campaign is incomplete')
    require((campaign['actual_generate_calls'], campaign['actual_consumed_examples']) ==
            (4, 4), 'Exactly four saved calls/consumptions required')
    require(campaign['automatic_training'] is False, 'Unexpected training authorization')
    prior = Path(reference['prior_singleton']['worker_dir'])
    paths = {args.model_reference.resolve(), run / 'campaign.json'}

    def freeze(path, expected=None, size=None):
        path = Path(path)
        require(not path.name.endswith('.safetensors'), 'No weight content rehash permitted')
        actual = sha(path)
        if expected is not None:
            require(actual == expected, f'Artifact/source SHA differs: {path}')
        if size is not None:
            require(path.stat().st_size == size, f'Artifact size differs: {path}')
        paths.add(path.resolve())
        return actual

    for record in reference['prior_singleton']['files'].values():
        freeze(record['path'], record['sha256'], record['bytes'])
    prior_identity = reference['anchors']['prior_identity']
    freeze(prior_identity['path'], prior_identity['sha256'])
    old_inputs, old_metadata = read(prior / 'inputs.json'), read(prior / 'metadata.json')
    old_effective = read(prior / 'effective-generation.json')
    require(old_inputs == reference['prior_singleton']['inputs'], 'Old input anchor differs')
    old_tokens = torch.load(prior / 'raw-tokens.pt', map_location='cpu', weights_only=True)
    old_input_tensors = torch.load(prior / 'inputs.pt', map_location='cpu', weights_only=True)
    for path, expected in reference['shared_identity']['files'].items():
        freeze(path, expected)
    for name, expected in reference['shared_identity']['packages'].items():
        require(importlib.metadata.version(name) == expected, f'Package drift: {name}')
    parser_path = Path(inspect.getsourcefile(parse_asr_output))
    freeze(parser_path, reference['shared_identity']['files'][str(parser_path)])
    normalizer = EnglishTextNormalizer(read(reference['shared_identity']['spelling']))
    helper_path = REPO / 'src/asr_lab/repetition_metrics.py'
    helper = load_metrics_helper(helper_path)
    # Bind comparison helper to the frozen runner receipt, not an unanchored live source.
    freeze(helper_path, campaign['source_sha256'][str(helper_path)])
    stage_records, tensors, effective_rules, metadata_rows = [], {}, {}, {}

    def count(ref, hyp):
        s = jiwer.process_words(ref, hyp)
        return dict(substitutions=s.substitutions, deletions=s.deletions,
                    insertions=s.insertions, hits=s.hits,
                    reference_words=s.hits+s.substitutions+s.deletions)

    for stage, step in zip(ORDER, STEPS):
        outer, model = run / stage, reference['models'][stage]
        directory = outer / 'target-single'
        result, terminal, receipt = (read(directory / name) for name in
                                     ('result.json', 'status.json', 'stage.json'))
        require(result['status'] == terminal['status'] == receipt['status'] == 'complete'
                and receipt['exit_code'] == 0, f'{stage}: incomplete worker')
        require((result['generate_calls'], result['consumed_examples'], result['optimizer_updates'])
                == (1, 1, 0), f'{stage}: call/update count differs')
        require(not result['raw_generation_missing'], f'{stage}: raw evidence missing')
        for name, item in result['artifacts'].items():
            freeze(directory / name, item['sha256'], item['bytes'])
        for name in ('result.json', 'status.json', 'stage.json'):
            freeze(directory / name)
        identity_path = outer / 'identity.json'
        freeze(identity_path, result['identity_sha256'])
        identity = read(identity_path)
        require(identity['checkpoint'] == model['path'], f'{stage}: model path differs')
        require(identity['onset']['model_reference_sha256'] == sha(args.model_reference),
                f'{stage}: model reference binding differs')
        verification = read(outer / 'model-verification.json')
        freeze(outer / 'model-verification.json')
        require(verification['status'] == 'passed_with_post_worker_stat_match',
                f'{stage}: incomplete model identity checks')
        require(set(verification['files']) == set(model['files']), 'Model inventory differs')
        if stage != 'base':
            marker_path = Path(model['path']) / 'prefix-complete.json'
            freeze(marker_path, model['historical_anchor']['prefix_complete']['sha256'])
            marker = read(marker_path)
            require(marker['step'] == step, 'Historical checkpoint step differs')
            freeze(Path(model['path']) / 'checkpoint.json',
                   model['historical_anchor']['checkpoint_metadata']['sha256'])
        base_new_captures = []
        for name, item in verification['files'].items():
            expected = model['files'][name]
            require(item['expected_sha256'] == expected['sha256'], 'Expected hash rebased')
            require(item['historical_expected_sha256'] == expected['historical_expected_sha256'],
                    'Historical expected hash rebased')
            require(item['before'] == item['after_hash'] == verification['post_worker_stats'][name],
                    f'{stage}: model stat drift')
            require(item['before']['size'] == expected['bytes'], 'Model size differs')
            for key in ('logical_path', 'resolved_path', 'dev', 'inode', 'size',
                        'mtime_ns', 'ctime_ns', 'is_symlink', 'symlink_target', 'symlink_lstat'):
                require(key in item['before'], f'Missing complete stat field: {key}')
            require(item['captured_in_this_stage'] is True and
                    len(item['actual_sha256']) == 64, 'Missing one-time stage content hash')
            if expected['sha256'] is None:
                require(stage == 'base' and name.endswith('.safetensors') and item['base_first_capture'],
                        'Unanchored hash outside declared base shards')
                base_new_captures.append(name)
            else:
                require(item['actual_sha256'] == expected['sha256'], 'Stage hash mismatch')
            if stage != 'base':
                require(item['actual_sha256'] == marker['files'][name]['sha256'],
                        'Current checkpoint hash differs from historical marker')
        inputs, raw, decoded, metadata, config = (read(directory / name) for name in
            ('inputs.json', 'raw-tokens.json', 'decoded.json', 'metadata.json', 'effective-generation.json'))
        require(inputs == old_inputs, f'{stage}: frozen input JSON drift')
        require(all(metadata[k] == old_metadata[k] for k in COMMON_FIELDS),
                f'{stage}: runtime/input metadata drift')
        require(metadata['generation_kwargs'] == KWARGS and metadata['ids'] == [TARGET]
                and metadata['optimizer_updates'] == metadata['backward_calls'] == 0
                and metadata['train_called'] is False and metadata['model_training'] is False
                and metadata['inference_mode'] is True, f'{stage}: inference contract differs')
        require(len(config) == len(old_effective) == 1, 'Missing effective configuration')
        rules = {k: config[0][k] for k in ('effective', 'special_tokens', 'logits_processor_types')}
        require(rules == {k: old_effective[0][k] for k in rules}, 'Effective rule drift')
        require(all(rules['effective'][k] == v for k, v in KWARGS.items()), 'Effective kwargs differ')
        require(rules['logits_processor_types'] == [], 'Unexpected logits processors')
        eos = set(rules['special_tokens']['_eos_token_tensor'])
        pad = rules['special_tokens']['_pad_token_tensor']
        require(eos == {151643, 151645}, 'Configured EOS differs')
        ipt = torch.load(directory / 'inputs.pt', map_location='cpu', weights_only=True)
        rpt = torch.load(directory / 'raw-tokens.pt', map_location='cpu', weights_only=True)
        require(set(ipt) == set(old_input_tensors), 'Input tensor key drift')
        for key, tensor in ipt.items():
            require(tensor.device.type == 'cpu' and torch.equal(tensor, old_input_tensors[key]),
                    f'{stage}: input tensor drift: {key}')
            fingerprint = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                numel=tensor.numel(), sha256=hashlib.sha256(
                    tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            require(fingerprint == inputs['tensors'][key], 'Input tensor fingerprint mismatch')
            if tensor.is_floating_point():
                require(bool(torch.isfinite(tensor).all()), 'Nonfinite input features')
        width = inputs['prompt_padded_width']
        require(ipt['input_ids'].tolist() == inputs['input_ids'] and
                ipt['attention_mask'].tolist() == inputs['attention_mask'], 'Input PT/JSON differs')
        require(ipt['attention_mask'].sum(-1).tolist() == inputs['effective_prompt_tokens'],
                'Effective prompt count differs')
        for mask in inputs['attention_mask']:
            require(mask == sorted(mask) and set(mask) <= {0, 1}, 'Invalid left padding')
        require(raw['prompt_padded_width'] == width == ipt['input_ids'].shape[1], 'Prompt width differs')
        require(rpt['sequences'].tolist() == raw['full_sequences'] and
                rpt['continuation'].tolist() == raw['continuation_token_ids'], 'Raw PT/JSON differs')
        require(torch.equal(rpt['sequences'][:, :width], ipt['input_ids']) and
                torch.equal(rpt['sequences'][:, width:], rpt['continuation']), 'Prefix/slice mismatch')
        require(len(decoded) == 1 and decoded[0]['id'] == TARGET and rpt['sequences'].shape[0] == 1,
                'Expected one target singleton')
        d, tokens = decoded[0], raw['continuation_token_ids'][0]
        require(0 < len(tokens) <= 512 and tokens == d['token_ids'], 'Invalid continuation')
        eos_positions = [i for i, t in enumerate(tokens) if t in eos]
        first = eos_positions[0] if eos_positions else None
        require(d['effective_eos_positions'] == d['legacy_eos_positions'] == eos_positions
                and d['first_effective_eos_position'] == first, 'EOS position mismatch')
        require(d['pad_id'] == pad and d['pad_id_positions'] ==
                [i for i, t in enumerate(tokens) if t == pad], 'PAD position mismatch')
        require(d['special_token_positions'] == [dict(position=i, token_id=t)
                for i, t in enumerate(tokens) if t in metadata['tokenizer']['all_special_ids']],
                'Special-token position mismatch')
        require(d['positions_after_first_eos'] ==
                (list(range(first+1, len(tokens))) if first is not None else []), 'EOS tail mismatch')
        require(d['legacy_generation_truncated'] == d['effective_eos_generation_truncated'] ==
                (len(tokens) >= 512 and not eos_positions), 'Truncation rule mismatch')
        tokenizer = AutoTokenizer.from_pretrained(model['path'], local_files_only=True,
                                                  fix_mistral_regex=True)
        keep = tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        skip = tokenizer.decode(tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        require((keep, skip) == (d['decoded_keep_special'], d['decoded_skip_special']) and
                tokenizer.convert_ids_to_tokens(tokens) == d['token_pieces'], 'Independent decode mismatch')
        parsed = list(parse_asr_output(skip, user_language='English'))
        fixed = detect_and_fix_repetitions(skip.strip())
        require(d['parser_input'] == skip and d['parser_kwargs'] == dict(user_language='English')
                and parsed == d['parser_output'] and fixed == parsed[1] ==
                d['repetition_fix_observation']['output'], 'Parser/repetition helper mismatch')
        ref = normalizer(d['reference']).strip()
        hyp = normalizer(parsed[1]).strip()
        require(d['reference'] == reference['target']['manifest_row']['text'] and
                (ref, hyp) == (d['reference_normalized'], d['hypothesis_normalized']),
                'Reference/normalizer mismatch')
        counts = count(ref, hyp)
        require(counts == d['counts'] and counts['reference_words'] > 0, 'S/D/I/H mismatch')
        require(metadata['audio'][0]['pcm_sha256'] == reference['target']['pcm_sha256'] and
                metadata['audio'][0]['file_sha256'] == reference['target']['file_sha256'],
                'Target PCM/FLAC identity mismatch')
        independently = independent_observation(tokens, eos, skip)
        implementation = helper.repetition_observation(tokens, eos, skip)
        saved = read(outer / 'observation.json')
        freeze(outer / 'observation.json')
        for key, value in independently.items():
            require(value == implementation[key] == saved[key], f'Periodicity mismatch: {key}')
        require(saved['stage'] == stage and saved['step'] == step and saved['counts'] == counts,
                'Saved observation metadata/counts mismatch')
        guard = read(directory / 'input-guard.json')
        require(guard['status'] == 'input_gate_passed' and guard['actual_generate_calls'] == 1,
                'Missing pre-generation identity guard')
        stage_records.append(dict(stage=stage, step=step, all_assertions_passed=True,
            tokenizer_class=str(type(tokenizer)), tokenizer_source=source(inspect.getsourcefile(type(tokenizer))),
            artifacts_hash_checked=len(result['artifacts']), inference_calls=1, optimizer_updates=0,
            raw_pt_json_prefix_and_slice_match=True, all_inputs_equal_prior_singleton=True,
            runtime_and_effective_equal_prior_singleton=True, eos_positions=eos_positions,
            pad_id=pad, preparse_word_frequency=dict(Counter(skip.split())), parser_output=parsed,
            normalized_hypothesis=hyp, counts=counts,
            preparse_counts=count(ref, normalizer(skip).strip()),
            independent_periodicity=independently, source_helper_and_saved_periodicity_match=True,
            base_shards_newly_captured_without_historical_sha=base_new_captures,
            model_hash_policy='Compared saved single-stage content hashes and stable before/after/post-worker stat; no weight content rehash during audit.'))
        tensors[stage], effective_rules[stage], metadata_rows[stage] = rpt, rules, metadata
    summary = read(run / 'onset-summary.json')
    freeze(run / 'onset-summary.json')
    onset = {}
    for key in ('strong_periodic_repetition', 'budget_loop'):
        flags = [s['independent_periodicity'][key] for s in stage_records]
        positive = next((i for i, value in enumerate(flags) if value), None)
        reversal = any(flags[i] and not all(flags[i:]) for i in range(4))
        onset[key] = dict(earliest_observed_stage=ORDER[positive] if positive is not None else None,
            earliest_observed_step=STEPS[positive] if positive is not None else None,
            previous_sampled_negative_step=STEPS[positive-1] if positive is not None and positive > 0 else None,
            observed_positive_then_negative=reversal)
        require(all(summary[key][k] == v for k, v in onset[key].items()), 'Onset summary mismatch')
    require(summary['exact_first_update_identified'] is False and summary['training_cause_identified'] is False,
            'Unsupported exact onset/cause claim')
    return dict(status='passed',audited_utc=datetime.now(timezone.utc).isoformat(),
        audit_mode='Independent CPU-only saved-evidence audit; no new generation or weight-content rehash.',
        source_script=source(__file__), command=shlex.join([sys.executable, *sys.argv]),
        CPU_VISIBLE_DEVICE_CONTRACT=os.environ['CUDA_VISIBLE_DEVICES'],
        package_versions={n: importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','qwen-asr','jiwer')},
        parser_source=source(parser_path),normalizer_source=source(inspect.getsourcefile(EnglishTextNormalizer)),
        jiwer_source=source(inspect.getsourcefile(jiwer.process_words)),helper_source=source(helper_path),
        audited_generate_calls=4,audited_consumed_examples=4,optimizer_updates=0,
        stages=stage_records, independently_recomputed_onset=onset,
        step100_full_sequences_equal_old_singleton=torch.equal(tensors['step100']['sequences'], old_tokens['sequences']),
        step100_continuation_equal_old_singleton=torch.equal(tensors['step100']['continuation'], old_tokens['continuation']),
        sources=[source(p) for p in sorted(paths)],
        limits=['Four saved sampled checkpoints only; exact first update, monotonicity and training cause remain unknown.',
                'Base pin/path are historical; its shard SHA is a new stage capture, not a historical file hash.',
                'Model-content hashes are checked against saved stage verification; audit does not repeat expensive weight scans.',
                'Old target-single is an observed fresh-worker reference, not the original early100 full-dev token stream.',
                'No GPU/model loading, decoding generation, training, or extra sample inference occurred in this audit.'])


def markdown(report):
    lines = ['# Repetition onset 独立 CPU 审计', '',
             '状态：**passed**。四个保存阶段的张量、切分、EOS、tokenizer/parser/normalizer、S/D/I及文件指纹通过独立复算。', '',
             '| 阶段 | 返回token数 | 首EOS位置 | 强重复 | 512无EOS loop | 解析结果 |',
             '|---|---:|---:|---|---|---|']
    for row in report['stages']:
        o = row['independent_periodicity']
        text = row['parser_output'][1].replace('|', '\\|').replace('\n', ' ')
        lines.append(f"| {row['stage']} | {o['returned_width']} | {o['first_effective_eos_position']} | {o['strong_periodic_repetition']} | {o['budget_loop']} | {text} |")
    lines += ['', '暴力枚举每个起点和完整周期，独立核对源码helper与保存统计；未将合成重复测试当作实际token证据。', '',
              '4次generate / 4次样本消费 / 0训练更新；本审计没有新增推理。Base shard SHA是本轮首次捕获；25/50/100对照历史prefix-complete。', '',
              '仅观察保存采样点，不能确定精确首发update、持续性、单调性或训练根因。旧singleton参考不等于原Early100全dev原始token重放。', '',
              f"step100完整sequences与旧singleton一致：{report['step100_full_sequences_equal_old_singleton']}。", '',
              '[完整命令、环境、逐项证据与SHA](independent-audit.json)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--model-reference', type=Path,
                        default=REPO/'reports/repetition-onset/inputs/model-reference.json')
    parser.add_argument('--output-json', type=Path,
                        default=REPO/'reports/repetition-onset/independent-audit.json')
    parser.add_argument('--output-md', type=Path,
                        default=REPO/'reports/repetition-onset/independent-audit.md')
    parser.add_argument('--self-test', action='store_true',
                        help='Only synthetic CPU periodicity algorithm tests; no run/model/tokenizer imports')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.run_dir is None:
        parser.error('--run-dir required for actual saved-evidence audit')
    try:
        report = audit(args)
    except Exception as error:
        report = dict(status='failed', error=f'{type(error).__name__}: {error}',
                      source_script=source(__file__),
                      command=shlex.join([sys.executable, *sys.argv]),
                      no_new_generation_performed=True)
        args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        args.output_md.write_text('# Repetition onset 独立 CPU 审计\n\n状态：**failed**。\n\n'+report['error']+'\n')
        raise
    args.output_json.write_text(json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False)+'\n')
    args.output_md.write_text(markdown(report))
    print(json.dumps(dict(status=report['status'], output_json=str(args.output_json),
                          output_md=str(args.output_md)), indent=2))


if __name__ == '__main__':
    main()
