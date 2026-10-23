#!/usr/bin/env python3
"""Paired corpus-WER deltas and percentile bootstrap from saved predictions.

Uses the saved normalized text, without running any model or lexical normalizer.
Negative delta means candidate improvement. Speaker clusters require a manifest.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import random
import sys

import jiwer

COUNT_KEYS = ('hits', 'substitutions', 'deletions', 'insertions')
TEXT_KEYS = ('reference', 'hypothesis', 'reference_normalized', 'hypothesis_normalized')
DEFAULT_SEED = 20261022
DEFAULT_REPLICATES = 10000


def load_jsonl(path):
    rows = []
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f'{path}:{number}: invalid JSON') from error
    return rows


def index_predictions(rows, label):
    result = {}
    for row in rows:
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f'{label}: missing/non-string utterance ID')
        if identifier in result:
            raise ValueError(f'{label}: duplicate utterance ID {identifier}')
        if any(not isinstance(row.get(key), str) for key in TEXT_KEYS):
            raise ValueError(f'{label}: {identifier}: requires raw and normalized reference/hypothesis strings')
        if not row['reference_normalized'].split():
            raise ValueError(f'{label}: {identifier}: zero reference-word denominator')
        result[identifier] = row
    if not result:
        raise ValueError(f'{label}: empty predictions')
    return result


def count_edits(reference, hypothesis):
    # Match the stored whitespace-delimited words. No punctuation/case/spelling changes.
    transform = jiwer.ReduceToListOfListOfWords()
    result = jiwer.process_words(' '.join(reference.split()), ' '.join(hypothesis.split()),
        reference_transform=transform, hypothesis_transform=transform)
    counts = {key: int(getattr(result, key)) for key in COUNT_KEYS}
    counts['reference_words'] = counts['hits'] + counts['substitutions'] + counts['deletions']
    counts['edits'] = counts['substitutions'] + counts['deletions'] + counts['insertions']
    if counts['reference_words'] <= 0:
        raise ValueError('zero reference-word denominator')
    return counts


def summarize(counts):
    result = {key: sum(row[key] for row in counts) for key in (*COUNT_KEYS, 'reference_words', 'edits')}
    if result['reference_words'] <= 0:
        raise ValueError('zero corpus reference-word denominator')
    result['wer'] = result['edits'] / result['reference_words']
    result['perfect_utterances'] = sum(row['edits'] == 0 for row in counts)
    return result


def verify_sidecar(metadata, score, identifiers, label):
    for key in (*COUNT_KEYS, 'reference_words', 'edits'):
        if key in metadata and (type(metadata[key]) is not int or metadata[key] != score[key]):
            raise ValueError(f'{label} metrics mismatch for {key}: declared {metadata[key]}, recomputed {score[key]}')
    if 'wer' in metadata:
        if type(metadata['wer']) not in (int,float):
            raise ValueError(f'{label} metrics WER must be numeric')
        value = float(metadata['wer'])
        if not math.isfinite(value) or not math.isclose(value, score['wer'], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'{label} metrics WER mismatch')
    if 'examples' in metadata and (type(metadata['examples']) is not int or metadata['examples'] != len(identifiers)):
        raise ValueError(f'{label} metrics example-count mismatch')
    declared_ids = metadata.get('selected_ids', metadata.get('subset', {}).get('selected_ids'))
    if declared_ids is not None and (len(declared_ids) != len(set(declared_ids)) or set(declared_ids) != set(identifiers)):
        raise ValueError(f'{label} metrics selected IDs mismatch')
    if 'generation_truncated_count' in metadata:
        count = metadata['generation_truncated_count']
        if type(count) is not int or not 0 <= count <= len(identifiers):
            raise ValueError(f'{label} invalid generation_truncated_count')
        if 'generation_truncated_fraction' in metadata:
            fraction = metadata['generation_truncated_fraction']
            if type(fraction) not in (int,float) or not math.isfinite(fraction) or not math.isclose(fraction,count/len(identifiers),abs_tol=1e-12):
                raise ValueError(f'{label} truncation fraction/count mismatch')
    required = (*COUNT_KEYS, 'reference_words', 'wer', 'examples')
    return dict(verified_fields=[key for key in (*required, 'edits') if key in metadata],
        complete=all(key in metadata for key in required))


def metadata_consistency(baseline, candidate):
    fields = ('normalizer', 'normalizer_revision')
    checks = {}
    issues = []
    for field in fields:
        left, right = baseline.get(field), candidate.get(field)
        if left is not None and right is not None and left != right:
            raise ValueError(f'Normalizer metadata mismatch for {field}: {left!r} != {right!r}')
        checks[field] = dict(baseline=left, candidate=right,
            consistent=True if left is not None and right is not None else None)
        if any(not isinstance(value,str) or not value.strip() for value in (left,right)):
            issues.append(field + '_missing_or_invalid')
    checks['status'] = 'verified_declared_metadata' if all(checks[key]['consistent'] for key in fields) else 'metadata_incomplete'
    if 'normalizer_settings' in baseline or 'normalizer_settings' in candidate:
        left, right = baseline.get('normalizer_settings'), candidate.get('normalizer_settings')
        if left != right:
            issues.append('normalizer_settings_mismatch_or_missing')
        checks['normalizer_settings'] = dict(baseline=left, candidate=right, equal=left == right)
    if issues:
        checks['status'] = 'metadata_incomplete'
    checks['interpretation'] = 'Exact saved normalized references are verified; missing metadata cannot establish identical hypothesis normalization.'
    for field in ('manifest_sha256', 'evaluation_batch_size', 'generation_max_new_tokens', 'generation_truncated_count'):
        left, right = baseline.get(field), candidate.get(field)
        checks[field] = dict(baseline=left, candidate=right,
            equal=left == right if left is not None and right is not None else None)
        if left is None or right is None:
            issues.append(field + '_missing')
            continue
        elif left != right:
            issues.append(field + '_mismatch')
        if field == 'manifest_sha256':
            if any(not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value) for value in (left,right)):
                issues.append('manifest_sha256_invalid')
        elif any(type(value) is not int or value < (0 if field=='generation_truncated_count' else 1) for value in (left,right)):
            issues.append(field + '_invalid')
    if baseline.get('generation_truncated_count') != 0 or candidate.get('generation_truncated_count') != 0:
        issues.append('generation_truncation_present_or_unknown')
    checks['issues'] = list(dict.fromkeys(issues))
    return checks


def speaker_mapping(manifest_rows, predictions):
    mapping, seen = {}, set()
    for row in manifest_rows:
        identifier = row.get('id')
        if identifier in seen:
            raise ValueError(f'Manifest duplicate ID: {identifier}')
        seen.add(identifier)
        if identifier not in predictions:
            continue
        speaker = row.get('speaker_id')
        if speaker is None or not str(speaker).strip():
            raise ValueError(f'Manifest lacks speaker_id for {identifier}')
        if 'text' in row and row['text'] != predictions[identifier]['reference']:
            raise ValueError(f'Manifest raw reference mismatch for {identifier}')
        mapping[identifier] = str(speaker)
    missing = set(predictions) - set(mapping)
    if missing:
        raise ValueError(f'Manifest missing {len(missing)} paired IDs, e.g. {min(missing)}')
    return mapping


def percentile(sorted_values, probability):
    location = (len(sorted_values) - 1) * probability
    lower = math.floor(location)
    upper = math.ceil(location)
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (location - lower)


def paired_bootstrap(rows, unit='utterance', replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED):
    if replicates < 100:
        raise ValueError('At least 100 bootstrap replicates are required')
    groups = defaultdict(lambda: [0, 0, 0])
    for row in rows:
        key = row['id'] if unit == 'utterance' else row.get('speaker_id')
        if key is None:
            raise ValueError('Speaker-cluster bootstrap requires manifest speaker_id')
        group = groups[key]
        group[0] += row['baseline']['edits']
        group[1] += row['candidate']['edits']
        group[2] += row['baseline']['reference_words']
    units = [groups[key] for key in sorted(groups)]
    if not units:
        raise ValueError('No bootstrap units')
    rng, deltas = random.Random(seed), []
    for _ in range(replicates):
        baseline_edits = candidate_edits = reference_words = 0
        for _ in units:
            baseline, candidate, words = units[rng.randrange(len(units))]
            baseline_edits += baseline
            candidate_edits += candidate
            reference_words += words
        if reference_words <= 0:
            raise ValueError('Zero denominator in bootstrap resample')
        deltas.append((candidate_edits - baseline_edits) / reference_words)
    deltas.sort()
    interval = [percentile(deltas, .025), percentile(deltas, .975)]
    return dict(unit='speaker_cluster' if unit == 'speaker' else 'utterance', units=len(units),
        replicates=replicates, seed=seed, confidence_level=.95,
        method='paired percentile bootstrap; corpus edits/reference words recomputed for each resample',
        wer_delta_95_ci=interval, wer_delta_percentage_points_95_ci=[100*value for value in interval],
        scope='Conditional on saved predictions; does not estimate variation across training seeds.')


def analyze_records(baseline_rows, candidate_rows, manifest_rows=None, baseline_metrics=None,
                    candidate_metrics=None, bootstrap_unit='auto', replicates=DEFAULT_REPLICATES, seed=DEFAULT_SEED,
                    manifest_sha256=None):
    baseline = index_predictions(baseline_rows, 'baseline')
    candidate = index_predictions(candidate_rows, 'candidate')
    if set(baseline) != set(candidate):
        raise ValueError(f'Paired ID sets differ: baseline-only={len(set(baseline)-set(candidate))}, candidate-only={len(set(candidate)-set(baseline))}')
    normalizer = metadata_consistency(baseline_metrics or {}, candidate_metrics or {})
    speakers = speaker_mapping(manifest_rows, baseline) if manifest_rows is not None else None
    rows = []
    for identifier in sorted(baseline):
        left, right = baseline[identifier], candidate[identifier]
        for key in ('reference', 'reference_normalized'):
            if left[key] != right[key]:
                raise ValueError(f'Paired {key} mismatch for {identifier}')
        left_score = count_edits(left['reference_normalized'], left['hypothesis_normalized'])
        right_score = count_edits(right['reference_normalized'], right['hypothesis_normalized'])
        for label, prediction, score in (('baseline',left,left_score),('candidate',right,right_score)):
            for field in (*COUNT_KEYS, 'reference_words', 'edits'):
                if field in prediction and (type(prediction[field]) is not int or prediction[field] != score[field]):
                    raise ValueError(f'{label} stored {field} mismatch for {identifier}')
        if left_score['reference_words'] != right_score['reference_words']:
            raise ValueError(f'Paired denominator mismatch for {identifier}')
        rows.append(dict(id=identifier, speaker_id=speakers[identifier] if speakers else None,
            baseline=left_score, candidate=right_score, edit_delta=right_score['edits']-left_score['edits'],
            normalized_hypothesis_changed=left['hypothesis_normalized'] != right['hypothesis_normalized']))
    left_score, right_score = summarize([r['baseline'] for r in rows]), summarize([r['candidate'] for r in rows])
    left_validation = verify_sidecar(baseline_metrics or {}, left_score, baseline, 'baseline')
    right_validation = verify_sidecar(candidate_metrics or {}, right_score, candidate, 'candidate')
    issues = list(normalizer['issues'])
    for label, validation in (('baseline',left_validation),('candidate',right_validation)):
        if not validation['complete']:
            issues.append(label + '_stored_metrics_incomplete')
    actual_manifest_match = None
    if manifest_sha256 is not None:
        actual_manifest_match = all(meta.get('manifest_sha256') == manifest_sha256
            for meta in (baseline_metrics or {}, candidate_metrics or {}))
        if not actual_manifest_match:
            issues.append('supplied_manifest_file_hash_mismatch')
    else:
        issues.append('manifest_file_hash_not_verified')
    for label, predictions, metadata in (('baseline',baseline,baseline_metrics or {}),('candidate',candidate,candidate_metrics or {})):
        flags = [prediction.get('generation_truncated') for prediction in predictions.values()]
        if any(flag is not None and type(flag) is not bool for flag in flags):
            raise ValueError(label + ' invalid stored generation_truncated flag')
        if any(flag is True for flag in flags):
            issues.append(label + '_stored_generation_truncation_present')
        if all(flag is not None for flag in flags) and 'generation_truncated_count' in metadata:
            if sum(flags) != metadata['generation_truncated_count']:
                raise ValueError(label + ' stored truncation flags disagree with metrics')
    if bootstrap_unit not in ('auto','utterance','speaker','both'):
        raise ValueError('Invalid bootstrap unit')
    primary = 'speaker' if speakers is not None and bootstrap_unit in ('auto','both') else bootstrap_unit
    if primary == 'auto':
        primary = 'utterance'
    units = ['utterance','speaker'] if bootstrap_unit == 'both' else [primary]
    if 'speaker' in units and speakers is None:
        raise ValueError('Speaker bootstrap requires --manifest with speaker_id')
    delta = right_score['wer'] - left_score['wer']
    report = dict(comparison_valid=not issues, comparison_invalid_reasons=list(dict.fromkeys(issues)),
        provenance=provenance(), paired_utterances=len(rows), speakers=len(set(speakers.values())) if speakers else None,
        validation=dict(exact_id_sets=True, exact_raw_references=True, exact_normalized_references=True,
            identical_reference_word_denominators=True, stored_normalized_text_used=True,
            lexical_normalizer_rerun=False, normalizer_metadata=normalizer,
            baseline_stored_metrics=left_validation,candidate_stored_metrics=right_validation,
            supplied_manifest_file_hash_match=actual_manifest_match),
        baseline=left_score, candidate=right_score,
        comparison=dict(delta_definition='candidate WER minus baseline WER; negative is improvement',
            wer_delta=delta, wer_delta_percentage_points=100*delta,
            relative_wer_reduction=(left_score['wer']-right_score['wer'])/left_score['wer'] if left_score['wer'] else None),
        utterance_changes=dict(improved=sum(r['edit_delta']<0 for r in rows),
            regressed=sum(r['edit_delta']>0 for r in rows), tied=sum(r['edit_delta']==0 for r in rows),
            normalized_hypothesis_changed=sum(r['normalized_hypothesis_changed'] for r in rows)),
        edit_deltas={key:right_score[key]-left_score[key] for key in ('substitutions','deletions','insertions')},
        primary_bootstrap_unit='speaker_cluster' if primary=='speaker' else primary,
        bootstrap={unit:paired_bootstrap(rows,unit,replicates,seed) for unit in units})
    return report, rows


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def provenance():
    return dict(python_version=sys.version.split()[0],
        packages={name:importlib.metadata.version(name) for name in ('jiwer','rapidfuzz')},
        analysis_source_sha256=sha256(__file__),
        jiwer_process_source_sha256=sha256(inspect.getsourcefile(jiwer.process_words)))


def read_metrics(predictions, explicit=None):
    path = Path(explicit) if explicit else Path(predictions).with_suffix('.metrics.json')
    if explicit or path.exists():
        return json.loads(path.read_text()), path
    return {}, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--manifest', type=Path, help='Speaker IDs; may contain additional corpus utterances')
    parser.add_argument('--baseline-metrics', type=Path)
    parser.add_argument('--candidate-metrics', type=Path)
    parser.add_argument('--bootstrap-unit', choices=('auto','utterance','speaker','both'), default='auto')
    parser.add_argument('--replicates', type=int, default=DEFAULT_REPLICATES)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--require-valid', action='store_true', help='Write report then exit2 if comparison validity gates are unmet')
    parser.add_argument('--details', type=Path, help='Optional per-utterance edit-count JSONL')
    args = parser.parse_args()
    left_metadata, left_path = read_metrics(args.baseline,args.baseline_metrics)
    right_metadata, right_path = read_metrics(args.candidate,args.candidate_metrics)
    report, details = analyze_records(load_jsonl(args.baseline),load_jsonl(args.candidate),
        load_jsonl(args.manifest) if args.manifest else None,left_metadata,right_metadata,
        args.bootstrap_unit,args.replicates,args.seed,
        manifest_sha256=sha256(args.manifest) if args.manifest else None)
    report['inputs'] = {label:dict(path=str(path.resolve()),sha256=sha256(path)) for label,path in
        (('baseline',args.baseline),('candidate',args.candidate),('manifest',args.manifest),
         ('baseline_metrics',left_path),('candidate_metrics',right_path)) if path is not None}
    input_paths = {path.resolve() for path in (args.baseline,args.candidate,args.manifest,left_path,right_path) if path is not None}
    if args.output.resolve() in input_paths or (args.details and (args.details.resolve() in input_paths or args.details.resolve()==args.output.resolve())):
        parser.error('Output/details must not overwrite prediction, manifest, or metrics inputs')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix+'.tmp')
    temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    temporary.replace(args.output)
    if args.details:
        args.details.parent.mkdir(parents=True,exist_ok=True)
        args.details.write_text(''.join(json.dumps(row,allow_nan=False)+'\n' for row in details))
    print(json.dumps(dict(comparison_valid=report['comparison_valid'],
        comparison_invalid_reasons=report['comparison_invalid_reasons'],baseline_wer=report['baseline']['wer'],candidate_wer=report['candidate']['wer'],
        wer_delta=report['comparison']['wer_delta'],bootstrap=report['bootstrap'],output=str(args.output)),indent=2))
    if args.require_valid and not report['comparison_valid']:
        raise SystemExit(2)


if __name__=='__main__':
    main()
