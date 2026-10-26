#!/usr/bin/env python3
"""Verify a corrected derived overlay against immutable saved CPU evidence.

The original independent FP64 auditor is imported from SHA-pinned source bytes.
Every original result artifact is still read/hashed/loaded by that auditor before
selected derived JSON values are substituted. No arithmetic/tolerance is changed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import time
import types

ORIGINAL_AUDITOR_SHA256 = 'ebc8db219efce0f05db3859976cf0d41834d4cfac2160837cb6418fe8b6e921d'
ORIGINAL_WORKER_SHA256 = 'acc34e309468c950b02e930ee7be7d45a0394b092ad0b0a35e07f83016f3a43b'
STAGES = ('base', 'step050', 'step100')
REPO = Path(__file__).resolve().parents[3]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    raw = Path(path).read_bytes()
    return json.loads(raw), dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def import_original():
    path = Path(__file__).with_name('audit_saved_evidence.py')
    source = path.read_bytes()
    require(hashlib.sha256(source).hexdigest() == ORIGINAL_AUDITOR_SHA256,
            'Original auditor source SHA differs; do not modify arithmetic or tolerances')
    module = types.ModuleType('frozen_original_token_probability_audit')
    module.__file__ = str(path)
    exec(compile(source, str(path), 'exec'), module.__dict__)
    return module


def expected_keys():
    return {f'{s}/worker/{name}' for s in STAGES for name in (
        'generation/derived.json', 'own-trajectory-comparison.json',
        *(f'teacher/{t}/derived.json' for t in STAGES))}


def validate_manifest(overlay, run, reference):
    overlay, run, reference = overlay.resolve(), run.resolve(), reference.resolve()
    require(not overlay.is_relative_to(run) and not run.is_relative_to(overlay),
            'Overlay and immutable original run must be disjoint directories')
    manifest, identity = read_json(overlay/'manifest.json')
    require(manifest.get('schema_version') == 1, 'Unsupported overlay manifest schema')
    require(Path(manifest['original_run']).is_absolute() and
            Path(manifest['original_run']).resolve() == run, 'Overlay original run differs')
    _, reference_identity = read_json(reference)
    require(manifest['reference_sha256'] == reference_identity['sha256'],
            'Frozen reference SHA differs')
    sources = manifest['original_source_sha256']
    require(isinstance(sources, dict) and
            sources.get('src/asr_lab/token_probability.py') == ORIGINAL_WORKER_SHA256,
            'Original worker provenance differs')
    campaign, campaign_identity = read_json(run/'campaign.json')
    require(campaign['status'] == 'complete', 'Original inference run is incomplete')
    for name, sha in sources.items():
        matches = [value for path,value in campaign['source_sha256'].items()
                   if path == name or path.endswith('/'+name)]
        require(matches and all(value == sha for value in matches),
                'Original source lacks campaign SHA anchor: '+name)
    artifacts = manifest['corrected_artifacts']
    require(set(artifacts) == expected_keys(), 'Overlay must contain exactly12 derived and3 own-comparison JSONs')
    originals, corrected_paths = {}, set()
    for stage in STAGES:
        result, _ = read_json(run/stage/'worker/result.json')
        require(result['status'] == 'complete' and result['stage'] == stage,
                'Original worker result is incomplete')
        for name, record in result['artifacts'].items():
            originals[f'{stage}/worker/{name}'] = record
    redirects = {}
    for name, entry in artifacts.items():
        require(name in originals, 'Overlay file lacks original result artifact: '+name)
        require(entry['original_sha256'] == originals[name]['sha256'] and
                entry['original_bytes'] == originals[name]['bytes'],
                'Overlay original identity differs from immutable result manifest: '+name)
        relative = Path(entry['path'])
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Corrected path must be relative and contained in overlay')
        corrected = (overlay/relative).resolve()
        require(corrected.is_relative_to(overlay) and corrected.suffix == '.json' and
                corrected not in corrected_paths, 'Escaping/duplicate/non-JSON corrected path')
        require(isinstance(entry['bytes'], int) and entry['bytes'] > 0 and
                isinstance(entry['sha256'], str) and len(entry['sha256']) == 64,
                'Invalid corrected file identity')
        corrected_paths.add(corrected)
        redirects[(run/name).resolve()] = (corrected, entry)
    return manifest, identity, campaign_identity, redirects


def overlay_evidence_class(original, redirects):
    class OverlayEvidence(original.Evidence):
        def load(self, path, expected_sha=None, expected_bytes=None):
            path = Path(path).resolve()
            # Always verify/load the immutable original from its original bytes first.
            value = super().load(path, expected_sha, expected_bytes)
            if path not in redirects:
                return value
            corrected, entry = redirects[path]
            record = self.records[path]
            require(record['sha256'] == entry['original_sha256'] and
                    record['bytes'] == entry['original_bytes'],
                    'Actual original derived bytes differ from overlay provenance')
            return super().load(corrected, entry['sha256'], entry['bytes'])
    return OverlayEvidence


def checked_auxiliary(overlay, manifest):
    entries = manifest['auxiliary_artifacts']
    require(set(entries) == {'comparisons.json', 'change-summary.json'}, 'Auxiliary artifact set differs')
    values = {}
    for name, entry in entries.items():
        relative = Path(entry['path'])
        path = (overlay/relative).resolve()
        require(not relative.is_absolute() and '..' not in relative.parts and
                path.is_relative_to(overlay.resolve()), 'Auxiliary path escapes overlay')
        value, identity = read_json(path)
        require(identity['sha256'] == entry['sha256'] and identity['bytes'] == entry['bytes'],
                'Auxiliary artifact identity differs: '+name)
        values[name] = value
    return values


def validate_comparisons(original, manifest, overlay, reference, comparisons, check):
    """Python-only summary reconstruction from previously FP64-audited corrected rows."""
    maxima = {}
    def close(value, expected, tolerance, field):
        original.close(value, expected, tolerance, 'comparisons.'+field, maxima)
    logtol = original.TOLERANCES['abs_logprob_entropy_margin']
    ptol = original.TOLERANCES['abs_probability']
    dtol = original.TOLERANCES['abs_difference_of_two_logprobs']
    trajectories = {s: reference['trajectories'][s]['token_ids'] for s in STAGES}
    require(comparisons['free_trajectory_forks'] == original.independent_forks(trajectories), 'Free trajectory fork summary differs')
    require(set(comparisons['fixed_trajectory_comparisons']) == set(STAGES), 'Comparison trajectory set differs')
    for trajectory in STAGES:
        check()
        data = {}
        for stage in STAGES:
            name = f'{stage}/worker/teacher/{trajectory}/derived.json'
            entry = manifest['corrected_artifacts'][name]
            value, identity = read_json(overlay/entry['path'])
            require(identity['sha256'] == entry['sha256'] and identity['bytes'] == entry['bytes'], 'Corrected derived changed before summary validation')
            data[stage] = value
        summary = comparisons['fixed_trajectory_comparisons'][trajectory]
        require(len(summary['rows']) == len(trajectories[trajectory]), 'Comparison row count differs')
        require(set(summary['per_model_nll']) == set(STAGES), 'NLL model set differs')
        for j, row in enumerate(summary['rows']):
            check()
            baseline = data['base']['rows'][j]
            require(row['j'] == j and row['target_id'] == trajectories[trajectory][j] and
                    row['prefix'] == baseline['prefix'], 'Comparison causal prefix/index/target differs')
            require(set(row['models']) == set(STAGES), 'Comparison model set differs')
            for stage in STAGES:
                saved, expected = row['models'][stage], data[stage]['rows'][j]
                require(expected['prefix'] == row['prefix'] and expected['target_id'] == row['target_id'], 'Cross-model prefix differs')
                require(saved['target_rank'] == expected['target']['rank'], 'Summary target rank differs')
                for field, value, tol in (
                    ('target_logprob', expected['target']['logprob'], logtol),
                    ('target_prob', expected['target']['prob'], ptol),
                    ('target_logprob_delta_vs_base', expected['target']['logprob']-baseline['target']['logprob'], dtol),
                    ('eos_total_prob', expected['eos']['total_prob'], ptol),
                    ('eos_total_logprob', expected['eos']['total_logprob'], logtol),
                    ('eos_total_logprob_delta_vs_base', expected['eos']['total_logprob']-baseline['eos']['total_logprob'], dtol)):
                    close(saved[field], value, tol, field)
                require([c['id'] for c in saved['candidate_deltas']] == [c['id'] for c in baseline['candidates']] == [c['id'] for c in expected['candidates']], 'Summary candidate mapping differs')
                for candidate, left, right in zip(saved['candidate_deltas'], baseline['candidates'], expected['candidates']):
                    require(candidate['rank'] == right['rank'] and candidate['base_rank'] == left['rank'], 'Summary candidate rank differs')
                    close(candidate['logprob_delta'], right['logprob']-left['logprob'], dtol, 'candidate_logprob_delta')
        for stage in STAGES:
            nll = summary['per_model_nll'][stage]
            close(nll['total'], data[stage]['total_target_nll'], len(trajectories[trajectory])*logtol, 'nll_total')
            close(nll['mean'], data[stage]['mean_target_nll'], logtol, 'nll_mean')
    return maxima


def arm_remaining(deadline):
    remaining = deadline-time.monotonic()
    require(remaining > 0, 'Overlay whole-audit deadline exhausted')
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Overlay whole-audit deadline')))
    signal.setitimer(signal.ITIMER_REAL, remaining)


def atomic_accept(output, pending, result, markdown, check):
    """Only publish accepted JSON after all validation and enrichment succeeded."""
    pending.write_text(json.dumps(result, indent=2)+'\n')
    pending.with_suffix('.md').write_text(markdown)
    check()
    os.replace(pending.with_suffix('.md'), output.with_suffix('.md'))
    check()
    os.replace(pending, output)


def install_progress(original, deadline):
    directory = os.environ.get('CPU_EVIDENCE_STAGE_DIR')
    if not directory:
        return
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    function = original.full_vocab_statistics
    completed = 0
    matrices = 0
    def publish(phase, rows):
        arm_remaining(deadline)
        pending = directory/'progress.pending.json'
        pending.write_text(json.dumps(dict(phase=phase, completed_rows=completed,
            completed_matrices=matrices, current_matrix_rows=rows,
            progress_kind='independent_fp64_recomputation_not_final_acceptance',
            observed_unix=time.time()))+'\n')
        os.replace(pending, directory/'progress.json')
    def observed(*args, **kwargs):
        nonlocal completed, matrices
        count = len(args[1] if len(args)>1 else kwargs['targets'])
        publish('matrix_enter', count)
        result = function(*args, **kwargs)
        completed += count
        matrices += 1
        publish('matrix_return', count)
        return result
    original.full_vocab_statistics = observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', required=True, type=Path)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--tokenizer-dir', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--seconds', type=int, default=600)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    require(0 < args.seconds <= 600 and 0 < args.threads <= 4,
            'Overlay audit requires <=600seconds and <=4CPU threads')
    output = args.output.resolve()
    require(not output.is_relative_to(args.run.resolve()), 'Never write audit output into immutable original run')
    pending = output.with_name(output.stem+'.pending.json')
    require(not any(p.exists() for p in (output, output.with_suffix('.md'), pending, pending.with_suffix('.md'))),
            'Refusing to overwrite an existing audit result')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    started = time.monotonic()
    deadline = started+args.seconds
    arm_remaining(deadline)
    original = import_original()
    manifest, identity, campaign_identity, redirects = validate_manifest(args.overlay, args.run, args.reference)
    original.Evidence = overlay_evidence_class(original, redirects)
    install_progress(original, deadline)
    # Reserve preprocessing time within the same frozen600s cap. FP64 functions and
    # every numeric comparison/tolerance remain exactly those of the pinned auditor.
    remaining = math.floor(args.seconds-(time.monotonic()-started))
    require(remaining > 0, 'Overlay audit budget exhausted during manifest validation')
    args.seconds = remaining
    args.output = pending
    original.audit(args)
    # The pinned audit disarms its timer on return; immediately rearm the original
    # absolute deadline before auxiliary checks, enrichment or publication.
    arm_remaining(deadline)
    auxiliaries = checked_auxiliary(args.overlay, manifest)
    reference, _ = read_json(args.reference)
    summary_maxima = validate_comparisons(original, manifest, args.overlay, reference,
        auxiliaries['comparisons.json'], lambda: arm_remaining(deadline))
    result = json.loads(pending.read_text())
    result.update(audited_evidence='corrected_derived_overlay_against_immutable_original_matrices',
        original_auditor_sha256=ORIGINAL_AUDITOR_SHA256,
        overlay_manifest=dict(path=str((args.overlay/'manifest.json').resolve()), **identity),
        original_campaign_sha256=campaign_identity['sha256'],
        original_source_sha256=manifest['original_source_sha256'],
        corrected_source_sha256=manifest.get('corrected_source_sha256'),
        overlay_corrected_artifacts=len(redirects), original_artifacts_verified_before_overlay=True,
        comparisons_independently_validated=True, comparison_max_absolute_errors=summary_maxima,
        auxiliary_artifacts=manifest['auxiliary_artifacts'],
        change_summary_boundary='Artifact SHA verified only; diagnostic counts are not independently certified.',
        original_failed_audit_preserved=True,
        whole_overlay_audit_elapsed_seconds=time.monotonic()-started)
    markdown = pending.with_suffix('.md').read_text()+(
            '\nThis pass applies to the corrected derived overlay only. Every original artifact '
            'was independently verified against its immutable worker result manifest before selecting '
            'corrected derived JSON. Original matrices, raw run and failed original audit remain unchanged. '
            'FP64 arithmetic, exact rank/tie/top10 checks and tolerances are from the SHA-pinned original auditor. '
            'Cross-model comparisons were independently reconstructed. Change-summary diagnostic counts '
            'are only hash-verified, not independently certified.\n')
    atomic_accept(output, pending, result, markdown, lambda: arm_remaining(deadline))
    signal.setitimer(signal.ITIMER_REAL, 0)
    print(json.dumps(dict(status='passed_corrected_overlay',corrected_artifacts=len(redirects),
                         elapsed_seconds=result['whole_overlay_audit_elapsed_seconds'])))


if __name__ == '__main__':
    main()
