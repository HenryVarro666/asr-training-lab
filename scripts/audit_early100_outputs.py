#!/usr/bin/env python3
"""CPU-only audit of the five saved Early100 prediction sets; never runs inference."""
from __future__ import annotations
import argparse
from collections import Counter
import importlib.metadata
import inspect
import json
import math
from pathlib import Path
import shlex
import sys

import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
from audit_saved_predictions import flags,read_rows,digest

REPO=Path(__file__).resolve().parents[1]
REVISION='7278e1e70fe206f11671096ffdd38061171dd6e5'
NORMALIZER_REVISION='973afd24965f72e36ca33b3055d56a652f456b4d'
STAGES=((25,'fixed128'),(50,'fixed128'),(100,'fixed128'),(100,'dev_clean'),(100,'dev_other'))
COUNTS=('substitutions','deletions','insertions','hits','reference_words')


def unique_rows(path):
    rows=read_rows(path)
    by_id={row['id']:row for row in rows}
    if not rows or len(by_id)!=len(rows):raise ValueError(f'Empty or duplicate IDs:{path}')
    return rows,by_id


def audit_pair(base_path,candidate_path,manifest_path,normalizer,expected_manifest_hash):
    base,b=unique_rows(base_path);candidate,c=unique_rows(candidate_path);manifest,m=unique_rows(manifest_path)
    if not set(b)==set(c)==set(m):raise ValueError('Baseline/candidate/manifest ID populations differ')
    if [row['id'] for row in base]!=[row['id'] for row in candidate]:raise ValueError('Paired prediction order differs')
    actual_hash=digest(manifest_path)
    if actual_hash!=expected_manifest_hash:raise ValueError('Original receipt manifest hash differs')
    result=dict(examples=len(candidate),manifest_sha256=actual_hash,identical_paired_ids_and_order=True,
        raw_references_match_manifest=True,normalization_mismatches=[],sides={})
    for side,rows,path in (('baseline',base,base_path),('candidate',candidate,candidate_path)):
        metrics=json.loads(path.with_suffix('.metrics.json').read_text())
        expected=dict(examples=len(rows),manifest_sha256=actual_hash,evaluation_batch_size=16,seed=42,
            normalizer='Whisper EnglishTextNormalizer',normalizer_revision=NORMALIZER_REVISION,generation_max_new_tokens=512)
        if any(metrics.get(key)!=value for key,value in expected.items()):raise ValueError(f'{side} evaluation metadata differs')
        if metrics['selected_ids']!=[row['id'] for row in rows]:raise ValueError(f'{side} metric selected IDs differ')
        if side=='baseline' and metrics.get('model_revision')!=REVISION:raise ValueError('Baseline model revision differs')
        counters=Counter();styles=Counter();totals=Counter()
        for row in rows:
            if row['reference']!=m[row['id']]['text']:raise ValueError(f'Raw reference differs:{row["id"]}')
            if not isinstance(row.get('generation_truncated'),bool):raise ValueError('Missing explicit truncation flag')
            for raw,norm in (('reference','reference_normalized'),('hypothesis','hypothesis_normalized')):
                if normalizer(row[raw]).strip()!=row[norm]:
                    result['normalization_mismatches'].append(dict(side=side,id=row['id'],field=norm))
            score=jiwer.process_words(row['reference_normalized'],row['hypothesis_normalized'])
            totals.update(substitutions=score.substitutions,deletions=score.deletions,insertions=score.insertions,
                          hits=score.hits,reference_words=score.hits+score.substitutions+score.deletions)
            counters.update({key:int(value) for key,value in flags(row).items()})
            raw=row['hypothesis']
            style='all_uppercase' if raw.isupper() else ('all_lowercase' if raw.islower() else ('mixed_case' if any(ch.lower()!=ch.upper() for ch in raw) else 'no_cased_letters'))
            styles[style]+=1
        if any(totals[key]!=metrics.get(key) for key in COUNTS):raise ValueError(f'{side} recomputed edit counts differ')
        wer=sum(totals[key] for key in ('substitutions','deletions','insertions'))/totals['reference_words']
        if not math.isclose(wer,float(metrics['wer']),rel_tol=1e-14,abs_tol=1e-15):raise ValueError(f'{side} recomputed WER differs')
        if counters['generation_truncated']!=metrics['generation_truncated_count']:raise ValueError('Truncation count differs')
        if not math.isclose(counters['generation_truncated']/len(rows),metrics['generation_truncated_fraction'],abs_tol=1e-15):raise ValueError('Truncation fraction differs')
        result['sides'][side]=dict(recomputed_counts=dict(totals),recomputed_wer=wer,flags=dict(counters),
            raw_case_categories=dict(styles),uppercase_fraction=counters['all_uppercase']/len(rows),
            normalization_checked_raw_reference_and_parsed_hypothesis=len(rows)*2)
    result['passed']=not result['normalization_mismatches']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=REPO/'reports/early100/run')
    parser.add_argument('--receipt',type=Path,default=REPO/'reports/qwen-campaign-20261022-a/campaign.json')
    parser.add_argument('--fixed-manifest',type=Path,default=REPO/'reports/early100/inputs/dev-clean-fixed128.jsonl')
    parser.add_argument('--full-manifests',type=Path,default=REPO/'reports/diagnosis/inputs')
    parser.add_argument('--fixed-baseline',type=Path,default=REPO/'reports/baselines/qwen-pinned-b16/dev-clean-fixed128.jsonl')
    parser.add_argument('--full-baselines',type=Path,default=REPO/'reports/qwen-campaign-20261022-a/evaluation')
    parser.add_argument('--spelling',type=Path,default=REPO/'reports/diagnosis/inputs/whisper-english-spelling-normalizer.json')
    parser.add_argument('--normalizer-receipt',type=Path,default=REPO/'reports/diagnosis/text-audit.json')
    parser.add_argument('--output',type=Path,default=REPO/'reports/early100/output-audit.json')
    args=parser.parse_args()
    files=[Path(__file__),Path(__file__).with_name('audit_saved_predictions.py'),args.receipt,args.run/'campaign.json',args.spelling,args.normalizer_receipt,
           args.run/'training/run.json',Path(inspect.getsourcefile(EnglishTextNormalizer))]
    result=dict(status='failed',command=shlex.join([sys.executable,*sys.argv]),script_sha256=digest(Path(__file__)),stages={},
        normalizer=dict(class_name='Whisper EnglishTextNormalizer',revision=NORMALIZER_REVISION,spelling_sha256=digest(args.spelling),
            transformers=importlib.metadata.version('transformers'),jiwer=importlib.metadata.version('jiwer')),
        interpretation='Text-only checks on saved parsed hypotheses, not acoustic truth or causal attribution. Flags are multi-label and may overlap. Raw generation text before parse_asr_output and token IDs were not saved by the evaluator. No bootstrap or inference is performed.')
    try:
        original_normalizer=json.loads(args.normalizer_receipt.read_text())['normalizer']
        if original_normalizer['spelling_sha256']!=digest(args.spelling) or original_normalizer['transformers']!=importlib.metadata.version('transformers'):
            raise ValueError('Frozen normalizer spelling map or implementation version differs')
        campaign=json.loads((args.run/'campaign.json').read_text())
        if campaign.get('status')!='complete' or campaign.get('stage')!='five_evaluations_complete':raise ValueError('Five-stage campaign is not complete')
        training=json.loads((args.run/'training/run.json').read_text())
        if training.get('revision')!=REVISION:raise ValueError('Candidate training initialization revision differs')
        receipt=json.loads(args.receipt.read_text())
        hashes={Path(key).name:value for key,value in receipt['manifests'].items()}
        normalizer=EnglishTextNormalizer(json.loads(args.spelling.read_text()))
        for step,split in STAGES:
            name=f'step{step}-{split}'
            candidate=args.run/'evaluation'/name/'predictions.jsonl'
            manifest=args.fixed_manifest if split=='fixed128' else args.full_manifests/(split.replace('_','-')+'.jsonl')
            baseline=args.fixed_baseline if split=='fixed128' else args.full_baselines/('baseline-'+split.replace('_','-'))/'predictions.jsonl'
            files.extend([manifest,baseline,baseline.with_suffix('.metrics.json'),candidate,candidate.with_suffix('.metrics.json')])
            result['stages'][name]=audit_pair(baseline,candidate,manifest,normalizer,hashes[manifest.name])
        result['status']='passed' if all(stage['passed'] for stage in result['stages'].values()) else 'failed'
    except Exception as error:
        result['error']=f'{type(error).__name__}: {error}'
    result['input_and_source_sha256']={str(path.resolve()):digest(path) for path in sorted(set(files)) if path.is_file()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],stages=len(result['stages']),output=str(args.output))))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
