#!/usr/bin/env python3
"""CPU-only audit of saved ASR predictions; no model inference or data mutation."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import re
import jiwer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repeated_trigram(text):
    words = text.split()
    return any(words[i:i+3] == words[i+3:i+6] == words[i+6:i+9] for i in range(len(words)-8))


def flags(row):
    hypothesis = row['hypothesis']
    normalized = row['hypothesis_normalized']
    reference = row['reference_normalized']
    return dict(empty_normalized=not normalized.split(),
        parsed_prefix_leak=bool(re.search(r'<\|[^>]+\|>|<asr_text>|^language English', hypothesis)),
        repeated_trigram_not_in_reference=repeated_trigram(normalized) and not repeated_trigram(reference),
        all_uppercase=bool(hypothesis.isupper()),
        words_over_1_5x_reference=len(normalized.split()) > 1.5*len(reference.split()),
        generation_truncated=bool(row.get('generation_truncated', False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261022)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    spelling = args.inputs/'whisper-english-spelling-normalizer.json'
    normalizer = EnglishTextNormalizer(json.loads(spelling.read_text()))
    result = dict(seed=args.seed, selection_counts=dict(regressed=20,improved=5,tied=5),
        normalizer=dict(class_name='Whisper EnglishTextNormalizer',spelling_sha256=digest(spelling),
            transformers=importlib.metadata.version('transformers'),jiwer=importlib.metadata.version('jiwer')),
        script_sha256=digest(Path(__file__)),splits={},
        interpretation='Text-only diagnostic flags, not verified acoustic errors or causal attribution. Raw outputs before parse_asr_output were not saved.')
    for split in ['dev-clean','dev-other']:
        bp=args.campaign/'evaluation'/f'baseline-{split}'/'predictions.jsonl'
        cp=args.campaign/'evaluation'/f'final-{split.replace("-","_")}'/'predictions.jsonl'
        mp=args.inputs/f'{split}.jsonl'
        b_rows,c_rows,manifest=read_rows(bp),read_rows(cp),read_rows(mp)
        b={x['id']:x for x in b_rows};c={x['id']:x for x in c_rows};m={x['id']:x for x in manifest}
        assert len(b)==len(b_rows) and len(c)==len(c_rows) and len(m)==len(manifest)
        assert set(b)==set(c)==set(m)
        counters={side:Counter() for side in ['baseline','candidate']}
        mismatches=[];details=[];speaker_deltas=Counter();bucket_deltas=Counter()
        for identifier in sorted(b):
            left,right=b[identifier],c[identifier]
            assert left['reference']==right['reference']==m[identifier]['text']
            scores=[]
            for side,row in [('baseline',left),('candidate',right)]:
                for raw,norm in [('reference','reference_normalized'),('hypothesis','hypothesis_normalized')]:
                    if normalizer(row[raw]).strip()!=row[norm]:mismatches.append(dict(id=identifier,side=side,field=norm))
                score=jiwer.process_words(row['reference_normalized'],row['hypothesis_normalized'])
                scores.append(dict(substitutions=score.substitutions,deletions=score.deletions,insertions=score.insertions,
                    reference_words=score.hits+score.substitutions+score.deletions))
                counters[side].update({key:int(value) for key,value in flags(row).items()})
            delta=sum(scores[1][k]-scores[0][k] for k in ['substitutions','deletions','insertions'])
            label='regressed' if delta>0 else ('improved' if delta<0 else 'tied')
            speaker_deltas[str(m[identifier]['speaker_id'])]+=delta
            duration=m[identifier]['duration'];bucket='<5s' if duration<5 else ('5-15s' if duration<15 else '15-30s')
            bucket_deltas[bucket]+=delta
            details.append(dict(id=identifier,change=label,edit_delta=delta,speaker_id=str(m[identifier]['speaker_id']),
                duration=duration,audio_filepath=m[identifier]['audio_filepath'],reference=left['reference'],
                reference_normalized=left['reference_normalized'],baseline_hypothesis=left['hypothesis'],
                candidate_hypothesis=right['hypothesis'],baseline_normalized=left['hypothesis_normalized'],
                candidate_normalized=right['hypothesis_normalized'],baseline_counts=scores[0],candidate_counts=scores[1],
                baseline_flags=flags(left),candidate_flags=flags(right)))
        assert not mismatches, f'{split}: normalization mismatch: {mismatches[:3]}'
        rng=random.Random(args.seed)
        selected=[]
        for label,limit in result['selection_counts'].items():
            pool=[row for row in details if row['change']==label]
            selected.extend(rng.sample(pool,min(limit,len(pool))))
        selected.sort(key=lambda r:(r['change'],r['id']))
        (args.output/f'{split}.review-samples.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in selected))
        (args.output/f'{split}.all-utterances.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in details))
        changes=Counter(row['change'] for row in details)
        result['splits'][split]=dict(examples=len(details),normalization_mismatches=0,
            raw_reference_matches_manifest=True,flags=counters,utterance_changes=changes,
            speaker_edit_deltas=dict(sorted(speaker_deltas.items())),duration_bucket_edit_deltas=dict(bucket_deltas),
            input_sha256={str(p):digest(p) for p in [bp,cp,mp]},sample_ids=[row['id'] for row in selected])
        print(split,dict(changes),dict(counters))
    (args.output/'text-audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
