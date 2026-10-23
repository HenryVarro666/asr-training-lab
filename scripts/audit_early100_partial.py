#!/usr/bin/env python3
"""CPU-only failure audit: retain the failed truncation gate and explicitly incomplete five-stage control."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path
import shlex
import sys

import jiwer
from audit_early100_outputs import audit_pair,EnglishTextNormalizer,STAGES,digest,read_rows,REPO


def edits(row):
    score=jiwer.process_words(row['reference_normalized'],row['hypothesis_normalized'])
    return dict(substitutions=score.substitutions,deletions=score.deletions,insertions=score.insertions,
                hits=score.hits,reference_words=score.hits+score.substitutions+score.deletions)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=REPO)
    args=parser.parse_args()
    root=args.root.resolve();run=root/'reports/early100/run';output=root/'reports/early100'
    campaign_path=run/'campaign.json';receipt_path=root/'reports/qwen-campaign-20261022-a/campaign.json'
    spelling=root/'reports/diagnosis/inputs/whisper-english-spelling-normalizer.json'
    normalizer_receipt=root/'reports/diagnosis/text-audit.json'
    campaign=json.loads(campaign_path.read_text());receipt=json.loads(receipt_path.read_text())
    old_normalizer=json.loads(normalizer_receipt.read_text())['normalizer']
    if old_normalizer['spelling_sha256']!=digest(spelling) or old_normalizer['transformers']!=importlib.metadata.version('transformers'):
        raise ValueError('Frozen normalizer changed')
    if campaign.get('status')!='failed' or 'Generation truncation' not in campaign.get('error',''):
        raise ValueError('This script audits only the observed failed truncation gate')
    normalizer=EnglishTextNormalizer(json.loads(spelling.read_text()))
    hashes={Path(key).name:value for key,value in receipt['manifests'].items()}
    files=[Path(__file__),Path(__file__).with_name('audit_early100_outputs.py'),Path(__file__).with_name('audit_saved_predictions.py'),
        root/'src/asr_lab/qwen_evaluate.py',campaign_path,receipt_path,spelling,normalizer_receipt]
    result=dict(status='partial_failed_quality_gate',comparison_valid=False,
        campaign_status=campaign['status'],campaign_stage=campaign['stage'],stop_reason=campaign['error'],
        command=shlex.join([sys.executable,*sys.argv]),script_sha256=digest(Path(__file__)),
        normalizer=dict(spelling_sha256=digest(spelling),transformers=importlib.metadata.version('transformers'),jiwer=importlib.metadata.version('jiwer')),
        expected_generation_stages=5,stages={},missing_stages=[],
        interpretation='Text-only forensic audit. Saved parsed hypothesis text is not raw generation text or audio truth. No inference, retry, bootstrap, causal attribution, or retroactive quality-gate relaxation.')
    for step,split in STAGES:
        name=f'step{step}-{split}';candidate=run/'evaluation'/name/'predictions.jsonl'
        if not candidate.is_file():result['missing_stages'].append(name);continue
        if split=='fixed128':
            manifest=root/'reports/early100/inputs/dev-clean-fixed128.jsonl'
            baseline=root/'reports/baselines/qwen-pinned-b16/dev-clean-fixed128.jsonl'
        else:
            manifest=root/'reports/diagnosis/inputs'/(split.replace('_','-')+'.jsonl')
            baseline=receipt_path.parent/'evaluation'/('baseline-'+split.replace('_','-'))/'predictions.jsonl'
        audit=audit_pair(baseline,candidate,manifest,normalizer,hashes[manifest.name])
        audit['text_integrity_passed']=audit.pop('passed')
        audit['accepted_by_original_generation_gate']=audit['sides']['candidate']['flags']['generation_truncated']==0
        result['stages'][name]=audit
        files.extend([manifest,baseline,baseline.with_suffix('.metrics.json'),candidate,candidate.with_suffix('.metrics.json')])
    result['saved_prediction_stages']=len(result['stages'])
    if result['missing_stages']!=['step100-dev_other']:raise ValueError('Unexpected partial completion layout')
    clean=run/'evaluation/step100-dev_clean/predictions.jsonl'
    base_path=receipt_path.parent/'evaluation/baseline-dev-clean/predictions.jsonl'
    rows=read_rows(clean);base={row['id']:row for row in read_rows(base_path)}
    truncated=[row for row in rows if row['generation_truncated']]
    if len(truncated)!=1:raise ValueError('Expected exactly one observed clean truncated prediction')
    row=truncated[0];baseline=base[row['id']]
    target_index=next(index for index,value in enumerate(rows) if value['id']==row['id'])
    batch_start=target_index//16*16
    batch=rows[batch_start:batch_start+16]
    result['truncation_rule_audit']=dict(source='src/asr_lab/qwen_evaluate.py',lines=[58,60,62,63,65,67],
        rule='Per utterance: len(continuation_tokens)>=512 AND no token in {151645,151643}',
        padded_batch_width_used=True,per_utterance_eos_checked=True,
        padding_interpretation='Width alone is shared by the padded batch; preserved recognized EOS prevents a completed utterance being marked solely because another item reaches512. Saved token IDs are unavailable for independent EOS verification.',
        target_zero_based_index=target_index,batch_zero_based_index=target_index//16,
        same_batch=[dict(id=value['id'],generation_truncated=value['generation_truncated'],parsed_hypothesis_words=len(value['hypothesis_normalized'].split())) for value in batch],
        other_same_batch_count=len(batch)-1,other_same_batch_truncated=sum(value['generation_truncated'] for value in batch if value['id']!=row['id']))
    candidate_counts=edits(row);baseline_counts=edits(baseline)
    error_keys=('substitutions','deletions','insertions')
    target_errors=sum(candidate_counts[key] for key in error_keys)
    baseline_target_errors=sum(baseline_counts[key] for key in error_keys)
    audit=result['stages']['step100-dev_clean'];sides=audit['sides']
    totals=sides['candidate']['recomputed_counts'];base_totals=sides['baseline']['recomputed_counts']
    total_errors=sum(totals[key] for key in error_keys);base_errors=sum(base_totals[key] for key in error_keys)
    words=totals['reference_words'];remaining_words=words-candidate_counts['reference_words']
    result['truncated_utterance']=dict(id=row['id'],candidate=row,baseline=baseline,candidate_counts=candidate_counts,baseline_counts=baseline_counts,
        parsed_hypothesis_words=len(row['hypothesis_normalized'].split()),
        generation_tokens=dict(saved_token_ids=False,saved_token_length=False,independently_verified_length=None,
            inferred_generated_width=512,inference='Frozen evaluator maximum512 plus generation_truncated=True implies continuation width512 and no recognized EOS; token IDs/raw pre-parse generation were not persisted. This is a source-rule inference, not an observed token-length record.'))
    result['clean_error_contribution']=dict(all_examples=len(rows),reference_words=words,candidate_errors=total_errors,baseline_errors=base_errors,
        candidate_wer=total_errors/words,baseline_wer=base_errors/words,
        difference_percentage_points=100*(total_errors-base_errors)/words,
        flagged_utterance_errors=target_errors,baseline_same_utterance_errors=baseline_target_errors,
        flagged_utterance_contribution_to_corpus_wer_percentage_points=100*target_errors/words,
        flagged_utterance_share_of_candidate_errors=target_errors/total_errors,
        flagged_utterance_share_of_candidate_minus_baseline_errors=(target_errors-baseline_target_errors)/(total_errors-base_errors),
        excluding_same_utterance_from_both=dict(examples=len(rows)-1,reference_words=remaining_words,
            candidate_errors=total_errors-target_errors,baseline_errors=base_errors-baseline_target_errors,
            candidate_wer=(total_errors-target_errors)/remaining_words,baseline_wer=(base_errors-baseline_target_errors)/remaining_words,
            interpretation='Descriptive paired exclusion sensitivity only; it does not replace the failed full-population evaluation or establish the cause of errors.'))
    result['all_saved_stage_text_integrity_passed']=all(stage['text_integrity_passed'] for stage in result['stages'].values())
    result['input_and_source_sha256']={str(path.resolve()):digest(path) for path in sorted(set(files))}
    output.mkdir(parents=True,exist_ok=True)
    (output/'truncation-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    contribution=result['clean_error_contribution'];excluding=contribution['excluding_same_utterance_from_both']
    lines=['# Early100 截断失败审计','',
        '状态：部分完成且质量 gate 失败；comparison_valid=false。未重跑、未修改训练或 runner。',
        '实际保存四份预测；step100-dev_other 未启动。完整五阶段控制没有完成。','',
        f"唯一截断 ID：`{row['id']}`。",
        f"- REF：{row['reference']}",f"- step100 解析后 HYP：{row['hypothesis']}",
        f"- baseline HYP：{baseline['hypothesis']}",
        f"- step100：S={candidate_counts['substitutions']}、D={candidate_counts['deletions']}、I={candidate_counts['insertions']}；参考{candidate_counts['reference_words']}词；baseline同句0词错误且未截断。",'',
        f"冻结 evaluator 的58–67行逐条检查continuation，判定为长度≥512且没有EOS151645/151643。长度使用batch补齐后的宽度，但每句保留正常EOS时不会仅因同batch长输出而被标截断。该句位于0-based第{target_index}条、batch{target_index//16}；同batch其余{len(batch)-1}条均未截断。没有证据支持仅由整batch统一长度导致误标；原token未保存，仍不能独立验证EOS或长度。",'',
        '实际 token IDs/长度及 parse_asr_output 之前的生成文本未被 evaluator 保存，不能从两词 HYP 推断其内容。按冻结源码的512上限与截断标记逻辑，可推断continuation触达512且无识别的EOS；这不是独立保存的token长度证据，也不能断言发生了哪种循环。','',
        f"已完成 clean 的2694条逐句 raw REF/HYP→固定 normalizer 与保存文本一致：{audit['text_integrity_passed']}；所有四份已保存预测文本完整性通过：{result['all_saved_stage_text_integrity_passed']}。",
        f"clean 复算 S/D/I={totals['substitutions']}/{totals['deletions']}/{totals['insertions']}，{total_errors}/{words}={100*total_errors/words:.6f}% WER；baseline={100*base_errors/words:.6f}%。原零截断质量 gate 仍失败。",'',
        f"该截断句计{target_errors}词错误，占candidate总错误{100*target_errors/total_errors:.4f}%，对完整clean WER的算术贡献为{100*target_errors/words:.6f}个百分点；占相对baseline新增错误的{100*(target_errors-baseline_target_errors)/(total_errors-base_errors):.4f}%。",
        f"同时从两侧移除此句后，剩余{excluding['examples']}条/{remaining_words}参考词：candidate={100*excluding['candidate_wer']:.6f}%，baseline={100*excluding['baseline_wer']:.6f}%。这只是配对排除敏感性，不是替代正式指标或通过gate；该句也不能解释其余词错误差异。",'',
        'style/uppercase、前缀、重复、长度、空输出和截断 flags 的计数保存在JSON（多标签可重叠）。这是文本诊断，不能判定音频真值、灾难性遗忘或学习率因果。',
        '原完整五项 audit_early100_outputs.py 仍保持原严格完成条件。',
        '',f"复算命令：`{result['command']}`。输入、脚本和复用依赖SHA256见truncation-audit.json。"]
    (output/'truncation-audit.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(status=result['status'],stages=result['saved_prediction_stages'],id=row['id'],candidate_errors=target_errors,clean_wer=total_errors/words)))


if __name__=='__main__':main()
