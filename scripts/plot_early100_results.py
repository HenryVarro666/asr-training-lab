#!/usr/bin/env python3
"""CPU-only figures for an original-schedule early100 diagnostic.

Default mode requires all five accepted evaluations. Explicit partial mode only
shows accepted fixed128 results and completed training after a truncation failure.
Never fills missing checkpoints or extrapolates WER.
The original plotting script, figures and their fingerprints remain untouched.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys

import plot_experiment_results as shared
from plot_experiment_results import (COLORS,REPO,finite,load_training,plt,read_json,
                                    relative,rolling_mean,save_figure,source,style_axes,validate_wer)


def completed_stage(directory):
    path=Path(directory)/'stage.json'
    receipt=read_json(path)
    if receipt.get('status')!='complete' or receipt.get('exit_code')!=0:
        raise ValueError(f'Requires an actually completed stage: {path}')
    return path


def validate_partial_failure(args,campaign):
    """Recognize the observed failure without accepting its full-dev WER."""
    if (campaign.get('status'),campaign.get('stage'))!=('failed','step100-dev_clean') or 'Generation truncation prevents an accepted evaluation' not in campaign.get('error',''):
        raise ValueError('Partial mode only accepts the recorded full-clean truncation failure')
    directory=args.early100/'evaluation/step100-dev_clean'
    stage=completed_stage(directory)
    path=directory/'predictions.metrics.json'
    failed=read_json(path)
    if (failed.get('examples'),failed.get('generation_max_new_tokens'),failed.get('evaluation_batch_size'))!=(2694,512,16) or failed.get('generation_truncated_count')!=1:
        raise ValueError('Partial failure receipt differs from the observed one-truncation full-clean evaluation')
    if (args.early100/'evaluation/step100-dev_other').exists():
        raise ValueError('Partial mode requires full-other not to have run; inspect any unexpected directory')
    return [stage,path]


def validate_inputs(args):
    run=args.early100
    campaign_path=run/'campaign.json'
    campaign=read_json(campaign_path)
    failure_sources=[]
    if args.partial_diagnostic:
        failure_sources=validate_partial_failure(args,campaign)
    elif campaign.get('status')!='complete' or campaign.get('stage')!='five_evaluations_complete':
        raise ValueError('Early100 campaign and all five evaluations must be complete before rendering')
    training=run/'training'
    terminal_path=training/'prefix-status.json'
    proof_path=training/'trajectory-validation.json'
    plan_path=training/'prefix-plan.json'
    terminal,proof,plan=map(read_json,(terminal_path,proof_path,plan_path))
    if (terminal.get('status'),terminal.get('actual_step'),terminal.get('original_total_steps'),terminal.get('completed_epoch'))!=('completed_bounded_prefix',100,892,False):
        raise ValueError('Missing completed 100-update prefix; it must not claim one completed epoch')
    if (proof.get('status'),proof.get('validated_steps'),proof.get('original_total_steps'),proof.get('warmup_steps'))!=('passed',100,892,45):
        raise ValueError('Original 892-step schedule / 45-step warmup not validated')
    if (plan.get('original_total_steps'),plan.get('warmup_steps'),plan.get('actual_stop'))!=(892,45,100):
        raise ValueError('Prefix plan differs from the predeclared 892/45/100 contract')
    original=load_training('Original run',args.reports/'qwen-campaign-20261022-a/main-epoch1')
    observed=load_training('Fresh prefix run',training)
    prefix=[r for r in original['train'] if 1<=r['step']<=100]
    if [r['step'] for r in prefix]!=list(range(1,101)) or [r['step'] for r in observed['train']]!=list(range(1,101)):
        raise ValueError('Original and new loss plots require exactly the observed first 100 updates')
    trajectory_path=training/'prefix-trajectory.jsonl'
    trajectory=[json.loads(line) for line in trajectory_path.read_text().splitlines() if line.strip()]
    if [r['step'] for r in trajectory]!=list(range(1,101)):
        raise ValueError('Missing/duplicate observed prefix-trajectory steps')
    for original_row,new_row,actual in zip(prefix,observed['train'],trajectory):
        used=finite(actual['used_learning_rate'],'actual used learning rate')
        if not math.isclose(used,original_row['learning_rate'],rel_tol=8*sys.float_info.epsilon,abs_tol=0):
            raise ValueError(f'Original/new used learning rates differ at update {actual["step"]}')
        if not math.isclose(float(new_row['learning_rate']),used,rel_tol=8*sys.float_info.epsilon,abs_tol=0):
            raise ValueError('Logged new LR is not the validated update-used LR')
        if actual['scheduler_last_epoch']!=actual['step'] or actual['examples']!=original_row['examples']:
            raise ValueError('Scheduler/update or sample-consumption boundary differs')
        if abs(finite(actual['data_seconds'],'audio duration')-original_row['data_seconds'])>1e-9:
            raise ValueError('Original/new consumed audio duration differs')
    common=[campaign_path,terminal_path,proof_path,plan_path,trajectory_path,completed_stage(training)]+failure_sources
    return original,observed,prefix,trajectory,common


def matching_metrics(paths,expected_examples):
    metrics=[validate_wer(path) for path in paths]
    for key in ('examples','reference_words','selected_ids','manifest_sha256','normalizer','normalizer_revision',
                'evaluation_batch_size','generation_max_new_tokens'):
        if metrics[0].get(key) is None or any(row.get(key)!=metrics[0][key] for row in metrics[1:]):
            raise ValueError(f'Cannot compare generation checkpoints with different {key}')
    if metrics[0]['examples']!=expected_examples or metrics[0]['evaluation_batch_size']!=16 or metrics[0]['generation_max_new_tokens']!=512:
        raise ValueError('Sample coverage or decoding contract differs from the declared diagnostic')
    if any(len(row['selected_ids'])!=len(set(row['selected_ids'])) or len(row['selected_ids'])!=expected_examples for row in metrics):
        raise ValueError('Duplicate/incomplete generation IDs')
    return metrics


def validate_generation_inputs(args):
    campaign=args.reports/'qwen-campaign-20261022-a'
    fixed=[args.reports/'baselines/qwen-pinned-b16/dev-clean-fixed128.metrics.json']+[args.early100/'evaluation'/f'step{step}-fixed128/predictions.metrics.json' for step in (25,50,100)]
    matching_metrics(fixed,128)
    if args.partial_diagnostic:
        return
    for split,count in (('dev-clean',2694),('dev-other',2857)):
        paths=[campaign/'evaluation'/f'baseline-{split}/predictions.metrics.json',
            args.early100/'evaluation'/f'step100-{split.replace("-","_")}/predictions.metrics.json',
            args.reports/'diagnosis/checkpoint800'/split/'predictions.metrics.json',
            campaign/'evaluation'/f'final-{split.replace("-","_")}/predictions.metrics.json']
        matching_metrics(paths,count)


def prefix_figure(args,inputs):
    original,observed,prefix,trajectory,common=inputs
    fig,axes=plt.subplots(2,1,figsize=(10.5,8),sharex=True,gridspec_kw={'height_ratios':[1.5,1]})
    fig.suptitle('First 100 updates: original and fresh prefix runs',x=.10,y=.975,ha='left',fontsize=16,fontweight='bold')
    fig.text(.10,.928,'Full scheduler: 892 updates; warmup: 45 updates; observation stops at 100',fontsize=10,color='#47515b')
    if args.partial_diagnostic:
        fig.text(.10,.895,'PARTIAL DIAGNOSTIC — campaign failed on full-clean truncation; full-other not run',fontsize=9,color='#9b3423')
    for records,label,color,linestyle in [(prefix,'Original',COLORS[0],'-'),(observed['train'],'Fresh prefix',COLORS[1],'--')]:
        steps=[r['step'] for r in records];values=[r['loss'] for r in records]
        axes[0].plot(steps,values,color=color,alpha=.25,linewidth=.8)
        axes[0].plot(steps,rolling_mean(values,5),color=color,linestyle=linestyle,linewidth=1.8,label=label+' / trailing 5-update mean')
    axes[1].plot([r['step'] for r in prefix],[r['learning_rate']*1e5 for r in prefix],color=COLORS[0],linewidth=3,label='Original / update-used LR')
    axes[1].plot([r['step'] for r in trajectory],[r['used_learning_rate']*1e5 for r in trajectory],color=COLORS[1],linestyle='--',linewidth=1.7,label='Fresh prefix / update-used LR')
    axes[0].set_ylabel('Training loss\n(microbatch-mean CE)')
    axes[1].set_ylabel('Learning rate\n(× 1e-5)');axes[1].set_xlabel('Optimizer update')
    axes[1].set_ylim(0,1.12);axes[1].set_xlim(0,104);axes[1].set_xticks([0,25,45,50,75,100])
    for axis in axes:
        style_axes(axis);axis.legend(loc='upper right',frameon=False,fontsize=8)
        for point in (25,50,100):axis.axvline(point,color='#adb5bc',linestyle=':',linewidth=.7)
    axes[1].legend(loc='lower right',frameon=False,fontsize=8)
    fig.subplots_adjust(left=.10,right=.98,top=.84 if args.partial_diagnostic else .87,bottom=.17,hspace=.25)
    caption='Observed prefix only. The scheduler was not rebuilt with max_steps=100: LR remains nonzero at update 100.\nWarmup has 45 scheduler steps; the update-used LR first reaches its peak at update 46. Pale lines are raw losses; dark lines are trailing means.\nMatched configuration and aggregate data counts do not prove the original per-ID order or bitwise CUDA replay.'
    sources=common+original['sources']+observed['sources']
    derived=dict(observed_steps=[1,100],original_total_steps=892,warmup_steps=45,planned_stop=100,
        final_used_learning_rate=trajectory[-1]['used_learning_rate'],max_lr_difference=max(abs(a['used_learning_rate']-b['learning_rate']) for a,b in zip(trajectory,prefix)),
        loss_smoothing='trailing 5-update mean; partial window at beginning',
        original_per_id_order_recorded=False,claimed_bitwise_replay=False)
    return save_figure(fig,'first100_training','Observed original/new first100 loss and learning rate',caption,sources,derived,args)


def fixed_wer_figure(args,inputs):
    baseline=args.reports/'baselines/qwen-pinned-b16/dev-clean-fixed128.metrics.json'
    paths=[baseline]+[args.early100/'evaluation'/f'step{step}-fixed128/predictions.metrics.json' for step in (25,50,100)]
    metrics=matching_metrics(paths,128)
    stages=[completed_stage(path.parent) for path in paths[1:]]
    steps=[0,25,50,100];values=[100*row['wer'] for row in metrics]
    fig,axis=plt.subplots(figsize=(10.5,6.4))
    fig.suptitle('Fixed128 generation WER: measured early checkpoints',x=.10,y=.972,ha='left',fontsize=16,fontweight='bold')
    fig.text(.10,.917,'Same 128 utterances / 2,905 reference words; exploratory development diagnosis',fontsize=10,color='#47515b')
    if args.partial_diagnostic:
        fig.text(.10,.877,'PARTIAL DIAGNOSTIC — campaign failed on full-clean truncation; full-other not run',fontsize=9,color='#9b3423')
    axis.plot(steps,values,marker='o',markersize=7,color=COLORS[0],linewidth=1.8)
    for step,value in zip(steps,values):axis.annotate(f'{value:.4f}%',(step,value),xytext=(0,10),textcoords='offset points',ha='center',fontsize=11,color=COLORS[0])
    axis.set_xticks(steps,['Base\n0','Checkpoint\n25','Checkpoint\n50','Checkpoint\n100'])
    axis.set_xlim(-8,108);axis.set_ylim(0,max(values)*1.25+.1)
    axis.set_ylabel('Corpus word error rate (%)');axis.set_xlabel('Measured optimizer update')
    style_axes(axis);fig.subplots_adjust(left=.10,right=.98,top=.81 if args.partial_diagnostic else .85,bottom=.23)
    caption='Only base / 25 / 50 / 100 generation results are measured here; lines are visual guides, not a continuous WER trajectory.\nFixed128 is a development subset, not full dev-clean or independent test. Same IDs/normalizer, greedy, batch 16, budget 512; zero truncation.\nRepeated development inspection is exploratory; this chart neither selects a winning checkpoint nor establishes the cause of any change.'
    derived=dict(steps=steps,wer_percent=values,reference_words=metrics[0]['reference_words'],examples=128,
        edits=[sum(row[k] for k in ('substitutions','deletions','insertions')) for row in metrics])
    return save_figure(fig,'fixed128_early_wer','Observed base25/50/100 fixed128 WER',caption,paths+stages+inputs[-1],derived,args)


def full_wer_figure(args,inputs):
    campaign=args.reports/'qwen-campaign-20261022-a'
    fig,axis=plt.subplots(figsize=(10.5,6.6));sources=[];derived=[]
    for split,color,examples in [('dev-clean',COLORS[0],2694),('dev-other',COLORS[1],2857)]:
        paths=[campaign/'evaluation'/f'baseline-{split}/predictions.metrics.json',
            args.early100/'evaluation'/f'step100-{split.replace("-","_")}/predictions.metrics.json',
            args.reports/'diagnosis/checkpoint800'/split/'predictions.metrics.json',
            campaign/'evaluation'/f'final-{split.replace("-","_")}/predictions.metrics.json']
        metrics=matching_metrics(paths,examples);values=[100*row['wer'] for row in metrics]
        axis.plot(range(4),values,marker='o',markersize=7,color=color,linewidth=1.8,label=f'{split}: {examples:,} eligible utterances')
        for index,value in enumerate(values):axis.annotate(f'{value:.4f}%',(index,value),xytext=(0,10),textcoords='offset points',ha='center',fontsize=10,color=color)
        sources+=paths+[completed_stage(paths[1].parent)]
        derived.append(dict(split=split,steps=[0,100,800,892],wer_percent=values,examples=examples,
            reference_words=metrics[0]['reference_words'],step100_origin='new fresh prefix run',step800_892_origin='original full-epoch run'))
    axis.set_xticks(range(4),['Base\n0','Fresh prefix\n100','Original run\n800','Original run\n892'])
    axis.set_xlim(-.25,3.30);axis.set_ylim(0,max(max(row['wer_percent']) for row in derived)*1.22+.2)
    axis.set_ylabel('Corpus word error rate (%)');style_axes(axis);axis.legend(loc='upper left',frameon=False,fontsize=9)
    fig.suptitle('Eligible full-development WER: only measured checkpoints',x=.10,y=.97,ha='left',fontsize=15,fontweight='bold')
    fig.subplots_adjust(left=.10,right=.98,top=.85,bottom=.26)
    caption='Step 100 comes from the separate fresh-prefix run; steps 800/892 come from the original run. Connecting lines do not establish one identical replay.\nNo full-development result exists at 25 or 50 in this diagnostic; none is interpolated or inferred. Same IDs/references/normalizer and decoding; zero truncation.\nEligible ≤30 s only: clean 2,694/2,703; other 2,857/2,864. Not the unfiltered official benchmark or independent test.'
    return save_figure(fig,'full_dev_step100_comparison','Measured full eligible dev base100/800/892 WER',caption,sources+inputs[-1],derived,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--early100',required=True,type=Path,help='Completed local run directory with training/ and evaluation/ receipts')
    parser.add_argument('--reports',type=Path,default=REPO/'reports')
    parser.add_argument('--output',type=Path,default=REPO/'reports/figures/early100')
    parser.add_argument('--dpi',type=int,default=220)
    parser.add_argument('--partial-diagnostic',action='store_true',help='Only accepted fixed128 + completed training; requires the recorded failed full-clean truncation and absent full-other')
    parser.add_argument('--check-only',action='store_true',help='Validate completed inputs without rendering or writing outputs')
    args=parser.parse_args();args.early100=args.early100.resolve();args.reports=args.reports.resolve();args.output=args.output.resolve()
    inputs=validate_inputs(args)
    # Full mode still requires all five accepted stages; partial mode cannot render full dev.
    for step in (25,50,100):completed_stage(args.early100/'evaluation'/f'step{step}-fixed128')
    if not args.partial_diagnostic:
        for split in ('dev_clean','dev_other'):completed_stage(args.early100/'evaluation'/f'step100-{split}')
    validate_generation_inputs(args)
    if args.check_only:
        print(('Partial diagnostic: completed training and accepted fixed128 only; failed full-clean excluded, full-other not run.' if args.partial_diagnostic else 'Complete early100 campaign: all five accepted evaluations validated.')+' Trajectory and comparison settings validated; no figures written.')
        return
    if args.dpi<72:parser.error('DPI must be at least 72')
    if args.output==(args.reports/'figures').resolve():
        parser.error('Use a separate early100 output directory; original figure manifest/README must remain untouched')
    if args.partial_diagnostic and any(args.output.glob('full_dev_step100_comparison.*')):
        parser.error('Partial output directory contains a full-dev chart; use a fresh directory to avoid stale results')
    args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#26323d',
        'text.color':'#172532','xtick.color':'#47515b','ytick.color':'#47515b','axes.edgecolor':'#b5bec5',
        'svg.fonttype':'none','svg.hashsalt':'asr-lab-early100-figures-v1','pdf.fonttype':42})
    figures={}
    renderers=[('first100_training',prefix_figure),('fixed128_early_wer',fixed_wer_figure)]
    if not args.partial_diagnostic:
        renderers.append(('full_dev_step100_comparison',full_wer_figure))
    for name,renderer in renderers:
        figures[name]=renderer(args,inputs)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),script=source(__file__),helper=source(shared.__file__),
        mode='partial_diagnostic' if args.partial_diagnostic else 'complete_campaign',
        campaign_status='failed' if args.partial_diagnostic else 'complete',
        excluded_evaluations={'step100-dev_clean':'rejected: one 512-token generation truncation','step100-dev_other':'not run'} if args.partial_diagnostic else {},
        python_version=sys.version.split()[0],matplotlib_version=shared.matplotlib.__version__,
        command=shlex.join([sys.executable,relative(__file__),*sys.argv[1:]]),figures=figures)
    (args.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    notice=('诊断部分结果：campaign因full-clean一条512-token截断而failed；full-other未执行。这里只展示已通过的fixed128与完成的100步训练；不接受或绘制full-dev结果。' if args.partial_diagnostic else '本地同步的campaign、训练prefix与五个生成stage全部通过后生成。')
    command=['.venv/bin/python','scripts/plot_early100_results.py','--early100',relative(args.early100)]
    if args.partial_diagnostic:command.append('--partial-diagnostic')
    lines=['# Early100真实结果图片','',notice+' 本目录不修改原三组图与其manifest。','',
        '复算（仓库根目录，依赖固定于`requirements-plots.txt`）：','',
        '```bash',shlex.join(command),'```','',
        '每个图的输入清单/完整SHA256、实际指标摘要和PNG/SVG/PDF输出指纹见同名metadata；脚本和复用helper指纹见`figure_manifest.json`。','']
    for name,figure in figures.items():
        lines += ['## '+figure['title'],'',f'![{name}]({name}.png)','',
            f'[PNG]({name}.png) · [SVG]({name}.svg) · [PDF]({name}.pdf) · [来源/hash]({name}.metadata.json)','',figure['caption'].replace('\n',' '),'']
    (args.output/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(output=str(args.output),figures=list(figures)),indent=2))


if __name__=='__main__':main()
