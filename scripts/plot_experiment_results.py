#!/usr/bin/env python3
"""Render auditable ASR experiment figures from existing local receipts; CPU only."""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile

os.environ.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'asr-lab-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO=Path(__file__).resolve().parents[1]
COLORS=['#0072B2','#D55E00','#009E73','#8C4E93']


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def relative(path):
    path=Path(path).resolve()
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def source(path):
    return dict(path=relative(path),sha256=sha256(path),bytes=Path(path).stat().st_size)


def finite(value,label):
    number=float(value)
    if not math.isfinite(number):
        raise ValueError(f'Nonfinite plotted value: {label}')
    return number


def rolling_mean(values,window):
    active=deque()
    total=0.0
    result=[]
    for value in values:
        active.append(value);total+=value
        if len(active)>window:
            total-=active.popleft()
        result.append(total/len(active))
    return result


def load_training(label,path):
    path=Path(path)
    metrics=path if path.is_file() else path/'metrics.jsonl'
    rows=[json.loads(line) for line in metrics.read_text().splitlines() if line.strip()]
    training=[r for r in rows if r.get('phase')=='train' and 'loss' in r]
    dev=[r for r in rows if 'eval_loss' in r]
    if not training:
        raise ValueError(f'No actual optimizer-step loss records: {metrics}')
    for records in (training,dev):
        steps=[int(row['step']) for row in records]
        if steps!=sorted(set(steps)):
            raise ValueError(f'Unsorted or duplicate plotted steps: {metrics}')
    for row in training:
        finite(row['loss'],'train loss');finite(row['learning_rate'],'learning rate')
    for row in dev:
        finite(row['eval_loss'],'development loss')
    paths=[metrics]
    if (metrics.parent/'run.json').exists():
        paths.append(metrics.parent/'run.json')
    return dict(label=label,train=training,dev=dev,sources=paths)


def style_axes(axis):
    axis.spines[['top','right']].set_visible(False)
    axis.grid(axis='y',color='#e3e7eb',linewidth=.7)
    axis.set_axisbelow(True)


def save_figure(fig,name,title,caption,paths,derived,args):
    files={}
    fig.text(.02,.017,caption,ha='left',va='bottom',fontsize=8,color='#424a52')
    for extension in ('png','svg','pdf'):
        path=args.output/f'{name}.{extension}'
        metadata={'Title':title,'Creator':'ASR Lab / plot_experiment_results.py'}
        if extension=='pdf':
            metadata.update(CreationDate=None,ModDate=None)
        elif extension=='svg':
            metadata.update(Date=None,Description=caption)
        else:
            metadata={'Title':title,'Description':caption,'Software':'matplotlib '+matplotlib.__version__}
        fig.savefig(path,dpi=args.dpi,metadata=metadata,facecolor='white')
        files[extension]=source(path)
    plt.close(fig)
    record=dict(title=title,caption=caption,sources=[source(path) for path in dict.fromkeys(paths)],
        derived_data=derived,outputs=files)
    sidecar=args.output/f'{name}.metadata.json'
    sidecar.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    record['metadata']=source(sidecar)
    return record


def training_figure(args):
    campaign=args.reports/'qwen-campaign-20261022-a'
    runs=[load_training('Original full epoch (892 updates)',campaign/'main-epoch1')]
    for specification in args.additional_run:
        label,separator,path=specification.partition('=')
        if not separator or not label or not path:
            raise ValueError('--additional-run requires LABEL=RUN_DIRECTORY_OR_METRICS_JSONL')
        runs.append(load_training(label,path))
    fig,axes=plt.subplots(3,1,figsize=(11,9.3),sharex=True,gridspec_kw={'height_ratios':[1.6,1,1]})
    fig.suptitle('Qwen3-ASR: observed training dynamics',x=.08,y=.975,ha='left',fontsize=17,fontweight='bold')
    fig.text(.08,.934,'Full-parameter SFT; original run: global batch 32, learning rate 1e-5, one epoch',fontsize=10,color='#47515b')
    sources=[];summary=[]
    for index,run in enumerate(runs):
        color=COLORS[index%len(COLORS)]
        records=run['train'];steps=[row['step'] for row in records]
        loss=[row['loss'] for row in records]
        axes[0].plot(steps,loss,color=color,alpha=.22,linewidth=.7,label=run['label']+' / raw')
        axes[0].plot(steps,rolling_mean(loss,args.smoothing_window),color=color,linewidth=1.8,
            label=run['label']+f' / trailing {args.smoothing_window}-update mean')
        if run['dev']:
            axes[1].plot([row['step'] for row in run['dev']],[row['eval_loss'] for row in run['dev']],
                marker='o',markersize=4,color=color,linewidth=1.6,label=run['label'])
        axes[2].plot(steps,[float(row['learning_rate'])*1e5 for row in records],color=color,linewidth=1.8,label=run['label'])
        sources+=run['sources']
        summary.append(dict(label=run['label'],train_records=len(records),first_step=steps[0],last_step=steps[-1],
            eval_records=len(run['dev']),eval_points=[dict(step=r['step'],eval_loss=r['eval_loss']) for r in run['dev']],
            smoothing=dict(method='trailing arithmetic mean; partial window at the beginning',window=args.smoothing_window)))
    axes[0].set_ylabel('Training loss\n(microbatch-mean CE)')
    axes[1].set_ylabel('Development loss\n(fixed 128 examples)')
    axes[2].set_ylabel('Learning rate\n(× 1e-5)');axes[2].set_xlabel('Optimizer update')
    for axis in axes:
        style_axes(axis)
        axis.legend(loc='upper right',fontsize=8,frameon=False)
        axis.axvline(800,color='#9ba3aa',linewidth=.8,linestyle=':')
    axes[2].set_ylim(bottom=0)
    fig.subplots_adjust(left=.10,right=.98,top=.90,bottom=.115,hspace=.24)
    caption='Original run: 892 training records; its development loss is observed only at updates 100–800. No endpoint dev-loss point is invented.\nLoss is not generation WER. Raw losses and explicitly labeled trailing means are both shown; learning-rate values are logged observations.'
    return save_figure(fig,'training_dynamics','Observed training loss, development loss and learning rate',caption,sources,summary,args)


def validate_wer(path):
    metrics=read_json(path)
    words=int(metrics['reference_words'])
    edits=sum(int(metrics[key]) for key in ('substitutions','deletions','insertions'))
    if words<=0 or not math.isclose(finite(metrics['wer'],'WER'),edits/words,rel_tol=1e-12,abs_tol=1e-12):
        raise ValueError(f'Invalid corpus-WER accounting: {path}')
    if metrics.get('generation_truncated_count')!=0:
        raise ValueError(f'Truncated generation cannot be presented as accepted WER: {path}')
    return metrics


def wer_figure(args):
    campaign=args.reports/'qwen-campaign-20261022-a'
    fig,axis=plt.subplots(figsize=(10.5,6.4))
    sources=[];summary=[]
    for split,color in [('dev-clean',COLORS[0]),('dev-other',COLORS[1])]:
        paths=[campaign/'evaluation'/f'baseline-{split}'/'predictions.metrics.json',
               args.reports/'diagnosis/checkpoint800'/split/'predictions.metrics.json',
               campaign/'evaluation'/f'final-{split.replace("-","_")}'/'predictions.metrics.json']
        metrics=[validate_wer(path) for path in paths]
        for key in ('reference_words','examples','selected_ids','manifest_sha256','normalizer','normalizer_revision',
                    'evaluation_batch_size','generation_max_new_tokens'):
            if metrics[0].get(key) is None or any(row.get(key)!=metrics[0][key] for row in metrics[1:]):
                raise ValueError(f'WER checkpoint comparison setting differs: {split}/{key}')
        values=[100*row['wer'] for row in metrics]
        axis.plot(range(3),values,marker='o',markersize=7,color=color,linewidth=2,label=f'{split}: {metrics[0]["examples"]:,} eligible utterances')
        for step,value in enumerate(values):
            axis.annotate(f'{value:.4f}%',(step,value),xytext=(0,10),textcoords='offset points',ha='center',fontsize=11,color=color)
        sources+=paths
        summary.append(dict(split=split,checkpoints=[0,800,892],wer_percent=values,reference_words=metrics[0]['reference_words'],
            examples=metrics[0]['examples'],edits=[sum(row[key] for key in ('substitutions','deletions','insertions')) for row in metrics],
            generation_batch=metrics[0]['evaluation_batch_size'],max_new_tokens=metrics[0]['generation_max_new_tokens']))
    axis.set_xticks(range(3),['Pretrained base\n(step 0)','Checkpoint 800','Checkpoint 892\n(epoch 1 complete)'])
    axis.set_ylabel('Corpus word error rate (%)');axis.set_ylim(0,6.2);axis.set_xlim(-.22,2.30)
    axis.set_title('Generation WER at the three measured checkpoints',loc='left',fontsize=16,fontweight='bold',pad=19)
    axis.legend(loc='upper left',frameon=False,fontsize=10)
    style_axes(axis)
    fig.subplots_adjust(left=.09,right=.98,top=.88,bottom=.23)
    caption='Eligible ≤30 s development data only: dev-clean 2,694/2,703; dev-other 2,857/2,864. Not the full unfiltered official benchmark or independent test.\nSame IDs, references, normalizer, greedy decoding, batch 16 and 512-token budget; zero truncation. Lines join measured points, not an observed continuous trajectory.'
    return save_figure(fig,'checkpoint_wer','Eligible-development generation WER by checkpoint',caption,sources,summary,args)


def throughput_figure(args):
    campaign=args.reports/'qwen-campaign-20261022-a'
    receipt=campaign/'experiment_results.json'
    rows=read_json(receipt)
    lookup={row['name']:row for row in rows}
    names=['E1-single-b2','E2-dual-b2','E3-dual-b4','E4-dual-b8']
    values=[];labels=[];summary=[];sources=[receipt]
    for name in names:
        row=lookup[name];path=campaign/name/'run.json';metadata=read_json(path)
        timing_path=campaign/name/'benchmark_summary.json';timing=read_json(timing_path)
        speed=finite(timing['audio_seconds_per_wall_second'],'throughput')
        if not row.get('valid') or row.get('status')!='measured' or metadata['effective_batch_size']!=32:
            raise ValueError('Throughput chart requires accepted measured global-batch-32 experiments')
        if timing['warmup_steps']!=10 or timing['measured_steps']!=30 or not math.isclose(speed,timing['total_measured_audio_seconds']/timing['total_measured_step_seconds'],rel_tol=1e-12):
            raise ValueError('Inconsistent throughput measurement window or denominator')
        world=metadata['world_size'];batch=metadata['config']['batch_size'];acc=metadata['config']['gradient_accumulation_steps']
        if world*batch*acc!=32:
            raise ValueError('Inconsistent effective batch')
        values.append(speed);labels.append(f'{world} GPU'+('s' if world>1 else '')+f'\nmicro {batch} × accum {acc}')
        sources += [path,timing_path]
        summary.append(dict(name=name,world_size=world,microbatch=batch,accumulation=acc,effective_batch=32,
            audio_seconds_per_wall_second=speed,project_peak_gib=row['peak_gib'],minimum_free_gib=row['min_free_gib']))
    fig,axis=plt.subplots(figsize=(10.5,6.5))
    bars=axis.bar(range(4),values,width=.62,color=['#94a5b4','#0072B2','#2684AE','#009E73'],zorder=3)
    axis.bar_label(bars,labels=[f'{value:.2f}' for value in values],padding=6,fontsize=12)
    axis.set_xticks(range(4),labels);axis.set_ylim(0,max(values)*1.20)
    axis.set_ylabel('Audio seconds / wall-clock second')
    fig.suptitle('Measured training throughput on one and two H100 GPUs',x=.11,y=.967,ha='left',fontsize=15,fontweight='bold')
    style_axes(axis)
    fig.text(.11,.90,f'Same microbatch: dual / single = {values[1]/values[0]:.2f}×     Best dual / single = {values[3]/values[0]:.2f}× (also changes microbatch)',fontsize=10,color='#47515b')
    fig.subplots_adjust(left=.11,right=.98,top=.845,bottom=.23)
    caption='Same pinned model, 2,048-example subset and global batch 32; updates 11–40 after 10 warmup updates.\nUpdate timing excludes model loading, checkpoint writes and generation evaluation. GPU0 also hosts PersonaPlex; NODE topology, no assumed NVLink.'
    return save_figure(fig,'gpu_throughput','Controlled H100 throughput measurements',caption,sources,summary,args)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,default=REPO/'reports')
    parser.add_argument('--output',type=Path,default=REPO/'reports/figures')
    parser.add_argument('--additional-run',action='append',default=[],help='Optional LABEL=RUN_DIRECTORY_OR_METRICS_JSONL for an observed early100 curve')
    parser.add_argument('--smoothing-window',type=int,default=25)
    parser.add_argument('--dpi',type=int,default=220)
    args=parser.parse_args()
    if args.smoothing_window<1 or args.dpi<72:
        parser.error('Smoothing window must be positive and DPI at least 72')
    args.reports=args.reports.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.labelcolor':'#26323d',
        'text.color':'#172532','xtick.color':'#47515b','ytick.color':'#47515b','axes.edgecolor':'#b5bec5',
        'svg.fonttype':'none','svg.hashsalt':'asr-lab-figures-v1','pdf.fonttype':42})
    figures={}
    for name,renderer in [('training_dynamics',training_figure),('checkpoint_wer',wer_figure),('gpu_throughput',throughput_figure)]:
        figures[name]=renderer(args)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),script=source(__file__),
        python_version=sys.version.split()[0],matplotlib_version=matplotlib.__version__,
        command=shlex.join([sys.executable,relative(__file__),*sys.argv[1:]]),
        defaults=dict(smoothing_window=args.smoothing_window,dpi=args.dpi,additional_runs=args.additional_run),figures=figures)
    (args.output/'figure_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),figures=list(figures),matplotlib=matplotlib.__version__),indent=2))


if __name__=='__main__':
    main()
