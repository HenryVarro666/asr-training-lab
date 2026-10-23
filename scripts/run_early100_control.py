#!/usr/bin/env python3
"""One original-schedule100-update prefix and exactly five generation evaluations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import time

import yaml

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_qwen_campaign import Campaign,BudgetExpired,file_hash,read_json,manifest_ids,validate_evaluation
from asr_lab.prepare_data import check_space,tree_size
from asr_lab.training_utils import atomic_json
from asr_lab.qwen_roundtrip_control import freeze_receipt
from asr_lab.qwen_early100 import validate_config

REPO=Path(__file__).resolve().parents[1]
TOTAL_SECONDS=5400
TRAIN_SECONDS=1200
EVAL_SECONDS=600
INCREMENT_BYTES=80*1024**3


def bounded_deadline(now,total_deadline,seconds):
    if seconds<=0 or now>=total_deadline:raise BudgetExpired('No stage budget remains')
    return min(total_deadline,now+seconds)


def check_terminal(run):
    run=Path(run)
    terminal=read_json(run/'prefix-status.json')
    trajectory=read_json(run/'trajectory-validation.json')
    if terminal.get('status')!='completed_bounded_prefix' or terminal.get('actual_step')!=100 or terminal.get('original_total_steps')!=892 or terminal.get('completed_epoch') is not False:
        raise ValueError('Missing planned bounded-prefix completion')
    if trajectory.get('status')!='passed' or trajectory.get('validated_steps')!=100 or trajectory.get('original_total_steps')!=892 or trajectory.get('warmup_steps')!=45:
        raise ValueError('Missing validated original892/45 trajectory')
    markers={}
    for step in (25,50,100):
        path=run/f'checkpoint-{step}'
        marker=read_json(path/'prefix-complete.json')
        if marker.get('status')!='complete' or marker.get('step')!=step or marker.get('original_total_steps')!=892 or marker.get('model_only')!=(step!=100) or marker.get('resumable')!=(step==100):
            raise ValueError(f'Incomplete or incorrect checkpoint{step}')
        for name,expected in marker['files'].items():
            file=path/name
            if file.stat().st_size!=expected['bytes'] or file_hash(file)!=expected['sha256']:
                raise ValueError(f'Checkpoint integrity changed:{file}')
        proofs=read_json(path/'rng-preservation.json')
        if sorted(p['rank'] for p in proofs)!=[0,1] or any(p['before']!=p['restored'] for p in proofs):
            raise ValueError('Missing two-rank RNG preservation proof')
        if step==100 and any(p['before']!=p['observed_after_native_save'] for p in proofs):
            raise ValueError('Full-checkpoint RNG differs from resumed active state')
        markers[str(step)]=marker
    return dict(terminal=terminal,trajectory=trajectory,checkpoints=markers)


def validate_decode(metrics,base=None):
    expected=dict(evaluation_batch_size=16,seed=42,normalizer='Whisper EnglishTextNormalizer',
        normalizer_revision='973afd24965f72e36ca33b3055d56a652f456b4d',generation_max_new_tokens=512)
    if any(metrics.get(k)!=v for k,v in expected.items()):raise ValueError('Decode/normalization settings differ')
    if base and any(metrics.get(k)!=base.get(k) for k in (*expected,'selected_ids','reference_words')):
        raise ValueError('Paired evaluation order/reference/decode differs')


class Early100Campaign(Campaign):
    def __init__(self,args):
        self.stage_deadline=None
        self.next_space_check=0.
        self.original_project_bytes=tree_size(args.project_root.resolve())
        super().__init__(args)
        self.state.update(kind='original_schedule_early100',original_total_steps=892,actual_stop=100,
                          automatic_followup_training=False,evaluations={})
        self.state.pop('selected_config',None);self.state.pop('main_lr_matrix_status',None)
        self.write_state()

    def remaining(self):
        left=super().remaining()
        now=time.monotonic()
        if self.stage_deadline is not None:
            left=min(left,self.stage_deadline-now)
            if left<=0:raise BudgetExpired('Stage wall-time budget exhausted')
        if now>=self.next_space_check:
            project_bytes=tree_size(self.root)
            if max(tree_size(self.directory),project_bytes-self.original_project_bytes)>INCREMENT_BYTES:
                raise RuntimeError('Early100 exceeded80GiB incremental storage cap')
            check_space(self.root,500,300,0)
            self.next_space_check=now+5
        return left

    def execute_stage(self,name,command,run,gpus,seconds,projected_gib=0):
        self.stage_deadline=bounded_deadline(time.monotonic(),self.deadline,seconds)
        try:
            result=self.execute(name,command,run,gpus,projected_gib)
            self.remaining()
            return result
        finally:
            self.stage_deadline=None

    def report(self):
        lines=['# Original-schedule Early100 control','',f"Status: {self.state['status']}; stage: {self.state['stage']}.",'',
            'One fresh initialization; original892-step scheduler and45-step warmup; planned stop after100 updates.',
            '25/50 are model-only exports.100 contains full recovery state, structurally verified; no new resume-training test.',
            'No follow-up training or automatic retry. Full development results cover only eligible audio of at most30 seconds.',
            'Matched aggregate counts/audio do not prove the original per-ID order or bitwise model trajectory.']
        if self.state.get('evaluations'):
            lines+=['','| Evaluation | N | Base WER | Prefix WER | Difference (percentage points) | S / D / I |',
                    '|---|---:|---:|---:|---:|---|']
            for name,row in self.state['evaluations'].items():
                lines.append(f"| {name} | {row['examples']} | {100*row['baseline_wer']:.4f}% | {100*row['wer']:.4f}% | {100*(row['wer']-row['baseline_wer']):+.4f} | {row['substitutions']} / {row['deletions']} / {row['insertions']} |")
            lines+=['','All three preset intermediate points are exploratory diagnostics; no checkpoint is selected as a winner.']
        if self.state.get('error'):lines+=['','Failure: '+self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def run(self):
        self.remaining()
        check_space(self.root,500,300,INCREMENT_BYTES)
        original=read_json(self.args.original_run/'run.json')
        receipt=read_json(self.args.receipt)
        fixed=Path(original['config']['eval_manifest'])
        frozen=freeze_receipt(self.args.receipt,fixed)
        self.state.update(frozen)
        self.state['source_sha256'].update({str(REPO/path):file_hash(REPO/path) for path in
            ('src/asr_lab/qwen_early100.py','scripts/run_early100_control.py','scripts/run_qwen_campaign.py',
             'scripts/monitor.py','src/asr_lab/qwen_roundtrip_control.py')})
        for name,expected in original['source_sha256'].items():
            path=REPO/name
            if file_hash(path)!=expected:raise ValueError(f'Original semantic source changed:{path}')
            self.state['source_sha256'][str(path)]=expected
        for path in (self.args.receipt,self.args.original_run/'run.json',self.args.original_run/'metrics.jsonl'):
            self.state['source_sha256'][str(path.resolve())]=file_hash(path)
        if original['model_id']!=frozen['model_id'] or original['revision']!=frozen['model_revision']:
            raise ValueError('Original model receipt differs')
        if Path(original['config']['project_root']).resolve()!=self.root:
            raise ValueError('Canonical project root differs from original')
        self.state['manifests']=dict(receipt['manifests'])
        for path,expected in self.state['manifests'].items():
            if file_hash(path)!=expected:raise ValueError('Original manifest changed')
        manifests={Path(k).name:Path(k) for k in self.state['manifests']}
        selected=dict(fixed128=fixed,dev_clean=manifests['dev-clean.jsonl'],dev_other=manifests['dev-other.jsonl'])
        if {name:len(manifest_ids(path)) for name,path in selected.items()}!=dict(fixed128=128,dev_clean=2694,dev_other=2857):
            raise ValueError('Original eligible evaluation populations differ')
        baselines=dict(fixed128=Path(receipt['baseline_fixed128']),
            dev_clean=self.args.receipt.parent/'evaluation/baseline-dev-clean/predictions.metrics.json',
            dev_other=self.args.receipt.parent/'evaluation/baseline-dev-other/predictions.metrics.json')
        base={}
        for name,path in baselines.items():
            base[name]=validate_evaluation(path,selected[name],frozen['model_revision'])
            validate_decode(base[name])
            for file in (path,path.with_name(path.name.removesuffix('.metrics.json')+'.jsonl')):
                self.state['source_sha256'][str(file.resolve())]=file_hash(file)
        config=dict(original['config'],max_wall_seconds=TRAIN_SECONDS)
        validate_config(original['config'],config)
        config_path=self.directory/'config.yaml'
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        self.state['source_sha256'][str(config_path)]=file_hash(config_path)
        self.write_state()
        run=self.directory/'training'
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
            '-m','asr_lab.qwen_early100','--config',str(config_path),'--run-dir',str(run),
            '--original-run',str(self.args.original_run)]
        self.execute_stage('training-prefix100',command,run,'0,1',TRAIN_SECONDS,projected_gib=60)
        evidence=check_terminal(run)
        self.state['training']=evidence
        self.write_state()
        for step,split in ((25,'fixed128'),(50,'fixed128'),(100,'fixed128'),(100,'dev_clean'),(100,'dev_other')):
            name=f'step{step}-{split}'
            output_dir=self.directory/'evaluation'/name
            command=[sys.executable,'-m','asr_lab.qwen_evaluate','--model',str(run/f'checkpoint-{step}'),
                '--revision',frozen['model_revision'],'--manifest',str(selected[split]),
                '--output',str(output_dir/'predictions.jsonl'),'--all','--batch-size','16','--max-new-tokens','512']
            self.execute_stage(name,command,output_dir,'1',EVAL_SECONDS)
            metrics=validate_evaluation(output_dir/'predictions.metrics.json',selected[split])
            validate_decode(metrics,base[split])
            if metrics['model']!=str(run/f'checkpoint-{step}'):
                raise ValueError('Evaluated model path differs')
            self.state['evaluations'][name]={key:metrics[key] for key in
                ('wer','examples','reference_words','substitutions','deletions','insertions','generation_truncated_count')}
            self.state['evaluations'][name]['baseline_wer']=base[split]['wer']
            self.write_state()
        self.remaining()
        self.state.update(status='complete',stage='five_evaluations_complete',completed_unix=time.time(),
            completion_reason='completed_bounded_prefix_and_all_preset_evaluations',completed_epoch=False)
        self.write_state()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--receipt',required=True,type=Path,help='Original campaign.json')
    parser.add_argument('--original-run',required=True,type=Path,help='Original main-epoch1 directory')
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--max-wall-seconds',type=float,default=TOTAL_SECONDS)
    args=parser.parse_args()
    if not 0<args.max_wall_seconds<=TOTAL_SECONDS:parser.error('Total budget must be in(0,5400]seconds')
    args.campaign_dir=args.run_dir
    def stop(*_):raise KeyboardInterrupt('Early100 interrupted')
    signal.signal(signal.SIGTERM,stop)
    control=Early100Campaign(args)
    try:
        control.run()
    except BaseException as error:
        control.state.update(status='budget_stopped' if isinstance(error,BudgetExpired) else ('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'),
                             error=f'{type(error).__name__}: {error}',automatic_followup_training=False)
        control.write_state()
        raise
    finally:
        control.lock.close();control.global_lock.close()


if __name__=='__main__':main()
