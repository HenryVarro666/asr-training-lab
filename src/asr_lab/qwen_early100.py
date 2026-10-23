"""Thin isolated prefix control around the frozen Qwen trainer; no scheduler/data truncation."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import torch.distributed as dist
import yaml

from . import qwen_train as frozen
from .qwen_data import QwenCollator as OriginalCollator
from .training_utils import atomic_json,sha256_file
from .prepare_data import check_space

TOTAL=892
WARMUP=45
STOP=100
_CONTROL={}
OriginalTrainer=frozen.SafeTrainer
OriginalCallback=frozen.ExperimentCallback


def save_policy(step):
    if not 0<=step<=STOP:raise ValueError('Outside planned prefix100')
    return dict(save=step in (25,50,100),model_only=step in (25,50),stop=step==100)


def validate_schedule(args,state):
    if args.max_steps>0 or state.max_steps!=TOTAL or args.get_warmup_steps(state.max_steps)!=WARMUP:
        raise ValueError('Actual schedule must remain892 total updates and45 warmup; no positive max_steps')
    if state.global_step!=0:raise ValueError('This prefix must start fresh from original model')


def validate_config(original,current):
    ignored={'max_wall_seconds'}
    if {k:v for k,v in original.items() if k not in ignored}!={k:v for k,v in current.items() if k not in ignored}:
        raise ValueError('Original training config invariant changed')
    if current.get('max_steps',-1)>0 or current.get('limit_train') is not None:
        raise ValueError('Do not cap total steps or shorten the training data')


def validate_step(current,original,used_lr):
    if int(current['step'])!=int(original['step']):raise ValueError('Step alignment differs')
    expected=float(original['learning_rate'])
    if not math.isfinite(used_lr) or not math.isclose(used_lr,expected,rel_tol=8*sys.float_info.epsilon,abs_tol=0.):
        raise ValueError(f'Used LR differs atstep{current["step"]}: {used_lr} != {expected}')
    if int(current['examples'])!=int(original['examples']):raise ValueError('Consumed example count differs')
    difference=abs(float(current['data_seconds'])-float(original['data_seconds']))
    if not math.isfinite(difference) or difference>1e-9:raise ValueError('Consumed audio seconds differ')
    for key in ('loss','grad_norm'):
        if not math.isfinite(float(current[key])):raise FloatingPointError(f'Nonfinite {key}')
    return dict(lr_abs_difference=abs(used_lr-expected),audio_seconds_abs_difference=difference,
                loss_difference=float(current['loss'])-float(original['loss']),
                grad_norm_difference=float(current['grad_norm'])-float(original['grad_norm']))


def capture_rng():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(torch.cuda.current_device()) if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['cpu'])
    if state['cuda'] is not None:torch.cuda.set_rng_state(state['cuda'],torch.cuda.current_device())


def rng_digest(state):
    digest=hashlib.sha256(repr(state['python']).encode())
    n=state['numpy'];digest.update(n[0].encode());digest.update(n[1].tobytes());digest.update(repr(n[2:]).encode())
    digest.update(state['cpu'].numpy().tobytes())
    if state['cuda'] is not None:digest.update(state['cuda'].cpu().numpy().tobytes())
    return digest.hexdigest()


class PrefixCollator:
    def __init__(self,processor,delegate_class=OriginalCollator):self.delegate=delegate_class(processor)
    def __call__(self,rows):
        result=self.delegate(rows)
        result['_early100_ids']=[str(row['id']) for row in rows]
        return result


def gather(value):
    if not dist.is_initialized():return [value]
    values=[None]*dist.get_world_size();dist.all_gather_object(values,value);return values


def verify_checkpoint(path,step,model_only):
    path=Path(path)
    required=['config.json','preprocessor_config.json','trainer_state.json','checkpoint.json']
    for name in required:
        if not (path/name).is_file():raise ValueError(f'Incomplete checkpoint missing{name}')
    weights=sorted(path.glob('*.safetensors'))
    if not weights:raise ValueError('Missing model weights')
    from safetensors import safe_open
    for file in weights:
        with safe_open(file,framework='pt',device='cpu') as handle:
            if any(handle.get_slice(key).get_dtype() in ('F16','BF16') for key in handle.keys()):
                raise ValueError('Expected FP32 model export')
    state=json.loads((path/'trainer_state.json').read_text())
    if state['global_step']!=step or state['max_steps']!=TOTAL:raise ValueError('Checkpoint step/schedule differs')
    metadata=json.loads((path/'checkpoint.json').read_text())
    if metadata['global_step']!=step:raise ValueError('Checkpoint metadata step differs')
    recovery=['optimizer.pt','scheduler.pt']
    if model_only:
        if any((path/name).exists() for name in recovery) or list(path.glob('rng_state*.pth')):
            raise ValueError('Model-only export unexpectedly contains recovery state')
    else:
        if any(not (path/name).is_file() for name in recovery):raise ValueError('Missing full recovery state')
        for rank in (0,1):
            rng_path=path/f'rng_state_{rank}.pth'
            if not rng_path.is_file():raise ValueError('Missing rank RNG state')
            saved=torch.load(rng_path,map_location='cpu',weights_only=False)
            if not {'python','numpy','cpu','cuda'}<=set(saved):raise ValueError('Incomplete rank RNG state')
        scheduler=torch.load(path/'scheduler.pt',map_location='cpu',weights_only=True)
        if scheduler['last_epoch']!=step:raise ValueError('Scheduler checkpoint step differs')
        optimizer=torch.load(path/'optimizer.pt',map_location='cpu',weights_only=True,mmap=True)
        if not optimizer['state']:raise ValueError('Empty optimizer state')
        for value in optimizer['state'].values():
            if int(value['step'].item())!=step:raise ValueError('Optimizer checkpoint step differs')
            if any(value[key].dtype!=torch.float32 for key in ('exp_avg','exp_avg_sq')):
                raise ValueError('Optimizer moments are notFP32')
        del optimizer
    return dict(status='complete',step=step,original_total_steps=TOTAL,model_only=model_only,
        resumable=not model_only,verification='files/structure/steps/dtypes; no new resume training test',
        files={file.name:dict(bytes=file.stat().st_size,sha256=sha256_file(file)) for file in sorted(path.iterdir()) if file.is_file()})


class PrefixTrainer(OriginalTrainer):
    def compute_loss(self,model,inputs,return_outputs=False,num_items_in_batch=None):
        ids=inputs.pop('_early100_ids',None)
        if ids is None:raise ValueError('Missing actual consumed IDs')
        if model.training:
            self.prefix_consumed.append(list(ids))
        return super().compute_loss(model,inputs,return_outputs,num_items_in_batch)

    def _save_checkpoint(self,model,trial):
        if (self.args.load_best_model_at_end or self.args.metric_for_best_model is not None or
                self.state.best_global_step is not None or self.is_deepspeed_enabled or self.is_fsdp_enabled):
            raise ValueError('Checkpoint coordination requires ordinaryDDP without best-model barriers')
        before=capture_rng();error=None
        try:
            super()._save_checkpoint(model,trial)
        except BaseException as exception:
            error=f'{type(exception).__name__}: {exception}'
        finally:
            observed=rng_digest(capture_rng())
            restore_rng(before)
        proof=dict(rank=int(os.environ.get('RANK',0)),before=rng_digest(before),
                   observed_after_native_save=observed,restored=rng_digest(capture_rng()))
        errors=gather(error);proofs=gather(proof)
        if any(errors):raise RuntimeError(f'Coordinated checkpoint failure: {errors}')
        if any(p['before']!=p['restored'] for p in proofs):raise ValueError('RNG restoration failed')
        if self.state.global_step==100 and any(p['before']!=p['observed_after_native_save'] for p in proofs):
            raise ValueError('Full checkpoint native RNG differs from active restored RNG')
        frozen.rank_zero_action(lambda:atomic_json(Path(self.args.output_dir)/f'checkpoint-{self.state.global_step}'/'rng-preservation.json',proofs))


class PrefixCallback(OriginalCallback):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.records=[];self.planned=False
        self.reference=_CONTROL['reference']
        original=_CONTROL['original']
        validate_config(original['config'],self.config)
        for key in ('model_id','revision','official_code_revision','world_size','effective_batch_size',
                    'train_manifest_sha256','eval_manifest_sha256','train_ids_sha256','eval_ids','parameters','packages','package_source_sha256'):
            if self.metadata.get(key)!=original.get(key):raise ValueError(f'Original initialization/data invariant differs:{key}')
    def on_train_begin(self,args,state,control,**kwargs):
        validate_schedule(args,state)
        args.save_total_limit=3
        if self.trainer.optimizer is None:raise ValueError('Optimizer not initialized')
        self.trainer.prefix_consumed=[]
        frozen.rank_zero_action(lambda:atomic_json(self.run/'prefix-plan.json',dict(original_total_steps=TOTAL,
            warmup_steps=WARMUP,actual_stop=STOP,model_only_steps=[25,50],full_checkpoint_step=100,
            rng_control='Python/NumPy/CPU/local CUDA preserved around exports; nativeHF full RNG files retained',
            original_per_id_order_recorded=False,reference_metrics_sha256=_CONTROL['metrics_sha256'])))
        return super().on_train_begin(args,state,control,**kwargs)
    def on_step_begin(self,args,state,control,**kwargs):
        if state.global_step>=STOP:raise ValueError('Training attempted beyond prefix100')
        self.trainer.prefix_consumed=[]
        self.used_lr=float(kwargs['optimizer'].param_groups[0]['lr'])
        return super().on_step_begin(args,state,control,**kwargs)
    def on_step_end(self,args,state,control,**kwargs):
        control=super().on_step_end(args,state,control,**kwargs)
        policy=save_policy(state.global_step)
        local=list(self.trainer.prefix_consumed)
        per_rank=gather(local)
        flat=[id for rank in per_rank for micro in rank for id in micro]
        if len(flat)!=int(self.last_step['examples']):raise ValueError('Actual ID count differs from consumed examples')
        current=dict(self.last_step,step=state.global_step,loss=0.,grad_norm=0.)
        # LR/data checked now; actual loss/grad norm are checked at original on_log boundary.
        validate_step(current,self.reference[state.global_step],self.used_lr)
        scheduler=kwargs['lr_scheduler']
        if scheduler.last_epoch!=state.global_step:raise ValueError('Optimizer/scheduler boundary differs')
        self.current_ids=per_rank
        self.scheduler_after=float(scheduler.get_last_lr()[0])
        if policy['save']:
            args.save_only_model=policy['model_only']
            control.should_save=True
            frozen.rank_zero_action(lambda:check_space(Path(self.config['project_root']),500,300,self.checkpoint_bytes))
        if policy['stop']:
            self.planned=True
            control.should_training_stop=True
            control.should_save=True
        return control
    def on_log(self,args,state,control,logs=None,**kwargs):
        control=super().on_log(args,state,control,logs,**kwargs)
        if logs and 'loss' in logs:
            current=dict(self.last_step,**logs,step=state.global_step)
            drift=validate_step(current,self.reference[state.global_step],self.used_lr)
            record=dict(step=state.global_step,used_learning_rate=self.used_lr,
                scheduler_after_learning_rate=self.scheduler_after,scheduler_last_epoch=state.global_step,
                original_learning_rate=self.reference[state.global_step]['learning_rate'],
                actual_ids_by_rank_and_microbatch=self.current_ids,
                ids_sha256=hashlib.sha256(json.dumps(self.current_ids,separators=(',',':')).encode()).hexdigest(),
                examples=current['examples'],data_seconds=current['data_seconds'],loss=current['loss'],grad_norm=current['grad_norm'],**drift)
            self.records.append(record)
            def write():
                with (self.run/'prefix-trajectory.jsonl').open('a') as stream:stream.write(json.dumps(record,allow_nan=False)+'\n')
            frozen.rank_zero_action(write)
        return control
    def on_save(self,args,state,control,**kwargs):
        before=capture_rng()
        try:
            control=super().on_save(args,state,control,**kwargs)
            policy=save_policy(state.global_step)
            def finish():
                path=self.run/f'checkpoint-{state.global_step}'
                marker=verify_checkpoint(path,state.global_step,policy['model_only'])
                atomic_json(path/'prefix-complete.json',marker)
            frozen.rank_zero_action(finish)
        finally:
            restore_rng(before)
            args.save_only_model=False
        return control
    def on_train_end(self,args,state,control,**kwargs):
        if state.global_step!=STOP or not self.planned or len(self.records)!=STOP:
            raise ValueError('Bounded prefix did not reach exactly100 validated updates')
        def finish():
            for step in (25,50,100):
                marker=json.loads((self.run/f'checkpoint-{step}'/'prefix-complete.json').read_text())
                if marker['step']!=step or marker['model_only']!=(step!=100):raise ValueError('Invalid export completion marker')
            atomic_json(self.run/'trajectory-validation.json',dict(status='passed',validated_steps=100,
                original_total_steps=TOTAL,warmup_steps=WARMUP,
                max_lr_absolute_difference=max(r['lr_abs_difference'] for r in self.records),
                max_audio_seconds_absolute_difference=max(r['audio_seconds_abs_difference'] for r in self.records),
                total_consumed_examples=sum(r['examples'] for r in self.records),
                total_consumed_audio_seconds=sum(r['data_seconds'] for r in self.records),
                new_actual_ids_recorded=True,original_per_id_order_recorded=False,
                limitation='Matched config, counts and audio seconds do not prove old per-ID order or bitwise model replay.'))
        frozen.rank_zero_action(finish)
        return control


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--original-run',required=True,type=Path)
    args=parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='0,1' or int(os.environ.get('WORLD_SIZE','0'))!=2:
        raise ValueError('This control requires exactly physical GPUs0,1 and worldsize2')
    original=json.loads((args.original_run/'run.json').read_text())
    rows=[json.loads(line) for line in (args.original_run/'metrics.jsonl').read_text().splitlines()]
    selected=[row for row in rows if row.get('phase')=='train' and 1<=int(row['step'])<=100]
    reference={int(row['step']):row for row in selected}
    if len(selected)!=100:raise ValueError('Original first100 metrics duplicated/incomplete')
    if sorted(reference)!=list(range(1,101)):raise ValueError('Original first100 metrics incomplete')
    _CONTROL.update(original=original,reference=reference,metrics_sha256=sha256_file(args.original_run/'metrics.jsonl'))
    validate_config(original['config'],yaml.safe_load(args.config.read_text()))
    frozen.ExperimentCallback=PrefixCallback
    frozen.SafeTrainer=PrefixTrainer
    frozen.QwenCollator=PrefixCollator
    sys.argv=[sys.argv[0],'--config',str(args.config),'--run-dir',str(args.run_dir)]
    frozen.main()
    if int(os.environ.get('RANK',0))==0:
        terminal=json.loads((args.run_dir/'status.json').read_text())
        if terminal['status']!='interrupted' or terminal['step']!=100:raise ValueError('Expected original interrupted status at planned100')
        atomic_json(args.run_dir/'prefix-status.json',dict(status='completed_bounded_prefix',actual_step=100,
            original_total_steps=TOTAL,original_training_status=terminal['status'],completed_epoch=False,
            trajectory_validation=str(args.run_dir/'trajectory-validation.json')))


if __name__=='__main__':main()
