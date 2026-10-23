#!/usr/bin/env python3
"""Four bounded singleton comparisons, reusing the frozen inference worker with a CPU input guard."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_qwen_campaign import BudgetExpired,file_hash,read_json
from run_truncation_probe import ProbeCampaign,validate_worker
from asr_lab import qwen_truncation_probe as frozen
from asr_lab.repetition_metrics import repetition_observation,summarize_onset
from asr_lab.training_utils import atomic_json

REPO=Path(__file__).resolve().parents[1]
ORDER=('base','step025','step050','step100')
STEPS=(0,25,50,100)


def file_stat(path):
    path=Path(path);stat=path.stat();link=path.lstat();is_link=path.is_symlink()
    return dict(logical_path=str(path.absolute()),is_symlink=is_link,symlink_target=os.readlink(path) if is_link else None,
        symlink_lstat=dict(dev=link.st_dev,inode=link.st_ino,size=link.st_size,mtime_ns=link.st_mtime_ns,ctime_ns=link.st_ctime_ns) if is_link else None,
        resolved_path=str(path.resolve()),dev=stat.st_dev,inode=stat.st_ino,size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,ctime_ns=stat.st_ctime_ns)


def bounded_sha256(path,remaining):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            remaining();block=stream.read(1024**2);remaining()
            if not block:break
            digest.update(block)
    return digest.hexdigest()


def verify_model(stage,model,remaining,receipt_path):
    records={};receipt=dict(stage=stage,status='hashing',files=records,
        policy='One full content SHA per current model before load; stable resolved path/dev/inode/size/mtime/ctime through worker. No final content rehash claim.')
    if model['step']!=STEPS[ORDER.index(stage)] or model['revision']!='7278e1e70fe206f11671096ffdd38061171dd6e5':
        raise ValueError('Model stage/revision differs')
    path=Path(model['path'])
    if stage=='base':
        if path.name!=model['revision']:raise ValueError('Base local pinned snapshot path differs')
    else:
        marker_path=path/'prefix-complete.json'
        if file_hash(marker_path)!=model['historical_anchor']['prefix_complete']['sha256']:
            raise ValueError('Historical prefix-complete marker changed')
        marker=read_json(marker_path)
        if marker['status']!='complete' or marker['step']!=model['step']:raise ValueError('Historical checkpoint marker differs')
    if not any(name.endswith('.safetensors') for name in model['files']):raise ValueError('Missing model shard inventory')
    atomic_json(receipt_path,receipt)
    try:
        for name,expected in model['files'].items():
            remaining()
            if Path(name).name!=name:raise ValueError('Model file inventory must use basenames')
            target=path/name;before=file_stat(target)
            if before['size']!=expected['bytes']:raise ValueError('Model file size changed')
            if stage!='base' and marker['files'][name]['sha256']!=expected['sha256']:
                raise ValueError('Model SHA does not match historical marker anchor')
            if expected['sha256'] is None and not(stage=='base' and name.endswith('.safetensors')):
                raise ValueError('Missing expected SHA outside base first-capture shards')
            receipt['current_file']=name;atomic_json(receipt_path,receipt)
            hash_started=time.monotonic();hash_started_unix=time.time()
            actual=bounded_sha256(target,remaining)
            hash_elapsed=time.monotonic()-hash_started
            after=file_stat(target)
            if before!=after:raise ValueError('Model file stat drift during content hash')
            records[name]=dict(expected_sha256=expected['sha256'],actual_sha256=actual,
                expected_origin=expected['expected_origin'],historical_expected_sha256=expected['historical_expected_sha256'],
                captured_in_this_stage=True,base_first_capture=expected['sha256'] is None,before=before,after_hash=after,
                hash_started_unix=hash_started_unix,hash_elapsed_seconds=hash_elapsed)
            if expected['sha256'] is not None and actual!=expected['sha256']:raise ValueError('Model content SHA differs')
            atomic_json(receipt_path,receipt)
        receipt.update(status='passed',current_file=None)
        atomic_json(receipt_path,receipt)
        return receipt
    except BaseException as error:
        receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
        atomic_json(receipt_path,receipt);raise


def verify_stats(model,verification):
    for name,record in verification['files'].items():
        if file_stat(Path(model['path'])/name)!=record['before']:raise ValueError('Model stat identity drift after content hash')


def common_metadata(metadata):
    fields=('runtime','generation_kwargs','audio','model_class','processor_class','inference_mode','model_training',
            'optimizer_updates','backward_calls','train_called','logits_collected','seed_reset')
    if any(key not in metadata for key in fields):raise ValueError('Missing shared runtime evidence')
    return {key:metadata[key] for key in fields}


def compare_common(inputs,metadata,expected_inputs,expected_metadata):
    if inputs!=expected_inputs:raise ValueError('Frozen singleton input drift before generation')
    if common_metadata(metadata)!=common_metadata(expected_metadata):raise ValueError('Shared runtime/kwargs identity drift')


def effective_rules(directory):
    records=read_json(directory/'effective-generation.json')
    if len(records)!=1:raise ValueError('Missing single effective generation capture')
    return {key:records[0][key] for key in ('effective','special_tokens','logits_processor_types')}


@contextmanager
def guarded_generator(identity,output):
    original=frozen.generate_and_persist_raw;calls=0
    receipt=dict(status='not_called',actual_generate_calls=0,wrapper_source_sha256=file_hash(__file__))
    def guarded(*args,**kwargs):
        nonlocal calls
        try:
            if calls:raise ValueError('Only one generation call per model authorized')
            inputs=read_json(output/'inputs.json');metadata=read_json(output/'metadata.json')
            for directory in identity['onset']['compare_dirs']:
                expected=Path(directory)
                compare_common(inputs,metadata,read_json(expected/'inputs.json'),read_json(expected/'metadata.json'))
            calls+=1;receipt.update(status='input_gate_passed',actual_generate_calls=calls)
            atomic_json(output/'input-guard.json',receipt)
            return original(*args,**kwargs)
        except BaseException as error:
            receipt.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:atomic_json(output/'input-guard.json',receipt)
    frozen.generate_and_persist_raw=guarded
    try:yield receipt
    finally:
        frozen.generate_and_persist_raw=original
        receipt['actual_generate_calls']=calls
        atomic_json(output/'input-guard.json',receipt)


def child(args):
    identity=read_json(args.identity)
    saved_argv=sys.argv
    try:
        with guarded_generator(identity,args.output) as receipt:
            sys.argv=[sys.argv[0],'--identity',str(args.identity),'--stage','target-single','--output',str(args.output)]
            frozen.main()
        if receipt['actual_generate_calls']!=1:raise ValueError('Missing single authorized generation call')
    finally:sys.argv=saved_argv


def stage_gate(run,comparison_dirs):
    inputs=read_json(run/'inputs.json');metadata=read_json(run/'metadata.json')
    for expected in comparison_dirs:
        compare_common(inputs,metadata,read_json(expected/'inputs.json'),read_json(expected/'metadata.json'))
        if effective_rules(run)!=effective_rules(expected):raise ValueError('Effective generation/special-token/processor drift')
    guard=read_json(run/'input-guard.json')
    if guard['actual_generate_calls']!=1 or guard['status']!='input_gate_passed':raise ValueError('Missing pre-generation input gate')


class OnsetCampaign(ProbeCampaign):
    def __init__(self,args):
        super().__init__(args)
        self.state.update(kind='repetition_onset',preset_stages=list(ORDER),expected_generate_calls=4,expected_consumed_examples=4,
            automatic_training=False,observations=[])
        self.write_state()
    def report(self):
        lines=['# Four-model singleton repetition onset probe','',f"Status: {self.state['status']}; stage: {self.state['stage']}.",'',
            'base → step025 → step050 → step100; four fresh singleton calls at most, zero training; source worker unchanged.',
            'Strong repetition and512/no-EOS budget exhaustion are separate observations. No retries or follow-up queue.',
            'Each current model is content-hashed once within its240-second stage, then protected by stat checks; base shard capture has no historical file-SHA anchor.']
        for row in self.state.get('observations',[]):
            lines.append(f"- {row['stage']}: {row['operational_category']}; strong={row['strong_periodic_repetition']}; loop={row['budget_loop']}.")
        if self.state.get('error'):lines+=['','Failure: '+self.state['error']]
        lines+=['','Only saved sampled points are observed; exact first update, monotonicity and training cause are not established.']
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def prepare(self):
        reference=read_json(self.args.model_reference)
        if reference['stage_order']!=list(ORDER) or reference['target_id']!=frozen.TARGET or reference['generation_kwargs']!=frozen.GENERATION_KWARGS:
            raise ValueError('Frozen four-stage/singleton/generation contract differs')
        if (reference['expected_total_generate_calls'],reference['expected_total_consumed_examples'],reference['expected_optimizer_updates'])!=(4,4,0):
            raise ValueError('Only4 calls/4 consumptions/0 updates permitted')
        shared=dict(reference['shared_identity']['files'])
        def add(path,expected=None):
            actual=file_hash(path)
            if expected is not None and actual!=expected:raise ValueError(f'Shared frozen file changed:{path}')
            shared[str(Path(path).resolve())]=actual
        add(self.args.model_reference)
        prior_identity=reference['anchors']['prior_identity'];add(prior_identity['path'],prior_identity['sha256'])
        for record in reference['prior_singleton']['files'].values():add(record['path'],record['sha256'])
        for path in (Path(__file__),REPO/'src/asr_lab/repetition_metrics.py'):add(path)
        for path,expected in list(shared.items()):self.remaining();add(path,expected)
        prior=Path(reference['prior_singleton']['worker_dir'])
        if read_json(prior/'inputs.json')!=reference['prior_singleton']['inputs']:raise ValueError('Prior singleton input anchor differs')
        if reference['target']['manifest_row']['sha256']!=reference['target']['pcm_sha256']:raise ValueError('PCM identity semantics differ')
        if file_hash(reference['target']['manifest_row']['audio_filepath'])!=reference['target']['file_sha256']:raise ValueError('Target FLAC changed')
        self.shared=shared;self.reference=reference;self.prior=prior
        self.state['source_sha256']=shared
        self.write_state()
    def run(self):
        self.remaining();self.prepare()
        for stage in ORDER:
            self.stage_deadline=min(self.deadline,time.monotonic()+240)
            self.state['stage']=stage;self.write_state()
            try:
                outer=self.directory/stage;outer.mkdir()
                model=self.reference['models'][stage]
                verification=verify_model(stage,model,self.remaining,outer/'model-verification.json')
                identity=dict(self.reference['shared_identity'])
                files=dict(self.shared)
                for name,record in verification['files'].items():
                    if not name.endswith('.safetensors'):files[str(Path(model['path'])/name)]=record['actual_sha256']
                compare_dirs=[self.prior]
                if stage!='base':compare_dirs.append(self.directory/'base/target-single')
                for directory in compare_dirs:
                    for name in ('inputs.json','metadata.json','effective-generation.json'):
                        files[str(directory/name)]=file_hash(directory/name)
                identity.update(checkpoint=model['path'],files=files,model_files=verification['files'],
                    onset=dict(stage=stage,compare_dirs=[str(path) for path in compare_dirs],model_reference_sha256=file_hash(self.args.model_reference)))
                identity_path=outer/'identity.json';atomic_json(identity_path,identity)
                self.state['source_sha256']=dict(files,**{str(identity_path):file_hash(identity_path)})
                verify_stats(model,verification)
                self.state['gpu_preflight']=frozen.require_gpu1_idle();self.write_state()
                run=outer/'target-single'
                command=[sys.executable,str(Path(__file__).resolve()),'--worker','--identity',str(identity_path),'--output',str(run)]
                self.execute(stage,command,run,'1')
                self.remaining();verify_stats(model,verification)
                verification['status']='passed_with_post_worker_stat_match';verification['post_worker_stats']={name:file_stat(Path(model['path'])/name) for name in verification['files']}
                atomic_json(outer/'model-verification.json',verification)
                decoded=validate_worker(outer,'target-single');stage_gate(run,compare_dirs)
                target=decoded[0]
                eos=read_json(run/'result.json')['effective_eos_ids']
                observation=repetition_observation(target['token_ids'],eos,target['parser_input'])
                observation.update(stage=stage,step=model['step'],parser_input=target['parser_input'],parser_output=target['parser_output'],
                    counts=target['counts'],repetition_fix_observation=target['repetition_fix_observation'])
                atomic_json(outer/'observation.json',observation)
                self.state['observations'].append(observation);self.write_state()
            finally:self.stage_deadline=None
        summary=summarize_onset(self.state['observations'])
        atomic_json(self.directory/'onset-summary.json',summary)
        self.remaining()
        self.state.update(status='complete',stage='four_preset_singletons_complete',actual_generate_calls=4,actual_consumed_examples=4,
            onset=summary,completed_unix=time.time())
        self.write_state()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--identity',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--run-dir',type=Path)
    parser.add_argument('--model-reference',type=Path,default=REPO/'reports/repetition-onset/inputs/model-reference.json')
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--max-wall-seconds',type=float,default=1200)
    args=parser.parse_args()
    if args.worker:
        if not args.identity or not args.output:parser.error('Worker requires identity and output')
        child(args);return
    if not args.run_dir or not 0<args.max_wall_seconds<=1200:parser.error('Fresh run-dir and wall budget in(0,1200]required')
    args.campaign_dir=args.run_dir
    def stop(*_):raise KeyboardInterrupt('Onset probe interrupted')
    signal.signal(signal.SIGTERM,stop)
    campaign=OnsetCampaign(args)
    try:campaign.run()
    except BaseException as error:
        campaign.state.update(status='budget_stopped' if isinstance(error,BudgetExpired) else ('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'),
            error=f'{type(error).__name__}: {error}',automatic_training=False)
        campaign.write_state();raise
    finally:campaign.lock.close();campaign.global_lock.close()


if __name__=='__main__':main()
