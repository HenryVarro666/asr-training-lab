#!/usr/bin/env python3
"""Three models, each one native-observation generate plus three fixed teacher forwards."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_qwen_campaign import Campaign,BudgetExpired,file_hash,read_json
from run_repetition_onset import verify_model,verify_stats
from asr_lab.token_probability import ORDER,CAP,estimated_bytes,check_generation_rules,input_gate
from asr_lab.qwen_truncation_probe import require_gpu1_idle
from asr_lab.training_utils import atomic_json
from asr_lab.prepare_data import tree_size,check_space

REPO=Path(__file__).resolve().parents[1]


def validate_worker(run,stage,reference,remaining):
    remaining()
    result=read_json(run/'result.json');status=read_json(run/'status.json');metadata=read_json(run/'metadata.json')
    if result['status']!='complete' or status['status']!='complete' or result['stage']!=stage or (result['generate_calls'],result['teacher_forward_calls'],result['optimizer_updates'])!=(1,3,0):
        raise ValueError('Incomplete1 generate/3 teacher forward worker')
    if (metadata['generate_calls'],metadata['teacher_forward_calls'],metadata['logits_collected'])!=(1,3,True):raise ValueError('Incorrect invocation evidence')
    required={'inputs.pt','inputs.json','gpu-preflight.json','metadata.json','generation/raw-tokens.json','generation/raw-tokens.pt',
        'generation/effective-generation.json','generation/distributions.pt','generation/distribution-manifest.json','generation/derived.json','generation/decoded.json','own-trajectory-comparison.json','prefix-forks.json'}
    for trajectory in ORDER:
        required.update('teacher/'+trajectory+'/'+name for name in ('inputs.pt','inputs.json','distributions.pt','distribution-manifest.json','forward-receipt.json','derived.json'))
    if not required<=set(result['artifacts']):raise ValueError('Missing mandatory probability evidence')
    for name,expected in result['artifacts'].items():
        remaining()
        path=run/name
        if path.stat().st_size!=expected['bytes'] or file_hash(path)!=expected['sha256']:raise ValueError('Worker probability artifact identity changed')
        remaining()
    input_gate(read_json(run/'inputs.json'),metadata,reference)
    capture=read_json(run/'generation/effective-generation.json')
    if len(capture)!=1:raise ValueError('Expected one generation capture')
    check_generation_rules(capture[0],reference['effective_generation_rules'])
    if read_json(run/'generation/raw-tokens.json')['continuation_token_ids'][0]!=reference['trajectories'][stage]['token_ids']:
        raise ValueError('instrumented_trajectory_drift')
    for trajectory in ORDER:
        receipt=read_json(run/'teacher'/trajectory/'forward-receipt.json')
        if receipt['status']!='complete' or receipt['explicit_teacher_forward']!=1 or receipt['past_cache_returned'] or receipt['rope_deltas_after_reset'] is not None:
            raise ValueError('Missing empty-cache teacher forcing evidence')
    remaining()
    return result


def compare_saved(directory,remaining):
    remaining()
    trajectories={stage:read_json(directory/stage/'worker/generation/raw-tokens.json')['continuation_token_ids'][0] for stage in ORDER}
    from asr_lab.token_probability import trajectory_forks
    forks=trajectory_forks(trajectories)
    comparisons={}
    for trajectory in ORDER:
        remaining()
        data={stage:read_json(directory/stage/'worker/teacher'/trajectory/'derived.json') for stage in ORDER}
        baseline=data['base'];rows=[]
        for j,left in enumerate(baseline['rows']):
            remaining()
            record=dict(j=j,prefix=left['prefix'],target_id=left['target_id'],models={})
            for stage in ORDER:
                right=data[stage]['rows'][j]
                if right['prefix']!=left['prefix'] or right['target_id']!=left['target_id']:raise ValueError('Cross-model conditional prefix mismatch')
                if [x['id'] for x in right['candidates']]!=[x['id'] for x in left['candidates']]:raise ValueError('Candidate union differs')
                record['models'][stage]=dict(target_logprob=right['target']['logprob'],target_rank=right['target']['rank'],
                    target_logprob_delta_vs_base=right['target']['logprob']-left['target']['logprob'],
                    eos_total_prob=right['eos']['total_prob'],eos_total_logprob_delta_vs_base=right['eos']['total_logprob']-left['eos']['total_logprob'],
                    candidate_deltas=[dict(id=a['id'],logprob_delta=b['logprob']-a['logprob'],rank=b['rank'],base_rank=a['rank']) for a,b in zip(left['candidates'],right['candidates'])])
            rows.append(record)
        comparisons[trajectory]=dict(rows=rows,per_model_nll={stage:dict(total=data[stage]['total_target_nll'],mean=data[stage]['mean_target_nll']) for stage in ORDER})
    remaining()
    return dict(free_trajectory_forks=forks,fixed_trajectory_comparisons=comparisons,
        interpretation='Compare a fixed trajectory, row and prefix across models. Natural/uppercase are both correct16-word transcripts; token forks are not automatically semantic errors. Full-forward/cached differences are observations, not equality gates or training causes.')


class ProbabilityCampaign(Campaign):
    def __init__(self,args):
        self.stage_deadline=None;self.next_space_check=0.;self.initial_project_bytes=tree_size(args.project_root.resolve())
        super().__init__(args)
        self.env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        self.state.update(kind='token_probability',model_order=list(ORDER),expected_generate_calls=3,expected_teacher_forward_calls=9,automatic_training=False)
        self.state.pop('selected_config',None);self.state.pop('main_lr_matrix_status',None)
        self.write_state()
    def remaining(self):
        left=super().remaining();now=time.monotonic()
        if self.stage_deadline is not None:
            left=min(left,self.stage_deadline-now)
            if left<=0:raise BudgetExpired('Probability model stage exceeded240seconds')
        if now>=self.next_space_check:
            if max(tree_size(self.directory),tree_size(self.root)-self.initial_project_bytes)>CAP:raise RuntimeError('Probability incremental storage exceeded2GiB')
            check_space(self.root,500,300,0);self.next_space_check=now+5
        return left
    def report(self):
        lines=['# Frozen-trajectory token probability probe','',f"Status: {self.state['status']}; stage: {self.state['stage']}.",'',
            'At most3 native-observation generations and9 explicit empty-cache teacher forwards; no training, retries or additional samples.',
            'Three fixed saved trajectories remain27/37/512 IDs; no re-tokenization. Native output_logits/output_scores are the only allowed observation differences.',
            'Full-vocabulary FP32 continuation rows are archived; no prompt logits or full probability matrices are saved.',
            'Same-prefix probabilities describe model behavior; they do not identify training causes.']
        if self.state.get('error'):lines+=['','Failure: '+self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def prepare(self):
        reference=read_json(self.args.trajectory_reference)
        if reference['model_order']!=list(ORDER) or reference['trajectory_order']!=list(ORDER) or reference['target_id']!='652-129742-0006':raise ValueError('Fixed3x3 matrix differs')
        if [len(reference['trajectories'][stage]['token_ids']) for stage in ORDER]!=[27,37,512]:raise ValueError('Fixed saved trajectories changed')
        shared=dict(reference['shared_identity']['files'])
        def add(path,expected=None):
            self.remaining();actual=file_hash(path)
            if expected is not None and actual!=expected:raise ValueError(f'Frozen probability source changed:{path}')
            shared[str(Path(path).resolve())]=actual
        add(self.args.trajectory_reference)
        upstream=read_json(self.args.upstream_source);add(self.args.upstream_source)
        for item in upstream['files']:add(item['path'],item['sha256'])
        for stage in ORDER:
            trajectory=reference['trajectories'][stage]
            for item in trajectory['source_files'].values():add(item['path'],item['sha256'])
            old=reference['models'][stage]['prior_model_verification'];add(old['path'],old['sha256'])
            verified=read_json(old['path'])
            for name,item in reference['models'][stage]['files'].items():
                if item['sha256'] is None or item['sha256']!=verified['files'][name]['actual_sha256']:raise ValueError('Missing prior model SHA anchor')
        for path in (Path(__file__),REPO/'src/asr_lab/token_probability.py',REPO/'scripts/run_repetition_onset.py',REPO/'scripts/run_qwen_campaign.py'):
            add(path)
        for path,expected in list(shared.items()):add(path,expected)
        self.reference=reference;self.shared=shared;self.vocab=upstream['config']['vocab_size']
        lengths=[27,37,512]
        planned=estimated_bytes(self.vocab,lengths,lengths)
        worst=estimated_bytes(self.vocab,lengths,[512]*3)
        if worst+64*1024**2>CAP:raise ValueError('Full-vocabulary matrix estimate exceeds2GiB')
        check_space(self.root,500,300,worst+64*1024**2)
        self.state.update(source_sha256=shared,planned_matrix_bytes=planned,worst_matrix_bytes_with_equal_role_storage=worst,vocab_size=self.vocab)
        self.write_state()
    def run(self):
        self.remaining();self.prepare()
        for stage in ORDER:
            self.stage_deadline=min(self.deadline,time.monotonic()+240);self.state['stage']=stage;self.write_state()
            try:
                outer=self.directory/stage;outer.mkdir();model=self.reference['models'][stage]
                verification=verify_model(stage,model,self.remaining,outer/'model-verification.json')
                files=dict(self.shared)
                for name,item in verification['files'].items():
                    if not name.endswith('.safetensors'):files[str(Path(model['path'])/name)]=item['actual_sha256']
                identity=dict(self.reference['shared_identity'],files=files,stage=stage,checkpoint=model['path'],model_files=verification['files'],
                    trajectory_reference=str(self.args.trajectory_reference.resolve()),vocab_size=self.vocab,project_root=str(self.root),run_root=str(self.directory),initial_project_bytes=self.initial_project_bytes)
                identity_path=outer/'identity.json';atomic_json(identity_path,identity)
                self.state['source_sha256']=dict(files,**{str(identity_path):file_hash(identity_path)})
                verify_stats(model,verification);self.state['gpu_preflight']=require_gpu1_idle();self.write_state()
                run=outer/'worker';command=[sys.executable,'-m','asr_lab.token_probability','--identity',str(identity_path),'--output',str(run)]
                self.execute(stage,command,run,'1');self.remaining();verify_stats(model,verification)
                verification.update(status='passed_with_post_worker_stat_match',post_worker_stats={name:__import__('run_repetition_onset').file_stat(Path(model['path'])/name) for name in verification['files']})
                atomic_json(outer/'model-verification.json',verification)
                validate_worker(run,stage,self.reference,self.remaining)
                self.remaining()
            finally:self.stage_deadline=None
        comparison=compare_saved(self.directory,self.remaining)
        atomic_json(self.directory/'comparisons.json',comparison)
        self.remaining();self.state.update(status='complete',stage='three_models_nine_teacher_forwards_complete',actual_generate_calls=3,actual_teacher_forward_calls=9,completed_unix=time.time())
        self.write_state()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--trajectory-reference',type=Path,default=REPO/'reports/token-probability/inputs/trajectory-reference.json')
    parser.add_argument('--upstream-source',type=Path,default=REPO/'reports/token-probability/inputs/upstream-probability-source.json')
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--max-wall-seconds',type=float,default=1200)
    args=parser.parse_args()
    if not 0<args.max_wall_seconds<=1200:parser.error('Total budget must be in(0,1200]seconds')
    args.campaign_dir=args.run_dir
    def stop(*_):raise KeyboardInterrupt('Token probability probe interrupted')
    signal.signal(signal.SIGTERM,stop)
    campaign=ProbabilityCampaign(args)
    try:campaign.run()
    except BaseException as error:
        campaign.state.update(status='budget_stopped' if isinstance(error,BudgetExpired) else ('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'),error=f'{type(error).__name__}: {error}',automatic_training=False)
        campaign.write_state();raise
    finally:campaign.lock.close();campaign.global_lock.close()


if __name__=='__main__':main()
