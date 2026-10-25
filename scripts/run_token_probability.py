#!/usr/bin/env python3
"""Three models, each one native-observation generate plus three fixed teacher forwards."""
from __future__ import annotations
import time
ENTRY_STARTED_MONOTONIC=time.monotonic()
ENTRY_STARTED_UNIX=time.time()
import argparse
import json
import fcntl
import os
import subprocess
from pathlib import Path
import signal
import sys

class StartupBudgetExpired(RuntimeError):
    pass


def startup_receipt(args,error):
    """A failed dependency import cannot rely on any project/scientific imports."""
    directory=args.run_dir
    if directory is None or directory.exists() or directory.is_symlink():return
    directory.mkdir(parents=True,exist_ok=False)
    elapsed=time.monotonic()-ENTRY_STARTED_MONOTONIC
    status='budget_stopped' if isinstance(error,StartupBudgetExpired) else 'failed'
    state=dict(status=status,stage='importing_dependencies',error=f'{type(error).__name__}: {error}',
        started_unix=ENTRY_STARTED_UNIX,elapsed_seconds=elapsed,max_wall_seconds=args.max_wall_seconds,
        actual_generate_calls=0,actual_teacher_forward_calls=0,automatic_training=False,
        cleanup_elapsed_seconds=0,owned_processes_remaining=[],commands=[],import_failure=True)
    def save(path,value):
        temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2)+'\n');os.replace(temporary,path)
    monitor=directory/'preflight-monitor';monitor.mkdir()
    observation=dict(status='failed',phase='importing_dependencies',campaign_status=status,terminal=True,
        continuous_observation=False,error='Dependency import failed before sampler startup; no resource observation obtained',
        elapsed_seconds=elapsed,gpus=[],compute_apps=[],disk_free_bytes=None,training_progress_claim=False)
    (monitor/'metrics.jsonl').write_text(json.dumps(observation)+'\n');save(monitor/'status.json',observation)
    (directory/'REPORT.md').write_text('# Frozen-trajectory token probability probe\n\n'
        f'Status: {status}; dependency initialization failed.\n'
        'Planned maximum: 3 generations and 9 teacher forwards.\n'
        'Observed generate calls: 0; Observed teacher forward calls: 0.\n'
        'Archived generation matrices: 0; Archived teacher matrices: 0.\n')
    state['elapsed_seconds']=time.monotonic()-ENTRY_STARTED_MONOTONIC;save(directory/'campaign.json',state)


# Arm before numpy/torch/transformers or inherited campaign modules can initialize.
if __name__=='__main__':
    early_parser=argparse.ArgumentParser(add_help=False)
    early_parser.add_argument('--run-dir',type=Path)
    early_parser.add_argument('--max-wall-seconds',type=float,default=1200)
    early_args,_=early_parser.parse_known_args()
    if not 0<early_args.max_wall_seconds<=1200:early_parser.error('Total budget must be in(0,1200]seconds')
    def startup_expired(*_):raise StartupBudgetExpired('Entry dependency/argument budget exhausted')
    signal.signal(signal.SIGALRM,startup_expired)
    early_deadline=ENTRY_STARTED_MONOTONIC+early_args.max_wall_seconds
    signal.setitimer(signal.ITIMER_REAL,max(.000001,early_deadline-min(20.,early_args.max_wall_seconds*.2)-time.monotonic()))

sys.path.insert(0,str(Path(__file__).resolve().parent))
try:
    from run_qwen_campaign import Campaign,BudgetExpired,file_hash,read_json,campaign_environment
    from run_repetition_onset import verify_model,verify_stats
    from asr_lab.token_probability import ORDER,CAP,estimated_bytes,check_generation_rules,input_gate
    from asr_lab.probe_supervision import resource_snapshot,require_gpu_idle,write_observation,stop_owned
    from asr_lab.training_utils import atomic_json
    from asr_lab.prepare_data import tree_size,check_space
except BaseException as dependency_error:
    if __name__=='__main__':
        signal.setitimer(signal.ITIMER_REAL,max(.000001,early_deadline-time.monotonic()))
        try:startup_receipt(early_args,dependency_error)
        finally:signal.setitimer(signal.ITIMER_REAL,0)
    raise

REPO=Path(__file__).resolve().parents[1]


def validate_worker(run,stage,reference,remaining,physical_gpu=1):
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
    input_gate(read_json(run/'inputs.json'),metadata,reference,physical_gpu=physical_gpu)
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
        self.args=args;self.root=args.project_root.resolve();self.directory=args.campaign_dir.resolve()
        self.owns_directory=False;self.locks=[];self.owned=[];self.monitor=None;self.finalizing=False
        self.stage_deadline=None;self.next_space_check=0.;self.initial_project_bytes=None
        self.started_monotonic=getattr(args,'entry_started_monotonic',time.monotonic())
        self.started=getattr(args,'entry_started_unix',time.time())
        self.deadline=self.started_monotonic+args.max_wall_seconds
        self.cleanup_reserve=min(20.,args.max_wall_seconds*.2)
        self.work_deadline=self.deadline-self.cleanup_reserve
        self.results=[];self._matrix_cache={}
        self.state=dict(status='running',stage='initializing',started_unix=self.started,max_wall_seconds=args.max_wall_seconds,
            cleanup_reserve_seconds=self.cleanup_reserve,failure_receipt_reserve_seconds=min(1.,self.cleanup_reserve*.2),results=self.results,commands=[],kind='token_probability',
            model_order=list(ORDER),expected_generate_calls=3,expected_teacher_forward_calls=9,automatic_training=False,
            physical_gpu=getattr(args,'physical_gpu',1),resource_identity_exception=dict(frozen_reference_physical_gpu='1',selected_physical_gpu=str(getattr(args,'physical_gpu',1)),only_allowed_runtime_difference='physical_gpu'),planned_teacher_matrix_count=9,planned_generation_matrix_count=3)
        # Reject even an empty existing directory before opening a lock or writing a file.
        if args.campaign_dir.exists() or args.campaign_dir.is_symlink() or self.directory.exists():raise FileExistsError('Choose a fresh --run-dir; existing paths are preserved')
        self.directory.mkdir(parents=True,exist_ok=False);self.owns_directory=True
        self.write_state()
        for path in (self.directory/'campaign.lock',self.root/'campaign.lock'):
            handle=path.open('a+')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BaseException:handle.close();raise
            self.locks.append(handle)
        self.state['stage']='initial_resource_observation';self.write_state()
        self.observe()
        self.monitor=subprocess.Popen([sys.executable,'-m','asr_lab.probe_supervision','--run-dir',str(self.directory),
            '--project-root',str(self.root),'--deadline',str(self.work_deadline),'--started',str(self.started_monotonic),
            '--first-observation',str(time.monotonic())],cwd=REPO,env=dict(os.environ,PYTHONPATH=str(REPO/'src')),
            stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        self.owned.append(self.monitor);self.state['preflight_monitor_pid']=self.monitor.pid
        self.state['stage']='initial_project_tree_size';self.write_state()
        self.initial_project_bytes=tree_size(self.root)
        self.remaining()
        self.env=campaign_environment(self.root)
        self.env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        self.state['stage']='validating';self.write_state()

    def arm(self,deadline=None):
        target=self.work_deadline if deadline is None else deadline
        if self.stage_deadline is not None and not self.finalizing:target=min(target,self.stage_deadline)
        left=target-time.monotonic()
        if left<=0:raise BudgetExpired('Probe absolute/stage wall-time budget exhausted')
        signal.setitimer(signal.ITIMER_REAL,left)

    def remaining(self):
        target=min(getattr(self,'work_deadline',self.deadline),self.stage_deadline or self.deadline)
        left=target-time.monotonic()
        if left<=0:raise BudgetExpired('Probability absolute/stage wall-time budget exhausted')
        if self.monitor is not None and self.monitor.poll() is not None:
            raise RuntimeError(f'Preflight monitor exited unexpectedly: {self.monitor.returncode}')
        now=time.monotonic()
        if self.initial_project_bytes is not None and now>=self.next_space_check:
            if max(tree_size(self.directory),tree_size(self.root)-self.initial_project_bytes)>CAP:
                raise RuntimeError('Probability incremental storage exceeded2GiB')
            check_space(self.root,500,300,0);self.next_space_check=now+5
        if target-time.monotonic()<=0:raise BudgetExpired('Probability resource check crossed wall-time budget')
        return target-time.monotonic()

    def observe(self,terminal=False,campaign_status=None):
        prior_errors=self.state.get('cleanup_errors',[]) if terminal else []
        error='; '.join(prior_errors) if prior_errors else None
        if terminal and self.monitor is None:error=error or 'Preflight sampler never started'
        failure=None
        try:snapshot=resource_snapshot(self.root,self.deadline if terminal else self.work_deadline)
        except BaseException as problem:
            snapshot={};failure=problem;error=f'{type(problem).__name__}: {problem}'
        row=write_observation(self.directory,snapshot,started=self.started_monotonic,phase=self.state['stage'],
            campaign_status=('failed' if error and campaign_status=='complete' else (campaign_status or self.state['status'])),terminal=terminal,error=error)
        self.last_observation=row
        if isinstance(failure,(BudgetExpired,KeyboardInterrupt)):raise failure
        if error:raise RuntimeError('Preflight resource observation failed: '+error)
        return row

    def evidence(self):
        counts=dict(generate_calls=0,teacher_forward_calls=0,generation_matrices=0,teacher_matrices=0,
            present_generation_matrices=0,present_teacher_matrices=0,unknown_invocation_workers=[],validated_workers=[])
        for stage in ORDER:
            worker=self.directory/stage/'worker';metadata=worker/'metadata.json'
            launched=any(row.get('name')==stage and row.get('worker_started',True) for row in self.state.get('commands',[]))
            if metadata.exists():
                try:
                    item=read_json(metadata)
                    for field in ('generate_calls','teacher_forward_calls'):
                        value=item[field]
                        if type(value) is not int or value<0:raise ValueError('Invalid invocation count')
                    counts['generate_calls']+=item['generate_calls'];counts['teacher_forward_calls']+=item['teacher_forward_calls']
                except (OSError,ValueError,KeyError):counts['unknown_invocation_workers'].append(stage)
            elif launched:counts['unknown_invocation_workers'].append(stage)
            prior_archived=counts['generation_matrices']+counts['teacher_matrices']
            for role,folders in (('generation',[worker/'generation']),('teacher',[worker/'teacher'/name for name in ORDER])):
                for folder in folders:
                    payload=folder/'distributions.pt';receipt=folder/'distribution-manifest.json'
                    if payload.is_file():counts['present_'+role+'_matrices']+=1
                    if not payload.is_file() or not receipt.is_file():continue
                    try:
                        item=read_json(receipt);stat=payload.stat()
                        if item['bytes']!=stat.st_size:continue
                        key=(str(payload),stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns,item['sha256'])
                        # Finalization never starts a new large content hash. Previously checked files remain stat-anchored.
                        if key not in getattr(self,'_matrix_cache',{}):
                            if getattr(self,'finalizing',False):continue
                            actual=file_hash(payload)
                            if not hasattr(self,'_matrix_cache'):self._matrix_cache={}
                            self._matrix_cache[key]=actual==item['sha256']
                        if self._matrix_cache[key]:counts[role+'_matrices']+=1
                    except (OSError,ValueError,KeyError):continue
            if stage in self.state.get('validated_workers',[]) and counts['generation_matrices']+counts['teacher_matrices']-prior_archived==4:
                counts['validated_workers'].append(stage)
        return counts

    def write_state(self):
        self.report()
        self.state.update(updated_unix=time.time(),elapsed_seconds=time.monotonic()-self.started_monotonic)
        atomic_json(self.directory/'experiment_results.json',self.results)
        atomic_json(self.directory/'campaign.json',self.state)

    def report(self):
        evidence=self.evidence()
        self.state.update(observed_evidence=evidence,actual_generate_calls=evidence['generate_calls'] if not evidence['unknown_invocation_workers'] else None,
            actual_teacher_forward_calls=evidence['teacher_forward_calls'] if not evidence['unknown_invocation_workers'] else None)
        lines=['# Frozen-trajectory token probability probe','',f"Status: {self.state['status']}; stage: {self.state['stage']}.",'',
            'Planned maximum: 3 native-observation generations, 9 empty-cache teacher forwards, 3 generation matrices and 9 teacher matrices; 0 optimizer updates. No retries or additional samples.',
            f"Observed generate calls: {evidence['generate_calls']}; Observed teacher forward calls: {evidence['teacher_forward_calls']} (recorded invocation counters).",
            f"Workers with unknown invocation counts: {evidence['unknown_invocation_workers']}; these are not counted as zero.",
            f"Present generation matrix files: {evidence['present_generation_matrices']}; present teacher matrix files: {evidence['present_teacher_matrices']}.",
            f"Archived generation matrices: {evidence['generation_matrices']}; Archived teacher matrices: {evidence['teacher_matrices']} (file plus size/SHA receipt verified).",
            f"Fully validated workers: {evidence['validated_workers']}. Partial files and unchecked receipts do not establish a complete matrix.",
            'Frozen trajectory lengths: 27/37/512 IDs. Same-prefix probabilities describe model behavior; they do not identify training causes.']
        if self.state.get('planned_matrix_bytes') is not None:lines.append(f"Planned matrix bytes: {self.state['planned_matrix_bytes']}; this is an estimate, not observed storage.")
        if self.state.get('error'):lines+=['','Failure: '+self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def execute(self,name,command,run,gpus,projected_gib=0):
        self.remaining()
        for path,expected in self.state.get('source_sha256',{}).items():
            if file_hash(path)!=expected:raise ValueError(f'Campaign source identity changed: {path}')
            self.remaining()
        check_space(self.root,500,300,0);self.remaining()
        # This is deliberately after all potentially slow hashes and immediately before Popen.
        checked=require_gpu_idle(int(gpus),deadline=time.monotonic()+self.remaining(),expected_uuid=self.state.get('gpu_uuid'))
        self.state.update(gpu_preflight=checked,gpu_uuid=checked['uuid'])
        run=Path(run);run.mkdir(parents=True,exist_ok=False)
        record=dict(name=name,command=command,physical_gpus=gpus,run_dir=str(run),started_unix=time.time(),status='launch_pending',worker_started=False)
        self.state['commands'].append(record);self.write_state()
        process=monitor=None
        with (run/'stdout.log').open('w') as output,(run/'monitor.log').open('w') as monitor_output:
            try:
                self.remaining()
                checked=require_gpu_idle(int(gpus),deadline=time.monotonic()+self.remaining(),expected_uuid=self.state['gpu_uuid'])
                self.state['gpu_preflight']=checked
                process=subprocess.Popen(command,cwd=REPO,env=dict(self.env,CUDA_VISIBLE_DEVICES=gpus),stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
                self.owned.append(process);record.update(status='running',worker_started=True,pid=process.pid);self.write_state()
                monitor=subprocess.Popen([sys.executable,str(REPO/'scripts/monitor.py'),'--run-dir',str(run),'--project-root',str(self.root),
                    '--pid',str(process.pid),'--gpu',gpus,'--interval','10','--reserve-gib','500'],cwd=REPO,env=self.env,
                    stdout=monitor_output,stderr=subprocess.STDOUT,start_new_session=True)
                self.owned.append(monitor)
                while process.poll() is None:
                    if monitor.poll() is not None:raise RuntimeError('Per-worker resource monitor exited unexpectedly')
                    # Reserve teardown time inside both the stage and total deadlines.
                    left=self.remaining()
                    if left<=2:raise BudgetExpired('Reserved owned-worker cleanup time reached')
                    time.sleep(min(.25,left-2))
                record.update(exit_code=process.returncode,status='complete' if process.returncode==0 else 'failed')
                if monitor.poll() is not None and monitor.returncode!=0:raise RuntimeError('Per-worker resource monitor failed')
            finally:
                cleanup_started=time.monotonic()
                for child in (process,monitor):
                    if child is not None:
                        stop_owned(child,min(self.deadline,self.stage_deadline or self.deadline))
                        if child in self.owned:self.owned.remove(child)
                record.update(ended_unix=time.time(),exit_code=process.returncode if process is not None else None,cleanup_elapsed_seconds=time.monotonic()-cleanup_started)
                if record['status']!='complete':record['status']='failed'
                atomic_json(run/'stage.json',record);self.write_state()
        if record['exit_code']!=0:raise RuntimeError(f'{name} worker failed; see {run/"stdout.log"}')
        return record

    def finalize(self):
        self.finalizing=True;self.stage_deadline=None
        completion_pending=self.state['status']=='models_complete_pending_cleanup'
        cleanup_started=time.monotonic();errors=[]
        for child in list(self.owned):
            try:
                if child is self.monitor and child.poll() is not None:errors.append(f'Preflight monitor exited: {child.returncode}')
                stop_owned(child,self.deadline)
                self.owned.remove(child)
                if child is self.monitor and child.returncode not in (0,-signal.SIGTERM):
                    errors.append(f'Preflight monitor required forced/failed exit: {child.returncode}')
            except BaseException as error:errors.append(f'{type(error).__name__}: {error}')
        for handle in reversed(self.locks):handle.close()
        self.locks=[]
        self.state.update(cleanup_elapsed_seconds=time.monotonic()-cleanup_started,owned_processes_remaining=[child.pid for child in self.owned])
        if errors:
            self.state.update(status='failed' if self.state['status']=='complete' else self.state['status'],cleanup_errors=errors)
        try:self.observe(terminal=True,campaign_status='complete' if completion_pending and not errors else self.state['status'])
        except BaseException as error:
            errors.append(str(error));self.state.update(status='failed' if self.state['status']=='complete' else self.state['status'],monitor_error=str(error))
        self.state['cleanup_elapsed_seconds']=time.monotonic()-cleanup_started
        self.state['completed_unix']=time.time()
        if self.deadline-time.monotonic()<=0:raise BudgetExpired('Finalization exceeded absolute probe deadline')
        if completion_pending and not errors:self.state['status']='complete'
        self.write_state()
        if errors:raise RuntimeError('Probe cleanup/monitor failure: '+'; '.join(errors))
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
        for path in (Path(__file__),REPO/'src/asr_lab/token_probability.py',REPO/'src/asr_lab/probe_supervision.py',REPO/'scripts/run_repetition_onset.py',REPO/'scripts/run_qwen_campaign.py'):
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
            self.stage_deadline=min(getattr(self,'work_deadline',self.deadline),time.monotonic()+240);self.state['stage']=stage
            if hasattr(self.args,'entry_started_monotonic'):self.arm()
            self.write_state()
            try:
                outer=self.directory/stage;outer.mkdir();model=self.reference['models'][stage]
                verification=verify_model(stage,model,self.remaining,outer/'model-verification.json')
                files=dict(self.shared)
                for name,item in verification['files'].items():
                    if not name.endswith('.safetensors'):files[str(Path(model['path'])/name)]=item['actual_sha256']
                identity=dict(self.reference['shared_identity'],files=files,stage=stage,checkpoint=model['path'],model_files=verification['files'],
                    trajectory_reference=str(self.args.trajectory_reference.resolve()),vocab_size=self.vocab,project_root=str(self.root),run_root=str(self.directory),initial_project_bytes=self.initial_project_bytes,physical_gpu=getattr(self.args,'physical_gpu',1),gpu_uuid=self.state.get('gpu_uuid'))
                identity_path=outer/'identity.json';atomic_json(identity_path,identity)
                self.state['source_sha256']=dict(files,**{str(identity_path):file_hash(identity_path)})
                verify_stats(model,verification)
                selected=getattr(self.args,'physical_gpu',1)
                checked=require_gpu_idle(selected,deadline=time.monotonic()+self.remaining(),expected_uuid=self.state.get('gpu_uuid'))
                self.state.update(gpu_preflight=checked,gpu_uuid=checked['uuid'])
                identity['gpu_uuid']=checked['uuid'];atomic_json(identity_path,identity)
                self.state['source_sha256'][str(identity_path)]=file_hash(identity_path);self.write_state()
                run=outer/'worker';command=[sys.executable,'-m','asr_lab.token_probability','--identity',str(identity_path),'--output',str(run)]
                self.execute(stage,command,run,str(selected));self.remaining();verify_stats(model,verification)
                verification.update(status='passed_with_post_worker_stat_match',post_worker_stats={name:__import__('run_repetition_onset').file_stat(Path(model['path'])/name) for name in verification['files']})
                atomic_json(outer/'model-verification.json',verification)
                validate_worker(run,stage,self.reference,self.remaining,physical_gpu=selected)
                self.state.setdefault('validated_workers',[]).append(stage)
                self.write_state();self.remaining()
            finally:
                self.stage_deadline=None
                if hasattr(self.args,'entry_started_monotonic'):self.arm()
        comparison=compare_saved(self.directory,self.remaining)
        atomic_json(self.directory/'comparisons.json',comparison)
        self.remaining();self.state.update(status='models_complete_pending_cleanup',stage='three_models_nine_teacher_forwards_complete',actual_generate_calls=3,actual_teacher_forward_calls=9,completed_unix=time.time())
        self.write_state()


def run_campaign(args,started=None,started_unix=None):
    args.entry_started_monotonic=time.monotonic() if started is None else started
    args.entry_started_unix=time.time() if started_unix is None else started_unix
    args.campaign_dir=args.run_dir
    deadline=args.entry_started_monotonic+args.max_wall_seconds
    work_deadline=deadline-min(20.,args.max_wall_seconds*.2)
    previous=signal.getsignal(signal.SIGALRM);previous_timer=signal.getitimer(signal.ITIMER_REAL);timer_saved=time.monotonic()
    def expired(*_):raise BudgetExpired('Probability absolute/stage wall-time budget exhausted')
    signal.signal(signal.SIGALRM,expired)
    campaign=ProbabilityCampaign.__new__(ProbabilityCampaign)
    error=None
    try:
        left=work_deadline-time.monotonic()
        if left<=0:raise BudgetExpired('Entry/argument parsing exhausted probe budget')
        signal.setitimer(signal.ITIMER_REAL,left)
        campaign.__init__(args);campaign.run()
        if campaign.state['status']=='complete':
            campaign.state['status']='models_complete_pending_cleanup';campaign.write_state()
    except BaseException as problem:
        error=problem
        if getattr(campaign,'owns_directory',False):
            campaign.state.update(status='budget_stopped' if isinstance(problem,BudgetExpired) else ('interrupted' if isinstance(problem,KeyboardInterrupt) else 'failed'),
                error=f'{type(problem).__name__}: {problem}',automatic_training=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if getattr(campaign,'owns_directory',False):
            try:
                left=deadline-time.monotonic()
                if left<=0:raise BudgetExpired('Absolute budget exhausted before finalization')
                signal.setitimer(signal.ITIMER_REAL,max(.000001,left-min(1.,campaign.cleanup_reserve*.2)))
                campaign.finalize()
            except BaseException as problem:
                if error is None:error=problem
                terminal_error=error if error is not None else problem
                campaign.state.update(status='budget_stopped' if isinstance(terminal_error,BudgetExpired) else ('interrupted' if isinstance(terminal_error,KeyboardInterrupt) else 'failed'),
                    error=f'{type(terminal_error).__name__}: {terminal_error}',finalization_error=f'{type(problem).__name__}: {problem}')
                # This recovery window is reserved inside the original absolute cap.
                signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
                try:
                    write_observation(campaign.directory,getattr(campaign,'last_observation',{}),started=campaign.started_monotonic,phase=campaign.state['stage'],
                        campaign_status=campaign.state['status'],terminal=True,error=campaign.state['error'])
                    campaign.write_state()
                except BaseException:pass
            finally:signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,previous)
        previous_left=previous_timer[0]-(time.monotonic()-timer_saved)
        if previous_left>0:signal.setitimer(signal.ITIMER_REAL,previous_left,previous_timer[1])
    if error is not None:raise error
    return campaign.state


def main():
    started=ENTRY_STARTED_MONOTONIC if __name__=='__main__' else time.monotonic()
    started_unix=ENTRY_STARTED_UNIX if __name__=='__main__' else time.time()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--trajectory-reference',type=Path,default=REPO/'reports/token-probability/inputs/trajectory-reference.json')
    parser.add_argument('--upstream-source',type=Path,default=REPO/'reports/token-probability/inputs/upstream-probability-source.json')
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--max-wall-seconds',type=float,default=1200)
    parser.add_argument('--physical-gpu',type=int,choices=(0,1),default=1)
    args=parser.parse_args()
    if not 0<args.max_wall_seconds<=1200:parser.error('Total budget must be in(0,1200]seconds')
    def stop(*_):raise KeyboardInterrupt('Token probability probe interrupted')
    previous=signal.signal(signal.SIGTERM,stop)
    try:run_campaign(args,started,started_unix)
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':main()
