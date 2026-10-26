#!/usr/bin/env python3
"""One bounded, CPU-only saved-evidence command; Linux /proc resource observations."""
from __future__ import annotations
import time
ENTRY_STARTED=time.monotonic()
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys

GIB=1024**3
STAGES=('diagnostic','recompute','audit')
class BudgetExpired(RuntimeError):pass

def utc():return dt.datetime.now(dt.timezone.utc).isoformat()
def save(path,value):
    temporary=path.with_name(path.name+'.tmp');temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)
def cpu_environment():
    return dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
def validate(args):
    if args.stage not in STAGES or not 2<args.seconds<=(120 if args.stage=='diagnostic' else 600):raise ValueError('Stage cap must exceed cleanup2s and fit diagnostic120s/other600s')
    if not args.command or not re.fullmatch(r'python(?:\d+(?:\.\d+)?)?',Path(args.command[0]).name):raise ValueError('Specify one Python command after --')
    if (args.analysis_root/args.stage).exists() or (args.analysis_root/args.stage).is_symlink():raise FileExistsError('Stage directory must be new; no retries')
    prior=0.
    for index,name in enumerate(STAGES):
        folder=args.analysis_root/name
        if index>=STAGES.index(args.stage):
            if folder.exists():raise ValueError('Future/old stage exists; no reuse')
            continue
        record=json.loads((folder/'stage.json').read_text())
        elapsed=record['elapsed_seconds']
        if record['stage']!=name or record['status']!='complete' or record['exit_code']!=0 or not math.isfinite(elapsed) or elapsed<0:raise ValueError('Prior stage did not complete; stop')
        prior+=elapsed
    if prior+args.seconds>1500:raise ValueError('Cumulative1500s budget exceeded')
    return prior

def tree_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file() and not path.is_symlink())
def project_bytes(root,deadline):
    left=deadline-time.monotonic()
    if left<=0:raise BudgetExpired('Project check deadline exhausted')
    result=subprocess.run(['du','-sb','--',str(root)],capture_output=True,text=True,check=True,timeout=min(4.,left))
    return int(result.stdout.split()[0])
def processes(pgid):
    rows=[];ticks=os.sysconf('SC_CLK_TCK');pages=os.sysconf('SC_PAGE_SIZE')
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            values=path.read_text().rsplit(')',1)[1].split()
            if int(values[2])==pgid and values[0]!='Z':rows.append(dict(pid=int(path.parent.name),ppid=int(values[1]),rss_bytes=max(0,int(values[21]))*pages,cpu_seconds=(int(values[11])+int(values[12]))/ticks))
        except (OSError,ValueError,IndexError):continue
    return rows

def snapshot(args,process,baseline,initial_analysis,phase):
    rows=processes(process.pid) if process is not None else []
    analysis=tree_bytes(args.analysis_root);progress=args.analysis_root/args.stage/'progress.json'
    try:progress_value=json.loads(progress.read_text()) if progress.exists() else None
    except (OSError,ValueError):progress_value={'error':'Unreadable progress.json'}
    return dict(utc=utc(),phase=phase,pids=[row['pid'] for row in rows],rss_bytes=sum(row['rss_bytes'] for row in rows),
        cpu_seconds=sum(row['cpu_seconds'] for row in rows),analysis_bytes=analysis,disk_free_bytes=shutil.disk_usage(args.project_root).free,
        estimated_project_bytes=baseline+max(0,analysis-initial_analysis),progress=progress_value)
def check_resources(row):
    if row['rss_bytes']>8*GIB or row['analysis_bytes']>128*1024**2 or row['disk_free_bytes']<500*GIB or row['estimated_project_bytes']>300*GIB:raise RuntimeError('CPU analysis resource guard exceeded')
def wait_command(process,deadline,observe):
    next_sample=time.monotonic()+5
    while process.poll() is None:
        left=deadline-time.monotonic()
        if left<=0:raise BudgetExpired('CPU stage work deadline exhausted')
        if time.monotonic()>=next_sample:observe('running');next_sample+=5
        time.sleep(min(.1,left))
    return process.returncode

def stop_owned(process,deadline):
    # Popen(start_new_session=True) created this group; verify the live leader before signaling.
    if process.poll() is None and os.getpgid(process.pid)!=process.pid:raise RuntimeError('Owned PGID identity changed; refusing signal')
    if processes(process.pid):
        try:os.killpg(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        until=min(deadline-.1,time.monotonic()+.2)
        while process.poll() is None and time.monotonic()<until:time.sleep(.01)
        if processes(process.pid):
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
    process.wait(timeout=max(.001,deadline-time.monotonic()))
    if processes(process.pid):raise RuntimeError('Owned process group survived cleanup')

def run_stage(args,started=None):
    started=time.monotonic() if started is None else started
    prior=validate(args);args.analysis_root.mkdir(parents=True,exist_ok=True)
    lock=(args.analysis_root/'.supervisor.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BaseException:lock.close();raise
    process=None;owns_directory=False;directory=args.analysis_root/args.stage;deadline=started+args.seconds;work=deadline-2
    state=dict(stage=args.stage,status='running',command=args.command,started_utc=utc(),seconds=args.seconds,prior_elapsed_seconds=prior,
        environment={key:cpu_environment()[key] for key in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','PYTHONDONTWRITEBYTECODE')},
        attestation=dict(supervisor_gpu_queries=0,supervisor_model_calls=0,authorized_generate_calls=0,authorized_optimizer_updates=0,child_call_counts_independently_observed=False),exit_code=None,owned_cleanup='pending')
    old=signal.getsignal(signal.SIGALRM);old_term=signal.getsignal(signal.SIGTERM)
    def expired(*_):raise BudgetExpired('CPU stage absolute deadline reached')
    signal.signal(signal.SIGALRM,expired);signal.signal(signal.SIGTERM,expired)
    error=None;baseline=0;initial=0;last=None
    def observe(phase,terminal=False):
        nonlocal last
        row=snapshot(args,process,baseline,initial,phase);row.update(elapsed_seconds=time.monotonic()-started,terminal=terminal)
        if last is not None and 'cpu_seconds' in row:
            row['cpu_percent']=max(0.,row['cpu_seconds']-last.get('cpu_seconds',0))*100/max(.000001,row['elapsed_seconds']-last['elapsed_seconds'])
        with (directory/'resources.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        last=row;check_resources(row)
    try:
        directory.mkdir(exist_ok=False);owns_directory=True;save(directory/'stage.json',state)
        signal.setitimer(signal.ITIMER_REAL,max(.000001,work-time.monotonic()))
        observe('initializing');initial=tree_bytes(args.analysis_root);baseline=project_bytes(args.project_root,work);observe('preflight')
        sources={str(Path(__file__).resolve()):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        for token in args.command[1:]:
            path=Path(token)
            if path.suffix=='.py' and path.is_file():sources[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        state['source_sha256']=sources;save(directory/'stage.json',state)
        env=dict(cpu_environment(),CPU_EVIDENCE_STAGE_DIR=str(directory.resolve()))
        with (directory/'stdout.log').open('w') as output,(directory/'stderr.log').open('w') as errors:
            process=subprocess.Popen(args.command,env=env,stdout=output,stderr=errors,start_new_session=True)
            state['pid']=process.pid;state['pgid']=process.pid;save(directory/'stage.json',state);observe('running')
            state['exit_code']=wait_command(process,work,observe)
        if state['exit_code']!=0:raise RuntimeError(f'Python command exited{state["exit_code"]}')
        observe('final_project_check');actual=project_bytes(args.project_root,work)
        if actual>300*GIB:raise RuntimeError('Final project quota exceeded')
        state['final_project_bytes']=actual;state['status']='pending_cleanup'
    except BaseException as problem:
        error=problem;state.update(status='budget_stopped' if isinstance(problem,(BudgetExpired,subprocess.TimeoutExpired)) else 'failed',error=f'{type(problem).__name__}: {problem}')
        state['elapsed_seconds']=time.monotonic()-started
        if owns_directory:save(directory/'stage.json',state)
    finally:
        signal.setitimer(signal.ITIMER_REAL,max(.000001,deadline-time.monotonic()))
        try:
            if process is not None:stop_owned(process,deadline)
            if owns_directory:
                state['owned_cleanup']='complete';observe('terminal',terminal=True)
                if error is None:state['status']='complete'
        except BaseException as problem:
            if error is None:error=problem
            state.update(status='budget_stopped' if isinstance(error,(BudgetExpired,subprocess.TimeoutExpired)) else 'failed',cleanup_error=f'{type(problem).__name__}: {problem}')
        finally:
            state.update(ended_utc=utc(),elapsed_seconds=time.monotonic()-started,cumulative_elapsed_seconds=prior+time.monotonic()-started)
            try:
                if owns_directory:save(directory/'stage.json',state)
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,old);signal.signal(signal.SIGTERM,old_term);lock.close()
    if error is not None:raise error
    return state

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-root',required=True,type=Path);parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=STAGES);parser.add_argument('--seconds',required=True,type=float)
    parser.add_argument('command',nargs=argparse.REMAINDER);args=parser.parse_args()
    if args.command[:1]==['--']:args.command=args.command[1:]
    run_stage(args,ENTRY_STARTED)
if __name__=='__main__':main()
