"""Stdlib-only resource observations and owned-process teardown for probe A."""
from __future__ import annotations
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def remaining(deadline):
    left=deadline-time.monotonic()
    if left<=0:raise TimeoutError('Absolute probe deadline exhausted')
    return left


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def resource_snapshot(project_root,deadline):
    def query(kind,fields):
        result=subprocess.run(['nvidia-smi',f'--query-{kind}={fields}','--format=csv,noheader,nounits'],
            capture_output=True,text=True,check=True,timeout=min(5,remaining(deadline)))
        return list(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    devices=[]
    for row in query('gpu','index,uuid,utilization.gpu,memory.used,memory.total'):
        if len(row)!=5:raise ValueError('Malformed GPU observation')
        devices.append(dict(index=int(row[0]),uuid=row[1].strip(),utilization_percent=float(row[2]),
            memory_used_mib=float(row[3]),memory_total_mib=float(row[4])))
    if {device['index'] for device in devices if device['index'] in (0,1)}!={0,1}:
        raise ValueError('Missing physical GPU0/GPU1 observation')
    applications=[]
    for row in query('compute-apps','gpu_uuid,pid,process_name'):
        if not row:continue
        if len(row)!=3:raise ValueError('Malformed compute application observation')
        applications.append(dict(gpu_uuid=row[0].strip(),pid=int(row[1]),process_name=row[2].strip()))
    remaining(deadline)
    return dict(gpus=devices,compute_apps=applications,disk_free_bytes=shutil.disk_usage(project_root).free,observed_unix=time.time())


def require_gpu_idle(physical_gpu=1,ignore_own=False,deadline=None,expected_uuid=None):
    if physical_gpu not in (0,1):raise ValueError('Physical GPU must be 0 or 1')
    observation=resource_snapshot(Path('.'),deadline if deadline is not None else time.monotonic()+10)
    devices=[gpu for gpu in observation['gpus'] if gpu['index']==physical_gpu]
    if len(devices)!=1:raise ValueError('Cannot identify selected physical GPU UUID')
    uuid=devices[0]['uuid']
    if expected_uuid is not None and expected_uuid!=uuid:raise ValueError('Selected GPU UUID changed')
    own=os.getpid() if ignore_own else None
    busy=[app for app in observation['compute_apps'] if app['gpu_uuid']==uuid and app['pid']!=own]
    if busy:raise RuntimeError(f'Physical GPU{physical_gpu} has other compute applications; no launch: {busy}')
    if devices[0]['utilization_percent']!=0:
        raise RuntimeError(f'Physical GPU{physical_gpu} is not idle; utilization is nonzero')
    return dict(physical_gpu=physical_gpu,uuid=uuid,other_compute_apps=[],ignored_own_pid=own,
        checked_unix=observation['observed_unix'],gpu=devices[0],
        limitation='Point-in-time check cannot reserve a GPU against another scheduler.')


def worker_gpu_preflight(identity):
    selected=identity.get('physical_gpu',1)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(selected):raise ValueError('Selected GPU visibility differs')
    return require_gpu_idle(selected,ignore_own=True,expected_uuid=identity.get('gpu_uuid'))


def write_observation(run_dir,snapshot,*,started,phase,campaign_status,terminal=False,error=None):
    output=Path(run_dir)/'preflight-monitor';output.mkdir(exist_ok=True)
    row=dict(snapshot,phase=phase,campaign_status=campaign_status,elapsed_seconds=time.monotonic()-started,
        terminal=terminal,status='failed' if error else ('stopped' if terminal else 'running'),error=error,continuous_observation=not bool(error),
        observation_kind='read_only_resources',training_progress_claim=False)
    with (output/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    atomic_json(output/'status.json',row)
    return row


def group_alive(pgid,deadline):
    result=subprocess.run(['ps','-eo','pgid=,stat='],capture_output=True,text=True,check=True,
        timeout=min(.5,remaining(deadline)))
    return any(parts[0]==str(pgid) and not parts[1].startswith('Z')
        for line in result.stdout.splitlines() if len(parts:=line.split())>=2)


def stop_owned(process,deadline):
    """Only call for Popen children created with start_new_session=True here."""
    if process.poll() is not None and not group_alive(process.pid,deadline):return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:pass
    alive=True
    try:
        grace=min(deadline-.1,time.monotonic()+.2)
        while time.monotonic()<grace:
            process.poll()
            alive=group_alive(process.pid,deadline)
            if not alive:break
            time.sleep(min(.025,max(0,grace-time.monotonic())))
    finally:
        # Even if ps/observation fails, teardown must still reach the owned session.
        if alive:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
        process.wait(timeout=remaining(deadline))
    if group_alive(process.pid,deadline):raise RuntimeError('Owned process group still running after cleanup')


def sample_loop(args):
    # The supervisor writes the initial observation before starting this process.
    stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop)
    next_sample=args.first_observation+10
    try:
        while not stop:
            left=remaining(args.deadline)
            if time.monotonic()<next_sample:
                time.sleep(min(.1,left,next_sample-time.monotonic()));continue
            phase=json.loads((args.run_dir/'campaign.json').read_text())
            snapshot=resource_snapshot(args.project_root,args.deadline)
            write_observation(args.run_dir,snapshot,started=args.started,phase=phase['stage'],campaign_status=phase['status'])
            next_sample+=10
    except BaseException as error:
        if stop:return
        write_observation(args.run_dir,{},started=args.started,phase='monitor_failure',campaign_status='failed',error=f'{type(error).__name__}: {error}')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path);parser.add_argument('--project-root',required=True,type=Path)
    parser.add_argument('--deadline',required=True,type=float);parser.add_argument('--started',required=True,type=float)
    parser.add_argument('--first-observation',required=True,type=float)
    sample_loop(parser.parse_args())


if __name__=='__main__':main()
