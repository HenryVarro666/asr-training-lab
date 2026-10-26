#!/usr/bin/env python3
"""Manually restore the saved PersonaPlex baseline; no model runs with --check-only."""
import argparse,csv,hashlib,json,os,socket,subprocess,time
from pathlib import Path
here=Path(__file__).resolve().parent
saved=json.loads((here/'process-snapshot.json').read_text())
parser=argparse.ArgumentParser();parser.add_argument('--check-only',action='store_true');args=parser.parse_args()
cwd=Path(saved['cwd'])
assert os.getuid()==saved['uid'],'Run under the original exx account'
for item in saved['source_candidates']:
 p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],'Baseline module changed'
assert (cwd/'.venv/bin/python').exists(),'Saved environment is unavailable'
for key in ('HF_HOME','HF_HUB_CACHE'):
 if key in saved['allowed_environment']:assert Path(saved['allowed_environment'][key]).exists(),key+' missing'
print(json.dumps(dict(source_sha_matched=True,cwd=str(cwd),argv=saved['argv'],check_only=args.check_only)),flush=True)
if args.check_only:raise SystemExit(0)
query=lambda fields,kind:subprocess.check_output(['nvidia-smi','--query-'+kind+'='+fields,'--format=csv,noheader,nounits'],text=True,timeout=10)
devices=list(csv.reader(query('index,uuid,utilization.gpu','gpu').splitlines(),skipinitialspace=True))
gpu0=next(r for r in devices if r[0]=='0');assert float(gpu0[2])==0,'GPU0 busy; do not restore now'
apps=list(csv.reader(query('gpu_uuid,pid,process_name','compute-apps').splitlines(),skipinitialspace=True))
assert not [r for r in apps if r and r[0]==gpu0[1]],'GPU0 has another task; do not restore now'
with socket.socket() as probe:
 probe.bind(('127.0.0.1',8998))
env=dict(os.environ,**saved['allowed_environment']);log=here/('restore-'+str(int(time.time()))+'.log')
with log.open('ab') as stream:
 process=subprocess.Popen(saved['argv'],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
receipt=dict(pid=process.pid,dispatched_unix=time.time(),log=str(log),cwd=str(cwd),argv=saved['argv'],status='dispatched_not_functionally_verified')
(here/'restore-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
