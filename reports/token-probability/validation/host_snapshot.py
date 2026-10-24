"""Read-only host snapshot; preserves the exact script and SSH command in each receipt."""
import argparse
import json
import subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--output', required=True, type=Path)
args=parser.parse_args()
original=json.loads((HERE/'host-preflight.json').read_text())
script=original['stdin_python']
script=script.replace("print(json.dumps({", "probe_run=root/'runs/qwen-token-probability-20261023-a'\nold_campaign=root/'runs/qwen-early100-20261022-a/campaign.json'\nimport hashlib\nold_hash=hashlib.sha256(old_campaign.read_bytes()).hexdigest()\nprint(json.dumps({'probe_run_exists':probe_run.exists(),'old_early100_campaign_sha256':old_hash,")
result=subprocess.run(original['command'],input=script,text=True,capture_output=True,timeout=45,check=True)
receipt=json.loads(result.stdout)
receipt.update(command=original['command'],stdin_python=script,stderr=result.stderr)
args.output.write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:receipt[k] for k in ('captured_utc','gpu1_compute_apps','gpu0_service_checks','global_lock_available','probe_run_exists','old_early100_campaign_sha256','disk')},indent=2))
