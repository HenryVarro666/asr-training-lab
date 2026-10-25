"""CPU-only design evidence; freezes a stratified panel before new GPU output."""
from pathlib import Path
import hashlib
import json
import random

REPO=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
SEED=20261024
TARGET='652-129742-0006'
def read_rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_json(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

sources={};stages={};summary={};all_ids=None;base_records=None
for step in (25,50,100):
 path=REPO/f'reports/early100/analysis/step{step}-fixed128.edits.jsonl'
 rows=read_rows(path);ids=[r['id'] for r in rows]
 assert len(ids)==len(set(ids))==128
 assert all_ids is None or set(ids)==all_ids
 all_ids=set(ids)
 baseline={r['id']:r['baseline'] for r in rows}
 assert base_records is None or baseline==base_records
 base_records=baseline;stages[step]={r['id']:r for r in rows}
 sources[str(path.relative_to(REPO))]=digest(path)
 for r in rows:
  assert r['edit_delta']==r['candidate']['edits']-r['baseline']['edits']
  assert r['baseline']['reference_words']==r['candidate']['reference_words']
 total_ref=sum(r['baseline']['reference_words'] for r in rows)
 summary[str(step)]=dict(examples=len(rows),reference_words=total_ref,
  baseline_errors=sum(r['baseline']['edits'] for r in rows),candidate_errors=sum(r['candidate']['edits'] for r in rows),
  improved=sum(r['edit_delta']<0 for r in rows),edit_count_tied=sum(r['edit_delta']==0 for r in rows),worse=sum(r['edit_delta']>0 for r in rows))
assert TARGET not in all_ids
manifest_path=REPO/'reports/early100/inputs/dev-clean-fixed128.jsonl'
manifest={r['id']:r for r in read_rows(manifest_path)};assert set(manifest)==all_ids
reference_path=REPO/'reports/token-probability/inputs/trajectory-reference.json'
reference=json.loads(reference_path.read_text());target_row=reference['target']['manifest_row'];assert target_row['id']==TARGET
sources[str(manifest_path.relative_to(REPO))]=digest(manifest_path)
sources[str(reference_path.relative_to(REPO))]=digest(reference_path)
rng=random.Random(SEED)
pools={name:sorted(k for k,v in stages[25].items() if predicate(v['edit_delta'])) for name,predicate in [('worse',lambda x:x>0),('edit_count_tied',lambda x:x==0),('improved',lambda x:x<0)]}
assert {k:len(v) for k,v in pools.items()}=={'worse':47,'edit_count_tied':79,'improved':2}
selected={'sentinel':[TARGET],'worse':sorted(rng.sample(pools['worse'],5)),
 'edit_count_tied':sorted(rng.sample(pools['edit_count_tied'],8)),'improved':pools['improved']}
selected_ids=[i for ids in selected.values() for i in ids];assert len(selected_ids)==len(set(selected_ids))==16
panel=[]
for group,ids in selected.items():
 for identifier in ids:
  row=dict(target_row if identifier==TARGET else manifest[identifier]);row['design_group']=group
  panel.append(row)
panel.sort(key=lambda r:r['id'])
panel_path=OUT/'proposed-panel16.jsonl'
panel_path.write_text(''.join(json.dumps(row,ensure_ascii=False,sort_keys=True)+'\n' for row in panel))
write_json(OUT/'design-evidence.json',dict(status='cpu_design_only_no_new_model_calls',sources=sources,
 fixed128=summary,target_in_fixed128=False,seed=SEED,selection_source='Only existing base-versus-step25 edit deltas; step50/100 never used to select IDs.',
 selection_algorithm='Sort IDs within each stratum; random.Random(20261024), sample5 worse then8 tied; include both improved IDs and sentinel; final batch sorted by ID.',
 group_counts={k:len(v) for k,v in selected.items()},selected_ids_by_group=selected,batch_order=[r['id'] for r in panel],
 panel_sha256=digest(panel_path),new_gpu_calls=0,new_updates=0,
 limitations=['Outcome-stratified exploratory panel, not a representative WER or loop-prevalence estimate.',
 'Edit-count tie need not mean unchanged hypothesis or error-free recognition.',
 'Selection is frozen before any new probability/panel GPU output, but uses already observed development results.']))
print(json.dumps(dict(fixed128=summary,panel_sha256=digest(panel_path),selected_ids_by_group=selected),ensure_ascii=False,indent=2))
