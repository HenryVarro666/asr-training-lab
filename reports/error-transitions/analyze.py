"""Bounded standard-library CPU analysis of frozen fixed128 edit counts."""
import collections
import hashlib
import json
from pathlib import Path
import time

START=time.monotonic()
OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
STEPS=(25,50,100)
STATES=('improved','tied','worse')
def check():
    if time.monotonic()-START>60: raise TimeoutError('60-second CPU budget exceeded')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    check(); rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    assert len(rows)==len({r['id'] for r in rows})==128
    return {r['id']:r for r in rows}
def state(delta): return 'improved' if delta<0 else 'worse' if delta>0 else 'tied'
def write(name,obj):
    check(); (OUT/name).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')
paths={str(s):ROOT/f'reports/early100/analysis/step{s}-fixed128.edits.jsonl' for s in STEPS}
paths['manifest']=ROOT/'reports/early100/inputs/dev-clean-fixed128.jsonl'
data={s:read(paths[str(s)]) for s in STEPS}; manifest=read(paths['manifest'])
ids=sorted(manifest); assert all(set(d)==set(ids) for d in data.values())
base={i:data[25][i]['baseline'] for i in ids}
assert all(data[s][i]['baseline']==base[i] for s in STEPS for i in ids)
assert sum(x['reference_words'] for x in base.values())==2905
assert sum(x['edits'] for x in base.values())==46
assert [sum(data[s][i]['candidate']['edits'] for i in ids) for s in STEPS]==[122,108,93]
rows=[]
for i in ids:
    check(); speaker=str(manifest[i]['speaker_id']); n=base[i]['reference_words']
    assert all(str(data[s][i]['speaker_id'])==speaker for s in STEPS)
    assert all(data[s][i]['candidate']['reference_words']==n for s in STEPS)
    edits={'base':base[i]['edits'],**{str(s):data[s][i]['candidate']['edits'] for s in STEPS}}
    delta={str(s):edits[str(s)]-edits['base'] for s in STEPS}
    assert all(delta[str(s)]==data[s][i]['edit_delta'] for s in STEPS)
    rows.append(dict(id=i,speaker_id=speaker,reference_words=n,edits=edits,delta_vs_base=delta,state_vs_base={s:state(v) for s,v in delta.items()}))
transitions={}
for a,b in ((25,50),(50,100),(25,100)):
    a,b=str(a),str(b); key=a+'->'+b
    table={x:{y:[] for y in STATES} for x in STATES}
    actual={x:[] for x in ('decreased','unchanged','increased')}
    persistent=[];recovered=[];new=[]
    for r in rows:
        x,y=r['state_vs_base'][a],r['state_vs_base'][b]
        table[x][y].append(r['id']); diff=r['edits'][b]-r['edits'][a]
        actual['decreased' if diff<0 else 'increased' if diff>0 else 'unchanged'].append(r['id'])
        if x=='worse' and y=='worse': persistent.append(r['id'])
        if x=='worse' and y!='worse': recovered.append(r['id'])
        if x!='worse' and y=='worse': new.append(r['id'])
    transitions[key]=dict(state_counts={x:{y:len(v) for y,v in ys.items()} for x,ys in table.items()},state_ids=table,
        actual_error_change_counts={k:len(v) for k,v in actual.items()},actual_error_change_ids=actual,
        net_error_change=sum(r['edits'][b]-r['edits'][a] for r in rows),persistent_worse_ids=persistent,recovered_from_worse_ids=recovered,newly_worse_ids=new)
all_persistent=[r['id'] for r in rows if all(r['state_vs_base'][str(s)]=='worse' for s in STEPS)]
speakers=[];loso=[]
for sp in sorted({r['speaker_id'] for r in rows}):
    check(); group=[r for r in rows if r['speaker_id']==sp]; remaining=[r for r in rows if r['speaker_id']!=sp]
    counts={k:sum(r['edits'][k] for r in group) for k in ('base','25','50','100')}
    speakers.append(dict(speaker_id=sp,examples=len(group),reference_words=sum(r['reference_words'] for r in group),errors=counts,extra_errors_vs_base={str(s):counts[str(s)]-counts['base'] for s in STEPS}))
    words=sum(r['reference_words'] for r in remaining); errors={k:sum(r['edits'][k] for r in remaining) for k in counts}
    loso.append(dict(excluded_speaker=sp,remaining_examples=len(remaining),reference_words=words,errors=errors,
        wer_percent={k:100*v/words for k,v in errors.items()},delta_vs_base_pp={str(s):100*(errors[str(s)]-errors['base'])/words for s in STEPS}))
write('per-id.json',rows);write('transitions.json',transitions);write('speaker-contributions.json',speakers);write('leave-one-speaker-out.json',loso)
summary=dict(status='complete_cpu_only',unique_ids=128,reference_words=2905,errors=dict(base=46,step25=122,step50=108,step100=93),speakers=len(speakers),
    transitions={k:{field:v[field] for field in ('state_counts','actual_error_change_counts','net_error_change')} for k,v in transitions.items()},
    persistently_worse_all_three_ids=all_persistent,
    worse_sets={k:{name:len(v[name]) for name in ('persistent_worse_ids','recovered_from_worse_ids','newly_worse_ids')} for k,v in transitions.items()},
    loso_delta_pp_ranges={str(s):dict(min=min(x['delta_vs_base_pp'][str(s)] for x in loso),max=max(x['delta_vs_base_pp'][str(s)] for x in loso)) for s in STEPS},
    generate_calls=0,teacher_forward_calls=0,training_updates=0,elapsed_seconds=time.monotonic()-START,
    definitions=dict(recovered='worse versus base at start, tied or improved versus base at end; not necessarily error-free',newly_worse='tied or improved versus base at start, worse at end',persistent='worse versus base at both endpoints; all-three set also provided',state='sign of edit count difference versus base; same state need not mean same edit count or hypothesis'),
    limitations=['Exploratory repeatedly inspected fixed development subset; no significance or population/causal claims.','Leave-one-speaker-out is a descriptive sensitivity calculation, not a confidence interval.','No audio, model, GPU, SSH or training calls; prior references and normalization unchanged.'])
write('summary.json',summary)
lines=['# Fixed128 error transitions','', 'CPU-only descriptive analysis; no model calls, GPU or training. Input totals verified: 128 unique IDs, 2905 reference words, base/25/50/100 errors 46/122/108/93.','', 'State means improved/tied/worse edit count relative to base. Recovery means leaving worse, not necessarily becoming error-free. Same state does not mean unchanged error count or unchanged hypothesis.','', '| Interval | Errors decreased / unchanged / increased (IDs) | Net error change | Persistent worse / recovered / newly worse |','|---|---|---|---|']
for k,v in transitions.items():
    c=v['actual_error_change_counts']; w=summary['worse_sets'][k]
    lines.append(f"| {k} | {c['decreased']} / {c['unchanged']} / {c['increased']} | {v['net_error_change']} | {w['persistent_worse_ids']} / {w['recovered_from_worse_ids']} / {w['newly_worse_ids']} |")
lines+=['',f"Persistently worse across all three checkpoints: {len(all_persistent)} IDs. All exact IDs and complete 3×3 tables are in transitions.json and summary.json.",'', '| Candidate | Leave-one-speaker-out corpus WER delta range vs base (pp) |','|---|---|']
for s,v in summary['loso_delta_pp_ranges'].items(): lines.append(f"| {s} | {v['min']:.6f} to {v['max']:.6f} |")
lines+=['','Per-speaker absolute error contributions and every excluded-speaker calculation are saved separately. Positive ranges describe this subset after any one speaker is removed; they do not prove significance, independence or a training mechanism.','', 'Reproduce: `python3 reports/error-transitions/analyze.py` (standard library; internal 60-second deadline; overwrites only this derived directory).','',f"Analysis computation completed in {summary['elapsed_seconds']:.6f} seconds; 0 generate, 0 teacher forward, 0 updates. Output manifests exclude themselves to avoid self-hashing."]
check();(OUT/'README.md').write_text('\n'.join(lines)+'\n')
write('artifact-manifest.json',dict(inputs={str(p.relative_to(ROOT)):sha(p) for p in paths.values()},script=dict(path=str(Path(__file__).relative_to(ROOT)),sha256=sha(Path(__file__))),outputs={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name not in ('analyze.py','artifact-manifest.json')},elapsed_seconds=time.monotonic()-START))
print(json.dumps(summary,indent=2))
