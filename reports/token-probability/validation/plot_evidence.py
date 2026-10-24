#!/usr/bin/env python3
"""Standard static figures from saved derived JSON only; never loads models/PTs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STAGES = ('base', 'step050', 'step100')
COLORS = ('#0072B2', '#E69F00', '#CC79A7')


def read(path):
    return json.loads(path.read_text())


def figures(run, output, reference):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    output.mkdir(parents=True, exist_ok=True)
    frozen = read(reference)
    prompt = frozen['common_input']['input_ids'][0]
    data = {s: {t: read(run/s/'worker/teacher'/t/'derived.json') for t in STAGES} for s in STAGES}
    source = {}
    for s in STAGES:
        for t in STAGES:
            path = run/s/'worker/teacher'/t/'derived.json'
            source[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    def row(s,t,j):
        r = data[s][t]['rows'][j]
        ids = prompt+frozen['trajectories'][t]['token_ids'][:j]
        expected = hashlib.sha256(json.dumps(ids,separators=(',', ':')).encode()).hexdigest()
        assert r['prefix']['sha256'] == expected and r['prefix']['length'] == len(ids)
        return r
    def candidate(r, identifier):
        return next(c for c in r['candidates'] if c['id'] == identifier)['logprob']
    def save(fig, name):
        fig.tight_layout()
        for suffix in ('png','svg','pdf'):
            fig.savefig(output/(name+'.'+suffix),dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    events = [('step100',0,[34,5049],['C (34)','CA (5049)']),
              ('step100',1,[1094,4943],['UL (1094)','aul (4943)'])]
    for ax,(t,j,ids,labels) in zip(axes,events):
        for i,s in enumerate(STAGES):
            r = row(s,t,j)
            values = [candidate(r,k) for k in ids]+[r['eos']['total_logprob']]
            ax.bar([n+(i-1)*.23 for n in range(3)],values,width=.23,color=COLORS[i],label=s)
        ax.set_xticks(range(3),labels+['EOS sum'])
        ax.set_ylabel('Next-token log probability (natural log)')
        ax.set_title(f"Fixed loop prefix: j={j}, continuation prefix={frozen['trajectories'][t]['token_ids'][:j]}")
        ax.legend()
    fig.suptitle('Single BPE candidate events under identical conditioning')
    save(fig,'fixed-prefix-branches')
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for ax,t,j,label in zip(axes,('base','step050'),(26,36),('Natural correct EOS','Uppercase correct EOS')):
        vals = [row(s,t,j)['eos']['total_logprob'] for s in STAGES]
        ax.bar(STAGES,vals,color=COLORS)
        ax.set_ylabel('Distinct two-EOS total log probability')
        ax.set_title(f'{label}: j={j}; prefix length={len(prompt)+j}')
    save(fig,'correct-endings-eos')
    fig, axes = plt.subplots(3,1,figsize=(11,9),sharex=True)
    for s,color in zip(STAGES,COLORS):
        rows=[row(s,'step100',j) for j in range(512)]
        axes[0].plot(range(512),[r['target']['logprob'] for r in rows],label=s,color=color,lw=1)
        axes[1].plot(range(512),[r['eos']['total_logprob'] for r in rows],label=s,color=color,lw=1)
        axes[2].plot(range(512),[r['target']['rank'] for r in rows],label=s,color=color,lw=1)
    for ax,label in zip(axes,('Forced loop target log probability','Distinct two-EOS log probability','Forced loop target rank')):
        ax.set_ylabel(label);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('Target index j; conditioned on P + frozen loop tokens before j')
    axes[-1].set_yscale('log')
    fig.suptitle('Complete 512-row loop trajectory; abnormal forced prefixes change conditioning')
    save(fig,'full-loop-traces')
    (output/'sources.json').write_text(json.dumps(dict(derived_sha256=source,
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        interpretation='Natural log probabilities preserve underflowed events. Candidate pieces are single next-token events. Each comparison fixes trajectory, j and prefix. No training-cause claim.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'reports/token-probability/run')
    p.add_argument('--output',type=Path,default=ROOT/'reports/token-probability/figures')
    p.add_argument('--reference',type=Path,default=ROOT/'reports/token-probability/inputs/trajectory-reference.json')
    args=p.parse_args()
    figures(args.run,args.output,args.reference)

if __name__=='__main__':main()
