#!/usr/bin/env python3
"""Evaluate one existing checkpoint against a completed campaign; never train."""
import argparse
import signal
import sys
import time
from pathlib import Path

from run_qwen_campaign import Campaign, BudgetExpired, REPO, file_hash, read_json, validate_evaluation


class CheckpointDiagnostic(Campaign):
    def report(self):
        lines = ['# Existing checkpoint diagnostic', '',
            f"Status: {self.state['status']}; stage: {self.state['stage']}.", '',
            'No parameter updates. Same eligible development manifests and batch16 as the original campaign.', '',
            '| Split | Original WER | Step800 WER | Final892 WER |', '|---|---:|---:|---:|']
        for split, row in self.state.get('evaluations', {}).items():
            lines.append(f"| {split} | {100*row['baseline_wer']:.4f}% | {100*row['candidate_wer']:.4f}% | {100*row['final_wer']:.4f}% |")
        if self.state.get('error'):
            lines += ['', 'Failure: '+self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def run_evaluation(self):
        original = self.args.reference_campaign.resolve()
        receipt = read_json(original/'campaign.json')
        if receipt.get('status') != 'complete':
            raise ValueError('Reference campaign is not complete')
        checkpoint = self.args.checkpoint.resolve()
        if not checkpoint.is_relative_to((original/'main-epoch1').resolve()):
            raise ValueError('Checkpoint must belong to the reference main run')
        metadata = read_json(checkpoint/'checkpoint.json')
        if metadata.get('global_step') != 800:
            raise ValueError('This predeclared diagnostic is restricted to checkpoint800')
        if metadata.get('revision') != receipt['model_revision'] or not list(checkpoint.glob('*.safetensors')):
            raise ValueError('Checkpoint model revision or weights invalid')
        expected_sources = receipt['source_sha256']
        for source, expected in expected_sources.items():
            if file_hash(source) != expected:
                raise ValueError(f'Original evaluation source changed: {source}')
        manifests = {split:self.root/'data/manifests'/f'{split}.jsonl' for split in ['dev-clean','dev-other']}
        for split, manifest in manifests.items():
            if file_hash(manifest) != receipt['manifests'][str(manifest)]:
                raise ValueError(f'Manifest changed: {split}')
        self.state.update(kind='existing_checkpoint_generation_diagnostic', parameter_updates=0,
            reference_campaign=str(original), checkpoint=str(checkpoint), checkpoint_step=800,
            question='Was the observed WER regression already present by step800, before the final92 updates?',
            source_sha256=expected_sources, manifests={str(p):file_hash(p) for p in manifests.values()},
            model_revision=metadata['revision'], evaluations={}, evaluation_batch_size=16,
            original_campaign_sha256=file_hash(original/'campaign.json'),
            checkpoint_metadata_sha256=file_hash(checkpoint/'checkpoint.json'))
        self.write_state()
        for split, manifest in manifests.items():
            baseline = validate_evaluation(original/'evaluation'/f'baseline-{split}'/'predictions.metrics.json',manifest,metadata['revision'])
            final = validate_evaluation(original/'evaluation'/f'final-{split.replace("-","_")}'/'predictions.metrics.json',manifest)
            if any(x.get('evaluation_batch_size') != 16 for x in (baseline,final)):
                raise ValueError('Reference generation batch size changed')
            run = self.directory/split
            output = run/'predictions.jsonl'
            command = [sys.executable,'-m','asr_lab.qwen_evaluate','--model',str(checkpoint),
                '--manifest',str(manifest),'--output',str(output),'--all','--batch-size','16']
            self.execute('evaluate-'+split,command,run,'1',projected_gib=1)
            candidate = validate_evaluation(output.with_suffix('.metrics.json'),manifest)
            for key in ('selected_ids','reference_words','normalizer','normalizer_revision','evaluation_batch_size','generation_max_new_tokens'):
                if not candidate.get(key) == baseline.get(key) == final.get(key):
                    raise ValueError(f'Comparison settings differ: {key}')
            self.state['evaluations'][split] = dict(baseline_wer=baseline['wer'],candidate_wer=candidate['wer'],
                final_wer=final['wer'],examples=candidate['examples'],reference_words=candidate['reference_words'])
            self.write_state()
        self.state.update(status='complete',stage='diagnostic_evaluations_complete',completed_unix=time.time())
        self.write_state()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-campaign',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--campaign-dir',required=True,type=Path)
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--max-wall-seconds',type=int,default=1200)
    args=parser.parse_args()
    if not 0 < args.max_wall_seconds <= 1200:
        parser.error('This diagnostic is bounded to at most1200 seconds')
    def stop(*_):raise KeyboardInterrupt('Diagnostic interrupted')
    signal.signal(signal.SIGTERM,stop)
    job=CheckpointDiagnostic(args)
    try:
        job.run_evaluation()
    except BaseException as error:
        job.state.update(status='budget_stopped' if isinstance(error,BudgetExpired) else 'failed',error=f'{type(error).__name__}: {error}')
        job.write_state()
        raise


if __name__=='__main__':main()
