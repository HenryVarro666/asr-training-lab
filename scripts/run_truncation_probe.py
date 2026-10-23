#!/usr/bin/env python3
"""Three preset inference-only captures, GPU1, global lock and strict finite budgets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import signal
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from run_qwen_campaign import Campaign,BudgetExpired,file_hash,read_json
from asr_lab.prepare_data import check_space,tree_size
from asr_lab.training_utils import atomic_json
from asr_lab.qwen_truncation_probe import select_probe_rows,verify_identity,require_gpu1_idle,STAGES,TARGET

REPO=Path(__file__).resolve().parents[1]


def prepare_identity(args):
    reference=read_json(args.reference_batch)
    rows=select_probe_rows(reference,'batch16-first')
    if str(args.checkpoint.resolve())!=reference['checkpoint_path']:
        raise ValueError('Only the original checkpoint100 path is permitted')
    if file_hash(args.manifest)!=reference['manifest_sha256']:raise ValueError('Original fullclean manifest changed')
    all_rows=sorted([json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()],key=lambda row:row['id'])
    if all_rows[2192:2208]!=rows:raise ValueError('Manifest sorted indices2192:2208 differ from frozen reference')
    if reference.get('original_batch_index_zero_based')!=137:raise ValueError('Original batch137 required')
    marker_path=args.checkpoint/'prefix-complete.json';marker=read_json(marker_path)
    if marker.get('status')!='complete' or marker.get('step')!=100 or marker.get('original_total_steps')!=892 or marker.get('model_only') is not False or marker.get('resumable') is not True:
        raise ValueError('Checkpoint100 completion marker differs')
    metadata_path=args.checkpoint/'checkpoint.json';metadata=read_json(metadata_path)
    if metadata['revision']!='7278e1e70fe206f11671096ffdd38061171dd6e5' or metadata['global_step']!=100:
        raise ValueError('Checkpoint100 model identity differs')
    files={}
    def add(path,expected=None):
        path=Path(path).resolve();actual=file_hash(path)
        if expected is not None and actual!=expected:raise ValueError(f'Frozen file identity changed:{path}')
        files[str(path)]=actual
    # Read/hash only weights and tokenizer/config files; never copy parameters or optimizer states.
    selected=[path for path in args.checkpoint.iterdir() if path.suffix in ('.safetensors','.json','.txt','.model','.tiktoken','.jinja')
        and path.name not in ('checkpoint.json','trainer_state.json','prefix-complete.json','rng-preservation.json')]
    if not any(path.suffix=='.safetensors' for path in selected):raise ValueError('Missing checkpoint model shards')
    for path in selected:
        expected=marker['files'].get(path.name)
        if not expected or path.stat().st_size!=expected['bytes']:raise ValueError('Checkpoint file inventory differs')
        add(path,expected['sha256'])
    for name in marker['files']:
        if Path(name).suffix in ('.safetensors','.json','.txt','.model','.tiktoken','.jinja') and name not in ('checkpoint.json','trainer_state.json','prefix-complete.json','rng-preservation.json') and not(args.checkpoint/name).is_file():
            raise ValueError('Checkpoint model/tokenizer/config file missing')
    for relative,expected in metadata['source_sha256'].items():add(REPO/relative,expected)
    for path in (args.reference_batch,args.manifest,args.spelling,args.normalizer_receipt,marker_path,metadata_path,
        REPO/'src/asr_lab/qwen_truncation_probe.py',REPO/'scripts/run_truncation_probe.py',
        REPO/'scripts/run_qwen_campaign.py',REPO/'scripts/monitor.py'):
        add(path)
    for upstream in (args.upstream_generation,args.upstream_parser):
        add(upstream)
        captured=read_json(upstream)
        for record in captured.get('files',[captured]):add(record['path'],record['sha256'])
    normalizer=read_json(args.normalizer_receipt)['normalizer']
    if normalizer['spelling_sha256']!=file_hash(args.spelling) or normalizer['transformers']!=metadata['packages']['transformers']:
        raise ValueError('Fixed normalizer map/version differs')
    for row in rows:add(row['audio_filepath'])
    identity=dict(checkpoint=str(args.checkpoint.resolve()),reference=str(args.reference_batch.resolve()),
        spelling=str(args.spelling.resolve()),files=files,packages=metadata['packages'],
        package_source_sha256=metadata['package_source_sha256'],target_id=TARGET,
        original_sorted_batch_indices=[2192,2208],total_generate_calls=3,total_consumed_examples=33,
        pcm_hash_interpretation='Manifest sha256 is decoded mono16kHz little-endian float32 PCM; files map separately hashes FLAC bytes.')
    verify_identity(identity)
    return identity


def validate_worker(directory,stage):
    run=directory/stage
    result=read_json(run/'result.json');status=read_json(run/'status.json')
    expected=1 if stage=='target-single' else 16
    if result['status']!='complete' or status['status']!='complete' or result['generate_calls']!=1 or result['consumed_examples']!=expected or result['optimizer_updates']!=0 or result['raw_generation_missing']:
        raise ValueError('Incomplete worker evidence')
    required={'gpu-preflight.json','metadata.json','inputs.pt','inputs.json','effective-generation.json','raw-tokens.pt','raw-tokens.json','decoded.json'}
    if not required<=set(result['artifacts']):raise ValueError('Missing mandatory evidence inventory')
    for name,expected_file in result['artifacts'].items():
        if file_hash(run/name)!=expected_file['sha256'] or (run/name).stat().st_size!=expected_file['bytes']:
            raise ValueError('Worker artifact changed')
    decoded=read_json(run/'decoded.json')
    if len(decoded)!=expected or sum(row['id']==TARGET for row in decoded)!=1:raise ValueError('Missing required decoded rows')
    return decoded


def validate_repeat_identity(directory):
    inputs_equal=read_json(directory/'batch16-first/inputs.json')==read_json(directory/'batch16-repeat/inputs.json')
    def actual_rules(stage):
        records=read_json(directory/stage/'effective-generation.json')
        if len(records)!=1:raise ValueError('Repeat effective-rule drift: incomplete configuration capture')
        keys=('effective','special_tokens','logits_processor_types')
        if any(key not in records[0] for key in keys):raise ValueError('Repeat effective-rule drift: missing captured fields')
        return {key:records[0][key] for key in keys}
    effective_equal=actual_rules('batch16-first')==actual_rules('batch16-repeat')
    result=dict(status='passed' if inputs_equal and effective_equal else 'identity_drift',
        inputs_equal=inputs_equal,effective_rules_equal=effective_equal,
        compared_generation_fields=['effective','special_tokens','logits_processor_types'],
        raw_generation_metadata_is_not_a_gate=True,tokens_and_parser_are_observations_not_identity_gates=True)
    atomic_json(directory/'repeat-identity-check.json',result)
    if not inputs_equal or not effective_equal:
        raise ValueError('Repeat input/effective-rule drift; remaining probes stopped before target-single')
    return result


def compare_workers(directory):
    first=read_json(directory/'batch16-first/decoded.json');repeat=read_json(directory/'batch16-repeat/decoded.json')
    identity=validate_repeat_identity(directory)
    inputs_equal=identity['inputs_equal'];config_equal=identity['effective_rules_equal']
    tokens_equal=read_json(directory/'batch16-first/raw-tokens.json')==read_json(directory/'batch16-repeat/raw-tokens.json')
    parser_equal=[(x['parser_input'],x['parser_output']) for x in first]==[(x['parser_input'],x['parser_output']) for x in repeat]
    targets={}
    for stage in STAGES:
        decoded=validate_worker(directory,stage)
        target=next(record for record in decoded if record['id']==TARGET)
        targets[stage]=dict(legacy_generation_truncated=target['legacy_generation_truncated'],
            returned_continuation_width=target['returned_continuation_width'],effective_eos_positions=target['effective_eos_positions'],
            parser_input=target['parser_input'],parser_output=target['parser_output'],repetition_fix_observation=target['repetition_fix_observation'])
    return dict(status='three_preset_probes_complete',total_generate_calls=3,total_consumed_examples=33,optimizer_updates=0,
        repeated_batch_inputs_equal=inputs_equal,repeated_batch_effective_config_equal=config_equal,
        repeated_batch_full_tokens_equal=tokens_equal,repeated_batch_parser_equal=parser_equal,targets=targets,
        interpretation='Truncation is an observation, not a retry trigger. This does not pass or repair the prior failed campaign. Fresh workers did not replay the preceding137 batches. No logits were collected; numerical or training causes remain unproven.')


class ProbeCampaign(Campaign):
    def __init__(self,args):
        self.stage_deadline=None;self.next_space_check=0.
        self.initial_project_bytes=tree_size(args.project_root.resolve())
        super().__init__(args)
        self.env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1')
        self.state.update(kind='truncation_probe',optimizer_updates=0,automatic_training=False,preset_stages=list(STAGES))
        self.state.pop('selected_config',None);self.state.pop('main_lr_matrix_status',None)
        self.write_state()
    def remaining(self):
        left=super().remaining();now=time.monotonic()
        if self.stage_deadline is not None:
            left=min(left,self.stage_deadline-now)
            if left<=0:raise BudgetExpired('Probe worker exceeded240seconds')
        if now>=self.next_space_check:
            if max(tree_size(self.directory),tree_size(self.root)-self.initial_project_bytes)>1024**3:
                raise RuntimeError('Probe incremental storage exceeded1GiB')
            check_space(self.root,500,300,0);self.next_space_check=now+5
        return left
    def report(self):
        lines=['# Fixed checkpoint100 truncation probe','',f"Status: {self.state['status']}; stage: {self.state['stage']}.",'',
            'Three fresh workers only: original batch137 twice, then the target alone;33 sample consumptions, zero optimizer updates.',
            'Original checkpoint,512-token budget and generation kwargs retained. No scores/processors/stopping criteria added.',
            'No previous137 batches replayed. Probe completion does not repair the original failed quality gate.']
        if self.state.get('summary'):
            summary=self.state['summary'];lines+=['',f"Repeated batch full tokens identical: {summary['repeated_batch_full_tokens_equal']}."]
            for stage,target in summary['targets'].items():
                lines.append(f"- {stage}: returned width{target['returned_continuation_width']}; legacy truncated={target['legacy_generation_truncated']}; parser output={target['parser_output']}.")
        if self.state.get('error'):lines+=['','Failure: '+self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')
    def run(self):
        self.remaining();check_space(self.root,500,300,1024**3)
        identity=prepare_identity(self.args)
        path=self.directory/'identity.json';atomic_json(path,identity)
        self.state['source_sha256']=dict(identity['files'],**{str(path):file_hash(path)})
        self.write_state()
        for stage in STAGES:
            self.stage_deadline=min(self.deadline,time.monotonic()+240)
            try:
                command=[sys.executable,'-m','asr_lab.qwen_truncation_probe','--identity',str(path),
                    '--stage',stage,'--output',str(self.directory/stage)]
                self.state['gpu_preflight']=require_gpu1_idle()
                self.write_state()
                self.execute(stage,command,self.directory/stage,'1')
                self.remaining()
                validate_worker(self.directory,stage)
                if stage=='batch16-repeat':
                    self.state['repeat_identity']=validate_repeat_identity(self.directory)
                    self.write_state()
            finally:self.stage_deadline=None
        summary=compare_workers(self.directory)
        atomic_json(self.directory/'summary.json',summary)
        self.remaining()
        self.state.update(status='complete',stage='three_preset_probes_complete',summary=summary,completed_unix=time.time())
        self.write_state()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--reference-batch',required=True,type=Path)
    parser.add_argument('--checkpoint',required=True,type=Path)
    parser.add_argument('--manifest',type=Path,default=Path('/mnt/raid/asr-lab/data/manifests/dev-clean.jsonl'))
    parser.add_argument('--spelling',type=Path,default=REPO/'reports/diagnosis/inputs/whisper-english-spelling-normalizer.json')
    parser.add_argument('--normalizer-receipt',type=Path,default=REPO/'reports/diagnosis/text-audit.json')
    parser.add_argument('--upstream-generation',type=Path,default=REPO/'reports/truncation-probe/inputs/upstream-generation-source.json')
    parser.add_argument('--upstream-parser',type=Path,default=REPO/'reports/truncation-probe/inputs/upstream-parser-source.json')
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--max-wall-seconds',type=float,default=1200)
    args=parser.parse_args()
    if not 0<args.max_wall_seconds<=1200:parser.error('Total wall budget must be in(0,1200]seconds')
    args.campaign_dir=args.run_dir
    def stop(*_):raise KeyboardInterrupt('Probe interrupted')
    signal.signal(signal.SIGTERM,stop)
    campaign=ProbeCampaign(args)
    try:campaign.run()
    except BaseException as error:
        campaign.state.update(status='budget_stopped' if isinstance(error,BudgetExpired) else ('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'),
            error=f'{type(error).__name__}: {error}',automatic_training=False)
        campaign.write_state();raise
    finally:campaign.lock.close();campaign.global_lock.close()


if __name__=='__main__':main()
