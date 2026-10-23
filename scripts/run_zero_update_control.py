#!/usr/bin/env python3
"""One bounded zero-update control, GPU1 only; never starts training or follow-up work."""
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
from asr_lab.qwen_roundtrip_control import freeze_receipt,compare_evaluations

REPO=Path(__file__).resolve().parents[1]


class ZeroUpdateCampaign(Campaign):
    def __init__(self,args):
        super().__init__(args)
        self.state.update(kind='zero_update_control_B',optimizer_updates=0,automatic_training=False)
        self.state.pop('main_lr_matrix_status',None)
        self.state.pop('selected_config',None)
        self.write_state()

    def remaining(self):
        if tree_size(self.directory)>20*1024**3:
            raise RuntimeError('Zero-update control exceeded20GiB incremental storage limit')
        return super().remaining()

    def report(self):
        lines=['# Zero-update save/reload control B','',
            f"Status: {self.state['status']}; stage: {self.state['stage']}.",'',
            'Optimizer updates:0. No training or automatic follow-up experiment is launched.',
            'Preparation and every generation evaluation use isolated child processes on physical GPU1.',
            'Pass/fail refers only to this frozen128-input control, not all training implementation behavior.']
        if self.state.get('comparison'):
            result=self.state['comparison']
            lines+=['',f"Control passed: {result['passed']}; differences: {result['differences']}.",
                    f"Conclusion: {result['conclusion']}."]
        if self.state.get('error'):lines+=['','Failure: '+self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def worker(self,phase,name,source,snapshot=None,role='base'):
        run=self.directory/name
        command=[sys.executable,'-m','asr_lab.qwen_roundtrip_control',phase,
            '--receipt',str(self.args.receipt),'--training-receipt',str(self.args.training_receipt),
            '--manifest',str(self.args.manifest),'--source',str(source),'--output',str(run),
            '--role',role]
        if snapshot:command+=['--snapshot',str(snapshot)]
        self.execute(name,command,run,'1')
        return run

    def run(self):
        check_space(self.root,500,300,20*1024**3)
        frozen=freeze_receipt(self.args.receipt,self.args.manifest)
        original=read_json(self.args.training_receipt)
        if original.get('revision')!=frozen['model_revision'] or original.get('model_id')!=frozen['model_id']:
            raise ValueError('Original training receipt model revision differs')
        if original.get('eval_manifest_sha256')!=file_hash(self.args.manifest):
            raise ValueError('Original training fixed128 manifest differs')
        self.state.update(frozen)
        self.state['source_sha256'].update({str(REPO/'src/asr_lab/qwen_roundtrip_control.py'):file_hash(REPO/'src/asr_lab/qwen_roundtrip_control.py'),
                                           str(Path(__file__).resolve()):file_hash(__file__)})
        self.state['receipt_sha256']=file_hash(self.args.receipt)
        self.state['training_receipt_sha256']=file_hash(self.args.training_receipt)
        # Verify both immutable receipts before every child as well as the frozen source/manifest.
        self.state['source_sha256'].update({str(self.args.receipt.resolve()):file_hash(self.args.receipt),
                                           str(self.args.training_receipt.resolve()):file_hash(self.args.training_receipt)})
        source=Path(original['model_path']).resolve()
        if not source.is_dir() or source.name!=frozen['model_revision']:
            raise ValueError('Original pinned local model snapshot is missing or not the receipt revision')
        self.state['original_snapshot']=str(source)
        self.write_state()
        prepared=self.directory/'prepare'
        snapshot=prepared/'snapshot'
        self.worker('prepare','prepare',source,snapshot)
        preparation=read_json(prepared/'prepared.json')
        if preparation.get('optimizer_updates')!=0 or preparation.get('train_called') is not False:
            raise ValueError('Preparation does not prove zero updates')
        base=self.worker('evaluate','base',source,role='base')
        reloaded=self.worker('evaluate','reload',snapshot,role='reload')
        base_evidence=read_json(base/'evidence.json')
        reload_evidence=read_json(reloaded/'evidence.json')
        projection=read_json(prepared/'bf16_projection.json')
        result=compare_evaluations(base_evidence,reload_evidence,projection)
        result['prepared_bf16_projection_matches_base']=projection==base_evidence['model']
        result['prepared_bf16_projection_matches_reload']=projection==reload_evidence['model']
        result['base_persistent_state_unchanged']=base_evidence['model']==base_evidence['post_generation_model']
        result['reload_persistent_state_unchanged']=reload_evidence['model']==reload_evidence['post_generation_model']
        atomic_json(self.directory/'comparison.json',result)
        self.state['comparison']=result
        self.state['wer']=dict(base=base_evidence['metrics']['wer'],reload=reload_evidence['metrics']['wer'])
        self.write_state()
        if result['token_only_difference']:
            repeat=self.worker('evaluate','base-repeat',source,role='base-repeat')
            calibration=compare_evaluations(base_evidence,read_json(repeat/'evidence.json'),projection)
            atomic_json(self.directory/'base_repeat_calibration.json',calibration)
            self.state['base_repeat_calibration']=calibration
            # A repeat never makes a differing control pass, regardless of apparent numerical sensitivity.
        self.state.update(status='complete' if result['passed'] else 'control_not_passed',
            stage='control_finished',completed_unix=time.time(),automatic_training=False)
        self.write_state()
        if not result['passed']:
            raise ControlNotPassed('Control did not pass; cause undetermined; no training was started')


class ControlNotPassed(RuntimeError):
    pass


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True,type=Path)
    parser.add_argument('--receipt',required=True,type=Path,help='First completed campaign.json with the frozen4 source hashes')
    parser.add_argument('--training-receipt',required=True,type=Path,help='First main-epoch1/run.json for software/source and initialization config')
    parser.add_argument('--project-root',type=Path,default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--manifest',type=Path,default=Path('/mnt/raid/asr-lab/data/manifests/dev-clean-fixed128.jsonl'))
    parser.add_argument('--max-wall-seconds',type=float,default=1200)
    args=parser.parse_args()
    if not 0<args.max_wall_seconds<=1200:parser.error('Control wall budget must be in(0,1200] seconds')
    args.campaign_dir=args.run_dir
    def stop(*_):raise KeyboardInterrupt('Zero-update control interrupted')
    signal.signal(signal.SIGTERM,stop)
    control=ZeroUpdateCampaign(args)
    try:
        control.run()
    except BaseException as error:
        status='control_not_passed' if isinstance(error,ControlNotPassed) else ('budget_stopped' if isinstance(error,BudgetExpired) else ('interrupted' if isinstance(error,KeyboardInterrupt) else 'failed'))
        control.state.update(status=status,error=f'{type(error).__name__}: {error}',automatic_training=False)
        control.write_state()
        raise
    finally:
        control.lock.close()
        control.global_lock.close()


if __name__=='__main__':main()
