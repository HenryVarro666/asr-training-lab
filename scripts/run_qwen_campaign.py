#!/usr/bin/env python3
"""Sequential, bounded Qwen ASR experiment campaign. No remote control or uploads."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
import time

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
from asr_lab.benchmark import summarize_run
from asr_lab.prepare_data import check_space
from asr_lab.training_utils import atomic_json


class BudgetExpired(RuntimeError):
    pass


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def manifest_ids(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    ids = [row['id'] for row in rows]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or duplicate manifest IDs: {path}')
    return ids


def validate_evaluation(metrics_path, manifest, revision=None):
    metrics_path = Path(metrics_path)
    result = read_json(metrics_path)
    ids = manifest_ids(manifest)
    if result.get('manifest_sha256') != file_hash(manifest):
        raise ValueError('Evaluation manifest hash differs')
    if result.get('examples') != len(ids) or set(result.get('selected_ids', [])) != set(ids):
        raise ValueError('Evaluation does not cover exactly the expected manifest')
    if len(result['selected_ids']) != len(set(result['selected_ids'])):
        raise ValueError('Duplicate evaluation predictions')
    if not math.isfinite(float(result['wer'])) or float(result['wer']) < 0 or result.get('reference_words', 0) <= 0:
        raise ValueError('Invalid generation WER evidence')
    if result.get('generation_truncated_count', -1) != 0:
        raise ValueError('Generation truncation prevents an accepted evaluation')
    if revision is not None and result.get('model_revision') != revision:
        raise ValueError('Evaluation model revision differs')
    output = metrics_path.with_name(metrics_path.name.removesuffix('.metrics.json') + '.jsonl')
    references = {row['id']: row['text'] for row in (json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip())}
    predictions = [json.loads(line) for line in output.read_text().splitlines() if line.strip()]
    if len(predictions) != len(ids) or {row['id'] for row in predictions} != set(ids):
        raise ValueError('Raw prediction file does not cover the expected manifest')
    if any(row.get('reference') != references[row['id']] for row in predictions):
        raise ValueError('Raw prediction references differ from the manifest')
    return result


def is_oom(text):
    lower = text.lower()
    if re.search(r'\bnan\b|non[ -]?finite|invariant|leakage', lower):
        return False
    return 'cuda out of memory' in lower or 'torch.outofmemoryerror' in lower


def allow_batch8(result):
    return bool(result.get('valid') and result.get('world_size') == 2 and
                0 < result.get('peak_gib', float('inf')) < 60 and result.get('min_free_gib', 0) > 12)


def choose_dual(results):
    eligible = [row for row in results if allow_batch8(row) and
                math.isfinite(row.get('throughput', float('nan'))) and row['throughput'] > 0]
    if not eligible:
        raise ValueError('No valid dual-GPU experiment with measured safe headroom')
    return max(eligible, key=lambda row: row['throughput'])


def checkpoint_projection_gib(parameters, copies=1):
    if parameters <= 0 or copies < 1:
        raise ValueError('Invalid checkpoint projection')
    # FP32 weights + two FP32 Adam moments, plus one GiB/checkpoint for minor state.
    return math.ceil(parameters * 12 / 1024**3 + 1) * copies


def train_command(python, config, run, world_size, benchmark=False):
    command = [python, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={world_size}',
               '-m', 'asr_lab.qwen_train', '--config', str(config), '--run-dir', str(run)]
    if benchmark:
        command += ['--max-steps', '40', '--limit-train', '2048', '--limit-eval', '8']
    return command


def group_alive(pgid):
    # Inspect only the process group created by this supervisor; zombies consume no GPU work.
    result = subprocess.run(['ps', '-eo', 'pgid=,stat='], capture_output=True, text=True, check=True)
    return any(parts[0] == str(pgid) and not parts[1].startswith('Z')
               for line in result.stdout.splitlines() if len(parts := line.split()) >= 2)


def stop_owned(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 45
    while group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(.25)
    if group_alive(process.pid):
        os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while group_alive(process.pid) and time.monotonic() < deadline:
        time.sleep(.25)
    if group_alive(process.pid):
        raise RuntimeError('Owned child process group did not exit')


def memory_headroom(run):
    minima = {}
    with (Path(run) / 'monitor/resources.csv').open() as stream:
        for row in csv.DictReader(stream):
            try:
                index = str(int(float(row.get('gpu_index', 'nan'))))
            except (ValueError, OverflowError):
                continue
            if index not in ('0', '1'):
                continue
            try:
                free = (float(row['gpu_memory_total_mib']) - float(row['gpu_memory_used_mib'])) / 1024
            except (ValueError, TypeError):
                continue
            if math.isfinite(free):
                minima[index] = min(minima.get(index, float('inf')), free)
    if set(minima) != {'0', '1'}:
        raise ValueError('Missing measured free-memory traces for both physical GPUs')
    return min(minima.values())


def campaign_environment(root):
    env = os.environ.copy()
    paths = dict(TMPDIR=root/'tmp', XDG_CACHE_HOME=root/'cache', HF_HOME=root/'cache/huggingface',
        HF_HUB_CACHE=root/'cache/huggingface/hub', TORCH_HOME=root/'cache/torch',
        TRITON_CACHE_DIR=root/'cache/triton', TORCHINDUCTOR_CACHE_DIR=root/'cache/torchinductor',
        PIP_CACHE_DIR=root/'cache/pip', CUDA_CACHE_PATH=root/'cache/cuda')
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        env[name] = str(path)
    env.update(PYTHONPATH=str(REPO/'src'), TOKENIZERS_PARALLELISM='false', HF_HUB_DISABLE_IMPLICIT_TOKEN='1',
               HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1', OMP_NUM_THREADS='1', WANDB_DISABLED='true')
    return env


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = args.project_root.resolve()
        self.directory = args.campaign_dir.resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory/'campaign.lock').open('a+')
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.global_lock = (self.root/'campaign.lock').open('a+')
        try:
            fcntl.flock(self.global_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self.lock.close()
            self.global_lock.close()
            raise
        if (self.directory/'campaign.json').exists():
            raise FileExistsError('Campaign already has state; preserve it and choose a fresh --campaign-dir')
        if any(path.name != 'campaign.lock' for path in self.directory.iterdir()):
            raise FileExistsError('Campaign directory must be new and empty')
        self.env = campaign_environment(self.root)
        self.started = time.time()
        self.deadline = time.monotonic() + args.max_wall_seconds
        self.results = []
        self.state = dict(status='running', stage='validating', started_unix=self.started,
                          max_wall_seconds=args.max_wall_seconds, results=self.results,
                          main_lr_matrix_status='recommendation_pending_analysis',
                          selected_config=None, commands=[])
        self.write_state()

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise BudgetExpired('Campaign wall-time budget exhausted')
        return remaining

    def write_state(self):
        self.state['updated_unix'] = time.time()
        atomic_json(self.directory/'campaign.json', self.state)
        atomic_json(self.directory/'experiment_results.json', self.results)
        self.report()

    def report(self):
        lines = ['# Qwen ASR campaign', '', f"Status: {self.state['status']}; stage: {self.state['stage']}.", '',
            'Each benchmark starts from the same pinned base model. Training-update timing excludes final generation evaluation and checkpoint writes.', '',
            '| Experiment | GPUs | Status | Audio sec / wall sec | Peak project GiB | Min observed free GiB |',
            '|---|---:|---|---:|---:|---:|']
        for row in self.results:
            lines.append(f"| {row['name']} | {row['world_size']} | {row['status']} | {row.get('throughput', '')} | {row.get('peak_gib', '')} | {row.get('min_free_gib', '')} |")
        lines += ['', f"Selected dual-GPU configuration: {self.state.get('selected_config') or 'pending'}.", '',
            'A second learning-rate run is not launched. Its recommendation remains pending analysis of the first completed training and evaluation matrix.']
        if 'evaluations' in self.state:
            lines += ['', '| Evaluation | Baseline WER | Final WER | Final minus baseline (percentage points) |', '|---|---:|---:|---:|']
            for name, result in self.state['evaluations'].items():
                lines.append(f"| {name} | {100*result['baseline_wer']:.4f}% | {100*result['final_wer']:.4f}% | {100*(result['final_wer']-result['baseline_wer']):+.4f} |")
        if self.state.get('error'):
            lines += ['', 'Failure: ' + self.state['error']]
        (self.directory/'REPORT.md').write_text('\n'.join(lines)+'\n')

    def execute(self, name, command, run, gpus, projected_gib=0):
        self.remaining()
        for path, expected in {**self.state.get('manifests', {}), **self.state.get('source_sha256', {})}.items():
            if file_hash(path) != expected:
                raise ValueError(f'Campaign data invariant changed: {path}')
        check_space(self.root, 500, 300, int(projected_gib * 1024**3))
        run = Path(run)
        fingerprint = hashlib.sha256(json.dumps(dict(command=command,
            config_sha256=file_hash(command[command.index('--config')+1]) if '--config' in command else None), sort_keys=True).encode()).hexdigest()
        if run.exists():
            receipt = read_json(run/'stage.json')
            if receipt.get('status') == 'complete' and receipt.get('fingerprint') == fingerprint and receipt.get('exit_code') == 0:
                return receipt
            raise FileExistsError(f'Refusing to reuse unverified stage directory: {run}')
        run.mkdir(parents=True)
        self.state['stage'] = name
        record = dict(name=name, command=command, physical_gpus=gpus, run_dir=str(run), fingerprint=fingerprint,
                      started_unix=time.time(), status='running')
        self.state['commands'].append(record)
        self.write_state()
        env = dict(self.env, CUDA_VISIBLE_DEVICES=gpus)
        process = monitor = None
        with (run/'stdout.log').open('w') as output, (run/'monitor.log').open('w') as monitor_output:
            try:
                process = subprocess.Popen(command, cwd=REPO, env=env, stdout=output, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                monitor = subprocess.Popen([sys.executable, str(REPO/'scripts/monitor.py'), '--run-dir', str(run),
                    '--project-root', str(self.root), '--pid', str(process.pid), '--gpu', gpus,
                    '--interval', '10', '--reserve-gib', '500'], cwd=REPO, env=env,
                    stdout=monitor_output, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    self.remaining()
                    time.sleep(min(1, self.remaining()))
                process.wait()
                if group_alive(process.pid):
                    stop_owned(process)
                if monitor.poll() is not None and monitor.returncode != 0:
                    raise RuntimeError('Resource monitor failed')
                record.update(exit_code=process.returncode, status='complete' if process.returncode == 0 else 'failed')
            finally:
                if process is not None and (process.poll() is None or group_alive(process.pid)):
                    stop_owned(process)
                if monitor is not None and monitor.poll() is None:
                    stop_owned(monitor)
                record['ended_unix'] = time.time()
                if 'exit_code' not in record:
                    record.update(status='failed', exit_code=process.returncode if process else None)
                atomic_json(run/'stage.json', record)
                self.write_state()
        if record['exit_code'] != 0:
            tail = (run/'stdout.log').read_text(errors='replace')[-131072:]
            if name.startswith('E') and is_oom(tail):
                record['status'] = 'oom'
                atomic_json(run/'stage.json', record)
                return record
            raise RuntimeError(f'{name} failed, exit {record["exit_code"]}; see {run / "stdout.log"}')
        return record

    def derived_config(self, name, source, benchmark):
        if self.remaining() < 120:
            raise BudgetExpired('Less than two minutes remain before a new training stage')
        config = yaml.safe_load(source.read_text())
        if config.get('model_id') != self.model_id or config.get('revision') != self.revision or int(config.get('seed', 42)) != 42:
            raise ValueError('Candidate base-model revision or seed invariant differs')
        config.update(project_root=str(self.root), train_manifest=str(self.train_manifest),
                      eval_manifest=str(self.fixed_manifest), max_wall_seconds=max(1, int(self.remaining()-30)),
                      quota_gib=300, reserve_gib=500, learning_rate=1e-5, epochs=1)
        config.pop('max_steps', None)
        config.pop('limit_train', None)
        if benchmark:
            config.update(eval_steps=1000, save_steps=1000, warmup_ratio=.25)
        else:
            config.update(eval_limit=128)
        target = self.directory/'configs'/f'{name}.yaml'
        target.parent.mkdir(exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        return target

    def benchmark(self, name, source_name, world_size):
        source = REPO/'configs'/source_name
        config = self.derived_config(name, source, True)
        values = yaml.safe_load(config.read_text())
        if values['batch_size'] * values['gradient_accumulation_steps'] * world_size != 32:
            raise ValueError('Benchmark global batch invariant differs from 32')
        run = self.directory/name
        receipt = self.execute(name, train_command(sys.executable, config, run, world_size, True), run,
                               '1' if world_size == 1 else '0,1', projected_gib=36)
        if receipt['status'] == 'oom':
            result = dict(name=name, world_size=world_size, status='oom', valid=False, config_source=source_name)
        else:
            terminal = read_json(run/'status.json')
            if terminal.get('status') != 'complete' or int(terminal.get('step', -1)) != 40:
                if self.deadline - time.monotonic() < 60:
                    raise BudgetExpired('Benchmark reached the remaining campaign wall budget')
                raise RuntimeError(f'{name} did not complete exactly40 benchmark steps')
            summary = summarize_run(run, 10, 30)
            metadata = read_json(run/'run.json')
            if metadata.get('revision') != self.revision or metadata.get('effective_batch_size') != 32 or metadata.get('world_size') != world_size:
                raise ValueError('Benchmark model revision, topology or batch invariant differs')
            if any(metadata['config'].get(key) != expected for key, expected in
                   (('max_steps',40),('limit_train',2048),('eval_limit',8))):
                raise ValueError('Controlled benchmark step/subset invariants differ')
            for line in (run/'metrics.jsonl').read_text().splitlines():
                row = json.loads(line)
                if row.get('phase') == 'train':
                    for key in ('loss','grad_norm'):
                        if not math.isfinite(float(row[key])):
                            raise FloatingPointError(f'Nonfinite benchmark {key}')
            free = memory_headroom(run)
            peak = summary['cuda_peak_memory_gb']
            result = dict(name=name, world_size=world_size, status='measured', valid=True,
                config_source=source_name, throughput=summary['audio_seconds_per_wall_second'],
                peak_gib=peak, min_free_gib=free, parameters=int(metadata['parameters']), summary=summary)
            atomic_json(run/'benchmark_summary.json', summary)
        self.results.append(result)
        self.write_state()
        return result

    def evaluation(self, name, manifest, model, revision=None, existing=None):
        if existing is not None:
            result = validate_evaluation(existing, manifest, revision)
            if result.get('evaluation_batch_size') != self.args.eval_batch_size:
                raise ValueError('Reused baseline evaluation batch size differs or is missing')
            return result
        run = self.directory/'evaluation'/name
        output = run/'predictions.jsonl'
        command = [sys.executable, '-m', 'asr_lab.qwen_evaluate', '--model', str(model),
                   '--revision', self.revision, '--manifest', str(manifest), '--output', str(output),
                   '--all', '--batch-size', str(self.args.eval_batch_size)]
        self.execute('evaluate-'+name, command, run, '1')
        result = validate_evaluation(output.with_suffix('.metrics.json'), manifest, revision)
        if result.get('evaluation_batch_size') != self.args.eval_batch_size:
            raise ValueError('Evaluator did not record the requested batch size')
        return result

    def run(self):
        base = yaml.safe_load((REPO/'configs/qwen_main.yaml').read_text())
        self.revision, self.model_id = base['revision'], base['model_id']
        if len(self.revision) != 40:
            raise ValueError('Model revision must be a pinned commit')
        self.train_manifest = self.root/'data/manifests/train-clean-100.jsonl'
        self.fixed_manifest = self.root/'data/manifests/dev-clean-fixed128.jsonl'
        clean = self.root/'data/manifests/dev-clean.jsonl'
        other = self.root/'data/manifests/dev-other.jsonl'
        if len(manifest_ids(self.train_manifest)) < 2048 or len(manifest_ids(self.fixed_manifest)) != 128:
            raise ValueError('Campaign requires prepared train100 and exactly128 fixed development IDs')
        if set(manifest_ids(self.train_manifest)) & set(manifest_ids(clean)):
            raise ValueError('Train/dev manifest ID leakage')
        if not set(manifest_ids(self.fixed_manifest)) <= set(manifest_ids(clean)):
            raise ValueError('Fixed subset must belong to prepared dev-clean')
        fixed_baseline = validate_evaluation(self.args.baseline_fixed128, self.fixed_manifest, self.revision)
        if fixed_baseline.get('evaluation_batch_size') != self.args.eval_batch_size:
            raise ValueError('Fixed128 baseline evaluation batch size differs or is missing')
        self.state.update(model_id=self.model_id, model_revision=self.revision,
            source_sha256={str(REPO/'src/asr_lab'/name): file_hash(REPO/'src/asr_lab'/name) for name in
                           ('qwen_train.py','qwen_data.py','qwen_evaluate.py','benchmark.py')},
            manifests={str(path):file_hash(path) for path in (self.train_manifest,self.fixed_manifest,clean)},
            baseline_fixed128=str(self.args.baseline_fixed128))
        self.write_state()
        first = self.benchmark('E1-single-b2', 'qwen_single.yaml', 1)
        if first['status'] == 'oom':
            raise RuntimeError('Smallest per-GPU benchmark OOM; stop rather than try larger configurations')
        second = self.benchmark('E2-dual-b2', 'qwen_main.yaml', 2)
        third = self.benchmark('E3-dual-b4', 'qwen_batch4.yaml', 2) if allow_batch8(second) else None
        if third is not None and allow_batch8(third):
            self.benchmark('E4-dual-b8', 'qwen_batch8.yaml', 2)
        else:
            self.results.append(dict(name='E4-dual-b8', world_size=2, status='skipped_memory_guard', valid=False))
        chosen = choose_dual(self.results)
        self.state['selected_config'] = chosen['config_source']
        self.state['main_projected_peak_checkpoint_gib'] = checkpoint_projection_gib(chosen['parameters'], 3)
        self.write_state()
        baselines = dict(fixed128=fixed_baseline,
            dev_clean=self.evaluation('baseline-dev-clean', clean, self.model_id, self.revision, self.args.baseline_dev_clean))
        if other.exists():
            if set(manifest_ids(other)) & set(manifest_ids(self.train_manifest)):
                raise ValueError('Train/dev-other manifest ID leakage')
            self.state['manifests'][str(other)] = file_hash(other)
            self.write_state()
            baselines['dev_other'] = self.evaluation('baseline-dev-other', other, self.model_id, self.revision)
        main_config = self.derived_config('main-epoch1', REPO/'configs'/chosen['config_source'], False)
        main_run = self.directory/'main-epoch1'
        self.execute('main-epoch1', train_command(sys.executable, main_config, main_run, 2), main_run, '0,1', projected_gib=self.state['main_projected_peak_checkpoint_gib'])
        terminal = read_json(main_run/'status.json')
        if terminal.get('status') != 'complete':
            if self.deadline - time.monotonic() < 60:
                raise BudgetExpired('Formal training reached the remaining campaign wall budget')
            raise RuntimeError(f'Main training stopped before completion: {terminal.get("status")}')
        if float(terminal.get('epoch', 0)) < .999999 or terminal.get('completion_reason') != 'epochs':
            raise ValueError('Formal training lacks a verified completed epoch')
        metadata = read_json(main_run/'run.json')
        trained_config = metadata['config']
        if metadata.get('world_size') != 2 or metadata.get('revision') != self.revision or trained_config.get('limit_train') is not None or trained_config.get('max_steps',-1) not in (-1,None):
            raise ValueError('Formal training must be a fresh full-data dual-GPU epoch')
        checkpoint = Path(terminal['latest_checkpoint']).resolve()
        if not checkpoint.is_relative_to(main_run.resolve()) or not (checkpoint/'checkpoint.json').is_file() or not list(checkpoint.glob('*.safetensors')):
            raise ValueError('Final checkpoint is incomplete or outside its own run')
        self.state['final_checkpoint'] = str(checkpoint)
        self.state['evaluations'] = {}
        for name, manifest in (('fixed128',self.fixed_manifest), ('dev_clean',clean), ('dev_other',other)):
            if name not in baselines:
                continue
            result = self.evaluation('final-'+name, manifest, checkpoint)
            baseline = baselines[name]
            if result['reference_words'] != baseline['reference_words'] or result['selected_ids'] != baseline['selected_ids']:
                raise ValueError('Baseline/final evaluation denominators or ordering differ')
            self.state['evaluations'][name] = dict(baseline_wer=baseline['wer'], final_wer=result['wer'],
                examples=result['examples'], reference_words=result['reference_words'])
            self.write_state()
        self.state.update(status='complete', stage='evaluations_complete', completed_unix=time.time())
        self.write_state()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign-dir', required=True, type=Path)
    parser.add_argument('--project-root', type=Path, default=Path('/mnt/raid/asr-lab'))
    parser.add_argument('--baseline-fixed128', required=True, type=Path, help='Validated pinned-model fixed128 .metrics.json')
    parser.add_argument('--baseline-dev-clean', type=Path, help='Optional complete eligible-dev baseline .metrics.json')
    parser.add_argument('--max-wall-seconds', type=float, default=43200)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    args = parser.parse_args()
    if not 0 < args.max_wall_seconds <= 43200:
        parser.error('Campaign budget must be in (0, 43200] seconds')
    if args.eval_batch_size < 1:
        parser.error('--eval-batch-size must be positive')
    def interrupted(*_):
        raise KeyboardInterrupt('Campaign interrupted; cleaning up owned processes')
    signal.signal(signal.SIGTERM, interrupted)
    campaign = Campaign(args)
    try:
        campaign.run()
    except BaseException as error:
        campaign.state.update(status='budget_stopped' if isinstance(error, BudgetExpired) else ('interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'),
                              error=f'{type(error).__name__}: {error}')
        campaign.write_state()
        raise


if __name__ == '__main__':
    main()
