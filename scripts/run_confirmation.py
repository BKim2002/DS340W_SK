#!/usr/bin/env python3
"""Run the four predeclared original-LoRA confirmations sequentially in WSL."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from ds340w_repro.advance_fixed_cell import _verified_result
from ds340w_repro.build_matrix import build_confirmation_matrix
from ds340w_repro.config_schema import ResolvedConfig
from ds340w_repro.run_opendelta import enforce_compute_gate
from ds340w_repro.scaling_experiment import forecast, load

EXPERIMENTS = tuple(f'confirmation-r{rank}-seed{seed}' for rank in (8, 16) for seed in (17, 23))
STATUS = ROOT / 'artifacts/confirmation-status.json'


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def config_path(name: str) -> Path:
    if name not in EXPERIMENTS:
        raise ValueError('Only the four predeclared original confirmations are allowed')
    return ROOT / f'configs/resolved/confirmation/{name}.json'


def validate_plan() -> None:
    baseline = ResolvedConfig.from_json((ROOT / 'configs/resolved/baseline/baseline-r8-lr1e-4-eb16.json').read_text())
    expected = build_confirmation_matrix(Path(baseline.values['model_name_or_path']), Path('artifacts/runs'),
                                         baseline.values['per_device_train_batch_size'],
                                         baseline.values['per_device_eval_batch_size'], 16)
    for config in expected:
        path = config_path(config.experiment_id)
        actual = ResolvedConfig.from_json(path.read_text())
        if actual.sha256 != config.sha256:
            raise ValueError('Confirmation configuration differs from the predeclared original recipe')
        enforce_compute_gate(actual, ROOT / 'data/manifests/compute-gate.json', path)


def completed_training(name: str) -> bool:
    directory = ROOT / f'artifacts/runs/{name}'
    if not (directory / 'process-result.json').exists():
        if directory.exists() or (ROOT / f'runtime/{name}').exists():
            raise ValueError(f'Refusing to overwrite an existing incomplete run: {name}')
        return False
    config = ResolvedConfig.from_json(config_path(name).read_text())
    process = load(directory / 'process-result.json')
    state = load(directory / 'trainer_state.json')
    if (process.get('exit_code') != 0 or state.get('global_step') != 20000
            or process.get('resolved_config_sha256') != config.sha256):
        raise ValueError(f'Incomplete or mismatched training: {name}')
    return True


def adjusted_projection(base: dict, training_seconds: list[float], observer_seconds: float = 0) -> float:
    """Replace reserved confirmations with observed costs instead of counting twice."""
    p = base['projection']
    if len(training_seconds) > 4 or any(not math.isfinite(s) or s <= 0 for s in training_seconds):
        raise ValueError('Invalid completed confirmation costs')
    per_run = p['original_seconds_per_future_run']
    future = max([per_run, *training_seconds])
    return p['projected_seconds'] + p['reserve_multiplier'] * (
        sum(training_seconds) + (4-len(training_seconds))*future - 4*per_run
    ) + observer_seconds + 30*(4-len(training_seconds))


def budget() -> dict:
    base = forecast(ROOT)
    seconds, observers = [], 0.0
    for name in EXPERIMENTS:
        directory = ROOT / f'artifacts/runs/{name}'
        process_path = directory / 'process-result.json'
        if process_path.exists():
            # Failed attempts still consume compute and must not disappear.
            seconds.append(load(process_path)['wall_seconds'])
        manifest = ROOT / f'data/manifests/retained-{name}.json'
        if manifest.exists():
            result = _verified_result(manifest_path=manifest, config_path=config_path(name), expected_experiment_id=name)
            observers += result.get('observer_wall_seconds', 30)
    projected = adjusted_projection(base, seconds, observers)
    result = {'schema_version': 1, 'status': 'PASS' if projected <= 48*3600 else 'STOP',
              'projected_gpu_hours': projected/3600, 'cap_gpu_hours': 48,
              'confirmation_attempts_finished': len(seconds), 'base_expanded_plan': base,
              'config_sha256': [ResolvedConfig.from_json(config_path(n).read_text()).sha256 for n in EXPERIMENTS],
              'final_unseen_accessed': False}
    write(ROOT / 'artifacts/confirmation-compute-gate.json', result)
    return result


def publish(name: str) -> bool:
    run, retained = ROOT / f'artifacts/runs/{name}', ROOT / f'artifacts/retained/{name}'
    out = ROOT / f'data/results/{name}'
    out.mkdir(parents=True, exist_ok=True)
    for file in ('metrics.json', 'predictions.jsonl'):
        shutil.copy2(retained / file, out / file)
    for file in ('execution-tree.json', 'process-result.json', 'eval_results.json', 'train_results.json'):
        shutil.copy2(run / file, out / file)
    write(out / 'loss-history.json', {'log_history': load(run / 'trainer_state.json')['log_history']})
    manifest = f'data/manifests/retained-{name}.json'
    subprocess.run(['git', 'add', '--', manifest, str(out)], cwd=ROOT, check=True)
    if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode:
        subprocess.run(['git', 'commit', '-m', f'data: verify retained {name} result'], cwd=ROOT, check=True)
    # WSL Git has no stored HTTPS credentials; use the Windows credential helper.
    windows_root = subprocess.check_output(['wslpath', '-w', str(ROOT)], text=True).strip()
    pushed = subprocess.run(['/mnt/c/Program Files/Git/cmd/git.exe', '-C', windows_root,
                             'push', 'origin', 'HEAD'], cwd=ROOT)
    return pushed.returncode == 0


def queue() -> None:
    lock_path = ROOT / 'artifacts/confirmation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        # Acquire before writing status so a second controller cannot replace it.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done, pushed = [], True
        def status(state: str, **extra):
            write(STATUS, {'status': state, 'experiments': list(EXPERIMENTS), 'completed': done[:],
                           'pid': os.getpid(), 'updated_at_utc': datetime.now(timezone.utc).isoformat(), **extra})
        try:
            validate_plan()
            for name in EXPERIMENTS:
                gate = budget()
                if gate['status'] != 'PASS':
                    raise RuntimeError(f"48-hour cap conflict: {gate['projected_gpu_hours']:.2f} hours")
                manifest = ROOT / f'data/manifests/retained-{name}.json'
                status('running', active_experiment=name, phase='verification_or_training', projected_gpu_hours=gate['projected_gpu_hours'])
                if not manifest.exists():
                    if not completed_training(name):
                        logs = ROOT / 'artifacts/logs/confirmation'
                        logs.mkdir(parents=True, exist_ok=True)
                        command = [sys.executable, '-m', 'ds340w_repro.run_opendelta', '--project-root', str(ROOT),
                                   '--config', str(config_path(name)), '--runtime-dir', str(ROOT / f'runtime/{name}'),
                                   '--python', sys.executable]
                        with (logs / f'{name}.stdout.log').open('x') as out, (logs / f'{name}.stderr.log').open('x') as err:
                            subprocess.run(command, cwd=ROOT, stdout=out, stderr=err, check=True)
                    status('running', active_experiment=name, phase='collecting_and_reloading')
                    subprocess.run([sys.executable, 'scripts/collect_retained_result.py', '--config', str(config_path(name)),
                                    '--artifact-dir', f'artifacts/retained/{name}', '--result-manifest', str(manifest)],
                                   cwd=ROOT, check=True)
                _verified_result(manifest_path=manifest, config_path=config_path(name), expected_experiment_id=name)
                published = publish(name)
                pushed = pushed and published
                done.append(name)
                status('running', active_experiment=None, phase='between_runs', publication='pushed' if published else 'pending')
            gate = budget()
            status('complete' if pushed else 'complete_publication_pending', projected_gpu_hours=gate['projected_gpu_hours'])
        except Exception as error:
            status('failed', error=str(error))
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('budget', 'launch', 'queue'))
    args = parser.parse_args()
    for key in ('DS340W_PRIVATE_ROOT', 'DS340W_SST2_FINAL_FILE'):
        os.environ.pop(key, None)
    os.environ.update({'PYTHONPATH': str(ROOT/'src'), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
                       'HF_DATASETS_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_OFFLINE': '1'})
    if args.action == 'queue':
        queue()
        return
    validate_plan()
    gate = budget()
    if args.action == 'budget':
        print(json.dumps({k:v for k,v in gate.items() if k!='base_expanded_plan'}, indent=2))
        return
    if gate['status'] != 'PASS':
        raise RuntimeError('Confirmation plan exceeds the 48-hour cap')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
    log = ROOT / f'artifacts/confirmation-launch-{stamp}.log'
    with log.open('x') as stream:
        child = subprocess.Popen([sys.executable, __file__, 'queue'], cwd=ROOT,
                                 stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
    print(json.dumps({'pid': child.pid, 'log': str(log), 'projected_gpu_hours': gate['projected_gpu_hours']}))


if __name__ == '__main__':
    main()
