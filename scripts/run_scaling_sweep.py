#!/usr/bin/env python3
"""Prepare, verify, budget, and sequentially run the Week 11 scaling sweep."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ds340w_repro.scaling_experiment import RANKS, forecast, load, prepare_configs
from ds340w_repro.config_schema import ResolvedConfig


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify_probe(experiment_id: str) -> dict:
    from collect_retained_result import _compare_saved_tensors, _predict_test
    from ds340w_repro.run_opendelta import verify_course_data_binding, verify_model_snapshot_binding
    started = time.perf_counter()
    config = ResolvedConfig.from_json((ROOT / f"configs/resolved/scaling/{experiment_id}.json").read_text())
    directory = ROOT / config.values["output_dir"]
    if load(directory / "process-result.json")["exit_code"] != 0:
        raise ValueError("Probe training failed")
    state = load(directory / "trainer_state.json")
    checkpoint = Path(state["best_model_checkpoint"]).resolve(strict=True)
    _compare_saved_tensors(checkpoint, directory)
    expected_mode = config.values["lora_scaling_mode"]
    if load(checkpoint / "delta-config.json")["lora_scaling_mode"] != expected_mode:
        raise ValueError("Native checkpoint lost its scaling mode")
    runtime = Path(load(directory / "execution-tree.json")["root"])
    sys.path.insert(0, str(runtime))
    cache = Path.home() / ".cache/ds340w"
    verify_model_snapshot_binding(config, ROOT, cache)
    _, _, test_path = verify_course_data_binding(config, ROOT, cache)
    if config.values["max_val_samples"] != 436:
        result = {"passed": True, "scaling_mode": expected_mode,
                  "native_and_delta_tensors_identical": True, "smoke_only": True}
    else:
        ids, labels, predictions, loss = _predict_test(output_dir=directory, config=config, test_path=test_path)
        correct = sum(a == b for a, b in zip(labels, predictions))
        upstream = load(directory / "eval_results.json")
        if not math.isclose(100 * correct / 436, upstream["eval_accuracy"], abs_tol=1e-6) or abs(loss-upstream["eval_loss"]) > 1e-4:
            raise ValueError("Saved-delta evaluation differs from the original runner")
        with tempfile.TemporaryDirectory(prefix="ds340w-scaling-native-") as temp:
            delta = Path(temp) / "delta"
            delta.mkdir()
            shutil.copy2(checkpoint / "delta-config.json", delta / "config.json")
            shutil.copy2(checkpoint / "pytorch_model.bin", delta / "pytorch_model.bin")
            second = _predict_test(output_dir=Path(temp), config=config, test_path=test_path)
        if (ids, labels, predictions) != second[:3] or abs(loss - second[3]) > 1e-7:
            raise ValueError("Native checkpoint reload changed predictions")
        result = {"passed": True, "scaling_mode": expected_mode, "correct_count": correct,
                  "test_rows": 436, "eval_loss": loss, "all_predictions_match": True,
                  "native_and_delta_tensors_identical": True}
    result["wall_seconds"] = time.perf_counter() - started
    write(directory / "reload-verification.json", result)
    return result


def budget() -> dict:
    result = forecast(ROOT)
    write(ROOT / "data/manifests/scaling-compute-gate.json", result)
    return result


def run_one(experiment_id: str) -> None:
    log_dir = ROOT / "artifacts/logs/scaling"
    log_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "ds340w_repro.run_opendelta", "--project-root", str(ROOT),
               "--config", str(ROOT / f"configs/resolved/scaling/{experiment_id}.json"),
               "--runtime-dir", str(ROOT / f"runtime/{experiment_id}"), "--python", sys.executable]
    started = time.perf_counter()
    returncode = None
    try:
        with (log_dir / f"{experiment_id}.stdout.log").open("x") as out, (log_dir / f"{experiment_id}.stderr.log").open("x") as err:
            completed = subprocess.run(command, cwd=ROOT, stdout=out, stderr=err)
            returncode = completed.returncode
            completed.check_returncode()
    finally:
        with (ROOT / "artifacts/scaling-execution-attempts.jsonl").open("a") as stream:
            stream.write(json.dumps({"experiment_id": experiment_id, "exit_code": returncode,
                                     "wall_seconds": time.perf_counter()-started}) + "\n")


def completed_training(experiment_id: str) -> bool:
    """Recover a completed run without invoking training a second time."""
    directory = ROOT / f"artifacts/runs/{experiment_id}"
    process_path = directory / "process-result.json"
    if not process_path.exists():
        return False
    config = ResolvedConfig.from_json((ROOT / f"configs/resolved/scaling/{experiment_id}.json").read_text())
    process = load(process_path)
    state = load(directory / "trainer_state.json")
    if (process.get("exit_code") != 0 or state.get("global_step") != 20_000
            or process.get("resolved_config_sha256") != config.sha256):
        raise ValueError(f"Existing run is incomplete or its configuration differs: {experiment_id}")
    return True


def queue() -> None:
    job = ROOT / "artifacts/scaling-sweep-status.json"
    try:
        for rank in RANKS:
            experiment_id = f"scaling-r{rank}-seed42"
            manifest = f"data/manifests/retained-{experiment_id}.json"
            gate = budget()
            if gate["status"] != "PASS":
                raise RuntimeError(f"48-hour cap conflict: {gate['projection']['projected_gpu_hours']:.2f} hours")
            # The budget verifier has already checked any completed result.
            if (ROOT / manifest).exists():
                print(f"Reusing verified completed result: {experiment_id}", flush=True)
                continue
            write(job, {"status": "running", "active_experiment": experiment_id,
                        "pid": os.getpid(), "updated_at_utc": datetime.now(timezone.utc).isoformat()})
            if completed_training(experiment_id):
                print(f"Collecting already completed training: {experiment_id}", flush=True)
            else:
                run_one(experiment_id)
            subprocess.run([sys.executable, "scripts/collect_retained_result.py", "--config",
                            f"configs/resolved/scaling/{experiment_id}.json", "--artifact-dir",
                            f"artifacts/retained/{experiment_id}", "--result-manifest", manifest],
                           cwd=ROOT, check=True)
            # Publish lightweight evidence as result data, keeping checkpoints,
            # large logs, and course documents outside the Git repository.
            data_dir = ROOT / f"data/results/{experiment_id}"
            data_dir.mkdir(parents=True, exist_ok=False)
            run_dir = ROOT / f"artifacts/runs/{experiment_id}"
            retained = ROOT / f"artifacts/retained/{experiment_id}"
            for name in ("metrics.json", "predictions.jsonl"):
                shutil.copy2(retained / name, data_dir / name)
            for name in ("adapter-gradients.jsonl", "parameter-counts.json", "cuda-memory.json",
                         "execution-tree.json", "process-result.json", "delta-config.json"):
                shutil.copy2(run_dir / name, data_dir / name)
            write(data_dir / "loss-history.json", {"log_history": load(run_dir / "trainer_state.json")["log_history"]})
            subprocess.run(["git", "add", "--", manifest, str(data_dir)], cwd=ROOT, check=True)
            subprocess.run(["git", "commit", "-m", f"data: verify retained {experiment_id} result"], cwd=ROOT, check=True)
        subprocess.run(["git", "push", "origin", "HEAD"], cwd=ROOT, check=True)
        write(job, {"status": "complete", "experiments": [f"scaling-r{r}-seed42" for r in RANKS],
                    "updated_at_utc": datetime.now(timezone.utc).isoformat()})
    except Exception as error:
        write(job, {"status": "failed", "error": str(error), "pid": os.getpid(),
                    "updated_at_utc": datetime.now(timezone.utc).isoformat()})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "probe", "verify", "budget", "queue", "launch"))
    parser.add_argument("--experiment-id")
    args = parser.parse_args()
    for name in ("DS340W_PRIVATE_ROOT", "DS340W_SST2_FINAL_FILE"):
        os.environ.pop(name, None)
    os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    if args.action == "prepare":
        print("\n".join(str(p) for p in prepare_configs(ROOT)))
    elif args.action == "probe":
        if not args.experiment_id or not args.experiment_id.startswith(("scaling-pilot-", "scaling-smoke-")):
            raise ValueError("probe requires an exact bounded probe ID")
        run_one(args.experiment_id)
        subprocess.run([sys.executable, __file__, "verify", "--experiment-id", args.experiment_id], check=True, cwd=ROOT)
    elif args.action == "verify":
        print(json.dumps(verify_probe(args.experiment_id), indent=2))
    elif args.action == "budget":
        print(json.dumps(budget(), indent=2))
    elif args.action == "queue":
        queue()
    elif args.action == "launch":
        result = budget()
        if result["status"] != "PASS":
            raise RuntimeError("Expanded compute projection exceeds 48 hours")
        path = ROOT / "artifacts/scaling-sweep-launch.log"
        if path.exists():
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            path = path.with_name(f"scaling-sweep-launch-{stamp}.log")
        with path.open("x") as stream:
            child = subprocess.Popen([sys.executable, __file__, "queue"], cwd=ROOT,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({"pid": child.pid, "log": str(path)}))


if __name__ == "__main__":
    main()
