"""Week 11 scaling sweep and its separate measured compute authorization."""

from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
import math
from pathlib import Path
import re

from ds340w_repro.config_schema import ResolvedConfig, validate_nonfinal_config
from ds340w_repro.preflight import ComputeGateError

RANKS = (2, 4, 8, 16)
SERIES = "patches/scaling-series.json"
AUTHORITY = "approved Week 11 exploratory scaling design"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def original_config(root: Path, rank: int) -> ResolvedConfig:
    relative = "baseline/baseline-r8-lr1e-4-eb16" if rank == 8 else f"rank/rank-r{rank}"
    return ResolvedConfig.from_json((root / f"configs/resolved/{relative}.json").read_text())


def build_scaling_config(root: Path, rank: int, *, probe: str | None = None,
                         mode: str = "sqrt_rank") -> ResolvedConfig:
    if rank not in RANKS or mode not in ("original", "sqrt_rank"):
        raise ValueError("Invalid rank or mode")
    suffix = f"{probe}-{mode}-r{rank}" if probe else f"r{rank}-seed42"
    config = replace(original_config(root, rank), experiment_id=f"scaling-{suffix}")
    for key, value in {
        "output_dir": f"artifacts/runs/{config.experiment_id}",
        "lora_scaling_mode": mode,
        "patch_series": SERIES,
        "record_adapter_gradients": True,
    }.items():
        config = config.with_value(key, value, AUTHORITY)
    if probe == "smoke":
        changes = {"max_steps": 2, "eval_steps": 1, "save_steps": 1,
                   "max_train_samples": 32, "max_val_samples": 1, "logging_steps": 1}
    elif probe == "pilot":
        changes = {"max_steps": 200}
    elif probe is None:
        changes = {}
    else:
        raise ValueError("Unknown bounded probe")
    for key, value in changes.items():
        config = config.with_value(key, value, AUTHORITY)
    validate_nonfinal_config(config)
    return config


def enforce_scaling_gate(config: ResolvedConfig, gate_path: Path) -> None:
    """Only exact bounded probes or hash-authorized full configurations may run."""
    root = gate_path.resolve().parents[2]
    probe = re.fullmatch(r"scaling-(smoke|pilot)-(original|sqrt_rank)-r(2|4|8|16)", config.experiment_id)
    full = re.fullmatch(r"scaling-r(2|4|8|16)-seed42", config.experiment_id)
    if probe:
        kind, mode, rank = probe.groups()
        expected = build_scaling_config(root, int(rank), probe=kind, mode=mode)
        if config.sha256 != expected.sha256:
            raise ComputeGateError("Scaling probe differs from its bounded declaration")
        return
    if not full:
        raise ComputeGateError("Unsupported scaling experiment")
    expected = build_scaling_config(root, int(full.group(1)))
    if config.sha256 != expected.sha256:
        raise ComputeGateError("Scaling run differs from its locked configuration")
    gate = load(gate_path)
    hashes = [build_scaling_config(root, rank).sha256 for rank in RANKS]
    projection = gate.get("projection", {})
    if (gate.get("status") != "PASS" or gate.get("config_sha256") != hashes
            or gate.get("patch_series_sha256") != digest(root / SERIES)
            or projection.get("total_training_runs") != 12
            or projection.get("deferred_original_confirmation_runs") != 4
            or not 0 < projection.get("projected_seconds", math.inf) <= 48 * 3600):
        raise ComputeGateError("The expanded scaling plan lacks a passing 48-hour gate")
    for evidence in gate.get("pilot_evidence", []):
        path = root / evidence["path"]
        if digest(path) != evidence["sha256"]:
            raise ComputeGateError("Scaling pilot evidence has changed")
    if len(gate.get("pilot_evidence", [])) < 4:
        raise ComputeGateError("Scaling pilot evidence is incomplete")


def prepare_configs(root: Path) -> list[Path]:
    configs = [build_scaling_config(root, rank) for rank in RANKS]
    configs += [build_scaling_config(root, 16, probe="pilot", mode=mode)
                for mode in ("original", "sqrt_rank")]
    configs += [build_scaling_config(root, 2, probe="smoke")]
    paths = []
    for config in configs:
        path = root / f"configs/resolved/scaling/{config.experiment_id}.json"
        if path.exists() and path.read_text() != config.to_json():
            raise FileExistsError(f"Refusing to replace a differing configuration: {path}")
        config.write(path)
        paths.append(path)
    return paths


def forecast(root: Path) -> dict:
    """Use matched measured pilots, retaining four future original confirmations."""
    from ds340w_repro.advance_fixed_cell import _verified_result

    completed = []
    for rank in RANKS:
        name = "baseline-r8-lr1e-4-eb16" if rank == 8 else f"rank-r{rank}"
        relative = "baseline" if rank == 8 else "rank"
        result = _verified_result(
            manifest_path=root / f"data/manifests/retained-{name}.json",
            config_path=root / f"configs/resolved/{relative}/{name}.json",
            expected_experiment_id=name,
        )
        completed.append(result["original_process_wall_seconds"])
    gate = load(root / "data/manifests/compute-gate.json")
    evidence = []
    run_seconds = {}
    attempts_path = root / "artifacts/scaling-execution-attempts.jsonl"
    attempts = [json.loads(line) for line in attempts_path.read_text().splitlines()] if attempts_path.exists() else []
    probe_seconds = sum(row["wall_seconds"] for row in attempts
                        if row["experiment_id"].startswith(("scaling-pilot-", "scaling-smoke-")))
    if not probe_seconds:
        raise ValueError("Probe attempt ledger is missing")
    for mode in ("original", "sqrt_rank"):
        directory = root / f"artifacts/runs/scaling-pilot-{mode}-r16"
        process_path, timing_path = directory / "process-result.json", directory / "preflight-timing.json"
        process, timing = load(process_path), load(timing_path)
        if process["exit_code"] != 0 or timing["completed_optimizer_steps"] != 200:
            raise ValueError("Matched pilot did not complete")
        measurements = timing["evaluation_measurements_seconds"]
        if len(measurements) < 2:
            raise ValueError("Pilot lacks full Test evaluations")
        seconds_per_step = timing["non_evaluation_seconds_steps_21_to_end"] / 180
        evaluation_seconds = max(measurements)
        startup = max(0, process["wall_seconds"] - timing["elapsed_seconds_steps_21_to_end"])
        run_seconds[mode] = 20_000 * seconds_per_step + 200 * evaluation_seconds + startup
        for path in (process_path, timing_path, directory / "reload-verification.json"):
            if path.name == "reload-verification.json" and load(path).get("passed") is not True:
                raise ValueError("Pilot checkpoint reload did not pass")
            evidence.append({"path": path.relative_to(root).as_posix(), "sha256": digest(path)})
    original_per_run = max(run_seconds["original"], max(completed))
    modified_per_run = run_seconds["sqrt_rank"]
    modified_completed = []
    completed_observer_seconds = 0.0
    for rank in RANKS:
        name = f"scaling-r{rank}-seed42"
        path = root / f"data/manifests/retained-{name}.json"
        if path.exists():
            result = _verified_result(manifest_path=path,
                                      config_path=root / f"configs/resolved/scaling/{name}.json",
                                      expected_experiment_id=name)
            modified_completed.append(result["original_process_wall_seconds"])
            completed_observer_seconds += result.get("observer_wall_seconds", 30.0)
    verification_seconds = sum(load(path).get("wall_seconds", 0)
                               for path in (root / "artifacts/runs").glob("scaling-*/reload-verification.json"))
    old_preflight = gate["projection"]["actual_preflight_seconds"]
    historical_probes = []
    for path in (root / "artifacts").rglob("process-result.json"):
        relative = path.relative_to(root / "artifacts")
        if relative.parts[0] in ("runs", "preflight-2026-10-08-fixed-cell", "fixed-cell-smoke-2026-10-08"):
            continue
        value = load(path)
        seconds = value.get("wall_seconds")
        if type(seconds) in (int, float) and math.isfinite(seconds) and seconds >= 0:
            historical_probes.append({"path": path.relative_to(root).as_posix(),
                                      "sha256": digest(path), "wall_seconds": seconds})
    historical_probe_seconds = sum(item["wall_seconds"] for item in historical_probes)
    # Earlier ad hoc demo/reload calls have no elapsed-time ledger. Reserve ten
    # minutes for them explicitly instead of claiming they cost zero seconds.
    unmetered_history_allowance = 600.0
    final_evaluations = 2 * max(
        load(root / f"artifacts/runs/scaling-pilot-{mode}-r16/preflight-timing.json")["evaluation_measurements_seconds"][-1]
        for mode in ("original", "sqrt_rank")
    )
    projected = old_preflight + historical_probe_seconds + unmetered_history_allowance + probe_seconds + verification_seconds + completed_observer_seconds + 1.2 * (
        sum(completed) + sum(modified_completed) + 4 * original_per_run
        + (4 - len(modified_completed)) * (modified_per_run + 30.0) + final_evaluations
    )
    return {
        "schema_version": 1, "assessment_kind": "Week-11-expanded-scaling-plan",
        "status": "PASS" if projected <= 48 * 3600 else "STOP",
        "config_sha256": [build_scaling_config(root, rank).sha256 for rank in RANKS],
        "patch_series_sha256": digest(root / SERIES), "pilot_evidence": evidence,
        "historical_probe_evidence": historical_probes,
        "projection": {"total_training_runs": 12, "completed_original_runs": 4,
                       "new_modified_runs": 4, "deferred_original_confirmation_runs": 4,
                       "completed_modified_runs": len(modified_completed),
                       "remaining_modified_runs": 4-len(modified_completed),
                       "completed_training_seconds": sum(completed),
                       "previous_preflight_seconds": old_preflight, "new_probe_seconds": probe_seconds,
                       "additional_historical_probe_seconds": historical_probe_seconds,
                       "unmetered_historical_evaluation_allowance_seconds": unmetered_history_allowance,
                       "completed_modified_observer_seconds": completed_observer_seconds,
                       "future_reload_seconds_per_modified_run": 30.0,
                       "reload_verification_seconds": verification_seconds,
                       "original_seconds_per_future_run": original_per_run,
                       "modified_seconds_per_future_run": modified_per_run,
                       "reserve_multiplier": 1.2, "projected_seconds": projected,
                       "projected_gpu_hours": projected / 3600, "cap_gpu_hours": 48},
        "final_assessment": "Original rank-8 and selected original-rank seed-42 checkpoints remain the declared final candidates; no final Unseen access.",
    }
