"""Calibrate evaluation/training batches, run the pilot, and build the gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Callable

from ds340w_repro.build_matrix import (
    build_baseline_matrix,
    build_evaluation_calibration_config,
    build_pilot_config,
    build_rank_matrix,
    build_training_calibration_config,
)
from ds340w_repro.config_schema import ResolvedConfig
from ds340w_repro.preflight import (
    CALIBRATION_BATCH_CANDIDATES,
    DEFAULT_RETAINED_RUN_PLAN,
    EVALUATION_BATCH_CANDIDATES,
    CalibrationAttempt,
    EvaluationBatchAttempt,
    PilotEvidence,
    RetainedRunPlan,
    RetainedRunStage,
    authorize_rank_stage,
    build_compute_gate_manifest,
    load_week1_evidence,
    require_retained_training_authorization,
)


class PreflightExecutionError(RuntimeError):
    """Raised when measured preflight evidence is missing or malformed."""


@dataclass(frozen=True)
class ObservedRun:
    """Fresh-process observations collected outside the authors' runner."""

    returncode: int
    started_at_utc: str
    finished_at_utc: str
    wall_seconds: float
    minimum_free_memory_mib: float
    oom: bool
    output_dir: Path


def dry_run_plan(retained_run_plan: RetainedRunPlan) -> dict[str, Any]:
    """Return the bounded work declaration emitted by ``--dry-run``."""

    return {
        "evaluation_batch_candidates": list(EVALUATION_BATCH_CANDIDATES),
        "training_microbatch_candidates": list(CALIBRATION_BATCH_CANDIDATES),
        "pilot": {
            "effective_batch_size": 32,
            "evaluations_at_steps": [100, 200],
            "learning_rate": 1e-3,
            "optimizer_steps": 200,
            "rank": 16,
            "warmup_steps_excluded": 20,
        },
        "final_unseen_access": False,
        "retained_run_plan": retained_run_plan.to_dict(),
    }


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PreflightExecutionError(f"Missing or invalid evidence: {path}") from error
    if not isinstance(payload, dict):
        raise PreflightExecutionError(f"Evidence is not a JSON object: {path}")
    return payload


def _sample_free_memory_mib() -> float:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
            "--id=0",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    if completed.returncode != 0:
        raise PreflightExecutionError("nvidia-smi could not sample free GPU memory")
    try:
        value = float(completed.stdout.strip().splitlines()[0])
    except (IndexError, ValueError) as error:
        raise PreflightExecutionError("nvidia-smi returned invalid free-memory data") from error
    if not math.isfinite(value) or value < 0:
        raise PreflightExecutionError("nvidia-smi returned invalid free-memory data")
    return value


def _execute_fresh_process(
    *,
    config: ResolvedConfig,
    project_root: Path,
    workspace: Path,
    python_executable: Path,
) -> ObservedRun:
    config_dir = workspace / "configs"
    runtime_dir = workspace / "runtime" / config.experiment_id
    log_dir = workspace / "logs"
    config_path = config_dir / f"{config.experiment_id}.json"
    stdout_path = log_dir / f"{config.experiment_id}.stdout.log"
    stderr_path = log_dir / f"{config.experiment_id}.stderr.log"
    for path in (config_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)
    if runtime_dir.exists():
        raise PreflightExecutionError(
            f"Fresh runtime directory already exists: {runtime_dir}"
        )
    config.write(config_path)
    command = [
        str(python_executable),
        "-m",
        "ds340w_repro.run_opendelta",
        "--project-root",
        str(project_root),
        "--config",
        str(config_path),
        "--runtime-dir",
        str(runtime_dir),
        "--python",
        str(python_executable),
    ]
    started_at = datetime.now(timezone.utc)
    started_clock = time.perf_counter()
    minimum_free = _sample_free_memory_mib()
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, stderr_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as stderr:
        process = subprocess.Popen(
            command,
            cwd=project_root,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
        while process.poll() is None:
            minimum_free = min(minimum_free, _sample_free_memory_mib())
            time.sleep(0.25)
        returncode = process.wait()
    minimum_free = min(minimum_free, _sample_free_memory_mib())
    finished_at = datetime.now(timezone.utc)
    combined_log = (
        stdout_path.read_text(encoding="utf-8", errors="replace")
        + "\n"
        + stderr_path.read_text(encoding="utf-8", errors="replace")
    ).lower()
    oom = "out of memory" in combined_log or "cuda error: out of memory" in combined_log
    return ObservedRun(
        returncode=returncode,
        started_at_utc=started_at.isoformat(),
        finished_at_utc=finished_at.isoformat(),
        wall_seconds=time.perf_counter() - started_clock,
        minimum_free_memory_mib=minimum_free,
        oom=oom,
        output_dir=Path(str(config.values["output_dir"])),
    )


def _evaluation_evidence(
    config: ResolvedConfig, observed: ObservedRun
) -> EvaluationBatchAttempt:
    metrics_path = observed.output_dir / "eval_results.json"
    evaluated_samples = 0
    if observed.returncode == 0 and metrics_path.is_file():
        metrics = _read_json_object(metrics_path)
        runtime = metrics.get("eval_runtime")
        samples_per_second = metrics.get("eval_samples_per_second")
        if (
            type(runtime) in (int, float)
            and math.isfinite(runtime)
            and runtime >= 0
            and type(samples_per_second) in (int, float)
            and math.isfinite(samples_per_second)
            and samples_per_second >= 0
        ):
            evaluated_samples = round(runtime * samples_per_second)
    return EvaluationBatchAttempt(
        evaluation_batch_size=config.values["per_device_eval_batch_size"],
        config_sha256=config.sha256,
        started_at_utc=observed.started_at_utc,
        finished_at_utc=observed.finished_at_utc,
        fresh_process=True,
        evaluated_samples=evaluated_samples,
        exit_status=observed.returncode,
        oom=observed.oom,
        minimum_free_memory_mib=observed.minimum_free_memory_mib,
    )


def _training_evidence(
    config: ResolvedConfig, observed: ObservedRun
) -> CalibrationAttempt:
    completed_steps = 0
    state_path = observed.output_dir / "trainer_state.json"
    if observed.returncode == 0 and state_path.is_file():
        state = _read_json_object(state_path)
        global_step = state.get("global_step")
        if type(global_step) is int and global_step >= 0:
            completed_steps = global_step
    return CalibrationAttempt(
        physical_batch_size=config.values["per_device_train_batch_size"],
        config_sha256=config.sha256,
        started_at_utc=observed.started_at_utc,
        finished_at_utc=observed.finished_at_utc,
        fresh_process=True,
        rank=config.values["lora_r"],
        lora_alpha=config.values["lora_alpha"],
        effective_batch_size=config.values["effective_batch_size"],
        target_optimizer_steps=10,
        completed_optimizer_steps=completed_steps,
        exit_status=observed.returncode,
        oom=observed.oom,
        minimum_free_memory_mib=observed.minimum_free_memory_mib,
    )


def _pilot_evidence(config: ResolvedConfig, observed: ObservedRun) -> PilotEvidence:
    timing = _read_json_object(observed.output_dir / "preflight-timing.json")
    evaluations = timing.get("evaluation_measurements_seconds")
    non_evaluation = timing.get("non_evaluation_seconds_steps_21_to_end")
    if (
        observed.returncode != 0
        or timing.get("schema_version") != 1
        or timing.get("completed_optimizer_steps") != 200
        or timing.get("warmup_steps_excluded") != 20
        or not isinstance(evaluations, list)
        or len(evaluations) != 2
        or any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or value < 0
            for value in evaluations
        )
        or type(non_evaluation) not in (int, float)
        or not math.isfinite(non_evaluation)
        or non_evaluation < 0
    ):
        raise PreflightExecutionError("Pilot timing evidence is incomplete")
    return PilotEvidence(
        config_sha256=config.sha256,
        started_at_utc=observed.started_at_utc,
        finished_at_utc=observed.finished_at_utc,
        fresh_process=True,
        physical_batch_size=config.values["per_device_train_batch_size"],
        evaluation_batch_size=config.values["per_device_eval_batch_size"],
        rank=config.values["lora_r"],
        lora_alpha=config.values["lora_alpha"],
        effective_batch_size=config.values["effective_batch_size"],
        learning_rate=config.values["learning_rate"],
        optimizer_steps=200,
        warmup_steps_excluded=20,
        non_evaluation_seconds_steps_21_to_200=float(non_evaluation),
        full_test_evaluation_seconds=(float(evaluations[0]), float(evaluations[1])),
        do_test=config.values["do_test"],
        exit_status=observed.returncode,
    )


def _retained_plan(confirmation_runs: int) -> RetainedRunPlan:
    if confirmation_runs < 0:
        raise ValueError("Confirmation run count must be non-negative")
    stages = list(DEFAULT_RETAINED_RUN_PLAN.stages)
    if confirmation_runs:
        stages.append(
            RetainedRunStage("confirmation", training_runs=confirmation_runs)
        )
    return RetainedRunPlan(
        stages=tuple(stages),
        final_evaluations=DEFAULT_RETAINED_RUN_PLAN.final_evaluations,
    )


def run_preflight(
    *,
    project_root: Path,
    model_path: Path,
    workspace: Path,
    python_executable: Path,
    gate_manifest_path: Path,
    resolved_config_root: Path,
    retained_output_root: Path,
    retained_run_plan: RetainedRunPlan,
    execute_config: Callable[..., ObservedRun] = _execute_fresh_process,
) -> dict[str, Any]:
    """Run all preflight phases sequentially and write baseline authorization."""

    project_root = project_root.resolve(strict=True)
    workspace = workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    hardware, smoke = load_week1_evidence(
        project_root / "data/manifests/environment-lock.json",
        project_root / "data/manifests/week1-smoke.json",
    )
    preflight_started = time.perf_counter()
    evaluation_attempts: list[EvaluationBatchAttempt] = []
    selected_evaluation_batch: int | None = None
    for candidate in EVALUATION_BATCH_CANDIDATES:
        config = build_evaluation_calibration_config(
            model_path,
            workspace / "runs" / f"preflight-eval-eb{candidate}",
            candidate,
        )
        attempt = _evaluation_evidence(
            config,
            execute_config(
                config=config,
                project_root=project_root,
                workspace=workspace,
                python_executable=python_executable,
            ),
        )
        evaluation_attempts.append(attempt)
        if attempt.passed:
            selected_evaluation_batch = candidate
            break

    calibration_attempts: list[CalibrationAttempt] = []
    selected_physical_batch: int | None = None
    if selected_evaluation_batch is not None:
        for candidate in CALIBRATION_BATCH_CANDIDATES:
            config = build_training_calibration_config(
                model_path,
                workspace
                / "runs"
                / f"preflight-train-pb{candidate}-eb{selected_evaluation_batch}",
                candidate,
                selected_evaluation_batch,
            )
            attempt = _training_evidence(
                config,
                execute_config(
                    config=config,
                    project_root=project_root,
                    workspace=workspace,
                    python_executable=python_executable,
                ),
            )
            calibration_attempts.append(attempt)
            if attempt.passed:
                selected_physical_batch = candidate
                break

    pilot: PilotEvidence | None = None
    if selected_evaluation_batch is not None and selected_physical_batch is not None:
        pilot_config = build_pilot_config(
            model_path,
            workspace
            / "runs"
            / (
                f"preflight-pilot-r16-pb{selected_physical_batch}-"
                f"eb{selected_evaluation_batch}"
            ),
            selected_physical_batch,
            selected_evaluation_batch,
        )
        pilot = _pilot_evidence(
            pilot_config,
            execute_config(
                config=pilot_config,
                project_root=project_root,
                workspace=workspace,
                python_executable=python_executable,
            ),
        )

    baseline_configs: list[ResolvedConfig] = []
    if selected_evaluation_batch is not None and selected_physical_batch is not None:
        baseline_configs = build_baseline_matrix(
            model_path,
            retained_output_root,
            selected_physical_batch,
            selected_evaluation_batch,
        )
    manifest = build_compute_gate_manifest(
        hardware=hardware,
        smoke=smoke,
        evaluation_batch_attempts=evaluation_attempts,
        calibration_attempts=calibration_attempts,
        pilot=pilot,
        actual_preflight_seconds=(
            smoke.wall_seconds + time.perf_counter() - preflight_started
        ),
        baseline_config_sha256=tuple(config.sha256 for config in baseline_configs),
        retained_run_plan=retained_run_plan,
    )
    if baseline_configs:
        for config in baseline_configs:
            config.write(resolved_config_root / "baseline" / f"{config.experiment_id}.json")
    gate_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    gate_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def authorize_rank_configs(
    *,
    gate_manifest_path: Path,
    selected_baseline_config_path: Path,
    rank_eight_checkpoint: Path,
    resolved_config_root: Path,
    retained_output_root: Path,
) -> dict[str, Any]:
    """Derive and authorize ranks 2/4/16 from one authorized baseline config."""

    manifest = _read_json_object(gate_manifest_path)
    selected = ResolvedConfig.from_json(
        selected_baseline_config_path.read_text(encoding="utf-8")
    )
    require_retained_training_authorization(
        manifest,
        stage="baseline",
        resolved_config_sha256=selected.sha256,
    )
    if not selected.experiment_id.startswith("baseline-r8-"):
        raise PreflightExecutionError(
            "Rank authorization requires an authorized rank-8 baseline config"
        )
    rank_matrix = build_rank_matrix(
        Path(str(selected.values["model_name_or_path"])),
        retained_output_root,
        selected.values["per_device_train_batch_size"],
        selected.values["per_device_eval_batch_size"],
        selected.values["learning_rate"],
        selected.values["effective_batch_size"],
        rank_eight_checkpoint,
    )
    trained_rank_configs = tuple(
        config for config in rank_matrix if config.values["lora_r"] != 8
    )
    for config in trained_rank_configs:
        config.write(resolved_config_root / "rank" / f"{config.experiment_id}.json")
    updated = authorize_rank_stage(
        manifest,
        rank_config_sha256=tuple(config.sha256 for config in trained_rank_configs),
    )
    gate_manifest_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return updated


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--workspace", type=Path, default=Path("artifacts/preflight")
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--gate-manifest", type=Path, default=Path("data/manifests/compute-gate.json")
    )
    parser.add_argument(
        "--resolved-config-root", type=Path, default=Path("configs/resolved")
    )
    parser.add_argument(
        "--retained-output-root", type=Path, default=Path("artifacts/runs")
    )
    parser.add_argument(
        "--confirmation-runs",
        type=int,
        default=4,
        help=(
            "Additional retained runs reserved for three-seed confirmation. "
            "The conservative default assumes rank 8 and the selected rank "
            "are distinct and each already has its seed-42 search run."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    arguments = parser.parse_args()
    retained_run_plan = _retained_plan(arguments.confirmation_runs)
    if arguments.dry_run:
        print(json.dumps(dry_run_plan(retained_run_plan), indent=2, sort_keys=True))
        return
    project_root = arguments.project_root.resolve(strict=True)

    def under_project(path: Path) -> Path:
        return path if path.is_absolute() else project_root / path

    manifest = run_preflight(
        project_root=project_root,
        model_path=arguments.model_path,
        workspace=under_project(arguments.workspace),
        python_executable=arguments.python,
        gate_manifest_path=under_project(arguments.gate_manifest),
        resolved_config_root=under_project(arguments.resolved_config_root),
        retained_output_root=arguments.retained_output_root,
        retained_run_plan=retained_run_plan,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["decision"]["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    _main()
