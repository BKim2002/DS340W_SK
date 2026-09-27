"""Fail-closed compute controls for retained OpenDelta experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


OPTIMIZER_STEPS_PER_RUN = 20_000
EVALUATIONS_PER_RUN = 200
FINAL_EVALUATIONS = 2
RESERVE_MULTIPLIER = 1.20
GPU_TIME_CAP_SECONDS = 48 * 60 * 60
MINIMUM_FREE_MEMORY_MIB = 512
EVALUATION_BATCH_CANDIDATES = (32, 16, 8, 4, 2, 1)
CALIBRATION_BATCH_CANDIDATES = (8, 4, 2, 1)


class ComputeGateError(RuntimeError):
    """Raised when retained training is not backed by a valid PASS manifest."""


@dataclass(frozen=True)
class RetainedRunStage:
    """One homogeneous group of retained training runs in the compute budget."""

    name: str
    training_runs: int
    optimizer_steps_per_run: int = OPTIMIZER_STEPS_PER_RUN
    evaluations_per_run: int = EVALUATIONS_PER_RUN

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Retained run-stage name must be non-empty")
        for field_name in (
            "training_runs",
            "optimizer_steps_per_run",
            "evaluations_per_run",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int | str]:
        return {
            "name": self.name,
            "training_runs": self.training_runs,
            "optimizer_steps_per_run": self.optimizer_steps_per_run,
            "evaluations_per_run": self.evaluations_per_run,
        }


@dataclass(frozen=True)
class RetainedRunPlan:
    """Explicit retained work used by the conservative GPU-time projection."""

    stages: tuple[RetainedRunStage, ...]
    final_evaluations: int

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Retained run plan must contain at least one stage")
        names = tuple(stage.name for stage in self.stages)
        if len(set(names)) != len(names):
            raise ValueError("Retained run-stage names must be unique")
        if type(self.final_evaluations) is not int or self.final_evaluations < 0:
            raise ValueError("Final evaluation count must be a non-negative integer")

    @property
    def training_runs(self) -> int:
        return sum(stage.training_runs for stage in self.stages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "training_runs": self.training_runs,
            "final_evaluations": self.final_evaluations,
        }


DEFAULT_RETAINED_RUN_PLAN = RetainedRunPlan(
    stages=(
        RetainedRunStage("baseline", training_runs=6),
        RetainedRunStage("rank", training_runs=3),
        RetainedRunStage("confirmation", training_runs=4),
    ),
    final_evaluations=FINAL_EVALUATIONS,
)


def _retained_plan_is_conservative(plan: RetainedRunPlan) -> bool:
    """Return whether a plan reserves the approved 13-run worst case exactly."""

    return plan == DEFAULT_RETAINED_RUN_PLAN


def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


def validated_preflight_phase(config: Any) -> str | None:
    """Return the phase only for an exact, bounded preflight configuration."""

    values = config.values
    common = {
        "do_test": False,
        "max_val_samples": 436,
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 1e-3,
        "fp16": False,
        "bf16": False,
        "tf32": False,
    }

    evaluation_batch_size = values.get("per_device_eval_batch_size")
    if evaluation_batch_size in EVALUATION_BATCH_CANDIDATES:
        evaluation = {
            **common,
            "do_train": False,
            "do_eval": True,
            "max_steps": 1,
            "max_train_samples": 1,
            "per_device_train_batch_size": 1,
            "effective_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "evaluation_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }
        if (
            config.experiment_id == f"preflight-eval-eb{evaluation_batch_size}"
            and all(values.get(key) == value for key, value in evaluation.items())
        ):
            return "evaluation_batch_calibration"

    physical_batch_size = values.get("per_device_train_batch_size")
    if (
        physical_batch_size in CALIBRATION_BATCH_CANDIDATES
        and evaluation_batch_size in EVALUATION_BATCH_CANDIDATES
    ):
        training = {
            **common,
            "do_train": True,
            "do_eval": False,
            "max_steps": 10,
            "per_device_train_batch_size": physical_batch_size,
            "effective_batch_size": 32,
            "gradient_accumulation_steps": 32 // physical_batch_size,
            "evaluation_strategy": "no",
            "save_strategy": "no",
            "load_best_model_at_end": False,
        }
        if (
            config.experiment_id
            == f"preflight-train-pb{physical_batch_size}-eb{evaluation_batch_size}"
            and all(values.get(key) == value for key, value in training.items())
        ):
            return "microbatch_calibration"

        pilot = {
            **common,
            "do_train": True,
            "do_eval": True,
            "max_steps": 200,
            "per_device_train_batch_size": physical_batch_size,
            "effective_batch_size": 32,
            "gradient_accumulation_steps": 32 // physical_batch_size,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "eval_steps": 100,
            "save_steps": 100,
            "load_best_model_at_end": True,
        }
        if (
            config.experiment_id
            == (
                f"preflight-pilot-r16-pb{physical_batch_size}-"
                f"eb{evaluation_batch_size}"
            )
            and all(values.get(key) == value for key, value in pilot.items())
        ):
            return "pilot"
    return None


@dataclass(frozen=True)
class HardwareEvidence:
    """The one CUDA device on which preflight evidence was collected."""

    source_manifest_sha256: str
    gpu_name: str
    total_memory_mib: int
    driver_version: str
    cuda_available: bool
    probe_wall_seconds: float

    def __post_init__(self) -> None:
        if not _valid_sha256(self.source_manifest_sha256):
            raise ValueError(
                "Environment manifest checksum must be a lowercase SHA-256"
            )
        if not self.gpu_name or not self.driver_version:
            raise ValueError("GPU name and driver version must be non-empty")
        if type(self.total_memory_mib) is not int or self.total_memory_mib <= 0:
            raise ValueError("Total GPU memory must be a positive integer MiB value")
        if type(self.cuda_available) is not bool:
            raise TypeError("CUDA availability must be boolean")
        _require_finite_nonnegative(
            self.probe_wall_seconds, "Environment probe wall seconds"
        )


@dataclass(frozen=True)
class Week1SmokeEvidence:
    """Sanitized evidence from the verified one-step Week 1 smoke run."""

    source_manifest_sha256: str
    environment_lock_sha256: str
    resolved_config_sha256: str
    completed_at_utc: str
    exit_status: int
    optimizer_steps: int
    generation_path_exercised: bool
    upstream_peak_memory_gb: float
    wall_seconds: float
    used_fp32: bool
    final_unseen_access_prevented: bool
    checkpoint_reload_passed: bool

    def __post_init__(self) -> None:
        if not _valid_sha256(self.source_manifest_sha256):
            raise ValueError("Smoke manifest checksum must be a lowercase SHA-256")
        if not _valid_sha256(self.environment_lock_sha256):
            raise ValueError(
                "Smoke environment checksum must be a lowercase SHA-256"
            )
        if not _valid_sha256(self.resolved_config_sha256):
            raise ValueError("Smoke configuration checksum must be a lowercase SHA-256")
        if not self.completed_at_utc:
            raise ValueError("Smoke completion timestamp must be non-empty")
        if type(self.exit_status) is not int:
            raise TypeError("Smoke exit status must be an integer")
        if type(self.optimizer_steps) is not int or self.optimizer_steps < 0:
            raise ValueError("Smoke optimizer steps must be a non-negative integer")
        for name in (
            "generation_path_exercised",
            "used_fp32",
            "final_unseen_access_prevented",
            "checkpoint_reload_passed",
        ):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be boolean")
        _require_finite_nonnegative(
            self.upstream_peak_memory_gb, "Upstream peak memory GB"
        )
        _require_finite_nonnegative(self.wall_seconds, "Smoke wall seconds")

    @property
    def verified(self) -> bool:
        return (
            self.exit_status == 0
            and self.optimizer_steps >= 1
            and self.generation_path_exercised
            and self.used_fp32
            and self.final_unseen_access_prevented
            and self.checkpoint_reload_passed
        )


@dataclass(frozen=True)
class EvaluationBatchAttempt:
    """One fresh-process, full-Test evaluation-batch calibration attempt."""

    evaluation_batch_size: int
    config_sha256: str
    started_at_utc: str
    finished_at_utc: str
    fresh_process: bool
    evaluated_samples: int
    exit_status: int
    oom: bool
    minimum_free_memory_mib: float

    def __post_init__(self) -> None:
        if self.evaluation_batch_size not in EVALUATION_BATCH_CANDIDATES:
            raise ValueError(
                "Evaluation batch is not in the approved candidate order"
            )
        if not _valid_sha256(self.config_sha256):
            raise ValueError(
                "Evaluation calibration config checksum must be a lowercase SHA-256"
            )
        if not self.started_at_utc or not self.finished_at_utc:
            raise ValueError("Evaluation calibration timestamps must be non-empty")
        if type(self.fresh_process) is not bool or type(self.oom) is not bool:
            raise TypeError("Evaluation process and OOM flags must be boolean")
        if type(self.evaluated_samples) is not int or self.evaluated_samples < 0:
            raise ValueError("Evaluated sample count must be a non-negative integer")
        if type(self.exit_status) is not int:
            raise TypeError("Evaluation exit status must be an integer")
        _require_finite_nonnegative(
            self.minimum_free_memory_mib,
            "Evaluation calibration minimum free memory MiB",
        )

    @property
    def passed(self) -> bool:
        return (
            self.fresh_process
            and self.evaluated_samples == 436
            and self.exit_status == 0
            and not self.oom
            and self.minimum_free_memory_mib >= MINIMUM_FREE_MEMORY_MIB
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_sha256": self.config_sha256,
            "evaluated_samples": self.evaluated_samples,
            "evaluation_batch_size": self.evaluation_batch_size,
            "exit_status": self.exit_status,
            "finished_at_utc": self.finished_at_utc,
            "fresh_process": self.fresh_process,
            "minimum_free_memory_mib": self.minimum_free_memory_mib,
            "oom": self.oom,
            "passed": self.passed,
            "started_at_utc": self.started_at_utc,
        }


@dataclass(frozen=True)
class CalibrationAttempt:
    """One fresh-process physical-microbatch calibration attempt."""

    physical_batch_size: int
    config_sha256: str
    started_at_utc: str
    finished_at_utc: str
    fresh_process: bool
    rank: int
    lora_alpha: int
    effective_batch_size: int
    target_optimizer_steps: int
    completed_optimizer_steps: int
    exit_status: int
    oom: bool
    minimum_free_memory_mib: float

    def __post_init__(self) -> None:
        if self.physical_batch_size not in CALIBRATION_BATCH_CANDIDATES:
            raise ValueError("Calibration batch is not in the approved candidate order")
        if not _valid_sha256(self.config_sha256):
            raise ValueError("Calibration config checksum must be a lowercase SHA-256")
        if not self.started_at_utc or not self.finished_at_utc:
            raise ValueError("Calibration timestamps must be non-empty")
        if type(self.fresh_process) is not bool or type(self.oom) is not bool:
            raise TypeError("Calibration process and OOM flags must be boolean")
        for name in (
            "rank",
            "lora_alpha",
            "effective_batch_size",
            "target_optimizer_steps",
            "completed_optimizer_steps",
            "exit_status",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        if self.target_optimizer_steps <= 0 or self.completed_optimizer_steps < 0:
            raise ValueError("Calibration step counts are invalid")
        _require_finite_nonnegative(
            self.minimum_free_memory_mib, "Calibration minimum free memory MiB"
        )

    @property
    def passed(self) -> bool:
        return (
            self.fresh_process
            and self.rank == 16
            and self.lora_alpha == 32
            and self.effective_batch_size == 32
            and self.target_optimizer_steps == 10
            and self.completed_optimizer_steps == 10
            and self.exit_status == 0
            and not self.oom
            and self.minimum_free_memory_mib >= MINIMUM_FREE_MEMORY_MIB
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "config_sha256": self.config_sha256,
            "effective_batch_size": self.effective_batch_size,
            "exit_status": self.exit_status,
            "finished_at_utc": self.finished_at_utc,
            "fresh_process": self.fresh_process,
            "lora_alpha": self.lora_alpha,
            "minimum_free_memory_mib": self.minimum_free_memory_mib,
            "oom": self.oom,
            "passed": self.passed,
            "physical_batch_size": self.physical_batch_size,
            "rank": self.rank,
            "started_at_utc": self.started_at_utc,
            "target_optimizer_steps": self.target_optimizer_steps,
        }


@dataclass(frozen=True)
class PilotEvidence:
    """Measurements from the approved discarded 200-step pilot."""

    config_sha256: str
    started_at_utc: str
    finished_at_utc: str
    fresh_process: bool
    physical_batch_size: int
    rank: int
    lora_alpha: int
    effective_batch_size: int
    learning_rate: float
    optimizer_steps: int
    warmup_steps_excluded: int
    non_evaluation_seconds_steps_21_to_200: float
    full_test_evaluation_seconds: tuple[float, float]
    do_test: bool
    exit_status: int
    evaluation_batch_size: int

    def __post_init__(self) -> None:
        if not _valid_sha256(self.config_sha256):
            raise ValueError("Pilot config checksum must be a lowercase SHA-256")
        if not self.started_at_utc or not self.finished_at_utc:
            raise ValueError("Pilot timestamps must be non-empty")
        if type(self.fresh_process) is not bool or type(self.do_test) is not bool:
            raise TypeError("Pilot process and data-access flags must be boolean")
        for name in (
            "physical_batch_size",
            "rank",
            "lora_alpha",
            "effective_batch_size",
            "optimizer_steps",
            "warmup_steps_excluded",
            "exit_status",
            "evaluation_batch_size",
        ):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer")
        _require_finite_nonnegative(self.learning_rate, "Pilot learning rate")
        _require_finite_nonnegative(
            self.non_evaluation_seconds_steps_21_to_200,
            "Pilot non-evaluation wall seconds",
        )
        if len(self.full_test_evaluation_seconds) != 2:
            raise ValueError("Pilot must contain exactly two full Test evaluations")
        for value in self.full_test_evaluation_seconds:
            _require_finite_nonnegative(value, "Pilot Test evaluation seconds")

    def complete_for(
        self,
        selected_physical_batch_size: int | None,
        selected_evaluation_batch_size: int | None,
    ) -> bool:
        return (
            selected_physical_batch_size is not None
            and self.fresh_process
            and self.physical_batch_size == selected_physical_batch_size
            and self.evaluation_batch_size == selected_evaluation_batch_size
            and self.rank == 16
            and self.lora_alpha == 32
            and self.effective_batch_size == 32
            and self.learning_rate == 1e-3
            and self.optimizer_steps == 200
            and self.warmup_steps_excluded == 20
            and self.non_evaluation_seconds_steps_21_to_200 > 0
            and all(value > 0 for value in self.full_test_evaluation_seconds)
            and not self.do_test
            and self.exit_status == 0
        )

    @property
    def seconds_per_optimizer_step(self) -> float:
        return self.non_evaluation_seconds_steps_21_to_200 / 180

    @property
    def slower_full_test_evaluation_seconds(self) -> float:
        return max(self.full_test_evaluation_seconds)

    def to_dict(self, *, complete: bool) -> dict[str, Any]:
        return {
            "complete": complete,
            "config_sha256": self.config_sha256,
            "do_test": self.do_test,
            "effective_batch_size": self.effective_batch_size,
            "evaluation_batch_size": self.evaluation_batch_size,
            "evaluation_measurements_seconds": list(
                self.full_test_evaluation_seconds
            ),
            "exit_status": self.exit_status,
            "finished_at_utc": self.finished_at_utc,
            "fresh_process": self.fresh_process,
            "full_test_evaluation_seconds": (
                self.slower_full_test_evaluation_seconds
            ),
            "learning_rate": self.learning_rate,
            "lora_alpha": self.lora_alpha,
            "non_evaluation_seconds_steps_21_to_200": (
                self.non_evaluation_seconds_steps_21_to_200
            ),
            "optimizer_steps": self.optimizer_steps,
            "physical_batch_size": self.physical_batch_size,
            "rank": self.rank,
            "seconds_per_optimizer_step": self.seconds_per_optimizer_step,
            "started_at_utc": self.started_at_utc,
            "warmup_steps_excluded": self.warmup_steps_excluded,
        }


def project_gpu_seconds(
    *,
    actual_preflight_seconds: float,
    seconds_per_optimizer_step: float,
    full_test_evaluation_seconds: float,
    retained_run_plan: RetainedRunPlan,
) -> float:
    """Apply the approved conservative GPU-time projection formula."""

    inputs = (
        actual_preflight_seconds,
        seconds_per_optimizer_step,
        full_test_evaluation_seconds,
    )
    if any(not math.isfinite(value) or value < 0 for value in inputs):
        raise ValueError("Projection inputs must be finite non-negative seconds")
    retained_seconds = sum(
        stage.training_runs
        * (
            stage.optimizer_steps_per_run * seconds_per_optimizer_step
            + stage.evaluations_per_run * full_test_evaluation_seconds
        )
        for stage in retained_run_plan.stages
    )
    final_seconds = (
        retained_run_plan.final_evaluations * full_test_evaluation_seconds
    )
    return actual_preflight_seconds + RESERVE_MULTIPLIER * (
        retained_seconds + final_seconds
    )


def _validate_stage_hashes(
    values: Sequence[str], *, stage: str, maximum: int
) -> tuple[str, ...]:
    hashes = tuple(values)
    if len(hashes) > maximum:
        raise ValueError(f"Too many {stage} configuration checksums")
    if any(not _valid_sha256(value) for value in hashes):
        raise ValueError(
            f"{stage.capitalize()} config checksums must be lowercase SHA-256 values"
        )
    if len(set(hashes)) != len(hashes):
        raise ValueError(f"{stage.capitalize()} config checksums must be unique")
    return hashes


def build_compute_gate_manifest(
    *,
    hardware: HardwareEvidence,
    smoke: Week1SmokeEvidence | None,
    evaluation_batch_attempts: Sequence[EvaluationBatchAttempt] = (),
    calibration_attempts: Sequence[CalibrationAttempt] = (),
    pilot: PilotEvidence | None = None,
    actual_preflight_seconds: float | None = None,
    baseline_config_sha256: Sequence[str] = (),
    retained_run_plan: RetainedRunPlan = DEFAULT_RETAINED_RUN_PLAN,
) -> dict[str, Any]:
    """Build a deterministic gate decision; incomplete evidence always stops."""

    evaluation_attempts = tuple(evaluation_batch_attempts)
    if len(evaluation_attempts) > len(EVALUATION_BATCH_CANDIDATES):
        raise ValueError("Too many evaluation-batch calibration attempts")
    observed_evaluation_order = tuple(
        attempt.evaluation_batch_size for attempt in evaluation_attempts
    )
    expected_evaluation_order = EVALUATION_BATCH_CANDIDATES[
        : len(evaluation_attempts)
    ]
    if observed_evaluation_order != expected_evaluation_order:
        raise ValueError(
            "Evaluation calibration attempts must follow 32, 16, 8, 4, 2, 1 order"
        )
    if any(attempt.passed for attempt in evaluation_attempts[:-1]):
        raise ValueError(
            "Evaluation calibration must stop after its first passing attempt"
        )
    selected_evaluation_batch_size = (
        evaluation_attempts[-1].evaluation_batch_size
        if evaluation_attempts and evaluation_attempts[-1].passed
        else None
    )
    evaluation_calibration_complete = selected_evaluation_batch_size is not None

    attempts = tuple(calibration_attempts)
    if len(attempts) > len(CALIBRATION_BATCH_CANDIDATES):
        raise ValueError("Too many microbatch calibration attempts")
    observed_order = tuple(attempt.physical_batch_size for attempt in attempts)
    expected_order = CALIBRATION_BATCH_CANDIDATES[: len(attempts)]
    if observed_order != expected_order:
        raise ValueError("Calibration attempts must follow 8, 4, 2, 1 order")
    if any(attempt.passed for attempt in attempts[:-1]):
        raise ValueError("Calibration must stop after its first passing attempt")
    selected_physical_batch_size = (
        attempts[-1].physical_batch_size if attempts and attempts[-1].passed else None
    )
    calibration_complete = selected_physical_batch_size is not None
    pilot_complete = pilot is not None and pilot.complete_for(
        selected_physical_batch_size, selected_evaluation_batch_size
    )
    if actual_preflight_seconds is not None:
        _require_finite_nonnegative(
            actual_preflight_seconds, "Actual preflight seconds"
        )
    baseline_hashes = _validate_stage_hashes(
        baseline_config_sha256, stage="baseline", maximum=6
    )
    baseline_hashes_complete = len(baseline_hashes) == 6

    smoke_verified = smoke is not None and smoke.verified
    hardware_matches_target = (
        hardware.cuda_available
        and hardware.gpu_name == "NVIDIA GeForce RTX 4060"
        and hardware.total_memory_mib >= 8_000
        and (
            smoke is None
            or smoke.upstream_peak_memory_gb * 1_000 < hardware.total_memory_mib
        )
    )
    reasons: list[str] = []
    if not smoke_verified:
        reasons.append("week1_smoke_not_verified")
    if not hardware_matches_target:
        reasons.append("target_hardware_not_verified")
    if not evaluation_calibration_complete:
        reasons.append("evaluation_batch_calibration_incomplete")
    if not calibration_complete:
        reasons.append("microbatch_calibration_incomplete")
    if not pilot_complete:
        reasons.append("pilot_incomplete")
    projection_available = (
        evaluation_calibration_complete
        and calibration_complete
        and pilot_complete
        and actual_preflight_seconds is not None
    )
    projected_seconds = None
    if projection_available:
        assert pilot is not None
        projected_seconds = project_gpu_seconds(
            actual_preflight_seconds=actual_preflight_seconds,
            seconds_per_optimizer_step=pilot.seconds_per_optimizer_step,
            full_test_evaluation_seconds=(
                pilot.slower_full_test_evaluation_seconds
            ),
            retained_run_plan=retained_run_plan,
        )
    else:
        reasons.append("approved_projection_unavailable")
    if not baseline_hashes_complete:
        reasons.append("baseline_configuration_hashes_missing")
    if not _retained_plan_is_conservative(retained_run_plan):
        reasons.append("retained_run_plan_incomplete")
    if projected_seconds is not None and projected_seconds > GPU_TIME_CAP_SECONDS:
        reasons.append("projected_time_exceeds_48_gpu_hour_cap")
    baseline_authorized = not reasons
    retained_training_authorized = baseline_authorized
    projection_status = (
        "UNAVAILABLE"
        if projected_seconds is None
        else (
            "WITHIN_CAP"
            if projected_seconds <= GPU_TIME_CAP_SECONDS
            else "OVER_CAP"
        )
    )

    if pilot is None:
        pilot_payload: dict[str, Any] = {
            "complete": False,
            "required_optimizer_steps": 200,
            "warmup_steps_excluded": 20,
            "required_full_test_evaluations": 2,
            "seconds_per_optimizer_step": None,
            "full_test_evaluation_seconds": None,
        }
    else:
        pilot_payload = pilot.to_dict(complete=pilot_complete)

    return {
        "schema_version": 2,
        "assessment_kind": "compute-gate",
        "evidence_cutoff_utc": (
            pilot.finished_at_utc
            if pilot is not None
            else (smoke.completed_at_utc if smoke is not None else None)
        ),
        "hardware": {
            "source_manifest_sha256": hardware.source_manifest_sha256,
            "cuda_available": hardware.cuda_available,
            "driver_version": hardware.driver_version,
            "gpu_name": hardware.gpu_name,
            "probe_wall_seconds": hardware.probe_wall_seconds,
            "total_memory_mib": hardware.total_memory_mib,
            "compatibility": (
                (
                    "RETAINED_RUN_PREFLIGHT_COMPLETE"
                    if evaluation_calibration_complete
                    and calibration_complete
                    and pilot_complete
                    else "WEEK1_SMOKE_ONLY"
                )
                if hardware_matches_target and smoke_verified
                else "NOT_VERIFIED"
            ),
        },
        "week1_smoke": {
            "verified": smoke_verified,
            "source_manifest_sha256": (
                smoke.source_manifest_sha256 if smoke is not None else None
            ),
            "environment_lock_sha256": (
                smoke.environment_lock_sha256 if smoke is not None else None
            ),
            "resolved_config_sha256": (
                smoke.resolved_config_sha256 if smoke is not None else None
            ),
            "completed_at_utc": smoke.completed_at_utc if smoke is not None else None,
            "optimizer_steps": smoke.optimizer_steps if smoke is not None else None,
            "upstream_peak_memory_gb": (
                smoke.upstream_peak_memory_gb if smoke is not None else None
            ),
            "wall_seconds": smoke.wall_seconds if smoke is not None else None,
            "observed_preflight_seconds_lower_bound": (
                hardware.probe_wall_seconds + smoke.wall_seconds
                if smoke is not None
                else None
            ),
            "scope": "one-step non-result viability evidence",
        },
        "evaluation_batch_calibration": {
            "complete": evaluation_calibration_complete,
            "required_batch_candidates": list(EVALUATION_BATCH_CANDIDATES),
            "minimum_free_memory_mib": MINIMUM_FREE_MEMORY_MIB,
            "attempts": [attempt.to_dict() for attempt in evaluation_attempts],
            "selected_evaluation_batch_size": selected_evaluation_batch_size,
        },
        "microbatch_calibration": {
            "complete": calibration_complete,
            "required_batch_candidates": list(CALIBRATION_BATCH_CANDIDATES),
            "minimum_free_memory_mib": MINIMUM_FREE_MEMORY_MIB,
            "attempts": [attempt.to_dict() for attempt in attempts],
            "selected_physical_batch_size": selected_physical_batch_size,
        },
        "pilot": pilot_payload,
        "projection": {
            "status": projection_status,
            "formula": (
                "actual_preflight_seconds + 1.20 * "
                "(sum(stage_runs * (stage_steps * s + stage_evaluations * e)) "
                "+ final_evaluations * e)"
            ),
            "cap_seconds": GPU_TIME_CAP_SECONDS,
            "cap_gpu_hours": 48,
            "reserve_multiplier": RESERVE_MULTIPLIER,
            "retained_run_plan": retained_run_plan.to_dict(),
            "retained_training_runs": retained_run_plan.training_runs,
            "final_evaluations": retained_run_plan.final_evaluations,
            "seconds_per_optimizer_step": (
                pilot.seconds_per_optimizer_step if pilot_complete else None
            ),
            "full_test_evaluation_seconds": (
                pilot.slower_full_test_evaluation_seconds
                if pilot_complete
                else None
            ),
            "actual_preflight_seconds": (
                actual_preflight_seconds if projection_available else None
            ),
            "projected_seconds": projected_seconds,
            "projected_gpu_hours": (
                projected_seconds / 3_600 if projected_seconds is not None else None
            ),
        },
        "authorization": {
            "baseline": {
                "authorized": baseline_authorized,
                "expected_config_count": 6,
                "config_sha256": list(baseline_hashes),
            },
            "rank": {
                "authorized": False,
                "expected_config_count": 3,
                "config_sha256": [],
                "requires_stage": "baseline",
            },
        },
        "decision": {
            "status": "PASS" if retained_training_authorized else "STOP",
            "retained_training_authorized": retained_training_authorized,
            "reasons": reasons,
        },
    }


def authorize_rank_stage(
    manifest: Mapping[str, Any], *, rank_config_sha256: Sequence[str]
) -> dict[str, Any]:
    """Add three post-selection rank hashes to an authorized baseline gate."""

    rank_hashes = _validate_stage_hashes(
        rank_config_sha256, stage="rank", maximum=3
    )
    if len(rank_hashes) != 3:
        raise ComputeGateError(
            "Rank authorization requires exactly three configuration checksums"
        )
    authorization = manifest.get("authorization")
    if not isinstance(authorization, Mapping):
        raise ComputeGateError("Compute-gate stage authorization is incomplete")
    baseline = authorization.get("baseline")
    if not isinstance(baseline, Mapping):
        raise ComputeGateError("Compute-gate baseline stage is not authorized")
    baseline_hashes = baseline.get("config_sha256")
    if not isinstance(baseline_hashes, list) or not baseline_hashes:
        raise ComputeGateError("Compute-gate baseline stage is not authorized")
    require_retained_training_authorization(
        manifest,
        stage="baseline",
        resolved_config_sha256=baseline_hashes[0],
    )
    if set(baseline_hashes) & set(rank_hashes):
        raise ComputeGateError(
            "Rank configuration checksums must be distinct from baseline checksums"
        )
    updated = json.loads(json.dumps(manifest))
    updated["authorization"]["rank"] = {
        "authorized": True,
        "expected_config_count": 3,
        "config_sha256": list(rank_hashes),
        "requires_stage": "baseline",
    }
    return updated


def load_week1_evidence(
    environment_manifest_path: Path,
    smoke_manifest_path: Path,
) -> tuple[HardwareEvidence, Week1SmokeEvidence]:
    """Load only the locked, sanitized fields needed by the Week 1 gate."""

    environment_manifest_bytes = environment_manifest_path.read_bytes()
    try:
        environment = json.loads(environment_manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Environment manifest is not valid UTF-8 JSON") from error
    if not isinstance(environment, dict):
        raise ValueError("Environment manifest must contain a JSON object")
    environment_manifest_sha256 = sha256(environment_manifest_bytes).hexdigest()
    smoke_manifest_bytes = smoke_manifest_path.read_bytes()
    try:
        smoke = json.loads(smoke_manifest_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError("Smoke manifest is not valid UTF-8 JSON") from error
    if not isinstance(smoke, dict):
        raise ValueError("Smoke manifest must contain a JSON object")
    try:
        devices = environment["nvidia"]["devices"]
        if not isinstance(devices, list) or len(devices) != 1:
            raise ValueError("Environment manifest must identify exactly one GPU")
        device = devices[0]
        hardware = HardwareEvidence(
            source_manifest_sha256=environment_manifest_sha256,
            gpu_name=device["name"],
            total_memory_mib=device["memory_total_mib"],
            driver_version=environment["nvidia"]["driver_version"],
            cuda_available=environment["pytorch"]["cuda_available"],
            probe_wall_seconds=environment["probe_wall_seconds"],
        )
        precision = environment["precision_policy"]
        used_fp32 = (
            precision["torch_dtype"] == "float32"
            and precision["fp16"] is False
            and precision["bf16"] is False
            and precision["tf32"] is False
        )
        smoke_evidence = Week1SmokeEvidence(
            source_manifest_sha256=sha256(smoke_manifest_bytes).hexdigest(),
            environment_lock_sha256=smoke["environment_lock_sha256"],
            resolved_config_sha256=smoke["configuration"][
                "resolved_config_sha256"
            ],
            completed_at_utc=smoke["process"]["finished_at_utc"],
            exit_status=smoke["process"]["exit_code"],
            optimizer_steps=smoke["training"]["global_step"],
            generation_path_exercised=smoke["evaluation"][
                "generation_path_exercised"
            ],
            upstream_peak_memory_gb=smoke["measurement"][
                "upstream_peak_memory_gb"
            ],
            wall_seconds=smoke["process"]["wall_seconds"],
            used_fp32=used_fp32,
            final_unseen_access_prevented=smoke["data_access"][
                "final_unseen_access_prevented"
            ],
            checkpoint_reload_passed=smoke["checkpoint_reload"]["passed"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("Environment or smoke manifest is incomplete") from error
    if (
        environment.get("schema_version") != 1
        or smoke.get("schema_version") != 1
        or smoke.get("status") != "verified"
        or smoke.get("artifact_kind") != "week1-non-result-smoke"
        or smoke.get("environment_lock_sha256") != environment_manifest_sha256
        or smoke.get("evidence_gaps") != []
        or smoke.get("configuration", {}).get("max_steps") != 1
        or smoke.get("configuration", {}).get("rank") != 8
        or smoke.get("configuration", {}).get("alpha") != 16
        or smoke.get("configuration", {}).get("do_test") is not False
        or smoke.get("trainable_parameters")
        != {
            "injected_projections": 72,
            "layer_norm_parameters": 47_616,
            "lora_parameters": 884_736,
            "total_trainable_parameters": 932_352,
        }
    ):
        raise ValueError("Week 1 source manifests are not verified schema-v1 evidence")
    return hardware, smoke_evidence


def write_week1_stop_manifest(
    *,
    environment_manifest_path: Path,
    smoke_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Regenerate the deterministic fail-closed Week 1 compute manifest."""

    hardware, smoke = load_week1_evidence(
        environment_manifest_path, smoke_manifest_path
    )
    manifest = build_compute_gate_manifest(hardware=hardware, smoke=smoke)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Manifest is not valid UTF-8 JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path.name}")
    return value


def require_authorization_from_files(
    *, gate_manifest_path: Path, resolved_config_path: Path, stage: str
) -> None:
    """Apply the fail-closed retained-run check to concrete locked files."""

    manifest = _load_json_object(gate_manifest_path)
    resolved_config_sha256 = sha256(resolved_config_path.read_bytes()).hexdigest()
    require_retained_training_authorization(
        manifest, stage=stage, resolved_config_sha256=resolved_config_sha256
    )


def require_retained_training_authorization(
    manifest: Mapping[str, Any], *, stage: str, resolved_config_sha256: str
) -> None:
    """Fail closed unless complete measured evidence authorizes this stage/config."""

    if stage not in {"baseline", "rank"}:
        raise ComputeGateError(f"Unsupported retained-training stage: {stage}")
    if not _valid_sha256(resolved_config_sha256):
        raise ComputeGateError("Requested configuration checksum is invalid")
    decision = manifest.get("decision")
    if not isinstance(decision, Mapping):
        raise ComputeGateError("Compute-gate manifest is incomplete")
    if (
        decision.get("status") != "PASS"
        or decision.get("retained_training_authorized") is not True
    ):
        reasons = decision.get("reasons")
        suffix = f": {', '.join(reasons)}" if isinstance(reasons, list) else ""
        raise ComputeGateError(f"Compute gate is STOP{suffix}")

    hardware = manifest.get("hardware")
    smoke = manifest.get("week1_smoke")
    evaluation_calibration = manifest.get("evaluation_batch_calibration")
    calibration = manifest.get("microbatch_calibration")
    pilot = manifest.get("pilot")
    projection = manifest.get("projection")
    authorization = manifest.get("authorization")
    complete_sections = (
        manifest.get("schema_version") == 2
        and decision.get("reasons") == []
        and isinstance(hardware, Mapping)
        and hardware.get("compatibility") == "RETAINED_RUN_PREFLIGHT_COMPLETE"
        and isinstance(smoke, Mapping)
        and smoke.get("verified") is True
        and isinstance(evaluation_calibration, Mapping)
        and evaluation_calibration.get("complete") is True
        and evaluation_calibration.get("selected_evaluation_batch_size")
        in EVALUATION_BATCH_CANDIDATES
        and isinstance(calibration, Mapping)
        and calibration.get("complete") is True
        and calibration.get("selected_physical_batch_size")
        in CALIBRATION_BATCH_CANDIDATES
        and isinstance(pilot, Mapping)
        and pilot.get("complete") is True
        and isinstance(projection, Mapping)
        and projection.get("status") == "WITHIN_CAP"
        and projection.get("cap_seconds") == GPU_TIME_CAP_SECONDS
        and _projection_plan_is_complete(projection)
        and isinstance(authorization, Mapping)
    )
    if not complete_sections:
        raise ComputeGateError("Compute-gate PASS evidence is incomplete")
    projected_seconds = projection.get("projected_seconds")
    if (
        type(projected_seconds) not in (int, float)
        or not math.isfinite(projected_seconds)
        or projected_seconds < 0
        or projected_seconds > GPU_TIME_CAP_SECONDS
    ):
        raise ComputeGateError("Compute-gate projection is missing or above the cap")
    assert isinstance(authorization, Mapping)
    stage_authorization = authorization.get(stage)
    if not isinstance(stage_authorization, Mapping):
        raise ComputeGateError("Compute-gate stage authorization is incomplete")
    expected_count = 6 if stage == "baseline" else 3
    hashes = stage_authorization.get("config_sha256")
    stage_complete = (
        stage_authorization.get("authorized") is True
        and stage_authorization.get("expected_config_count") == expected_count
        and isinstance(hashes, list)
        and len(hashes) == expected_count
        and len(set(hashes)) == expected_count
        and all(isinstance(value, str) and _valid_sha256(value) for value in hashes)
    )
    if not stage_complete:
        raise ComputeGateError(f"Compute-gate {stage} stage is not authorized")
    if stage == "rank":
        baseline = authorization.get("baseline")
        if not isinstance(baseline, Mapping) or baseline.get("authorized") is not True:
            raise ComputeGateError("Compute-gate rank stage requires baseline authorization")
    if resolved_config_sha256 not in hashes:
        raise ComputeGateError(
            "Requested configuration is not listed in the compute-gate manifest"
        )


def _projection_plan_is_complete(projection: Mapping[str, Any]) -> bool:
    plan = projection.get("retained_run_plan")
    if not isinstance(plan, Mapping):
        return False
    stages = plan.get("stages")
    if not isinstance(stages, list):
        return False
    stage_specs: dict[str, tuple[int, int, int]] = {}
    for stage in stages:
        if not isinstance(stage, Mapping):
            return False
        name = stage.get("name")
        runs = stage.get("training_runs")
        steps = stage.get("optimizer_steps_per_run")
        evaluations = stage.get("evaluations_per_run")
        if (
            not isinstance(name, str)
            or not name
            or type(runs) is not int
            or runs < 0
            or type(steps) is not int
            or steps < 0
            or type(evaluations) is not int
            or evaluations < 0
            or name in stage_specs
        ):
            return False
        stage_specs[name] = (runs, steps, evaluations)
    expected = {
        "baseline": (6, OPTIMIZER_STEPS_PER_RUN, EVALUATIONS_PER_RUN),
        "rank": (3, OPTIMIZER_STEPS_PER_RUN, EVALUATIONS_PER_RUN),
        "confirmation": (4, OPTIMIZER_STEPS_PER_RUN, EVALUATIONS_PER_RUN),
    }
    return (
        stage_specs == expected
        and plan.get("training_runs") == 13
        and plan.get("final_evaluations") == FINAL_EVALUATIONS
    )
