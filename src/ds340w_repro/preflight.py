"""Fail-closed compute controls for retained OpenDelta experiments."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


RETAINED_TRAINING_RUNS = 9
OPTIMIZER_STEPS_PER_RUN = 20_000
EVALUATIONS_PER_RUN = 200
FINAL_EVALUATIONS = 2
RESERVE_MULTIPLIER = 1.20
GPU_TIME_CAP_SECONDS = 48 * 60 * 60
MINIMUM_FREE_MEMORY_MIB = 512
CALIBRATION_BATCH_CANDIDATES = (8, 4, 2, 1)


class ComputeGateError(RuntimeError):
    """Raised when retained training is not backed by a valid PASS manifest."""


def _valid_sha256(value: str) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _require_finite_nonnegative(value: float, name: str) -> None:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")


@dataclass(frozen=True)
class HardwareEvidence:
    """The one CUDA device on which preflight evidence was collected."""

    gpu_name: str
    total_memory_mib: int
    driver_version: str
    cuda_available: bool

    def __post_init__(self) -> None:
        if not self.gpu_name or not self.driver_version:
            raise ValueError("GPU name and driver version must be non-empty")
        if type(self.total_memory_mib) is not int or self.total_memory_mib <= 0:
            raise ValueError("Total GPU memory must be a positive integer MiB value")
        if type(self.cuda_available) is not bool:
            raise TypeError("CUDA availability must be boolean")


@dataclass(frozen=True)
class Week1SmokeEvidence:
    """Sanitized evidence from the verified one-step Week 1 smoke run."""

    source_manifest_sha256: str
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

    def complete_for(self, selected_physical_batch_size: int | None) -> bool:
        return (
            selected_physical_batch_size is not None
            and self.fresh_process
            and self.physical_batch_size == selected_physical_batch_size
            and self.rank == 16
            and self.lora_alpha == 32
            and self.effective_batch_size == 32
            and self.learning_rate == 1e-3
            and self.optimizer_steps == 200
            and self.warmup_steps_excluded == 20
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
) -> float:
    """Apply the approved conservative GPU-time projection formula."""

    inputs = (
        actual_preflight_seconds,
        seconds_per_optimizer_step,
        full_test_evaluation_seconds,
    )
    if any(not math.isfinite(value) or value < 0 for value in inputs):
        raise ValueError("Projection inputs must be finite non-negative seconds")
    return actual_preflight_seconds + RESERVE_MULTIPLIER * (
        RETAINED_TRAINING_RUNS
        * (
            OPTIMIZER_STEPS_PER_RUN * seconds_per_optimizer_step
            + EVALUATIONS_PER_RUN * full_test_evaluation_seconds
        )
        + FINAL_EVALUATIONS * full_test_evaluation_seconds
    )


def build_compute_gate_manifest(
    *,
    hardware: HardwareEvidence,
    smoke: Week1SmokeEvidence | None,
    calibration_attempts: Sequence[CalibrationAttempt] = (),
    pilot: PilotEvidence | None = None,
    actual_preflight_seconds: float | None = None,
    retained_config_sha256: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a deterministic gate decision; incomplete evidence always stops."""

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
    pilot_complete = (
        pilot is not None and pilot.complete_for(selected_physical_batch_size)
    )
    if actual_preflight_seconds is not None:
        _require_finite_nonnegative(
            actual_preflight_seconds, "Actual preflight seconds"
        )
    retained_hashes = tuple(retained_config_sha256)
    if any(not _valid_sha256(value) for value in retained_hashes):
        raise ValueError("Retained config checksums must be lowercase SHA-256 values")
    if len(set(retained_hashes)) != len(retained_hashes):
        raise ValueError("Retained config checksums must be unique")
    retained_hashes_complete = len(retained_hashes) == RETAINED_TRAINING_RUNS

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
    if not calibration_complete:
        reasons.append("microbatch_calibration_incomplete")
    if not pilot_complete:
        reasons.append("pilot_incomplete")
    projection_available = (
        calibration_complete
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
        )
    else:
        reasons.append("approved_projection_unavailable")
    if not retained_hashes_complete:
        reasons.append("retained_configuration_hashes_missing")
    if projected_seconds is not None and projected_seconds > GPU_TIME_CAP_SECONDS:
        reasons.append("projected_time_exceeds_48_gpu_hour_cap")
    retained_training_authorized = not reasons
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
        "schema_version": 1,
        "assessment_kind": "compute-gate",
        "evidence_cutoff_utc": (
            pilot.finished_at_utc
            if pilot is not None
            else (smoke.completed_at_utc if smoke is not None else None)
        ),
        "hardware": {
            "cuda_available": hardware.cuda_available,
            "driver_version": hardware.driver_version,
            "gpu_name": hardware.gpu_name,
            "total_memory_mib": hardware.total_memory_mib,
            "compatibility": (
                (
                    "RETAINED_RUN_PREFLIGHT_COMPLETE"
                    if calibration_complete and pilot_complete
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
                smoke.wall_seconds if smoke is not None else None
            ),
            "scope": "one-step non-result viability evidence",
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
                "(9 * (20000 * s + 200 * e) + 2 * e)"
            ),
            "cap_seconds": GPU_TIME_CAP_SECONDS,
            "cap_gpu_hours": 48,
            "reserve_multiplier": RESERVE_MULTIPLIER,
            "retained_training_runs": RETAINED_TRAINING_RUNS,
            "optimizer_steps_per_run": OPTIMIZER_STEPS_PER_RUN,
            "evaluations_per_run": EVALUATIONS_PER_RUN,
            "final_evaluations": FINAL_EVALUATIONS,
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
        "retained_config_sha256": list(retained_hashes),
        "decision": {
            "status": "PASS" if retained_training_authorized else "STOP",
            "retained_training_authorized": retained_training_authorized,
            "reasons": reasons,
        },
    }


def load_week1_evidence(
    environment_manifest_path: Path,
    smoke_manifest_path: Path,
) -> tuple[HardwareEvidence, Week1SmokeEvidence]:
    """Load only the locked, sanitized fields needed by the Week 1 gate."""

    environment = _load_json_object(environment_manifest_path)
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
            gpu_name=device["name"],
            total_memory_mib=device["memory_total_mib"],
            driver_version=environment["nvidia"]["driver_version"],
            cuda_available=environment["pytorch"]["cuda_available"],
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
    *, gate_manifest_path: Path, resolved_config_path: Path
) -> None:
    """Apply the fail-closed retained-run check to concrete locked files."""

    manifest = _load_json_object(gate_manifest_path)
    resolved_config_sha256 = sha256(resolved_config_path.read_bytes()).hexdigest()
    require_retained_training_authorization(
        manifest, resolved_config_sha256=resolved_config_sha256
    )


def require_retained_training_authorization(
    manifest: Mapping[str, Any], *, resolved_config_sha256: str
) -> None:
    """Fail closed unless complete measured evidence authorizes this config."""

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
    calibration = manifest.get("microbatch_calibration")
    pilot = manifest.get("pilot")
    projection = manifest.get("projection")
    hashes = manifest.get("retained_config_sha256")
    complete_sections = (
        manifest.get("schema_version") == 1
        and decision.get("reasons") == []
        and isinstance(hardware, Mapping)
        and hardware.get("compatibility") == "RETAINED_RUN_PREFLIGHT_COMPLETE"
        and isinstance(smoke, Mapping)
        and smoke.get("verified") is True
        and isinstance(calibration, Mapping)
        and calibration.get("complete") is True
        and calibration.get("selected_physical_batch_size")
        in CALIBRATION_BATCH_CANDIDATES
        and isinstance(pilot, Mapping)
        and pilot.get("complete") is True
        and isinstance(projection, Mapping)
        and projection.get("status") == "WITHIN_CAP"
        and projection.get("cap_seconds") == GPU_TIME_CAP_SECONDS
        and isinstance(hashes, list)
        and len(hashes) == RETAINED_TRAINING_RUNS
        and len(set(hashes)) == RETAINED_TRAINING_RUNS
        and all(isinstance(value, str) and _valid_sha256(value) for value in hashes)
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
    if resolved_config_sha256 not in hashes:
        raise ComputeGateError(
            "Requested configuration is not listed in the compute-gate manifest"
        )
