from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import pytest

from ds340w_repro.preflight import (
    CalibrationAttempt,
    ComputeGateError,
    HardwareEvidence,
    PilotEvidence,
    Week1SmokeEvidence,
    build_compute_gate_manifest,
    load_week1_evidence,
    project_gpu_seconds,
    require_authorization_from_files,
    require_retained_training_authorization,
    write_week1_stop_manifest,
)


SHA = "a" * 64
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _hardware() -> HardwareEvidence:
    return HardwareEvidence(
        gpu_name="NVIDIA GeForce RTX 4060",
        total_memory_mib=8_188,
        driver_version="610.88",
        cuda_available=True,
    )


def _smoke() -> Week1SmokeEvidence:
    return Week1SmokeEvidence(
        source_manifest_sha256=SHA,
        resolved_config_sha256="b" * 64,
        completed_at_utc="2026-09-06T16:19:56.408943+00:00",
        exit_status=0,
        optimizer_steps=1,
        generation_path_exercised=True,
        upstream_peak_memory_gb=0.977955078125,
        wall_seconds=20.53837343200007,
        used_fp32=True,
        final_unseen_access_prevented=True,
        checkpoint_reload_passed=True,
    )


def test_projection_uses_the_predeclared_conservative_formula() -> None:
    projection = project_gpu_seconds(
        actual_preflight_seconds=600.0,
        seconds_per_optimizer_step=0.5,
        full_test_evaluation_seconds=30.0,
    )

    expected = 600.0 + 1.20 * (9 * (20_000 * 0.5 + 200 * 30.0) + 2 * 30.0)
    assert projection == pytest.approx(expected)


def test_week1_smoke_proves_feasibility_but_does_not_authorize_long_runs() -> None:
    manifest = build_compute_gate_manifest(
        hardware=_hardware(),
        smoke=_smoke(),
    )

    assert manifest["hardware"]["compatibility"] == "WEEK1_SMOKE_ONLY"
    assert manifest["hardware"]["gpu_name"] == "NVIDIA GeForce RTX 4060"
    assert manifest["week1_smoke"]["verified"] is True
    assert manifest["microbatch_calibration"]["complete"] is False
    assert manifest["pilot"]["complete"] is False
    assert manifest["projection"]["status"] == "UNAVAILABLE"
    assert manifest["projection"]["cap_seconds"] == 172_800
    assert manifest["decision"] == {
        "reasons": [
            "microbatch_calibration_incomplete",
            "pilot_incomplete",
            "approved_projection_unavailable",
            "retained_configuration_hashes_missing",
        ],
        "retained_training_authorized": False,
        "status": "STOP",
    }


def test_complete_measured_evidence_under_the_cap_authorizes_the_grid() -> None:
    failed_batch_eight = CalibrationAttempt(
        physical_batch_size=8,
        config_sha256="c" * 64,
        started_at_utc="2026-09-07T00:00:00+00:00",
        finished_at_utc="2026-09-07T00:01:00+00:00",
        fresh_process=True,
        rank=16,
        lora_alpha=32,
        effective_batch_size=32,
        target_optimizer_steps=10,
        completed_optimizer_steps=0,
        exit_status=1,
        oom=True,
        minimum_free_memory_mib=100.0,
    )
    passed_batch_four = CalibrationAttempt(
        physical_batch_size=4,
        config_sha256="d" * 64,
        started_at_utc="2026-09-07T00:02:00+00:00",
        finished_at_utc="2026-09-07T00:04:00+00:00",
        fresh_process=True,
        rank=16,
        lora_alpha=32,
        effective_batch_size=32,
        target_optimizer_steps=10,
        completed_optimizer_steps=10,
        exit_status=0,
        oom=False,
        minimum_free_memory_mib=900.0,
    )
    pilot = PilotEvidence(
        config_sha256="e" * 64,
        started_at_utc="2026-09-07T00:05:00+00:00",
        finished_at_utc="2026-09-07T00:20:00+00:00",
        fresh_process=True,
        physical_batch_size=4,
        rank=16,
        lora_alpha=32,
        effective_batch_size=32,
        learning_rate=1e-3,
        optimizer_steps=200,
        warmup_steps_excluded=20,
        non_evaluation_seconds_steps_21_to_200=18.0,
        full_test_evaluation_seconds=(5.0, 4.5),
        do_test=False,
        exit_status=0,
    )
    retained_hashes = tuple(f"{number:064x}" for number in range(1, 10))

    manifest = build_compute_gate_manifest(
        hardware=_hardware(),
        smoke=_smoke(),
        calibration_attempts=(failed_batch_eight, passed_batch_four),
        pilot=pilot,
        actual_preflight_seconds=600.0,
        retained_config_sha256=retained_hashes,
    )

    assert manifest["microbatch_calibration"]["selected_physical_batch_size"] == 4
    assert manifest["pilot"]["seconds_per_optimizer_step"] == pytest.approx(0.1)
    assert manifest["pilot"]["full_test_evaluation_seconds"] == pytest.approx(5.0)
    assert manifest["projection"]["projected_seconds"] == pytest.approx(33_012.0)
    assert manifest["projection"]["status"] == "WITHIN_CAP"
    assert manifest["decision"] == {
        "reasons": [],
        "retained_training_authorized": True,
        "status": "PASS",
    }


def test_projection_above_48_gpu_hours_stops_retained_training() -> None:
    calibration = CalibrationAttempt(
        physical_batch_size=8,
        config_sha256="c" * 64,
        started_at_utc="2026-09-07T00:00:00+00:00",
        finished_at_utc="2026-09-07T00:02:00+00:00",
        fresh_process=True,
        rank=16,
        lora_alpha=32,
        effective_batch_size=32,
        target_optimizer_steps=10,
        completed_optimizer_steps=10,
        exit_status=0,
        oom=False,
        minimum_free_memory_mib=700.0,
    )
    slow_pilot = PilotEvidence(
        config_sha256="e" * 64,
        started_at_utc="2026-09-07T00:05:00+00:00",
        finished_at_utc="2026-09-07T01:00:00+00:00",
        fresh_process=True,
        physical_batch_size=8,
        rank=16,
        lora_alpha=32,
        effective_batch_size=32,
        learning_rate=1e-3,
        optimizer_steps=200,
        warmup_steps_excluded=20,
        non_evaluation_seconds_steps_21_to_200=90.0,
        full_test_evaluation_seconds=(30.0, 25.0),
        do_test=False,
        exit_status=0,
    )

    manifest = build_compute_gate_manifest(
        hardware=_hardware(),
        smoke=_smoke(),
        calibration_attempts=(calibration,),
        pilot=slow_pilot,
        actual_preflight_seconds=600.0,
        retained_config_sha256=tuple(f"{number:064x}" for number in range(1, 10)),
    )

    assert manifest["projection"]["projected_seconds"] == pytest.approx(173_472.0)
    assert manifest["projection"]["status"] == "OVER_CAP"
    assert manifest["decision"] == {
        "reasons": ["projected_time_exceeds_48_gpu_hour_cap"],
        "retained_training_authorized": False,
        "status": "STOP",
    }


def test_authorizer_fails_closed_for_stop_or_incomplete_pass_manifest() -> None:
    stopped = build_compute_gate_manifest(hardware=_hardware(), smoke=_smoke())
    forged_pass = {
        "schema_version": 1,
        "decision": {
            "status": "PASS",
            "retained_training_authorized": True,
            "reasons": [],
        },
    }

    with pytest.raises(ComputeGateError, match="STOP"):
        require_retained_training_authorization(
            stopped, resolved_config_sha256="1" * 64
        )
    with pytest.raises(ComputeGateError, match="incomplete"):
        require_retained_training_authorization(
            forged_pass, resolved_config_sha256="1" * 64
        )


def test_checked_environment_and_smoke_manifests_load_as_week1_evidence() -> None:
    smoke_manifest_path = (
        PROJECT_ROOT / "data" / "manifests" / "week1-smoke.json"
    )
    checked_smoke = json.loads(smoke_manifest_path.read_text(encoding="utf-8"))
    hardware, smoke = load_week1_evidence(
        PROJECT_ROOT / "data" / "manifests" / "environment-lock.json",
        smoke_manifest_path,
    )

    assert hardware == _hardware()
    assert smoke.verified is True
    assert checked_smoke["status"] == "verified"
    assert smoke.source_manifest_sha256 == sha256(
        smoke_manifest_path.read_bytes()
    ).hexdigest()
    assert smoke.resolved_config_sha256 == checked_smoke["configuration"][
        "resolved_config_sha256"
    ]
    assert smoke.optimizer_steps == checked_smoke["training"]["global_step"] == 1
    assert smoke.upstream_peak_memory_gb == pytest.approx(
        checked_smoke["measurement"]["upstream_peak_memory_gb"]
    )
    assert smoke.upstream_peak_memory_gb > 0


def test_authorizer_accepts_only_a_hash_listed_by_a_complete_pass() -> None:
    retained_hashes = tuple(f"{number:064x}" for number in range(1, 10))
    manifest = build_compute_gate_manifest(
        hardware=_hardware(),
        smoke=_smoke(),
        calibration_attempts=(
            CalibrationAttempt(
                physical_batch_size=8,
                config_sha256="c" * 64,
                started_at_utc="2026-09-07T00:00:00+00:00",
                finished_at_utc="2026-09-07T00:02:00+00:00",
                fresh_process=True,
                rank=16,
                lora_alpha=32,
                effective_batch_size=32,
                target_optimizer_steps=10,
                completed_optimizer_steps=10,
                exit_status=0,
                oom=False,
                minimum_free_memory_mib=700.0,
            ),
        ),
        pilot=PilotEvidence(
            config_sha256="e" * 64,
            started_at_utc="2026-09-07T00:05:00+00:00",
            finished_at_utc="2026-09-07T00:20:00+00:00",
            fresh_process=True,
            physical_batch_size=8,
            rank=16,
            lora_alpha=32,
            effective_batch_size=32,
            learning_rate=1e-3,
            optimizer_steps=200,
            warmup_steps_excluded=20,
            non_evaluation_seconds_steps_21_to_200=18.0,
            full_test_evaluation_seconds=(5.0, 4.5),
            do_test=False,
            exit_status=0,
        ),
        actual_preflight_seconds=600.0,
        retained_config_sha256=retained_hashes,
    )

    assert (
        require_retained_training_authorization(
            manifest, resolved_config_sha256=retained_hashes[0]
        )
        is None
    )
    with pytest.raises(ComputeGateError, match="not listed"):
        require_retained_training_authorization(
            manifest, resolved_config_sha256="f" * 64
        )


def test_week1_manifest_writer_is_deterministic_and_file_gate_stops_run(
    tmp_path: Path,
) -> None:
    output = tmp_path / "compute-gate.json"
    resolved_config = tmp_path / "resolved-config.json"
    resolved_config.write_bytes(b"{}\n")
    first = write_week1_stop_manifest(
        environment_manifest_path=(
            PROJECT_ROOT / "data" / "manifests" / "environment-lock.json"
        ),
        smoke_manifest_path=(
            PROJECT_ROOT / "data" / "manifests" / "week1-smoke.json"
        ),
        output_path=output,
    )
    first_bytes = output.read_bytes()
    second = write_week1_stop_manifest(
        environment_manifest_path=(
            PROJECT_ROOT / "data" / "manifests" / "environment-lock.json"
        ),
        smoke_manifest_path=(
            PROJECT_ROOT / "data" / "manifests" / "week1-smoke.json"
        ),
        output_path=output,
    )

    assert first == second
    assert output.read_bytes() == first_bytes
    with pytest.raises(ComputeGateError, match="STOP"):
        require_authorization_from_files(
            gate_manifest_path=output,
            resolved_config_path=resolved_config,
        )


def test_checked_compute_gate_matches_locked_week1_evidence() -> None:
    hardware, smoke = load_week1_evidence(
        PROJECT_ROOT / "data" / "manifests" / "environment-lock.json",
        PROJECT_ROOT / "data" / "manifests" / "week1-smoke.json",
    )
    expected = build_compute_gate_manifest(hardware=hardware, smoke=smoke)
    checked = json.loads(
        (PROJECT_ROOT / "data" / "manifests" / "compute-gate.json").read_text(
            encoding="utf-8"
        )
    )

    assert checked == expected
    assert checked["decision"]["status"] == "STOP"
