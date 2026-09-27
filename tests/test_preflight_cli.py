from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from ds340w_repro.preflight import DEFAULT_RETAINED_RUN_PLAN
from ds340w_repro.run_preflight import (
    ObservedRun,
    _evaluation_evidence,
    authorize_rank_configs,
    run_preflight,
)
from ds340w_repro.build_matrix import build_evaluation_calibration_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_dry_run_declares_all_three_preflight_phases(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ds340w_repro.run_preflight",
            "--project-root",
            str(PROJECT_ROOT),
            "--model-path",
            "/verified/model",
            "--workspace",
            str(tmp_path / "preflight"),
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)
    assert plan["evaluation_batch_candidates"] == [32, 16, 8, 4, 2, 1]
    assert plan["training_microbatch_candidates"] == [8, 4, 2, 1]
    assert plan["pilot"] == {
        "effective_batch_size": 32,
        "evaluations_at_steps": [100, 200],
        "learning_rate": 0.001,
        "optimizer_steps": 200,
        "rank": 16,
        "warmup_steps_excluded": 20,
    }
    assert plan["final_unseen_access"] is False
    assert plan["retained_run_plan"]["training_runs"] == 13
    assert plan["retained_run_plan"]["stages"][-1] == {
        "name": "confirmation",
        "training_runs": 4,
        "optimizer_steps_per_run": 20_000,
        "evaluations_per_run": 200,
    }


def test_cli_can_deduplicate_confirmations_when_rank_eight_is_selected(
    tmp_path: Path,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ds340w_repro.run_preflight",
            "--project-root",
            str(PROJECT_ROOT),
            "--model-path",
            "/verified/model",
            "--workspace",
            str(tmp_path / "preflight"),
            "--confirmation-runs",
            "2",
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    plan = json.loads(completed.stdout)["retained_run_plan"]
    assert plan["training_runs"] == 11
    assert plan["stages"][-1]["name"] == "confirmation"
    assert plan["stages"][-1]["training_runs"] == 2


def test_evaluation_calibration_counts_the_measured_samples(
    tmp_path: Path,
) -> None:
    config = build_evaluation_calibration_config(
        Path("/verified/model"), tmp_path / "eval", 16
    )
    output_dir = Path(config.values["output_dir"])
    output_dir.mkdir(parents=True)
    (output_dir / "eval_results.json").write_text(
        json.dumps(
            {
                "eval_runtime": 5.0,
                "eval_samples_per_second": 0.2,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    observed = ObservedRun(
        returncode=0,
        started_at_utc="2026-09-07T00:00:00+00:00",
        finished_at_utc="2026-09-07T00:01:00+00:00",
        wall_seconds=60.0,
        minimum_free_memory_mib=900.0,
        oom=False,
        output_dir=output_dir,
    )

    evidence = _evaluation_evidence(config, observed)

    assert evidence.evaluated_samples == 1
    assert evidence.passed is False


def test_preflight_orchestrates_fresh_attempts_and_writes_baseline_gate(
    tmp_path: Path,
) -> None:
    attempted: list[str] = []

    def execute_config(*, config, **_kwargs) -> ObservedRun:
        attempted.append(config.experiment_id)
        output_dir = Path(config.values["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        returncode = 0
        minimum_free = 900.0
        oom = False
        if config.experiment_id == "preflight-eval-eb32":
            returncode, minimum_free, oom = 1, 100.0, True
        elif config.experiment_id == "preflight-eval-eb16":
            (output_dir / "eval_results.json").write_text(
                '{"eval_runtime": 5.0, "eval_samples_per_second": 87.2}\n',
                encoding="utf-8",
            )
        elif config.experiment_id.startswith("preflight-train-pb8"):
            returncode, minimum_free, oom = 1, 100.0, True
        elif config.experiment_id.startswith("preflight-train-pb4"):
            (output_dir / "trainer_state.json").write_text(
                '{"global_step": 10}\n', encoding="utf-8"
            )
        elif config.experiment_id.startswith("preflight-pilot"):
            (output_dir / "preflight-timing.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "completed_optimizer_steps": 200,
                        "warmup_steps_excluded": 20,
                        "evaluation_measurements_seconds": [5.0, 4.5],
                        "non_evaluation_seconds_steps_21_to_end": 18.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        return ObservedRun(
            returncode=returncode,
            started_at_utc="2026-09-07T00:00:00+00:00",
            finished_at_utc="2026-09-07T00:01:00+00:00",
            wall_seconds=60.0,
            minimum_free_memory_mib=minimum_free,
            oom=oom,
            output_dir=output_dir,
        )

    gate_path = tmp_path / "compute-gate.json"
    config_root = tmp_path / "resolved"
    manifest = run_preflight(
        project_root=PROJECT_ROOT,
        model_path=Path("/verified/model"),
        workspace=tmp_path / "workspace",
        python_executable=Path(sys.executable),
        gate_manifest_path=gate_path,
        resolved_config_root=config_root,
        retained_output_root=Path("artifacts/runs"),
        retained_run_plan=DEFAULT_RETAINED_RUN_PLAN,
        execute_config=execute_config,
    )

    assert attempted == [
        "preflight-eval-eb32",
        "preflight-eval-eb16",
        "preflight-train-pb8-eb16",
        "preflight-train-pb4-eb16",
        "preflight-pilot-r16-pb4-eb16",
    ]
    assert manifest["decision"]["status"] == "PASS"
    assert manifest["authorization"]["baseline"]["authorized"] is True
    assert manifest["authorization"]["rank"]["authorized"] is False
    assert len(list((config_root / "baseline").glob("*.json"))) == 6
    assert json.loads(gate_path.read_text(encoding="utf-8")) == manifest

    selected_baseline = sorted((config_root / "baseline").glob("*.json"))[0]
    updated = authorize_rank_configs(
        gate_manifest_path=gate_path,
        selected_baseline_config_path=selected_baseline,
        rank_eight_checkpoint=Path("artifacts/selected-rank8"),
        resolved_config_root=config_root,
        retained_output_root=Path("artifacts/runs"),
    )

    rank_paths = sorted((config_root / "rank").glob("*.json"))
    assert [path.stem for path in rank_paths] == ["rank-r16", "rank-r2", "rank-r4"]
    assert updated["authorization"]["rank"]["authorized"] is True
    assert len(updated["authorization"]["rank"]["config_sha256"]) == 3

