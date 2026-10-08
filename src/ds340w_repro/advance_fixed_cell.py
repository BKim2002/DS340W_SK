"""Advance the predeclared fixed-cell study only from verified run artifacts."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Sequence

from ds340w_repro.build_matrix import build_confirmation_matrix
from ds340w_repro.checkpoints import verify_checkpoint_artifact
from ds340w_repro.config_schema import ResolvedConfig, validate_nonfinal_config
from ds340w_repro.preflight import authorize_confirmation_stage
from ds340w_repro.run_preflight import authorize_rank_configs


TEST_ROWS = 436
RANKS = (2, 4, 8, 16)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _directory_sha256(root: Path) -> str:
    digest = sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("Retained artifact is empty or contains symbolic links")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _verified_result(
    *, manifest_path: Path, config_path: Path, expected_experiment_id: str
) -> dict[str, Any]:
    result = _read_json(manifest_path)
    config_path = config_path.resolve(strict=True)
    config = ResolvedConfig.from_json(config_path.read_text(encoding="utf-8"))
    validate_nonfinal_config(config)
    if (
        result.get("schema_version") != 1
        or result.get("artifact_kind") != "retained-run-result"
        or result.get("status") != "verified"
        or result.get("experiment_id") != expected_experiment_id
        or config.experiment_id != expected_experiment_id
        or result.get("config_sha256") != config.sha256
        or sha256(config_path.read_bytes()).hexdigest() != config.sha256
        or result.get("final_unseen_accessed") is not False
    ):
        raise ValueError(f"Run result is not bound to {expected_experiment_id}")
    artifact_path = Path(str(result.get("artifact_path")))
    if not artifact_path.is_absolute():
        raise ValueError("Retained artifact path must be absolute")
    artifact_path = artifact_path.resolve(strict=True)
    verification = verify_checkpoint_artifact(artifact_path)
    if (
        verification["manifest_sha256"] != result.get("artifact_manifest_sha256")
        or _directory_sha256(artifact_path)
        != result.get("artifact_directory_sha256")
    ):
        raise ValueError("Retained artifact checksums do not match")
    if not isinstance(result.get("checkpoint_step"), int) or not 100 <= result["checkpoint_step"] <= 20_000 or result["checkpoint_step"] % 100:
        raise ValueError("Retained checkpoint step is invalid")
    count = result.get("correct_count")
    accuracy = result.get("accuracy_percent")
    if (
        type(count) is not int
        or not 0 <= count <= TEST_ROWS
        or type(accuracy) not in (int, float)
        or not math.isclose(accuracy, 100 * count / TEST_ROWS, abs_tol=1e-6)
    ):
        raise ValueError("Retained result is not a 436-row Test accuracy")
    predictions = (artifact_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    if len(predictions) != TEST_ROWS:
        raise ValueError("Retained result does not have 436 predictions")
    parsed = [json.loads(line) for line in predictions]
    if (
        len({row["test_id"] for row in parsed}) != TEST_ROWS
        or any(row["config_sha256"] != config.sha256 for row in parsed)
        or any(row["checkpoint_step"] != result["checkpoint_step"] for row in parsed)
        or sum(row["label"] == row["prediction"] for row in parsed) != count
    ):
        raise ValueError("Retained predictions disagree with result manifest")
    wall = result.get("original_process_wall_seconds")
    if type(wall) not in (int, float) or not math.isfinite(wall) or wall <= 0:
        raise ValueError("Retained process wall time is invalid")
    return result


def _check_runtime_cap(
    gate: dict[str, Any], *, completed_wall_seconds: Sequence[float]
) -> float:
    projection = gate["projection"]
    run_plan = projection["retained_run_plan"]
    total_runs = run_plan["training_runs"]
    if total_runs != 8 or not 0 < len(completed_wall_seconds) <= total_runs:
        raise ValueError("Runtime reassessment requires one to eight completed runs")
    per_run = (
        20_000 * projection["seconds_per_optimizer_step"]
        + 200 * projection["full_test_evaluation_seconds"]
    )
    projected = projection["actual_preflight_seconds"] + 1.20 * (
        sum(completed_wall_seconds)
        + (total_runs - len(completed_wall_seconds)) * per_run
        + run_plan["final_evaluations"] * projection["full_test_evaluation_seconds"]
    )
    if projected > projection["cap_seconds"]:
        raise ValueError(
            f"Updated projection {projected / 3600:.2f} GPU-hours exceeds the 48-hour cap"
        )
    return projected


def advance_to_rank(
    *, project_root: Path, baseline_result_path: Path,
    selection_manifest_path: Path
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    gate_path = project_root / "data/manifests/compute-gate.json"
    config_path = project_root / "configs/resolved/baseline/baseline-r8-lr1e-4-eb16.json"
    result = _verified_result(
        manifest_path=baseline_result_path,
        config_path=config_path,
        expected_experiment_id="baseline-r8-lr1e-4-eb16",
    )
    gate = _read_json(gate_path)
    revised_projection = _check_runtime_cap(
        gate, completed_wall_seconds=(result["original_process_wall_seconds"],)
    )
    selection = {
        "schema_version": 1,
        "artifact_kind": "baseline-selection",
        "status": "verified",
        "completed_config_sha256": [result["config_sha256"]],
        "selected_config_sha256": result["config_sha256"],
        "selected_checkpoint_path": result["artifact_path"],
        "selected_checkpoint_sha256": result["artifact_directory_sha256"],
        "selected_checkpoint_step": result["checkpoint_step"],
        "correct_count": result["correct_count"],
        "updated_projected_gpu_hours": revised_projection / 3600,
    }
    selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    authorize_rank_configs(
        gate_manifest_path=gate_path,
        baseline_selection_manifest_path=selection_manifest_path,
        selected_baseline_config_path=config_path,
        rank_eight_checkpoint=Path(result["artifact_path"]),
        resolved_config_root=project_root / "configs/resolved",
        retained_output_root=Path("artifacts/runs"),
    )
    return selection


def _select_rank(correct_by_rank: dict[int, int]) -> int:
    if set(correct_by_rank) != set(RANKS):
        raise ValueError("Rank selection requires all four declared ranks")
    best = max(correct_by_rank.values())
    return min(
        rank for rank, count in correct_by_rank.items()
        if 100 * (best - count) / TEST_ROWS <= 0.5
    )


def advance_to_confirmation(
    *, project_root: Path, rank_result_paths: dict[int, Path],
    baseline_result_path: Path, selection_manifest_path: Path
) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    if set(rank_result_paths) != {2, 4, 16}:
        raise ValueError("Rank stage requires exactly ranks 2, 4, and 16")
    baseline_config_path = project_root / "configs/resolved/baseline/baseline-r8-lr1e-4-eb16.json"
    baseline = _verified_result(
        manifest_path=baseline_result_path, config_path=baseline_config_path,
        expected_experiment_id="baseline-r8-lr1e-4-eb16",
    )
    results = {8: baseline}
    for rank, manifest_path in rank_result_paths.items():
        results[rank] = _verified_result(
            manifest_path=manifest_path,
            config_path=project_root / f"configs/resolved/rank/rank-r{rank}.json",
            expected_experiment_id=f"rank-r{rank}",
        )
    baseline_predictions = [
        json.loads(line)
        for line in (
            Path(baseline["artifact_path"]) / "predictions.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    reference_rows = [
        (row["test_id"], row["label"]) for row in baseline_predictions
    ]
    for rank in (2, 4, 16):
        comparison = [
            json.loads(line)
            for line in (
                Path(results[rank]["artifact_path"]) / "predictions.jsonl"
            ).read_text(encoding="utf-8").splitlines()
        ]
        if [(row["test_id"], row["label"]) for row in comparison] != reference_rows:
            raise ValueError("Rank results do not align on the same Test examples")
    gate_path = project_root / "data/manifests/compute-gate.json"
    gate = _read_json(gate_path)
    if gate["authorization"]["rank"]["authorized"] is not True:
        raise ValueError("Rank stage was not authorized")
    if set(gate["authorization"]["rank"]["config_sha256"]) != {
        results[rank]["config_sha256"] for rank in (2, 4, 16)
    }:
        raise ValueError("Completed rank results do not match authorized configurations")
    revised_projection = _check_runtime_cap(
        gate,
        completed_wall_seconds=[
            results[rank]["original_process_wall_seconds"] for rank in (8, 2, 4, 16)
        ],
    )
    correct_by_rank = {rank: results[rank]["correct_count"] for rank in RANKS}
    selected_rank = _select_rank(correct_by_rank)
    selection = {
        "schema_version": 1,
        "artifact_kind": "rank-selection",
        "status": "verified",
        "selected_rank": selected_rank,
        "correct_by_rank": {str(rank): correct_by_rank[rank] for rank in RANKS},
        "result_config_sha256_by_rank": {
            str(rank): results[rank]["config_sha256"] for rank in RANKS
        },
        "selected_checkpoint_path": results[selected_rank]["artifact_path"],
        "selected_checkpoint_sha256": results[selected_rank]["artifact_directory_sha256"],
        "updated_projected_gpu_hours": revised_projection / 3600,
        "final_unseen_accessed": False,
    }
    baseline_config = ResolvedConfig.from_json(
        baseline_config_path.read_text(encoding="utf-8")
    )
    configs = build_confirmation_matrix(
        Path(str(baseline_config.values["model_name_or_path"])),
        Path("artifacts/runs"),
        baseline_config.values["per_device_train_batch_size"],
        baseline_config.values["per_device_eval_batch_size"],
        selected_rank,
    )
    updated_gate = authorize_confirmation_stage(
        gate, selected_rank=selected_rank,
        confirmation_config_sha256=[config.sha256 for config in configs],
    )
    selection_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selection_manifest_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    for config in configs:
        config.write(
            project_root / "configs/resolved/confirmation" / f"{config.experiment_id}.json"
        )
    gate_path.write_text(
        json.dumps(updated_gate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    return selection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("rank", "confirmation"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve(strict=True)
    baseline_result = root / "data/manifests/retained-baseline-r8-lr1e-4-eb16.json"
    if args.stage == "rank":
        result = advance_to_rank(
            project_root=root, baseline_result_path=baseline_result,
            selection_manifest_path=root / "data/manifests/baseline-selection.json",
        )
    else:
        result = advance_to_confirmation(
            project_root=root,
            rank_result_paths={
                rank: root / f"data/manifests/retained-rank-r{rank}.json"
                for rank in (2, 4, 16)
            },
            baseline_result_path=baseline_result,
            selection_manifest_path=root / "data/manifests/rank-selection.json",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
