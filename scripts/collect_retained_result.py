#!/usr/bin/env python3
"""Verify and preserve one completed OpenDelta SST-2 retained run.

Training and checkpoint selection remain in the authors' pinned runner. This
observer reconstructs its selected 436 Test predictions from the saved delta
weights, checks them against the upstream metrics, and preserves native files.
It never reads the course's private Validation (Unseen) partition.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEST_ROWS = 436


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_sha256(root: Path) -> str:
    digest = sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("Checkpoint artifact is empty or contains a symlink")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _selected_checkpoint(state: dict[str, Any], output_dir: Path) -> tuple[Path, int, int]:
    from ds340w_repro.checkpoints import CheckpointEvaluation, select_checkpoint

    candidates = []
    for row in state.get("log_history", []):
        if "eval_accuracy" not in row:
            continue
        step = row.get("step")
        accuracy = row.get("eval_accuracy")
        if type(step) is not int or type(accuracy) not in (int, float):
            raise ValueError("Malformed upstream checkpoint evaluation history")
        correct = round(accuracy * TEST_ROWS / 100)
        if not math.isclose(accuracy, 100 * correct / TEST_ROWS, abs_tol=1e-6):
            raise ValueError("Upstream accuracy is not a 436-row correct count")
        candidates.append(
            CheckpointEvaluation(output_dir / f"checkpoint-{step}", step, correct)
        )
    if len(candidates) != 200 or state.get("global_step") != 20_000:
        raise ValueError("Retained run lacks its 200 checkpoint evaluations")
    selected = select_checkpoint(candidates)
    actual_best = Path(str(state.get("best_model_checkpoint"))).resolve(strict=True)
    if actual_best != selected.checkpoint_path.resolve(strict=True):
        raise ValueError("Upstream best checkpoint disagrees with the declared tie rule")
    if not math.isclose(
        float(state.get("best_metric")), 100 * selected.correct_count / TEST_ROWS,
        abs_tol=1e-6,
    ):
        raise ValueError("Upstream best metric disagrees with checkpoint history")
    return actual_best, selected.step, selected.correct_count


def _compare_saved_tensors(checkpoint: Path, output_dir: Path) -> None:
    import torch

    original = torch.load(checkpoint / "pytorch_model.bin", map_location="cpu")
    delta = torch.load(output_dir / "delta" / "pytorch_model.bin", map_location="cpu")
    if original.keys() != delta.keys() or not all(
        torch.equal(original[name], delta[name]) for name in original
    ):
        raise ValueError("Saved delta is not tensor-identical to the selected checkpoint")


def _predict_test(
    *, output_dir: Path, config: Any, test_path: Path
) -> tuple[list[str], list[int], list[int], float]:
    import torch
    from opendelta import AutoDeltaModel
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    values = config.values
    rows = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != TEST_ROWS:
        raise ValueError("Course Test file does not contain exactly 436 rows")
    indices = torch.randperm(
        len(rows), generator=torch.Generator().manual_seed(values["data_sample_seed"])
    ).tolist()
    selected_rows = [rows[position] for position in indices[:TEST_ROWS]]
    ids = [f"validation:{row['idx']}" for row in selected_rows]
    labels = [int(row["label"]) for row in selected_rows]
    if len(set(ids)) != TEST_ROWS or any(label not in (0, 1) for label in labels):
        raise ValueError("Course Test IDs or labels are invalid")

    model_path = values["model_name_or_path"]
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, config=model_config, local_files_only=True
    )
    model.resize_token_embeddings(len(tokenizer))
    AutoDeltaModel.from_finetuned(str(output_dir / "delta"), backbone_model=model)
    model.parallelize()
    model.eval()

    predictions: list[int] = []
    weighted_loss = 0.0
    batch_size = int(values["per_device_eval_batch_size"])
    target_max = max(len(tokenizer.encode(label)) for label in ("0", "1"))
    for start in range(0, TEST_ROWS, batch_size):
        batch = selected_rows[start : start + batch_size]
        texts = [f"sst2 sentence: {row['sentence']}" for row in batch]
        inputs = tokenizer(
            texts, max_length=values["max_source_length"], padding=True,
            truncation=True, return_tensors="pt",
        )
        with tokenizer.as_target_tokenizer():
            targets = tokenizer(
                [str(row["label"]) for row in batch], max_length=target_max,
                padding=True, truncation=True, return_tensors="pt",
            )["input_ids"]
        targets[targets == tokenizer.pad_token_id] = -100
        inputs = {key: tensor.to("cuda:0") for key, tensor in inputs.items()}
        targets = targets.to("cuda:0")
        with torch.no_grad():
            loss = model(**inputs, labels=targets).loss.item()
            generated = model.generate(
                **inputs,
                max_length=int(values["val_max_target_length"]),
                num_beams=int(values["num_beams"]),
            )
        weighted_loss += float(loss) * len(batch)
        decoded = [
            value.strip() for value in tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
        ]
        if any(value not in ("0", "1") for value in decoded):
            raise ValueError("Saved model generated a non-binary SST-2 prediction")
        predictions.extend(int(value) for value in decoded)
    return ids, labels, predictions, weighted_loss / TEST_ROWS


def collect(*, config_path: Path, artifact_dir: Path) -> dict[str, Any]:
    from ds340w_repro.checkpoints import (
        build_prediction_records, preserve_checkpoint_artifact,
        validate_checkpoint_reload, verify_checkpoint_artifact,
    )
    from ds340w_repro.config_schema import ResolvedConfig, validate_nonfinal_config
    from ds340w_repro.run_opendelta import (
        verify_course_data_binding, verify_model_snapshot_binding,
    )

    config_path = config_path.resolve(strict=True)
    config = ResolvedConfig.from_json(config_path.read_text(encoding="utf-8"))
    validate_nonfinal_config(config)
    if _file_sha256(config_path) != config.sha256:
        raise ValueError("Resolved configuration file is not canonical")
    values = config.values
    if values.get("do_test") is not False or values.get("max_val_samples") != TEST_ROWS:
        raise ValueError("Observer accepts only a full course-Test nonfinal run")
    output_dir = PROJECT_ROOT / values["output_dir"]
    output_dir = output_dir.resolve(strict=True)
    process = _read_json(output_dir / "process-result.json")
    if process.get("exit_code") != 0 or process.get("resolved_config_sha256") != config.sha256:
        raise ValueError("Original OpenDelta run did not finish cleanly")
    state = _read_json(output_dir / "trainer_state.json")
    checkpoint, step, expected_correct = _selected_checkpoint(state, output_dir)
    upstream_metrics = _read_json(output_dir / "eval_results.json")
    if not math.isclose(
        float(upstream_metrics["eval_accuracy"]),
        100 * expected_correct / TEST_ROWS, abs_tol=1e-6,
    ):
        raise ValueError("Final upstream evaluation disagrees with selected checkpoint")
    _compare_saved_tensors(checkpoint, output_dir)

    execution = _read_json(output_dir / "execution-tree.json")
    execution_root = Path(str(execution.get("root"))).resolve(strict=True)
    if not execution_root.is_relative_to(PROJECT_ROOT / "runtime"):
        raise ValueError("Original execution tree is outside this project")
    sys.path.insert(0, str(execution_root))

    cache_root = Path(
        os.environ.get("DS340W_CACHE_ROOT", Path.home() / ".cache" / "ds340w")
    )
    verify_model_snapshot_binding(config, PROJECT_ROOT, cache_root)
    _, _, test_path = verify_course_data_binding(config, PROJECT_ROOT, cache_root)
    ids, labels, predictions, observed_loss = _predict_test(
        output_dir=output_dir, config=config, test_path=test_path
    )
    observed_correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    validate_checkpoint_reload(
        expected_correct_count=expected_correct,
        expected_eval_loss=float(upstream_metrics["eval_loss"]),
        actual_correct_count=observed_correct,
        actual_eval_loss=observed_loss,
    )
    records = build_prediction_records(
        test_ids=ids, labels=labels, predictions=predictions,
        checkpoint_step=step, config_sha256=config.sha256,
    )
    metrics = {
        "accuracy_percent": 100 * observed_correct / TEST_ROWS,
        "correct_count": observed_correct,
        "eval_loss": observed_loss,
        "original_eval_loss": upstream_metrics["eval_loss"],
        "test_rows": TEST_ROWS,
        "upstream_peak_memory_gb": _read_json(output_dir / "performance_results.json")["peak_memory"],
        "wall_seconds": process["wall_seconds"],
    }
    preserved = preserve_checkpoint_artifact(
        source_checkpoint=checkpoint,
        resolved_config_path=config_path,
        trainer_state_path=output_dir / "trainer_state.json",
        predictions=records,
        metrics=metrics,
        artifact_dir=artifact_dir,
    )
    verified = verify_checkpoint_artifact(preserved.artifact_dir)
    return {
        "schema_version": 1,
        "artifact_kind": "retained-run-result",
        "status": "verified" if verified["valid"] else "invalid",
        "experiment_id": config.experiment_id,
        "config_sha256": config.sha256,
        "checkpoint_step": step,
        "correct_count": observed_correct,
        "accuracy_percent": metrics["accuracy_percent"],
        "artifact_path": str(preserved.artifact_dir),
        "artifact_manifest_sha256": preserved.manifest_sha256,
        "artifact_directory_sha256": _directory_sha256(preserved.artifact_dir),
        "original_process_wall_seconds": process["wall_seconds"],
        "final_unseen_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--result-manifest", type=Path, required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.pop("DS340W_PRIVATE_ROOT", None)
    os.environ.pop("DS340W_SST2_FINAL_FILE", None)
    result = collect(config_path=args.config, artifact_dir=args.artifact_dir)
    args.result_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.result_manifest.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8", newline="\n",
    )
    sys.stdout.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
