#!/usr/bin/env python3
"""Reload a Week 1 OpenDelta checkpoint and verify its retained evaluation."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_LOADER_REVISION = "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
LOSS_TOLERANCE = 1e-4


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_tensor_comparison(
    retained_path: Path, delta_path: Path
) -> tuple[bool, float]:
    import torch

    retained = torch.load(retained_path, map_location="cpu")
    delta = torch.load(delta_path, map_location="cpu")
    if retained.keys() != delta.keys():
        return False, float("inf")
    maximum = max(
        (retained[name] - delta[name]).abs().max().item() for name in retained
    )
    return True, float(maximum)


def _evaluate_reloaded_checkpoint(
    *, artifact_root: Path, values: dict[str, Any]
) -> tuple[str, str, str, float]:
    import torch
    from opendelta import AutoDeltaModel
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

    from ds340w_repro.data.materialize_sst2 import (
        TEST_FILENAME,
        verify_course_data,
    )

    model_path = values["model_name_or_path"]
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    config.dropout_rate = 0.0
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        config=config,
        local_files_only=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    AutoDeltaModel.from_finetuned(
        str(artifact_root / "delta"),
        backbone_model=model,
    )
    model.parallelize()
    model.eval()

    cache_root = Path(
        os.environ.get("DS340W_CACHE_ROOT", Path.home() / ".cache" / "ds340w")
    )
    course_data_root = cache_root / "course-data"
    course_manifest = PROJECT_ROOT / "data/manifests/sst2-course-data.json"
    test_indices_path = PROJECT_ROOT / "data/manifests/sst2-test-indices.json"
    course_report = verify_course_data(
        output_root=course_data_root,
        manifest_path=course_manifest,
        test_indices_path=test_indices_path,
    )
    if course_report["manifest_sha256"] != values["course_data_manifest_sha256"]:
        raise RuntimeError("isolated course-data manifest is not configuration-bound")
    if _file_sha256(test_indices_path) != values["test_indices_sha256"]:
        raise RuntimeError("course Test indices are not configuration-bound")
    rows = [
        json.loads(line)
        for line in (course_data_root / TEST_FILENAME).read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    generator = torch.Generator().manual_seed(int(values["data_sample_seed"]))
    selected_position = torch.randperm(len(rows), generator=generator).tolist()[0]
    row = rows[selected_position]
    stable_id = f"validation:{row['idx']}"
    source = f"sst2 sentence: {row['sentence']}"
    target = str(row["label"])
    inputs = tokenizer(
        source,
        max_length=int(values["max_source_length"]),
        padding=False,
        truncation=True,
        return_tensors="pt",
    )
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            target,
            max_length=2,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )["input_ids"]
    inputs = {name: tensor.to("cuda:0") for name, tensor in inputs.items()}
    labels = labels.to("cuda:0")
    with torch.no_grad():
        loss = float(model(**inputs, labels=labels).loss.item())
        generated = model.generate(
            **inputs,
            max_length=int(values["generation_max_length"]),
            num_beams=int(values["generation_num_beams"]),
        )
    prediction = tokenizer.batch_decode(
        generated,
        skip_special_tokens=True,
    )[0].strip()
    return stable_id, target, prediction, loss


def verify(artifact_root: Path) -> dict[str, Any]:
    from ds340w_repro.config_schema import ResolvedConfig
    from ds340w_repro.run_opendelta import verify_model_snapshot_binding

    artifact_root = artifact_root.resolve(strict=True)
    resolved_path = artifact_root / "resolved-config.json"
    resolved_text = resolved_path.read_text(encoding="utf-8")
    resolved = _read_json(resolved_path)
    resolved_config = ResolvedConfig.from_json(resolved_text)
    values = resolved.get("values")
    if not isinstance(values, dict):
        raise RuntimeError("resolved configuration has no values object")
    if values.get("do_test") is not False or values.get("max_val_samples") != 1:
        raise RuntimeError("reload verification only accepts the non-final one-row smoke")
    if values.get("dataset_loader_revision") != DATASET_LOADER_REVISION:
        raise RuntimeError("dataset loader revision is not the approved immutable pin")
    cache_root = Path(
        os.environ.get("DS340W_CACHE_ROOT", Path.home() / ".cache" / "ds340w")
    )
    model_input_binding = verify_model_snapshot_binding(
        resolved_config, PROJECT_ROOT, cache_root
    )

    tensor_sets_match, tensor_max_abs_difference = _checkpoint_tensor_comparison(
        artifact_root / "pytorch_model.bin",
        artifact_root / "delta" / "pytorch_model.bin",
    )
    stable_id, target, prediction, reloaded_loss = _evaluate_reloaded_checkpoint(
        artifact_root=artifact_root,
        values=values,
    )
    retained_metrics = _read_json(artifact_root / "eval_results.json")
    retained_loss = float(retained_metrics["eval_loss"])
    retained_accuracy = float(retained_metrics["eval_accuracy"])
    reloaded_accuracy = float(prediction == target)
    loss_difference = abs(retained_loss - reloaded_loss)
    passed = (
        tensor_sets_match
        and tensor_max_abs_difference == 0.0
        and retained_accuracy == reloaded_accuracy
        and loss_difference <= LOSS_TOLERANCE
    )
    return {
        "abs_eval_loss_difference": loss_difference,
        "checkpoint_relative_path": "delta",
        "checkpoint_tensor_keys_match": tensor_sets_match,
        "checkpoint_tensor_max_abs_difference": tensor_max_abs_difference,
        "passed": passed,
        "model_input_binding": model_input_binding,
        "prediction_reconstruction_basis": (
            "retained and delta checkpoint tensors are exactly equal"
        ),
        "predictions_match": tensor_sets_match
        and tensor_max_abs_difference == 0.0,
        "reloaded_correct_count": int(reloaded_accuracy),
        "reloaded_eval_loss": reloaded_loss,
        "reloaded_prediction": prediction,
        "resolved_config_sha256": _file_sha256(resolved_path),
        "retained_eval_loss": retained_loss,
        "retained_correct_count": int(retained_accuracy),
        "retained_prediction": prediction,
        "schema_version": 1,
        "stable_id": stable_id,
        "tolerance": LOSS_TOLERANCE,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--write-result", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    cache_root = Path(
        os.environ.get("DS340W_CACHE_ROOT", Path.home() / ".cache" / "ds340w")
    )
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache_root / "datasets"))
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.pop("DS340W_PRIVATE_ROOT", None)

    result = verify(args.artifact_root)
    serialized = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write_result is not None:
        args.write_result.parent.mkdir(parents=True, exist_ok=True)
        args.write_result.write_text(serialized, encoding="utf-8", newline="\n")
    sys.stdout.write(serialized)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
