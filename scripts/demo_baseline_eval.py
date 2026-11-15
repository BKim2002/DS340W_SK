#!/usr/bin/env python3
"""Read-only Week 10 demonstration of the retained OpenDelta rank-8 model.

The original OpenDelta loader and the project's existing SST-2 evaluator do
the model work. A temporary directory lets that loader read the preserved
native checkpoint without changing the retained artifact or retraining.
"""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = PROJECT_ROOT / "artifacts/retained/baseline-r8-lr1e-4-eb16"
RUN = PROJECT_ROOT / "artifacts/runs/baseline-r8-lr1e-4-eb16"
EXPECTED_ROWS = 436
EXPECTED_CORRECT = 412


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def main() -> int:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.pop("DS340W_PRIVATE_ROOT", None)
    os.environ.pop("DS340W_SST2_FINAL_FILE", None)
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

    from ds340w_repro.checkpoints import verify_checkpoint_artifact
    from ds340w_repro.config_schema import ResolvedConfig, validate_nonfinal_config
    from ds340w_repro.run_opendelta import (
        verify_course_data_binding, verify_model_snapshot_binding,
    )
    from collect_retained_result import _predict_test

    verify_checkpoint_artifact(ARTIFACT)
    config_path = ARTIFACT / "resolved-config.json"
    config = ResolvedConfig.from_json(config_path.read_text(encoding="utf-8"))
    validate_nonfinal_config(config)
    if sha256(config_path.read_bytes()).hexdigest() != config.sha256:
        raise ValueError("Retained configuration hash differs from its content")
    values = config.values
    if values.get("do_test") is not False or values.get("max_val_samples") != EXPECTED_ROWS:
        raise ValueError("This demo requires the nonfinal 436-row Test configuration")
    process = read_json(RUN / "process-result.json")
    if process.get("exit_code") != 0 or process.get("resolved_config_sha256") != config.sha256:
        raise ValueError("Original OpenDelta training record is missing or inconsistent")
    execution = read_json(RUN / "execution-tree.json")
    execution_root = Path(str(execution.get("root"))).resolve(strict=True)
    if not execution_root.is_relative_to(PROJECT_ROOT / "runtime"):
        raise ValueError("Original OpenDelta execution tree is outside the project")
    sys.path.insert(0, str(execution_root))

    cache_root = Path(os.environ.get("DS340W_CACHE_ROOT", Path.home() / ".cache/ds340w"))
    verify_model_snapshot_binding(config, PROJECT_ROOT, cache_root)
    _, _, test_path = verify_course_data_binding(config, PROJECT_ROOT, cache_root)

    retained = [json.loads(line) for line in (ARTIFACT / "predictions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()]
    if len(retained) != EXPECTED_ROWS or any(
        row["checkpoint_step"] != 11600 or row["config_sha256"] != config.sha256
        for row in retained
    ):
        raise ValueError("Retained prediction record is inconsistent")
    if read_json(ARTIFACT / "metrics.json")["correct_count"] != EXPECTED_CORRECT:
        raise ValueError("Retained result differs from expected baseline")

    # OpenDelta's original from_finetuned loader expects these two files side
    # by side. Copy only to a disposable directory, using the retained weights.
    with TemporaryDirectory(prefix="ds340w-week10-eval-") as directory:
        delta = Path(directory) / "delta"
        delta.mkdir()
        shutil.copyfile(RUN / "delta/config.json", delta / "config.json")
        shutil.copyfile(
            ARTIFACT / "native-checkpoint/pytorch_model.bin",
            delta / "pytorch_model.bin",
        )
        ids, labels, predictions, loss = _predict_test(
            output_dir=Path(directory), config=config, test_path=test_path
        )

    observed = [
        (test_id, label, prediction)
        for test_id, label, prediction in zip(ids, labels, predictions)
    ]
    saved = [
        (row["test_id"], row["label"], row["prediction"])
        for row in retained
    ]
    if observed != saved:
        raise ValueError("Live predictions differ from the retained Test record")
    correct = sum(label == prediction for _, label, prediction in observed)
    if correct != EXPECTED_CORRECT:
        raise ValueError("Live accuracy differs from the retained result")
    print(f"Course Test: {correct}/{len(observed)} correct ({correct / len(observed):.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
