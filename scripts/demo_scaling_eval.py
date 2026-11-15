#!/usr/bin/env python3
"""Read-only live evaluation of a retained Week 11 modified-scaling model."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, choices=(2, 4, 8, 16), default=8)
    args = parser.parse_args()
    os.environ.update({"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HF_DATASETS_OFFLINE": "1",
                       "TRANSFORMERS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"})
    for key in ("DS340W_PRIVATE_ROOT", "DS340W_SST2_FINAL_FILE"):
        os.environ.pop(key, None)
    from ds340w_repro.advance_fixed_cell import _verified_result
    from ds340w_repro.config_schema import ResolvedConfig
    from ds340w_repro.run_opendelta import verify_course_data_binding, verify_model_snapshot_binding
    from collect_retained_result import _predict_test
    name = f"scaling-r{args.rank}-seed42"
    config_path = ROOT / f"configs/resolved/scaling/{name}.json"
    result = _verified_result(manifest_path=ROOT / f"data/manifests/retained-{name}.json",
                              config_path=config_path, expected_experiment_id=name)
    artifact = Path(result["artifact_path"])
    execution = json.loads((artifact / "evidence/execution-tree.json").read_text())
    runtime = Path(execution["root"]).resolve(strict=True)
    if not runtime.is_relative_to(ROOT / "runtime"):
        raise ValueError("Saved execution tree is outside the project")
    sys.path.insert(0, str(runtime))
    config = ResolvedConfig.from_json(config_path.read_text())
    cache = Path.home() / ".cache/ds340w"
    verify_model_snapshot_binding(config, ROOT, cache)
    _, _, test_path = verify_course_data_binding(config, ROOT, cache)
    ids, labels, predictions, loss = _predict_test(output_dir=artifact, config=config, test_path=test_path)
    saved = [json.loads(line) for line in (artifact / "predictions.jsonl").read_text().splitlines()]
    if list(zip(ids, labels, predictions)) != [(r["test_id"], r["label"], r["prediction"]) for r in saved]:
        raise ValueError("Live predictions differ from the retained record")
    print(f"Modified scaling: rank {args.rank}, alpha {2*args.rank}, seed 42")
    print(f"Course Test: {result['correct_count']}/436 correct ({result['accuracy_percent']:.2f}%)")
    print(f"Selected checkpoint: step {result['checkpoint_step']}; evaluation loss {loss:.6f}")
    print("All 436 live predictions match the retained record. No training or final Unseen access.")


if __name__ == "__main__":
    main()
