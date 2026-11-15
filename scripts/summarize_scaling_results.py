#!/usr/bin/env python3
"""Verify the eight retained results and export the Week 11 paired comparison."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ds340w_repro.advance_fixed_cell import _verified_result
from ds340w_repro.config_schema import expected_trainable_counts
from collect_retained_result import _compare_saved_tensors, _selected_checkpoint


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def verified_run(rank: int, modified: bool) -> dict:
    import torch
    name = f"scaling-r{rank}-seed42" if modified else (
        "baseline-r8-lr1e-4-eb16" if rank == 8 else f"rank-r{rank}"
    )
    group = "scaling" if modified else "baseline" if rank == 8 else "rank"
    result = _verified_result(manifest_path=ROOT / f"data/manifests/retained-{name}.json",
                              config_path=ROOT / f"configs/resolved/{group}/{name}.json",
                              expected_experiment_id=name)
    artifact = Path(result["artifact_path"])
    metrics = read(artifact / "metrics.json")
    run = ROOT / f"artifacts/runs/{name}"
    state = read(run / "trainer_state.json")
    checkpoint, step, correct = _selected_checkpoint(state, run)
    _compare_saved_tensors(checkpoint, run)
    assert (step, correct) == (result["checkpoint_step"], result["correct_count"])
    weights = torch.load(artifact / "native-checkpoint/pytorch_model.bin", map_location="cpu", weights_only=True)
    count = sum(t.numel() for t in weights.values())
    assert count == expected_trainable_counts(rank)["trainable_parameters"]
    rows = [json.loads(line) for line in (artifact / "predictions.jsonl").read_text().splitlines()]
    assert all(row["label"] in (0, 1) and row["prediction"] in (0, 1) for row in rows)
    assert math.isclose(metrics["eval_loss"], read(run / "eval_results.json")["eval_loss"], abs_tol=1e-4)
    process = read(run / "process-result.json")
    assert process["exit_code"] == 0
    assert process["nonfinal_access_enforcement"]["final_unseen_file_exposed_to_child"] is False
    if modified:
        assert read(run / "delta/config.json")["lora_scaling_mode"] == "sqrt_rank"
        assert read(checkpoint / "delta-config.json")["lora_scaling_mode"] == "sqrt_rank"
        gradients = [json.loads(line) for line in (run / "adapter-gradients.jsonl").read_text().splitlines()]
        observed_steps = [row["step"] for row in gradients]
        expected_steps = [1, 10, *range(100, 20001, 100)]
        assert observed_steps == sorted(set(observed_steps))
        assert set(observed_steps).issubset(expected_steps)
        assert {1, 10, 20000}.issubset(observed_steps)
        assert all(row["gradient_accumulation_steps"] == 2 for row in gradients)
        assert all(row[side + "_elements"] == (count - 47616) // 2
                   and math.isfinite(row[side + "_rms"]) and row[side + "_rms"] >= 0
                   for row in gradients for side in ("A", "B"))
    # Upstream peak_memory is allocated bytes / 2**20 / 1000.
    allocated_gib = metrics["upstream_peak_memory_gb"] * 1000 / 1024
    return {"experiment_id": name, "rank": rank, "alpha": rank * 2, "seed": 42,
            "scaling_mode": "sqrt_rank" if modified else "original",
            "multiplier": 2 * math.sqrt(rank) if modified else 2,
            "correct_count": correct, "test_rows": 436,
            "accuracy_percent": 100 * correct / 436, "checkpoint_step": step,
            "eval_loss": metrics["eval_loss"], "runtime_seconds": process["wall_seconds"],
            "peak_allocated_gib": allocated_gib, "trainable_parameters": count,
            "config_sha256": result["config_sha256"], "project_git_commit": process["project_git_commit"],
            "gradient_samples": len(gradients) if modified else None,
            "missing_gradient_steps": sorted(set(expected_steps) - set(observed_steps)) if modified else None,
            "predictions": rows}


def main() -> None:
    table = []
    paired = []
    reference = None
    for rank, expected_correct in zip((2, 4, 8, 16), (413, 414, 412, 417)):
        original, modified = verified_run(rank, False), verified_run(rank, True)
        assert original["correct_count"] == expected_correct
        for result in (original, modified):
            keys = [(row["test_id"], row["label"]) for row in result["predictions"]]
            if reference is None:
                reference = keys
            assert keys == reference
        paired.append({"rank": rank, "original": original, "modified": modified})
        table.append({"rank": rank, "alpha": 2 * rank,
                      "original_correct": original["correct_count"],
                      "modified_correct": modified["correct_count"],
                      "original_accuracy_percent": original["accuracy_percent"],
                      "modified_accuracy_percent": modified["accuracy_percent"],
                      "difference_percentage_points": modified["accuracy_percent"] - original["accuracy_percent"],
                      "original_runtime_minutes": original["runtime_seconds"] / 60,
                      "modified_runtime_minutes": modified["runtime_seconds"] / 60,
                      "original_peak_allocated_gib": original["peak_allocated_gib"],
                      "modified_peak_allocated_gib": modified["peak_allocated_gib"],
                      "trainable_parameters_each": original["trainable_parameters"],
                      "original_checkpoint_step": original["checkpoint_step"],
                      "modified_checkpoint_step": modified["checkpoint_step"],
                      "original_eval_loss": original["eval_loss"],
                      "modified_eval_loss": modified["eval_loss"]})
    out = ROOT / "data/results/scaling-comparison"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    payload = {"verified": True, "aligned_test_rows": 436, "comparison": table, "runs": paired,
               "limits": ["One training seed per setting.", "alpha=2*rank in both methods.",
                          "At a fixed rank, ordinary LoRA with alpha increased by sqrt(rank) has the same multiplier.",
                          "Course Test selected the checkpoint; it is not an untouched final assessment.",
                          "Original full runs have no comparable recorded gradient histories.",
                          "Matched rank-16 pilot gradients cover steps 1, 10, 100, 200 only.",
                          "AdamW parameter updates are not proportional to gradient magnitude.",
                          "Original rank-8 and selected original-rank final candidates remain unchanged."]}
    (out / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
