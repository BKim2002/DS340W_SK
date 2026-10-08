from __future__ import annotations

from pathlib import Path

import pytest

from scripts.collect_retained_result import _selected_checkpoint


def _state(output_dir: Path, counts: list[int]) -> dict:
    best_count = max(counts)
    best_step = 100 * (counts.index(best_count) + 1)
    (output_dir / f"checkpoint-{best_step}").mkdir()
    return {
        "global_step": 20_000,
        "best_metric": 100 * best_count / 436,
        "best_model_checkpoint": str(output_dir / f"checkpoint-{best_step}"),
        "log_history": [
            {"step": (index + 1) * 100, "eval_accuracy": 100 * count / 436}
            for index, count in enumerate(counts)
        ],
    }


def test_observer_selects_earliest_exact_tie(tmp_path: Path) -> None:
    counts = [300] * 200
    counts[5] = 405
    counts[10] = 405
    state = _state(tmp_path, counts)

    checkpoint, step, correct = _selected_checkpoint(state, tmp_path)

    assert checkpoint == tmp_path / "checkpoint-600"
    assert (step, correct) == (600, 405)


def test_observer_rejects_missing_checkpoint_evaluations(tmp_path: Path) -> None:
    state = _state(tmp_path, [300] * 199)
    with pytest.raises(ValueError, match="200 checkpoint"):
        _selected_checkpoint(state, tmp_path)
