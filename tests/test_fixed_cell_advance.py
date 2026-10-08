from __future__ import annotations

import pytest

from ds340w_repro.advance_fixed_cell import _check_runtime_cap, _select_rank


def test_select_smallest_rank_within_half_percentage_point() -> None:
    # On 436 Test examples, two fewer correct is 0.459 percentage points.
    assert _select_rank({2: 400, 4: 402, 8: 401, 16: 402}) == 2
    assert _select_rank({2: 399, 4: 402, 8: 401, 16: 402}) == 4
    with pytest.raises(ValueError, match="all four"):
        _select_rank({2: 400, 8: 401})


def test_reassessment_fails_closed_above_the_48_hour_cap() -> None:
    gate = {
        "projection": {
            "retained_run_plan": {"training_runs": 8, "final_evaluations": 2},
            "seconds_per_optimizer_step": 0.5,
            "full_test_evaluation_seconds": 5.0,
            "actual_preflight_seconds": 600.0,
            "cap_seconds": 172_800,
        }
    }
    assert _check_runtime_cap(gate, completed_wall_seconds=(10_000.0,)) < 172_800
    with pytest.raises(ValueError, match="exceeds"):
        _check_runtime_cap(gate, completed_wall_seconds=(150_000.0,))
