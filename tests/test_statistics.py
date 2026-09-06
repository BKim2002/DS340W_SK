import pytest

from ds340w_repro.checkpoints import build_prediction_records
from ds340w_repro.statistics import (
    paired_bootstrap_accuracy_difference,
    wilson_interval,
)


@pytest.mark.parametrize(
    ("correct_count", "total_count", "expected_lower", "expected_upper"),
    [
        (0, 10, 0.0, 0.2775327998628892),
        (5, 10, 0.23659309051256394, 0.7634069094874361),
        (10, 10, 0.7224672001371107, 1.0),
    ],
)
def test_wilson_95_percent_interval_matches_worked_fixtures(
    correct_count: int,
    total_count: int,
    expected_lower: float,
    expected_upper: float,
) -> None:
    interval = wilson_interval(correct_count, total_count)

    assert interval.confidence_level == 0.95
    assert interval.lower == pytest.approx(expected_lower, abs=1e-12)
    assert interval.upper == pytest.approx(expected_upper, abs=1e-12)


def test_seeded_paired_bootstrap_is_deterministic_and_zero_for_identical_vectors(
) -> None:
    predictions = build_prediction_records(
        test_ids=[f"validation:{index}" for index in (718, 74, 168, 589)],
        labels=[1, 0, 1, 0],
        predictions=[1, 1, 0, 0],
        checkpoint_step=200,
        config_sha256="a" * 64,
    )

    first = paired_bootstrap_accuracy_difference(predictions, predictions)
    second = paired_bootstrap_accuracy_difference(predictions, predictions)

    assert first == second
    assert first.observed_difference == 0.0
    assert first.lower == 0.0
    assert first.upper == 0.0
    assert first.resamples == 10_000
    assert first.seed == 42


def test_paired_bootstrap_rejects_misaligned_stable_test_ids() -> None:
    reference = build_prediction_records(
        test_ids=["validation:718", "validation:74"],
        labels=[1, 0],
        predictions=[1, 0],
        checkpoint_step=200,
        config_sha256="a" * 64,
    )
    comparison = build_prediction_records(
        test_ids=["validation:718", "validation:168"],
        labels=[1, 0],
        predictions=[0, 0],
        checkpoint_step=300,
        config_sha256="b" * 64,
    )

    with pytest.raises(ValueError, match="stable Test IDs"):
        paired_bootstrap_accuracy_difference(reference, comparison)
