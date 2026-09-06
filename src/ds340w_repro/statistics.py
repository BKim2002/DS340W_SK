"""Descriptive uncertainty estimates for fixed-checkpoint SST-2 results."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from statistics import NormalDist
from typing import Sequence

from ds340w_repro.checkpoints import (
    PredictionRecord,
    validate_aligned_predictions,
)


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided interval on the accuracy scale."""

    lower: float
    upper: float
    confidence_level: float = 0.95


@dataclass(frozen=True)
class PairedBootstrapInterval:
    """A percentile interval for comparison-minus-reference accuracy."""

    observed_difference: float
    lower: float
    upper: float
    resamples: int
    seed: int
    confidence_level: float = 0.95


def wilson_interval(
    correct_count: int,
    total_count: int,
    *,
    confidence_level: float = 0.95,
) -> ConfidenceInterval:
    """Return the two-sided Wilson score interval for a binomial proportion."""

    if type(correct_count) is not int or type(total_count) is not int:
        raise TypeError("Correct and total counts must be integers")
    if total_count <= 0 or not 0 <= correct_count <= total_count:
        raise ValueError("Counts must satisfy 0 <= correct <= total and total > 0")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie strictly between zero and one")

    z = NormalDist().inv_cdf((1 + confidence_level) / 2)
    proportion = correct_count / total_count
    z_squared = z * z
    denominator = 1 + z_squared / total_count
    center = (proportion + z_squared / (2 * total_count)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total_count
            + z_squared / (4 * total_count * total_count)
        )
        / denominator
    )
    return ConfidenceInterval(
        lower=max(0.0, center - half_width),
        upper=min(1.0, center + half_width),
        confidence_level=confidence_level,
    )


def paired_bootstrap_accuracy_difference(
    reference: Sequence[PredictionRecord],
    comparison: Sequence[PredictionRecord],
    *,
    resamples: int = 10_000,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> PairedBootstrapInterval:
    """Bootstrap paired example-level accuracy differences by stable Test ID."""

    validate_aligned_predictions(reference, comparison)
    if type(resamples) is not int or resamples <= 0:
        raise ValueError("Bootstrap resamples must be a positive integer")
    if type(seed) is not int:
        raise TypeError("Bootstrap seed must be an integer")
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("Confidence level must lie strictly between zero and one")

    per_example_differences = tuple(
        int(comparison_record.prediction == comparison_record.label)
        - int(reference_record.prediction == reference_record.label)
        for reference_record, comparison_record in zip(reference, comparison)
    )
    sample_size = len(per_example_differences)
    observed = sum(per_example_differences) / sample_size
    generator = random.Random(seed)
    bootstrap_differences = sorted(
        sum(
            per_example_differences[generator.randrange(sample_size)]
            for _ in range(sample_size)
        )
        / sample_size
        for _ in range(resamples)
    )
    tail_probability = (1 - confidence_level) / 2
    return PairedBootstrapInterval(
        observed_difference=observed,
        lower=_linear_quantile(bootstrap_differences, tail_probability),
        upper=_linear_quantile(bootstrap_differences, 1 - tail_probability),
        resamples=resamples,
        seed=seed,
        confidence_level=confidence_level,
    )


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1 - fraction)
        + sorted_values[upper_index] * fraction
    )
