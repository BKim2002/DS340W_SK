"""Central access policy for the three course data partitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ds340w_repro.data.create_sst2_split import (
    SPLIT_MANIFEST_FILENAME,
    TEST_INDICES_FILENAME,
    UNSEEN_COMMITMENT_FILENAME,
)


class FinalUnseenAccessError(PermissionError):
    """Raised when final-unseen access is not explicitly authorized."""


@dataclass(frozen=True)
class PartitionSelection:
    """A source split and optional ordered row selection."""

    source_split: str
    source_indices: tuple[int, ...] | None


def resolve_partition_selection(
    course_partition: str,
    *,
    public_dir: Path,
    private_root: Path | None = None,
    evaluation_mode: str = "development",
    lock_artifact: Path | None = None,
) -> PartitionSelection:
    """Resolve an authorized course partition without loading dataset rows."""

    # Non-final routes intentionally never inspect private_root or lock_artifact.
    if course_partition == "Training":
        return PartitionSelection(source_split="train", source_indices=None)
    if course_partition == "Test":
        payload = json.loads(
            (public_dir / TEST_INDICES_FILENAME).read_text(encoding="utf-8")
        )
        if payload.get("source_split") != "validation":
            raise ValueError("Test index artifact has the wrong source split")
        indices = tuple(payload["source_indices"])
        if payload.get("row_count") != len(indices):
            raise ValueError("Test index artifact row count does not match")
        return PartitionSelection(
            source_split="validation", source_indices=indices
        )
    if course_partition == "Validation (Unseen)":
        if evaluation_mode != "final":
            raise FinalUnseenAccessError(
                "Validation (Unseen) requires explicit final evaluation mode"
            )
        if lock_artifact is None or not lock_artifact.is_file():
            raise FinalUnseenAccessError(
                "Validation (Unseen) requires an experiment lock artifact"
            )
        lock_payload = json.loads(lock_artifact.read_text(encoding="utf-8"))
        if lock_payload.get("locked") is not True:
            raise FinalUnseenAccessError(
                "Experiment lock artifact does not declare locked=true"
            )
        if lock_payload.get("schema_version") != 1:
            raise FinalUnseenAccessError(
                "Experiment lock artifact requires schema_version=1"
            )
        for required_field in (
            "git_commit",
            "private_unseen_indices_sha256",
            "resolved_config_sha256",
        ):
            value = lock_payload.get(required_field)
            if not isinstance(value, str) or not value:
                raise FinalUnseenAccessError(
                    f"Experiment lock artifact requires non-empty {required_field}"
                )
        commitment = json.loads(
            (public_dir / UNSEEN_COMMITMENT_FILENAME).read_text(encoding="utf-8")
        )
        public_manifest = json.loads(
            (public_dir / SPLIT_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        if (
            public_manifest["partitions"]["Validation (Unseen)"]
            != commitment
        ):
            raise FinalUnseenAccessError(
                "Unseen commitment does not match the public manifest"
            )
        if (
            lock_payload["private_unseen_indices_sha256"]
            != commitment.get("sha256")
        ):
            raise FinalUnseenAccessError(
                "Experiment lock references a different private index commitment"
            )
        raise FinalUnseenAccessError(
            "Final-unseen access is disabled until the complete audited "
            "FINAL_EVAL_LOCK contract is implemented after Week 1"
        )
    raise ValueError(f"Unknown course partition: {course_partition}")
