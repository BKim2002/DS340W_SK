"""Checkpoint selection and immutable artifact-preservation controls."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CheckpointEvaluation:
    """Selection evidence for one native upstream checkpoint."""

    checkpoint_path: Path
    step: int
    correct_count: int

    def __post_init__(self) -> None:
        if (
            type(self.step) is not int
            or self.step < 0
            or type(self.correct_count) is not int
            or self.correct_count < 0
        ):
            raise ValueError(
                "Checkpoint step and correct count must be non-negative integers"
            )


@dataclass(frozen=True)
class PredictionRecord:
    """One SST-2 prediction tied to stable data and experiment identities."""

    test_id: str
    label: int
    prediction: int
    checkpoint_step: int
    config_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return {
            "checkpoint_step": self.checkpoint_step,
            "config_sha256": self.config_sha256,
            "label": self.label,
            "prediction": self.prediction,
            "test_id": self.test_id,
        }


@dataclass(frozen=True)
class PreservedCheckpointArtifact:
    """Locations and top-level digest for one immutable artifact directory."""

    artifact_dir: Path
    manifest_path: Path
    manifest_sha256: str


def select_checkpoint(
    candidates: Sequence[CheckpointEvaluation],
) -> CheckpointEvaluation:
    """Select greatest raw correct count, then the earliest exact tie."""

    if not candidates:
        raise ValueError("At least one checkpoint evaluation is required")
    for candidate in candidates:
        if candidate.step < 0:
            raise ValueError("Checkpoint steps must be non-negative")
        if candidate.correct_count < 0:
            raise ValueError("Correct counts must be non-negative")
    return min(candidates, key=lambda item: (-item.correct_count, item.step))


def build_prediction_records(
    *,
    test_ids: Sequence[str],
    labels: Sequence[int],
    predictions: Sequence[int],
    checkpoint_step: int,
    config_sha256: str,
) -> tuple[PredictionRecord, ...]:
    """Validate and bind aligned predictions to their checkpoint identity."""

    row_count = len(test_ids)
    if row_count == 0:
        raise ValueError("At least one Test prediction is required")
    if len(labels) != row_count or len(predictions) != row_count:
        raise ValueError("Test IDs, labels, and predictions must have equal lengths")
    if len(set(test_ids)) != row_count or any(not value for value in test_ids):
        raise ValueError("Stable Test IDs must be non-empty and unique")
    if any(type(value) is not int or value not in (0, 1) for value in labels):
        raise ValueError("SST-2 labels must be integers in {0, 1}")
    if any(type(value) is not int or value not in (0, 1) for value in predictions):
        raise ValueError("SST-2 predictions must be integers in {0, 1}")
    if type(checkpoint_step) is not int or checkpoint_step < 0:
        raise ValueError("Checkpoint step must be a non-negative integer")
    if not re.fullmatch(r"[0-9a-f]{64}", config_sha256):
        raise ValueError("Configuration checksum must be a lowercase SHA-256")

    return tuple(
        PredictionRecord(
            test_id=test_id,
            label=label,
            prediction=prediction,
            checkpoint_step=checkpoint_step,
            config_sha256=config_sha256,
        )
        for test_id, label, prediction in zip(test_ids, labels, predictions)
    )


def validate_aligned_predictions(
    reference: Sequence[PredictionRecord],
    comparison: Sequence[PredictionRecord],
) -> tuple[str, ...]:
    """Return aligned stable IDs or reject an invalid paired comparison."""

    reference_ids = tuple(record.test_id for record in reference)
    comparison_ids = tuple(record.test_id for record in comparison)
    if not reference_ids:
        raise ValueError("Paired predictions must not be empty")
    if len(set(reference_ids)) != len(reference_ids):
        raise ValueError("Reference stable Test IDs must be unique")
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("Comparison stable Test IDs must be unique")
    if reference_ids != comparison_ids:
        raise ValueError("Paired predictions must have aligned stable Test IDs")
    if tuple(record.label for record in reference) != tuple(
        record.label for record in comparison
    ):
        raise ValueError("Paired predictions must have aligned labels")
    return reference_ids


def validate_checkpoint_reload(
    *,
    expected_correct_count: int,
    expected_eval_loss: float,
    actual_correct_count: int,
    actual_eval_loss: float,
    loss_tolerance: float = 1e-4,
) -> None:
    """Reject a reloaded checkpoint whose evaluation evidence has drifted."""

    if actual_correct_count != expected_correct_count:
        raise ValueError(
            "Reloaded checkpoint correct count differs from the retained result"
        )
    losses = (expected_eval_loss, actual_eval_loss, loss_tolerance)
    if not all(math.isfinite(value) for value in losses) or loss_tolerance < 0:
        raise ValueError("Evaluation losses and tolerance must be finite")
    loss_difference = abs(
        Decimal(str(actual_eval_loss)) - Decimal(str(expected_eval_loss))
    )
    if loss_difference > Decimal(str(loss_tolerance)):
        raise ValueError(
            "Reloaded checkpoint evaluation loss differs by more than "
            f"{loss_tolerance}"
        )


def preserve_checkpoint_artifact(
    *,
    source_checkpoint: Path,
    resolved_config_path: Path,
    trainer_state_path: Path,
    predictions: Sequence[PredictionRecord],
    metrics: Mapping[str, Any],
    artifact_dir: Path,
) -> PreservedCheckpointArtifact:
    """Copy a complete native result into a new checksummed artifact directory."""

    source_checkpoint = source_checkpoint.resolve(strict=True)
    resolved_config_path = resolved_config_path.resolve(strict=True)
    trainer_state_path = trainer_state_path.resolve(strict=True)
    artifact_dir = artifact_dir.resolve()
    if artifact_dir.exists():
        raise FileExistsError(f"Artifact directory already exists: {artifact_dir}")
    if artifact_dir == source_checkpoint or artifact_dir.is_relative_to(
        source_checkpoint
    ):
        raise ValueError("Artifact directory cannot be inside the source checkpoint")
    _validate_prediction_records(predictions)
    if _file_sha256(resolved_config_path) != predictions[0].config_sha256:
        raise ValueError(
            "Prediction config checksum does not match the resolved configuration"
        )
    correct_count = sum(record.prediction == record.label for record in predictions)
    if metrics.get("correct_count") != correct_count:
        raise ValueError("metrics correct_count does not match predictions")
    eval_loss = metrics.get("eval_loss")
    if type(eval_loss) not in (int, float) or not math.isfinite(eval_loss):
        raise ValueError("metrics eval_loss must be a finite number")

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{artifact_dir.name}-", dir=artifact_dir.parent
    ) as temporary_parent:
        staged = Path(temporary_parent) / "artifact"
        staged.mkdir()
        native_destination = staged / "native-checkpoint"
        if source_checkpoint.is_dir():
            shutil.copytree(source_checkpoint, native_destination)
        else:
            native_destination.mkdir()
            shutil.copy2(
                source_checkpoint, native_destination / source_checkpoint.name
            )
        shutil.copy2(resolved_config_path, staged / "resolved-config.json")
        shutil.copy2(trainer_state_path, staged / "trainer-state.json")
        (staged / "predictions.jsonl").write_text(
            "".join(
                json.dumps(
                    record.to_dict(),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
                for record in predictions
            ),
            encoding="utf-8",
            newline="\n",
        )
        (staged / "metrics.json").write_bytes(_canonical_json_bytes(dict(metrics)))

        files = {
            path.relative_to(staged).as_posix(): {
                "bytes": path.stat().st_size,
                "sha256": _file_sha256(path),
            }
            for path in sorted(staged.rglob("*"))
            if path.is_file()
        }
        manifest_path = staged / "manifest.json"
        manifest_path.write_bytes(
            _canonical_json_bytes({"files": files, "schema_version": 1})
        )
        staged.replace(artifact_dir)

    final_manifest = artifact_dir / "manifest.json"
    return PreservedCheckpointArtifact(
        artifact_dir=artifact_dir,
        manifest_path=final_manifest,
        manifest_sha256=_file_sha256(final_manifest),
    )


def verify_checkpoint_artifact(artifact_dir: Path) -> dict[str, bool | str]:
    """Verify that an artifact still exactly matches its preservation manifest."""

    artifact_dir = artifact_dir.resolve(strict=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not isinstance(
        manifest.get("files"), dict
    ):
        raise ValueError("Invalid checkpoint artifact manifest")
    expected_files = set(manifest["files"])
    actual_files = {
        path.relative_to(artifact_dir).as_posix()
        for path in artifact_dir.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != expected_files:
        raise ValueError("Artifact file set does not match the manifest")
    for relative_path, evidence in manifest["files"].items():
        path = (artifact_dir / relative_path).resolve(strict=True)
        if not path.is_relative_to(artifact_dir) or not path.is_file():
            raise ValueError(f"Unsafe artifact manifest path: {relative_path}")
        if _file_sha256(path) != evidence.get("sha256"):
            raise ValueError(f"Artifact checksum mismatch: {relative_path}")
        if path.stat().st_size != evidence.get("bytes"):
            raise ValueError(f"Artifact byte count mismatch: {relative_path}")
    return {
        "manifest_sha256": _file_sha256(manifest_path),
        "valid": True,
    }


def _validate_prediction_records(
    predictions: Sequence[PredictionRecord],
) -> None:
    if not predictions:
        raise ValueError("At least one Test prediction is required")
    test_ids = [record.test_id for record in predictions]
    if len(set(test_ids)) != len(test_ids) or any(not value for value in test_ids):
        raise ValueError("Stable Test IDs must be non-empty and unique")
    if len({record.checkpoint_step for record in predictions}) != 1:
        raise ValueError("Prediction records must use one checkpoint step")
    hashes = {record.config_sha256 for record in predictions}
    if len(hashes) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(hashes))):
        raise ValueError("Prediction records must use one valid config SHA-256")


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
