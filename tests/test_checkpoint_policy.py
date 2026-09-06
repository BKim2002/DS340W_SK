import hashlib
import json
from pathlib import Path

import pytest

from ds340w_repro.checkpoints import (
    CheckpointEvaluation,
    build_prediction_records,
    preserve_checkpoint_artifact,
    select_checkpoint,
    validate_aligned_predictions,
    validate_checkpoint_reload,
    verify_checkpoint_artifact,
)
from ds340w_repro.measurement import FreshProcessMeasurement


def test_checkpoint_selection_uses_raw_correct_count_then_earliest_step() -> None:
    candidates = [
        CheckpointEvaluation(Path("checkpoint-100"), step=100, correct_count=390),
        CheckpointEvaluation(Path("checkpoint-200"), step=200, correct_count=391),
        CheckpointEvaluation(Path("checkpoint-300"), step=300, correct_count=391),
    ]

    selected = select_checkpoint(candidates)

    assert selected.checkpoint_path == Path("checkpoint-200")
    assert selected.correct_count == 391
    assert selected.step == 200


@pytest.mark.parametrize(
    ("step", "correct_count"),
    [(-1, 1), (1, -1), (1.5, 1), (1, 1.5)],
)
def test_checkpoint_evidence_requires_nonnegative_integer_counts(
    step: object, correct_count: object
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        CheckpointEvaluation(
            Path("checkpoint-invalid"),
            step=step,  # type: ignore[arg-type]
            correct_count=correct_count,  # type: ignore[arg-type]
        )


def test_prediction_records_bind_stable_test_ids_to_checkpoint_and_config() -> None:
    records = build_prediction_records(
        test_ids=["validation:718", "validation:74"],
        labels=[1, 0],
        predictions=[1, 1],
        checkpoint_step=200,
        config_sha256="a" * 64,
    )

    assert [record.to_dict() for record in records] == [
        {
            "checkpoint_step": 200,
            "config_sha256": "a" * 64,
            "label": 1,
            "prediction": 1,
            "test_id": "validation:718",
        },
        {
            "checkpoint_step": 200,
            "config_sha256": "a" * 64,
            "label": 0,
            "prediction": 1,
            "test_id": "validation:74",
        },
    ]


def test_prediction_alignment_rejects_a_stable_test_id_mismatch() -> None:
    reference = build_prediction_records(
        test_ids=["validation:718", "validation:74"],
        labels=[1, 0],
        predictions=[1, 1],
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
        validate_aligned_predictions(reference, comparison)


def test_checkpoint_reload_accepts_exact_count_and_loss_within_tolerance() -> None:
    validate_checkpoint_reload(
        expected_correct_count=391,
        expected_eval_loss=0.12345,
        actual_correct_count=391,
        actual_eval_loss=0.12355,
    )


def test_checkpoint_reload_rejects_count_or_loss_drift() -> None:
    with pytest.raises(ValueError, match="correct count"):
        validate_checkpoint_reload(
            expected_correct_count=391,
            expected_eval_loss=0.12345,
            actual_correct_count=390,
            actual_eval_loss=0.12345,
        )

    with pytest.raises(ValueError, match="evaluation loss"):
        validate_checkpoint_reload(
            expected_correct_count=391,
            expected_eval_loss=0.12345,
            actual_correct_count=391,
            actual_eval_loss=0.1235501,
        )


def test_preservation_copies_native_outputs_and_writes_a_hash_manifest(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "run" / "checkpoint-200"
    checkpoint.mkdir(parents=True)
    native_weights = checkpoint / "pytorch_model.bin"
    native_weights.write_bytes(b"native author checkpoint")
    resolved_config = tmp_path / "run" / "resolved-config.json"
    resolved_config.write_text('{"experiment_id":"baseline-1"}\n', encoding="utf-8")
    config_sha256 = hashlib.sha256(resolved_config.read_bytes()).hexdigest()
    trainer_state = tmp_path / "run" / "trainer_state.json"
    trainer_state.write_text('{"global_step":200}\n', encoding="utf-8")
    predictions = build_prediction_records(
        test_ids=["validation:718", "validation:74"],
        labels=[1, 0],
        predictions=[1, 1],
        checkpoint_step=200,
        config_sha256=config_sha256,
    )
    artifact_dir = tmp_path / "artifacts" / "baseline-1" / "checkpoint-200"

    preserved = preserve_checkpoint_artifact(
        source_checkpoint=checkpoint,
        resolved_config_path=resolved_config,
        trainer_state_path=trainer_state,
        predictions=predictions,
        metrics={"correct_count": 1, "eval_loss": 0.25},
        artifact_dir=artifact_dir,
    )

    assert native_weights.read_bytes() == b"native author checkpoint"
    assert (artifact_dir / "native-checkpoint" / native_weights.name).read_bytes() == (
        native_weights.read_bytes()
    )
    assert json.loads((artifact_dir / "metrics.json").read_text("utf-8")) == {
        "correct_count": 1,
        "eval_loss": 0.25,
    }
    prediction_rows = [
        json.loads(line)
        for line in (artifact_dir / "predictions.jsonl").read_text("utf-8").splitlines()
    ]
    assert prediction_rows == [record.to_dict() for record in predictions]
    manifest = json.loads(preserved.manifest_path.read_text("utf-8"))
    assert manifest["files"]["native-checkpoint/pytorch_model.bin"] == {
        "bytes": len(native_weights.read_bytes()),
        "sha256": hashlib.sha256(native_weights.read_bytes()).hexdigest(),
    }
    assert set(manifest["files"]) == {
        "metrics.json",
        "native-checkpoint/pytorch_model.bin",
        "predictions.jsonl",
        "resolved-config.json",
        "trainer-state.json",
    }


def test_artifact_verification_rejects_post_preservation_changes(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "delta.pt").write_bytes(b"author bytes")
    config = tmp_path / "resolved.json"
    config.write_text("{}\n", encoding="utf-8")
    config_sha256 = hashlib.sha256(config.read_bytes()).hexdigest()
    trainer_state = tmp_path / "trainer.json"
    trainer_state.write_text("{}\n", encoding="utf-8")
    predictions = build_prediction_records(
        test_ids=["validation:718"],
        labels=[1],
        predictions=[1],
        checkpoint_step=100,
        config_sha256=config_sha256,
    )
    artifact_dir = tmp_path / "artifact"
    preserve_checkpoint_artifact(
        source_checkpoint=checkpoint,
        resolved_config_path=config,
        trainer_state_path=trainer_state,
        predictions=predictions,
        metrics={"correct_count": 1, "eval_loss": 0.1},
        artifact_dir=artifact_dir,
    )
    assert verify_checkpoint_artifact(artifact_dir)["valid"] is True

    (artifact_dir / "native-checkpoint" / "delta.pt").write_bytes(b"changed")

    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checkpoint_artifact(artifact_dir)


def test_preservation_rejects_predictions_from_a_different_config(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "delta.pt").write_bytes(b"author bytes")
    config = tmp_path / "resolved.json"
    config.write_text("{}\n", encoding="utf-8")
    trainer_state = tmp_path / "trainer.json"
    trainer_state.write_text("{}\n", encoding="utf-8")
    predictions = build_prediction_records(
        test_ids=["validation:718"],
        labels=[1],
        predictions=[1],
        checkpoint_step=100,
        config_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="resolved configuration"):
        preserve_checkpoint_artifact(
            source_checkpoint=checkpoint,
            resolved_config_path=config,
            trainer_state_path=trainer_state,
            predictions=predictions,
            metrics={"correct_count": 1, "eval_loss": 0.1},
            artifact_dir=tmp_path / "artifact",
        )


def test_preservation_rejects_a_raw_correct_count_that_disagrees_with_predictions(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "source-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "delta.pt").write_bytes(b"author bytes")
    config = tmp_path / "resolved.json"
    config.write_text("{}\n", encoding="utf-8")
    trainer_state = tmp_path / "trainer.json"
    trainer_state.write_text("{}\n", encoding="utf-8")
    predictions = build_prediction_records(
        test_ids=["validation:718", "validation:74"],
        labels=[1, 0],
        predictions=[1, 1],
        checkpoint_step=100,
        config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="correct_count"):
        preserve_checkpoint_artifact(
            source_checkpoint=checkpoint,
            resolved_config_path=config,
            trainer_state_path=trainer_state,
            predictions=predictions,
            metrics={"correct_count": 2, "eval_loss": 0.1},
            artifact_dir=tmp_path / "artifact",
        )


def test_measurement_record_captures_the_fresh_process_runtime_contract() -> None:
    measurement = FreshProcessMeasurement(
        phase="training_and_test",
        fresh_process=True,
        exit_status=0,
        wall_time_seconds=123.5,
        cuda_peak_allocated_bytes=6_000_000_000,
        cuda_peak_reserved_bytes=7_000_000_000,
        config_sha256="d" * 64,
        software_versions={
            "cuda": "12.1",
            "python": "3.10.12",
            "torch": "2.1.2+cu121",
        },
    )

    assert measurement.to_dict() == {
        "config_sha256": "d" * 64,
        "cuda_peak_allocated_bytes": 6_000_000_000,
        "cuda_peak_reserved_bytes": 7_000_000_000,
        "exit_status": 0,
        "fresh_process": True,
        "phase": "training_and_test",
        "schema_version": 1,
        "software_versions": {
            "cuda": "12.1",
            "python": "3.10.12",
            "torch": "2.1.2+cu121",
        },
        "wall_time_seconds": 123.5,
    }
