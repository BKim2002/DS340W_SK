"""Behavioral tests for isolated SST-2 course-data materialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from ds340w_repro.data.materialize_sst2 import (
    load_local_pinned_sst2,
    main,
    materialize_course_data,
    verify_course_data,
)
from ds340w_repro.source_pins import DatasetSourcePin, HubPin, PinnedFile


class InstrumentedSplit:
    """Small indexable split that records reads and rejects blinded rows."""

    def __init__(self, rows: list[dict[str, object]], forbidden: set[int] = set()):
        self.rows = rows
        self.forbidden = forbidden
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        if index in self.forbidden:
            raise AssertionError(f"blinded validation row {index} was read")
        self.accessed.append(index)
        return self.rows[index]


def write_test_indices(path: Path, indices: list[int]) -> None:
    path.write_text(
        json.dumps(
            {
                "row_count": len(indices),
                "source_indices": indices,
                "source_split": "validation",
            }
        ),
        encoding="utf-8",
    )


def pinned_file(url: str, relative_path: str, payload: bytes) -> PinnedFile:
    return PinnedFile(
        url=url,
        relative_path=relative_path,
        size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_materialization_reads_only_training_and_committed_test_rows(
    tmp_path: Path,
) -> None:
    training = InstrumentedSplit(
        [
            {"idx": index, "label": index % 2, "sentence": f"train {index}"}
            for index in range(4)
        ]
    )
    validation = InstrumentedSplit(
        [
            {"idx": index, "label": index % 2, "sentence": f"valid {index}"}
            for index in range(6)
        ],
        forbidden={1, 4, 5},
    )
    indices_path = tmp_path / "sst2-test-indices.json"
    write_test_indices(indices_path, [0, 3, 2])
    output_root = tmp_path / "course-data"
    manifest_path = tmp_path / "sst2-course-data.json"

    artifacts = materialize_course_data(
        {"train": training, "validation": validation},
        output_root=output_root,
        manifest_path=manifest_path,
        test_indices_path=indices_path,
        expected_training_rows=4,
        expected_validation_rows=6,
    )

    expected_training = b"".join(
        (
            json.dumps(
                {"idx": index, "label": index % 2, "sentence": f"train {index}"},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for index in range(4)
    )
    expected_test = b"".join(
        (
            json.dumps(
                {"idx": index, "label": index % 2, "sentence": f"valid {index}"},
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for index in (0, 3, 2)
    )
    assert artifacts.training.read_bytes() == expected_training
    assert artifacts.test.read_bytes() == expected_test
    assert training.accessed == [0, 1, 2, 3]
    assert validation.accessed == [0, 3, 2]

    manifest = json.loads(manifest_path.read_text("utf-8"))
    assert manifest["partitions"]["Training"] == {
        "file": "sst2-training.jsonl",
        "row_count": 4,
        "sha256": hashlib.sha256(expected_training).hexdigest(),
        "size_bytes": len(expected_training),
        "source_split": "train",
    }
    assert manifest["partitions"]["Test"] == {
        "file": "sst2-test.jsonl",
        "row_count": 3,
        "sha256": hashlib.sha256(expected_test).hexdigest(),
        "size_bytes": len(expected_test),
        "source_split": "validation",
    }
    serialized_manifest = json.dumps(manifest).lower()
    assert "unseen" not in serialized_manifest
    assert "private" not in serialized_manifest


def test_repeated_materialization_is_byte_identical_and_verifiable(
    tmp_path: Path,
) -> None:
    indices_path = tmp_path / "sst2-test-indices.json"
    write_test_indices(indices_path, [0, 3, 2])
    dataset = {
        "train": InstrumentedSplit(
            [
                {"idx": index, "label": index % 2, "sentence": f"train {index}"}
                for index in range(4)
            ]
        ),
        "validation": InstrumentedSplit(
            [
                {"idx": index, "label": index % 2, "sentence": f"valid {index}"}
                for index in range(6)
            ],
            forbidden={1, 4, 5},
        ),
    }

    snapshots: list[tuple[bytes, bytes, bytes]] = []
    for run_number in (1, 2):
        output_root = tmp_path / f"course-data-{run_number}"
        manifest_path = tmp_path / f"manifest-{run_number}.json"
        artifacts = materialize_course_data(
            dataset,
            output_root=output_root,
            manifest_path=manifest_path,
            test_indices_path=indices_path,
            expected_training_rows=4,
            expected_validation_rows=6,
        )
        snapshots.append(
            (
                artifacts.training.read_bytes(),
                artifacts.test.read_bytes(),
                artifacts.manifest.read_bytes(),
            )
        )
        assert verify_course_data(
            output_root=output_root,
            manifest_path=manifest_path,
            test_indices_path=indices_path,
            expected_training_rows=4,
            expected_validation_rows=6,
        ) == {
            "manifest_sha256": hashlib.sha256(
                artifacts.manifest.read_bytes()
            ).hexdigest(),
            "test_sha256": hashlib.sha256(artifacts.test.read_bytes()).hexdigest(),
            "training_sha256": hashlib.sha256(
                artifacts.training.read_bytes()
            ).hexdigest(),
            "valid": True,
        }

    assert snapshots[1] == snapshots[0]

    (tmp_path / "course-data-2" / "sst2-test.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="Test file"):
        verify_course_data(
            output_root=tmp_path / "course-data-2",
            manifest_path=tmp_path / "manifest-2.json",
            test_indices_path=indices_path,
            expected_training_rows=4,
            expected_validation_rows=6,
        )


def test_local_loader_verifies_inputs_seeds_raw_cache_and_stays_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loader_bytes = b"# exact local loader\n"
    infos_bytes = b'{}\n'
    archive_bytes = b"toy archive bytes"
    archive_url = "https://example.invalid/exact-sst2.zip"
    pin = DatasetSourcePin(
        repository_url="https://example.invalid/datasets.git",
        revision="1" * 40,
        dataset_name="glue",
        configuration="sst2",
        loader_script=pinned_file(
            "https://example.invalid/glue.py", "dataset/glue.py", loader_bytes
        ),
        dataset_infos=pinned_file(
            "https://example.invalid/dataset_infos.json",
            "dataset/dataset_infos.json",
            infos_bytes,
        ),
        raw_archive=pinned_file(
            archive_url, "dataset/raw/SST-2.zip", archive_bytes
        ),
        provenance_mirror=HubPin(
            repo_id="example/glue", revision="2" * 40, repo_type="dataset"
        ),
    )
    input_root = tmp_path / "inputs"
    for file_pin, payload in (
        (pin.loader_script, loader_bytes),
        (pin.dataset_infos, infos_bytes),
        (pin.raw_archive, archive_bytes),
    ):
        path = input_root / file_pin.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    monkeypatch.setenv("HF_DATASETS_OFFLINE", "before")
    datasets_cache = tmp_path / "otherwise-empty-datasets-cache"
    calls: list[tuple[str, str, dict[str, object]]] = []

    def loader(path: str, name: str, **kwargs: object) -> str:
        assert os.environ["HF_DATASETS_OFFLINE"] == "1"
        download_config = kwargs["download_config"]
        assert download_config.local_files_only is True  # type: ignore[union-attr]
        assert download_config.use_etag is False  # type: ignore[union-attr]
        assert Path(download_config.cache_dir) == datasets_cache / "downloads"  # type: ignore[union-attr]
        seeded = (
            datasets_cache
            / "downloads"
            / hashlib.sha256(archive_url.encode("utf-8")).hexdigest()
        )
        assert seeded.read_bytes() == archive_bytes
        calls.append((path, name, kwargs))
        return str(kwargs["split"])

    loaded = load_local_pinned_sst2(
        input_root=input_root,
        datasets_cache=datasets_cache,
        loader=loader,
        pin=pin,
    )

    assert loaded == {"train": "train", "validation": "validation"}
    assert os.environ["HF_DATASETS_OFFLINE"] == "before"
    assert [call[2]["split"] for call in calls] == ["train", "validation"]
    assert all(Path(call[0]) == (input_root / "dataset/glue.py").resolve() for call in calls)
    assert all(call[1] == "sst2" for call in calls)
    metadata_path = (
        datasets_cache
        / "downloads"
        / (hashlib.sha256(archive_url.encode("utf-8")).hexdigest() + ".json")
    )
    assert json.loads(metadata_path.read_text("utf-8")) == {
        "etag": None,
        "url": archive_url,
    }


def test_cli_verify_only_never_loads_or_prepares_source_data(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    indices_path = tmp_path / "sst2-test-indices.json"
    write_test_indices(indices_path, [0, 3, 2])
    output_root = tmp_path / "course-data"
    manifest_path = tmp_path / "sst2-course-data.json"
    materialize_course_data(
        {
            "train": InstrumentedSplit(
                [
                    {"idx": index, "label": index % 2, "sentence": f"train {index}"}
                    for index in range(4)
                ]
            ),
            "validation": InstrumentedSplit(
                [
                    {"idx": index, "label": index % 2, "sentence": f"valid {index}"}
                    for index in range(6)
                ],
                forbidden={1, 4, 5},
            ),
        },
        output_root=output_root,
        manifest_path=manifest_path,
        test_indices_path=indices_path,
        expected_training_rows=4,
        expected_validation_rows=6,
    )

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("verify-only must not load source data")

    assert (
        main(
            [
                "--verify-only",
                "--input-root",
                str(tmp_path / "missing-inputs"),
                "--output-root",
                str(output_root),
                "--manifest-path",
                str(manifest_path),
                "--test-indices-path",
                str(indices_path),
            ],
            loader=forbidden_loader,
            expected_training_rows=4,
            expected_validation_rows=6,
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["valid"] is True


def test_course_test_index_artifact_must_select_exactly_half_of_validation(
    tmp_path: Path,
) -> None:
    indices_path = tmp_path / "sst2-test-indices.json"
    write_test_indices(indices_path, [0, 2])
    training = InstrumentedSplit(
        [{"idx": 0, "label": 0, "sentence": "train"}]
    )
    validation = InstrumentedSplit(
        [
            {"idx": index, "label": index % 2, "sentence": f"valid {index}"}
            for index in range(6)
        ]
    )

    with pytest.raises(ValueError, match="exactly half"):
        materialize_course_data(
            {"train": training, "validation": validation},
            output_root=tmp_path / "course-data",
            manifest_path=tmp_path / "manifest.json",
            test_indices_path=indices_path,
            expected_training_rows=1,
            expected_validation_rows=6,
        )

    assert training.accessed == []
    assert validation.accessed == []


def test_course_test_indices_must_match_the_approved_seeded_permutation(
    tmp_path: Path,
) -> None:
    indices_path = tmp_path / "sst2-test-indices.json"
    write_test_indices(indices_path, [0, 2, 4])
    training = InstrumentedSplit(
        [{"idx": 0, "label": 0, "sentence": "train"}]
    )
    validation = InstrumentedSplit(
        [
            {"idx": index, "label": index % 2, "sentence": f"valid {index}"}
            for index in range(6)
        ]
    )

    with pytest.raises(ValueError, match="approved seeded permutation"):
        materialize_course_data(
            {"train": training, "validation": validation},
            output_root=tmp_path / "course-data",
            manifest_path=tmp_path / "manifest.json",
            test_indices_path=indices_path,
            expected_training_rows=1,
            expected_validation_rows=6,
        )

    assert training.accessed == []
    assert validation.accessed == []
