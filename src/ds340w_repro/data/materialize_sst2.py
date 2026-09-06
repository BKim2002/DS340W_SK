"""Materialize the public SST-2 course partitions as deterministic JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping

from ds340w_repro.data.create_sst2_split import (
    LEGACY_DATASET_LOADER_COMMIT,
    OFFICIAL_VALIDATION_ROWS,
    SST2_ARCHIVE_SHA256,
    SST2_ARCHIVE_SIZE_BYTES,
    SST2_ARCHIVE_URL,
    TEST_INDICES_FILENAME,
    create_course_split,
)
from ds340w_repro.source_pins import (
    DATASET_PIN,
    DatasetSourcePin,
    fetch_dataset_inputs,
)


OFFICIAL_TRAINING_ROWS = 67_349
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT_ROOT = Path.home() / ".cache" / "ds340w" / "inputs"
DEFAULT_OUTPUT_ROOT = Path.home() / ".cache" / "ds340w" / "course-data"
DEFAULT_DATASETS_CACHE = Path.home() / ".cache" / "ds340w" / "datasets"
DEFAULT_MANIFEST_PATH = REPOSITORY_ROOT / "data" / "manifests" / "sst2-course-data.json"
DEFAULT_TEST_INDICES_PATH = REPOSITORY_ROOT / "data" / "manifests" / TEST_INDICES_FILENAME
TRAINING_FILENAME = "sst2-training.jsonl"
TEST_FILENAME = "sst2-test.jsonl"


@dataclass(frozen=True)
class CourseDataArtifacts:
    """Paths written for the two public course partitions."""

    training: Path
    test: Path
    manifest: Path


def load_local_pinned_sst2(
    *,
    input_root: Path = DEFAULT_INPUT_ROOT,
    datasets_cache: Path = DEFAULT_DATASETS_CACHE,
    loader: Callable[..., object] | None = None,
    pin: DatasetSourcePin = DATASET_PIN,
) -> dict[str, object]:
    """Prepare SST-2 offline from three hash-verified local source inputs."""

    report = fetch_dataset_inputs(
        input_root,
        pin=pin,
        local_files_only=True,
    )
    datasets_cache = datasets_cache.resolve()
    downloads_cache = datasets_cache / "downloads"
    _seed_raw_download_cache(
        report.raw_archive_path,
        downloads_cache=downloads_cache,
        pin=pin,
    )

    with _datasets_offline():
        from datasets import DownloadConfig, load_dataset

        selected_loader = loader or load_dataset
        download_config = DownloadConfig(
            cache_dir=downloads_cache,
            local_files_only=True,
            use_etag=False,
        )
        common = {
            "cache_dir": str(datasets_cache),
            "download_config": download_config,
        }
        return {
            "train": selected_loader(
                str(report.loader_path),
                pin.configuration,
                **common,
                split="train",
            ),
            "validation": selected_loader(
                str(report.loader_path),
                pin.configuration,
                **common,
                split="validation",
            ),
        }


def materialize_course_data(
    dataset: Mapping[str, object],
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    test_indices_path: Path = DEFAULT_TEST_INDICES_PATH,
    expected_training_rows: int = OFFICIAL_TRAINING_ROWS,
    expected_validation_rows: int = OFFICIAL_VALIDATION_ROWS,
) -> CourseDataArtifacts:
    """Write Training and committed course Test without reading other rows."""

    training = dataset["train"]
    validation = dataset["validation"]
    if len(training) != expected_training_rows:  # type: ignore[arg-type]
        raise ValueError(
            "Training row count mismatch: "
            f"expected {expected_training_rows}, found {len(training)}"  # type: ignore[arg-type]
        )
    if len(validation) != expected_validation_rows:  # type: ignore[arg-type]
        raise ValueError(
            "Validation row count mismatch: "
            f"expected {expected_validation_rows}, found {len(validation)}"  # type: ignore[arg-type]
        )

    test_indices = _load_test_indices(
        test_indices_path, validation_size=expected_validation_rows
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    training_path = output_root / TRAINING_FILENAME
    test_path = output_root / TEST_FILENAME

    training_record = _write_partition(
        training,  # type: ignore[arg-type]
        range(expected_training_rows),
        destination=training_path,
        source_split="train",
    )
    test_record = _write_partition(
        validation,  # type: ignore[arg-type]
        test_indices,
        destination=test_path,
        source_split="validation",
    )
    manifest = {
        "dataset": _expected_dataset_manifest(),
        "partitions": {
            "Test": test_record,
            "Training": training_record,
        },
        "schema_version": 1,
    }
    manifest_path = manifest_path.resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write(manifest_path, _canonical_manifest_bytes(manifest))
    return CourseDataArtifacts(
        training=training_path,
        test=test_path,
        manifest=manifest_path,
    )


def verify_course_data(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    test_indices_path: Path = DEFAULT_TEST_INDICES_PATH,
    expected_training_rows: int = OFFICIAL_TRAINING_ROWS,
    expected_validation_rows: int = OFFICIAL_VALIDATION_ROWS,
) -> dict[str, str | bool]:
    """Verify canonical bytes, identities, and row selections without source data."""

    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    if set(manifest) != {"dataset", "partitions", "schema_version"}:
        raise ValueError("Course-data manifest has unexpected top-level fields")
    if manifest["schema_version"] != 1:
        raise ValueError("Course-data manifest schema version is unsupported")
    if manifest["dataset"] != _expected_dataset_manifest():
        raise ValueError("Course-data manifest does not match the pinned source inputs")
    if set(manifest["partitions"]) != {"Training", "Test"}:
        raise ValueError("Course-data manifest must contain only Training and Test")

    test_indices = _load_test_indices(
        test_indices_path, validation_size=expected_validation_rows
    )
    partition_specs = (
        (
            "Training",
            TRAINING_FILENAME,
            tuple(range(expected_training_rows)),
            "train",
        ),
        ("Test", TEST_FILENAME, test_indices, "validation"),
    )
    digests: dict[str, str] = {}
    for partition, filename, expected_indices, source_split in partition_specs:
        record = manifest["partitions"][partition]
        if set(record) != {
            "file",
            "row_count",
            "sha256",
            "size_bytes",
            "source_split",
        }:
            raise ValueError(f"{partition} manifest record has unexpected fields")
        expected_metadata = {
            "file": filename,
            "row_count": len(expected_indices),
            "source_split": source_split,
        }
        for key, expected in expected_metadata.items():
            if record.get(key) != expected:
                raise ValueError(f"{partition} manifest {key} mismatch")

        path = output_root.resolve() / filename
        payload = path.read_bytes()
        if len(payload) != record["size_bytes"]:
            raise ValueError(f"{partition} file size does not match its manifest")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"{partition} file SHA-256 does not match its manifest")
        _verify_jsonl(payload, expected_indices=expected_indices, partition=partition)
        digests[partition] = digest

    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "test_sha256": digests["Test"],
        "training_sha256": digests["Training"],
        "valid": True,
    }


def main(
    argv: list[str] | None = None,
    *,
    loader: Callable[..., object] | None = None,
    pin: DatasetSourcePin = DATASET_PIN,
    expected_training_rows: int = OFFICIAL_TRAINING_ROWS,
    expected_validation_rows: int = OFFICIAL_VALIDATION_ROWS,
) -> int:
    """Generate or verify the isolated public SST-2 course data."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--datasets-cache", type=Path, default=DEFAULT_DATASETS_CACHE
    )
    parser.add_argument(
        "--manifest-path", type=Path, default=DEFAULT_MANIFEST_PATH
    )
    parser.add_argument(
        "--test-indices-path", type=Path, default=DEFAULT_TEST_INDICES_PATH
    )
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)

    if not arguments.verify_only:
        dataset = load_local_pinned_sst2(
            input_root=arguments.input_root,
            datasets_cache=arguments.datasets_cache,
            loader=loader,
            pin=pin,
        )
        materialize_course_data(
            dataset,
            output_root=arguments.output_root,
            manifest_path=arguments.manifest_path,
            test_indices_path=arguments.test_indices_path,
            expected_training_rows=expected_training_rows,
            expected_validation_rows=expected_validation_rows,
        )
    report = verify_course_data(
        output_root=arguments.output_root,
        manifest_path=arguments.manifest_path,
        test_indices_path=arguments.test_indices_path,
        expected_training_rows=expected_training_rows,
        expected_validation_rows=expected_validation_rows,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _load_test_indices(path: Path, *, validation_size: int) -> tuple[int, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = payload.get("source_indices")
    if payload.get("source_split") != "validation" or not isinstance(indices, list):
        raise ValueError("Course Test index artifact has an invalid schema")
    if payload.get("row_count") != len(indices):
        raise ValueError("Course Test index row count does not match its indices")
    if validation_size % 2 or len(indices) != validation_size // 2:
        raise ValueError("Course Test must select exactly half of validation")
    if any(type(index) is not int for index in indices):
        raise ValueError("Course Test indices must be integers")
    if len(set(indices)) != len(indices):
        raise ValueError("Course Test indices must be unique")
    if any(index < 0 or index >= validation_size for index in indices):
        raise ValueError("Course Test index falls outside official validation")
    if tuple(indices) != create_course_split(validation_size=validation_size).test:
        raise ValueError("Course Test indices do not match the approved seeded permutation")
    return tuple(indices)


def _seed_raw_download_cache(
    source: Path,
    *,
    downloads_cache: Path,
    pin: DatasetSourcePin,
) -> Path:
    downloads_cache.mkdir(parents=True, exist_ok=True)
    filename = hashlib.sha256(pin.raw_archive.url.encode("utf-8")).hexdigest()
    destination = downloads_cache / filename
    expected = pin.raw_archive.sha256
    if not destination.is_file() or _sha256_file(destination) != expected:
        temporary = destination.with_name(f".{destination.name}.partial")
        try:
            temporary.write_bytes(source.read_bytes())
            if _sha256_file(temporary) != expected:
                raise ValueError("Raw-download cache seed does not match pinned archive")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    metadata = {"etag": None, "url": pin.raw_archive.url}
    _atomic_write(
        destination.with_name(f"{destination.name}.json"),
        _canonical_compact_json_bytes(metadata),
    )
    return destination


@contextmanager
def _datasets_offline() -> Iterator[None]:
    key = "HF_DATASETS_OFFLINE"
    was_present = key in os.environ
    previous = os.environ.get(key)
    os.environ[key] = "1"
    datasets_config = None
    previous_flag = None
    try:
        from datasets import config as datasets_config

        previous_flag = datasets_config.HF_DATASETS_OFFLINE
        datasets_config.HF_DATASETS_OFFLINE = True
        yield
    finally:
        if datasets_config is not None:
            datasets_config.HF_DATASETS_OFFLINE = previous_flag
        if was_present:
            assert previous is not None
            os.environ[key] = previous
        else:
            os.environ.pop(key, None)


def _verify_jsonl(
    payload: bytes, *, expected_indices: tuple[int, ...], partition: str
) -> None:
    lines = payload.splitlines(keepends=True)
    if len(lines) != len(expected_indices):
        raise ValueError(f"{partition} file row count does not match its manifest")
    for position, (line, expected_index) in enumerate(zip(lines, expected_indices)):
        try:
            decoded = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"{partition} file row {position} is not valid UTF-8 JSON"
            ) from error
        record = _canonical_row(decoded, expected_index=expected_index)
        canonical = (
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if line != canonical:
            raise ValueError(f"{partition} file row {position} is not canonical JSONL")


def _expected_dataset_manifest() -> dict[str, object]:
    return {
        "configuration": DATASET_PIN.configuration,
        "legacy_loader_commit": LEGACY_DATASET_LOADER_COMMIT,
        "name": DATASET_PIN.dataset_name,
        "source_inputs": {
            "dataset_infos": {
                "relative_path": DATASET_PIN.dataset_infos.relative_path,
                "sha256": DATASET_PIN.dataset_infos.sha256,
                "size_bytes": DATASET_PIN.dataset_infos.size,
            },
            "loader_script": {
                "relative_path": DATASET_PIN.loader_script.relative_path,
                "sha256": DATASET_PIN.loader_script.sha256,
                "size_bytes": DATASET_PIN.loader_script.size,
            },
            "raw_archive": {
                "relative_path": DATASET_PIN.raw_archive.relative_path,
                "sha256": SST2_ARCHIVE_SHA256,
                "size_bytes": SST2_ARCHIVE_SIZE_BYTES,
                "url": SST2_ARCHIVE_URL,
            },
        },
    }


def _write_partition(
    split: object,
    indices: Iterable[int],
    *,
    destination: Path,
    source_split: str,
) -> dict[str, object]:
    temporary = destination.with_name(f".{destination.name}.partial")
    digest = hashlib.sha256()
    row_count = 0
    try:
        with temporary.open("wb") as stream:
            for index in indices:
                row = split[index]  # type: ignore[index]
                record = _canonical_row(row, expected_index=index)
                encoded = (
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                stream.write(encoded)
                digest.update(encoded)
                row_count += 1
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "file": destination.name,
        "row_count": row_count,
        "sha256": digest.hexdigest(),
        "size_bytes": destination.stat().st_size,
        "source_split": source_split,
    }


def _canonical_row(row: object, *, expected_index: int) -> dict[str, object]:
    if not isinstance(row, Mapping):
        raise TypeError(f"SST-2 row {expected_index} must be a mapping")
    index = row.get("idx")
    if type(index) is not int or index != expected_index:
        raise ValueError(
            f"SST-2 row index mismatch: expected {expected_index}, found {index!r}"
        )
    label = row.get("label")
    if type(label) is not int or label not in (0, 1):
        raise ValueError(f"Unexpected SST-2 label {label!r} at source index {index}")
    sentence = row.get("sentence")
    if not isinstance(sentence, str):
        raise TypeError(f"SST-2 sentence must be text at source index {index}")
    return {"idx": index, "label": label, "sentence": sentence}


def _canonical_manifest_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    try:
        temporary.write_bytes(payload)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_compact_json_bytes(payload: object) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
