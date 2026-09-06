"""Create the deterministic, blinded SST-2 course partitions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch


OFFICIAL_VALIDATION_ROWS = 872
SPLIT_SEED = 42
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LEGACY_DATASET_LOADER_COMMIT = (
    "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
)
SST2_ARCHIVE_URL = "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip"
SST2_ARCHIVE_SIZE_BYTES = 7_439_277
SST2_ARCHIVE_SHA256 = (
    "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
)
PRIVATE_INDICES_FILENAME = "sst2-validation-unseen-indices.json"
UNSEEN_COMMITMENT_FILENAME = "sst2-validation-commitment.json"
TEST_INDICES_FILENAME = "sst2-test-indices.json"
SPLIT_MANIFEST_FILENAME = "sst2-split-manifest.json"


@dataclass(frozen=True)
class CourseSplit:
    """Source-row indices for the public Test and final-unseen partitions."""

    test: tuple[int, ...]
    final_unseen: tuple[int, ...]


@dataclass(frozen=True)
class SplitArtifacts:
    """Paths produced by the split-artifact generator."""

    private_indices: Path
    unseen_commitment: Path
    test_indices: Path
    manifest: Path


def namespaced_source_ids(
    source_split: str, indices: Iterable[int]
) -> set[str]:
    """Identify rows without conflating equal positions from different splits."""

    return {f"{source_split}:{index}" for index in indices}


def validate_course_split(
    split: CourseSplit,
    *,
    training_size: int,
    validation_size: int = OFFICIAL_VALIDATION_ROWS,
) -> dict[str, bool | int]:
    """Reject an incomplete or overlapping three-way course partition."""

    expected_half = validation_size // 2
    test_positions = set(split.test)
    unseen_positions = set(split.final_unseen)
    if len(split.test) != expected_half or len(test_positions) != expected_half:
        raise ValueError("Course Test must contain unique half-split indices")
    if (
        len(split.final_unseen) != expected_half
        or len(unseen_positions) != expected_half
    ):
        raise ValueError("Final-unseen must contain unique half-split indices")
    if test_positions & unseen_positions:
        raise ValueError("Course Test and final-unseen indices overlap")
    if test_positions | unseen_positions != set(range(validation_size)):
        raise ValueError("Course partitions do not cover official validation")

    training_ids = namespaced_source_ids("train", range(training_size))
    test_ids = namespaced_source_ids("validation", split.test)
    unseen_ids = namespaced_source_ids("validation", split.final_unseen)
    overlaps = {
        "training_test_overlap": len(training_ids & test_ids),
        "training_final_unseen_overlap": len(training_ids & unseen_ids),
        "test_final_unseen_overlap": len(test_ids & unseen_ids),
    }
    if any(overlaps.values()):
        raise ValueError("Namespaced course partitions overlap")
    return {
        "all_pairwise_disjoint": True,
        **overlaps,
        "validation_union_complete": True,
    }


def create_course_split(
    validation_size: int = OFFICIAL_VALIDATION_ROWS,
    seed: int = SPLIT_SEED,
) -> CourseSplit:
    """Apply the approved OpenDelta-style permutation to an integer row count."""

    if validation_size % 2:
        raise ValueError("The official validation row count must be even")

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(validation_size, generator=generator).tolist()
    midpoint = validation_size // 2
    return CourseSplit(
        test=tuple(permutation[:midpoint]),
        final_unseen=tuple(permutation[midpoint:]),
    )


def generate_split_artifacts(
    dataset: object,
    *,
    public_dir: Path,
    private_root: Path,
    torch_version: str | None = None,
) -> SplitArtifacts:
    """Write blind split artifacts without indexing any final-unseen row."""

    resolved_private_root = private_root.resolve()
    if (
        resolved_private_root == REPOSITORY_ROOT
        or resolved_private_root.is_relative_to(REPOSITORY_ROOT)
    ):
        raise ValueError("Private indices must be stored outside the Git repository")

    training = dataset["train"]
    validation = dataset["validation"]
    split = create_course_split(validation_size=len(validation))
    integrity = validate_course_split(split, training_size=len(training))

    # Materialization is deliberately limited to Training and course Test.
    training_summary = _summarize_examples(
        training, source_split="train", indices=range(len(training))
    )
    test_summary = _summarize_examples(
        validation, source_split="validation", indices=split.test
    )

    public_dir.mkdir(parents=True, exist_ok=True)
    private_root.mkdir(parents=True, exist_ok=True, mode=0o700)

    test_indices_path = public_dir / TEST_INDICES_FILENAME
    test_indices_path.write_bytes(
        _canonical_json_bytes(
            {
                "row_count": len(split.test),
                "source_indices": list(split.test),
                "source_split": "validation",
            }
        )
    )
    private_path = private_root / PRIVATE_INDICES_FILENAME
    private_bytes = _canonical_json_bytes(
        {
            "dataset": "glue/sst2",
            "partition": "Validation (Unseen)",
            "row_count": len(split.final_unseen),
            "schema_version": 1,
            "source_indices": list(split.final_unseen),
            "source_split": "validation",
        }
    )
    descriptor = os.open(
        private_path,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
        0o600,
    )
    try:
        os.write(descriptor, private_bytes)
    finally:
        os.close(descriptor)
    private_path.chmod(0o600)

    commitment_path = public_dir / UNSEEN_COMMITMENT_FILENAME
    commitment_path.write_bytes(
        _canonical_json_bytes(
            {
                "row_count": len(split.final_unseen),
                "sha256": hashlib.sha256(private_bytes).hexdigest(),
            }
        )
    )

    commitment = json.loads(commitment_path.read_text("utf-8"))
    manifest_path = public_dir / SPLIT_MANIFEST_FILENAME
    manifest_path.write_bytes(
        _canonical_json_bytes(
            {
                "dataset": {
                    "archive_sha256": SST2_ARCHIVE_SHA256,
                    "archive_size_bytes": SST2_ARCHIVE_SIZE_BYTES,
                    "archive_url": SST2_ARCHIVE_URL,
                    "configuration": "sst2",
                    "legacy_loader_commit": LEGACY_DATASET_LOADER_COMMIT,
                    "name": "glue",
                },
                "integrity": integrity,
                "partitions": {
                    "Test": {
                        **test_summary,
                        "indices_file": TEST_INDICES_FILENAME,
                        "indices_sha256": hashlib.sha256(
                            test_indices_path.read_bytes()
                        ).hexdigest(),
                    },
                    "Training": training_summary,
                    "Validation (Unseen)": commitment,
                },
                "schema_version": 1,
                "split_rule": {
                    "permutation_call": (
                        "torch.randperm(872, "
                        "generator=torch.Generator().manual_seed(42))"
                    ),
                    "seed": SPLIT_SEED,
                    "torch_version": str(torch_version or torch.__version__),
                },
            }
        )
    )
    return SplitArtifacts(
        private_indices=private_path,
        unseen_commitment=commitment_path,
        test_indices=test_indices_path,
        manifest=manifest_path,
    )


def verify_split_artifacts(
    *, public_dir: Path, private_root: Path
) -> dict[str, str | bool]:
    """Verify public artifacts and the opaque private-file commitment."""

    manifest_path = public_dir / SPLIT_MANIFEST_FILENAME
    test_indices_path = public_dir / TEST_INDICES_FILENAME
    commitment_path = public_dir / UNSEEN_COMMITMENT_FILENAME
    private_path = private_root / PRIVATE_INDICES_FILENAME

    manifest_bytes = manifest_path.read_bytes()
    test_indices_bytes = test_indices_path.read_bytes()
    commitment_bytes = commitment_path.read_bytes()
    private_bytes = private_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    test_payload = json.loads(test_indices_bytes)
    commitment = json.loads(commitment_bytes)

    split = create_course_split()
    expected_test_payload = {
        "row_count": len(split.test),
        "source_indices": list(split.test),
        "source_split": "validation",
    }
    if test_payload != expected_test_payload:
        raise ValueError("Public Test indices do not match the approved split")

    test_sha256 = hashlib.sha256(test_indices_bytes).hexdigest()
    private_sha256 = hashlib.sha256(private_bytes).hexdigest()
    expected_commitment = {
        "row_count": len(split.final_unseen),
        "sha256": private_sha256,
    }
    if commitment != expected_commitment:
        raise ValueError("Private indices do not match the public commitment")
    if manifest["partitions"]["Validation (Unseen)"] != expected_commitment:
        raise ValueError("Manifest contains a different unseen commitment")
    if manifest["partitions"]["Test"]["indices_sha256"] != test_sha256:
        raise ValueError("Manifest Test-index checksum does not match")
    if (
        manifest["dataset"]["legacy_loader_commit"]
        != LEGACY_DATASET_LOADER_COMMIT
    ):
        raise ValueError("Manifest does not use the pinned legacy loader")
    if stat.S_IMODE(private_path.stat().st_mode) != 0o600:
        raise PermissionError("Private index file permissions must be 0600")

    return {
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "private_indices_sha256": private_sha256,
        "test_indices_sha256": test_sha256,
        "valid": True,
    }


def load_pinned_sst2(
    *,
    cache_dir: Path,
    loader: Callable[..., object] | None = None,
) -> dict[str, object]:
    """Load only the two labeled official splits through the pinned script."""

    if loader is None:
        from datasets import load_dataset

        loader = load_dataset
    common = {
        "cache_dir": str(cache_dir),
        "revision": LEGACY_DATASET_LOADER_COMMIT,
    }
    return {
        "train": loader("glue", "sst2", **common, split="train"),
        "validation": loader("glue", "sst2", **common, split="validation"),
    }


def main(
    argv: list[str] | None = None,
    *,
    loader: Callable[..., object] | None = None,
    torch_version: str | None = None,
) -> int:
    """Generate or verify the approved split artifacts."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--public-dir", type=Path, default=Path("data/manifests")
    )
    parser.add_argument(
        "--private-root",
        type=Path,
        default=Path(
            os.environ.get(
                "DS340W_PRIVATE_ROOT",
                Path.home() / ".local" / "share" / "ds340w-private",
            )
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "ds340w" / "datasets",
    )
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)

    if not arguments.verify_only:
        dataset = load_pinned_sst2(
            cache_dir=arguments.cache_dir, loader=loader
        )
        generate_split_artifacts(
            dataset,
            public_dir=arguments.public_dir,
            private_root=arguments.private_root,
            torch_version=torch_version,
        )
    report = verify_split_artifacts(
        public_dir=arguments.public_dir,
        private_root=arguments.private_root,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def _canonical_json_bytes(payload: object) -> bytes:
    return (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _summarize_examples(
    split: object,
    *,
    source_split: str,
    indices: Iterable[int],
) -> dict[str, object]:
    digest = hashlib.sha256()
    class_counts = {"0": 0, "1": 0}
    row_count = 0
    for index in indices:
        row = split[index]
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(
                f"Unexpected SST-2 label {label} at {source_split}:{index}"
            )
        sentence = row["sentence"]
        if not isinstance(sentence, str):
            raise TypeError(
                f"SST-2 sentence must be text at {source_split}:{index}"
            )
        class_counts[str(label)] += 1
        record = {
            "label": label,
            "sentence": sentence,
            "source_id": f"{source_split}:{index}",
        }
        canonical_record = json.dumps(
            record,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        digest.update((canonical_record + "\n").encode("utf-8"))
        row_count += 1
    return {
        "class_counts": class_counts,
        "content_sha256": digest.hexdigest(),
        "row_count": row_count,
        "source_split": source_split,
    }


if __name__ == "__main__":
    raise SystemExit(main())
