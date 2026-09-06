import hashlib
import json
import stat
from collections.abc import Collection
from pathlib import Path

import pytest

from ds340w_repro.data.access_guard import (
    FinalUnseenAccessError,
    resolve_partition_selection,
)
from ds340w_repro.data.create_sst2_split import (
    create_course_split,
    generate_split_artifacts,
    load_pinned_sst2,
    main,
    namespaced_source_ids,
    validate_course_split,
    verify_split_artifacts,
)


class InstrumentedSplit:
    def __init__(self, size: int, forbidden: Collection[int] = ()) -> None:
        self.size = size
        self.forbidden = set(forbidden)
        self.accessed: list[int] = []

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, object]:
        if index in self.forbidden:
            raise AssertionError(f"forbidden row {index} was indexed")
        self.accessed.append(index)
        return {
            "sentence": f"sentence {index}",
            "label": index % 2,
            "idx": index,
        }


def write_valid_lock(path: Path, private_sha256: str) -> None:
    path.write_text(
        json.dumps(
            {
                "git_commit": "a" * 40,
                "locked": True,
                "private_unseen_indices_sha256": private_sha256,
                "resolved_config_sha256": "b" * 64,
                "schema_version": 1,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_course_split_matches_the_approved_seeded_permutation() -> None:
    split = create_course_split(validation_size=872, seed=42)

    assert split.test[:10] == (
        718,
        74,
        168,
        589,
        106,
        3,
        542,
        211,
        750,
        640,
    )
    assert split.test[-6:] == (289, 140, 708, 588, 40, 96)
    assert len(split.test) == 436
    assert len(split.final_unseen) == 436


def test_course_partitions_are_complete_unique_and_namespaced() -> None:
    split = create_course_split()

    assert len(set(split.test)) == 436
    assert len(set(split.final_unseen)) == 436
    assert set(split.test).isdisjoint(split.final_unseen)
    assert set(split.test) | set(split.final_unseen) == set(range(872))

    # Identical integer positions in different official splits are not overlap.
    assert namespaced_source_ids("train", [0]).isdisjoint(
        namespaced_source_ids("validation", [0])
    )
    assert validate_course_split(split, training_size=67_349) == {
        "all_pairwise_disjoint": True,
        "training_test_overlap": 0,
        "training_final_unseen_overlap": 0,
        "test_final_unseen_overlap": 0,
        "validation_union_complete": True,
    }


def test_artifact_generation_commits_to_unseen_indices_without_reading_rows(
    tmp_path: Path,
) -> None:
    split = create_course_split()
    training = InstrumentedSplit(6)
    validation = InstrumentedSplit(872, forbidden=split.final_unseen)
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"

    artifacts = generate_split_artifacts(
        {"train": training, "validation": validation},
        public_dir=public_dir,
        private_root=private_root,
        torch_version="2.1.2+cu121",
    )

    assert training.accessed == list(range(6))
    assert validation.accessed == list(split.test)
    assert set(validation.accessed).isdisjoint(split.final_unseen)

    private_bytes = artifacts.private_indices.read_bytes()
    commitment = json.loads(artifacts.unseen_commitment.read_text("utf-8"))
    assert commitment == {
        "row_count": 436,
        "sha256": hashlib.sha256(private_bytes).hexdigest(),
    }
    assert stat.S_IMODE(artifacts.private_indices.stat().st_mode) == 0o600


def test_public_manifest_contains_only_training_and_test_derived_metadata(
    tmp_path: Path,
) -> None:
    split = create_course_split()
    public_dir = tmp_path / "public"
    artifacts = generate_split_artifacts(
        {
            "train": InstrumentedSplit(6),
            "validation": InstrumentedSplit(872, split.final_unseen),
        },
        public_dir=public_dir,
        private_root=tmp_path / "private",
        torch_version="2.1.2+cu121",
    )

    test_indices = json.loads(artifacts.test_indices.read_text("utf-8"))
    assert test_indices == {
        "row_count": 436,
        "source_indices": list(split.test),
        "source_split": "validation",
    }

    manifest = json.loads(artifacts.manifest.read_text("utf-8"))
    assert manifest["dataset"] == {
        "archive_sha256": (
            "d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa"
        ),
        "archive_size_bytes": 7_439_277,
        "archive_url": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        "configuration": "sst2",
        "legacy_loader_commit": "dff6c92c80fbfac456349a531c0a4521c2b0c8f8",
        "name": "glue",
    }
    assert manifest["split_rule"] == {
        "permutation_call": (
            "torch.randperm(872, generator=torch.Generator().manual_seed(42))"
        ),
        "seed": 42,
        "torch_version": "2.1.2+cu121",
    }
    assert manifest["partitions"]["Training"] == {
        "class_counts": {"0": 3, "1": 3},
        "content_sha256": (
            "78d7b82dff58ed3c262a07adbc918ad3a9a6477a80cfd93081271a3b396ed72c"
        ),
        "row_count": 6,
        "source_split": "train",
    }
    assert manifest["partitions"]["Test"] == {
        "class_counts": {"0": 226, "1": 210},
        "content_sha256": (
            "c418587819e32bdaa4373927a2d976948831b6fa4d361f2a32f818bc87e84a62"
        ),
        "indices_file": "sst2-test-indices.json",
        "indices_sha256": hashlib.sha256(
            artifacts.test_indices.read_bytes()
        ).hexdigest(),
        "row_count": 436,
        "source_split": "validation",
    }
    unseen = manifest["partitions"]["Validation (Unseen)"]
    assert unseen == json.loads(artifacts.unseen_commitment.read_text("utf-8"))
    assert set(unseen) == {"row_count", "sha256"}
    assert manifest["integrity"] == {
        "all_pairwise_disjoint": True,
        "test_final_unseen_overlap": 0,
        "training_final_unseen_overlap": 0,
        "training_test_overlap": 0,
        "validation_union_complete": True,
    }


def test_non_final_access_does_not_require_the_private_index_file(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    artifacts = generate_split_artifacts(
        {
            "train": InstrumentedSplit(6),
            "validation": InstrumentedSplit(
                872, create_course_split().final_unseen
            ),
        },
        public_dir=public_dir,
        private_root=tmp_path / "private-generation-only",
        torch_version="2.1.2+cu121",
    )
    artifacts.private_indices.unlink()
    absent_private_root = tmp_path / "absent-private-root"

    training = resolve_partition_selection(
        "Training", public_dir=public_dir, private_root=absent_private_root
    )
    test = resolve_partition_selection(
        "Test", public_dir=public_dir, private_root=absent_private_root
    )

    assert training.source_split == "train"
    assert training.source_indices is None
    assert test.source_split == "validation"
    assert test.source_indices == create_course_split().test


@pytest.mark.parametrize(
    ("evaluation_mode", "write_lock", "message"),
    [
        ("development", True, "final evaluation mode"),
        ("final", False, "lock artifact"),
    ],
)
def test_final_unseen_requires_both_explicit_mode_and_a_lock_artifact(
    tmp_path: Path,
    evaluation_mode: str,
    write_lock: bool,
    message: str,
) -> None:
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"
    generate_split_artifacts(
        {
            "train": InstrumentedSplit(6),
            "validation": InstrumentedSplit(
                872, create_course_split().final_unseen
            ),
        },
        public_dir=public_dir,
        private_root=private_root,
        torch_version="2.1.2+cu121",
    )
    lock_artifact = tmp_path / "experiment-lock.json"
    if write_lock:
        lock_artifact.write_text('{"locked":true}\n', encoding="utf-8")

    with pytest.raises(FinalUnseenAccessError, match=message):
        resolve_partition_selection(
            "Validation (Unseen)",
            public_dir=public_dir,
            private_root=private_root,
            evaluation_mode=evaluation_mode,
            lock_artifact=lock_artifact,
        )


def test_final_unseen_rejects_an_incomplete_lock_artifact(tmp_path: Path) -> None:
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"
    generate_split_artifacts(
        {
            "train": InstrumentedSplit(6),
            "validation": InstrumentedSplit(
                872, create_course_split().final_unseen
            ),
        },
        public_dir=public_dir,
        private_root=private_root,
        torch_version="2.1.2+cu121",
    )
    lock_artifact = tmp_path / "experiment-lock.json"
    lock_artifact.write_text('{"locked":true}\n', encoding="utf-8")

    with pytest.raises(FinalUnseenAccessError, match="schema_version"):
        resolve_partition_selection(
            "Validation (Unseen)",
            public_dir=public_dir,
            private_root=private_root,
            evaluation_mode="final",
            lock_artifact=lock_artifact,
        )


def test_week1_guard_fails_closed_even_with_a_nominal_final_lock(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"
    generate_split_artifacts(
        {
            "train": InstrumentedSplit(6),
            "validation": InstrumentedSplit(
                872, create_course_split().final_unseen
            ),
        },
        public_dir=public_dir,
        private_root=private_root,
        torch_version="2.1.2+cu121",
    )
    lock_artifact = tmp_path / "experiment-lock.json"
    commitment = json.loads(
        (public_dir / "sst2-validation-commitment.json").read_text("utf-8")
    )
    write_valid_lock(lock_artifact, commitment["sha256"])

    with pytest.raises(FinalUnseenAccessError, match="disabled until"):
        resolve_partition_selection(
            "Validation (Unseen)",
            public_dir=public_dir,
            private_root=private_root,
            evaluation_mode="final",
            lock_artifact=lock_artifact,
        )


def test_final_access_requires_the_commitment_to_match_the_public_manifest(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"
    generate_split_artifacts(
        {
            "train": InstrumentedSplit(6),
            "validation": InstrumentedSplit(
                872, create_course_split().final_unseen
            ),
        },
        public_dir=public_dir,
        private_root=private_root,
        torch_version="2.1.2+cu121",
    )
    commitment = json.loads(
        (public_dir / "sst2-validation-commitment.json").read_text("utf-8")
    )
    lock_artifact = tmp_path / "experiment-lock.json"
    write_valid_lock(lock_artifact, commitment["sha256"])
    manifest_path = public_dir / "sst2-split-manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["partitions"]["Validation (Unseen)"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(FinalUnseenAccessError, match="public manifest"):
        resolve_partition_selection(
            "Validation (Unseen)",
            public_dir=public_dir,
            private_root=private_root,
            evaluation_mode="final",
            lock_artifact=lock_artifact,
        )


def test_repeated_generation_is_byte_identical_and_verifiable(
    tmp_path: Path,
) -> None:
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"

    def generate() -> dict[str, bytes]:
        artifacts = generate_split_artifacts(
            {
                "train": InstrumentedSplit(6),
                "validation": InstrumentedSplit(
                    872, create_course_split().final_unseen
                ),
            },
            public_dir=public_dir,
            private_root=private_root,
            torch_version="2.1.2+cu121",
        )
        paths = (
            artifacts.manifest,
            artifacts.test_indices,
            artifacts.unseen_commitment,
            artifacts.private_indices,
        )
        return {path.name: path.read_bytes() for path in paths}

    first = generate()
    second = generate()

    assert second == first
    verification = verify_split_artifacts(
        public_dir=public_dir, private_root=private_root
    )
    assert verification["valid"] is True
    assert verification["manifest_sha256"] == hashlib.sha256(
        first["sst2-split-manifest.json"]
    ).hexdigest()
    assert verification["test_indices_sha256"] == hashlib.sha256(
        first["sst2-test-indices.json"]
    ).hexdigest()
    assert verification["private_indices_sha256"] == hashlib.sha256(
        first["sst2-validation-unseen-indices.json"]
    ).hexdigest()


def test_dataset_loader_uses_the_exact_legacy_commit_and_explicit_splits(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def loader(*args: object, **kwargs: object) -> str:
        calls.append((args, kwargs))
        return str(kwargs["split"])

    loaded = load_pinned_sst2(cache_dir=tmp_path / "cache", loader=loader)

    assert loaded == {"train": "train", "validation": "validation"}
    common = {
        "cache_dir": str(tmp_path / "cache"),
        "revision": "dff6c92c80fbfac456349a531c0a4521c2b0c8f8",
    }
    assert calls == [
        (("glue", "sst2"), {**common, "split": "train"}),
        (("glue", "sst2"), {**common, "split": "validation"}),
    ]


def test_cli_can_generate_then_verify_without_loading_data_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    public_dir = tmp_path / "public"
    private_root = tmp_path / "private"
    cache_dir = tmp_path / "cache"
    split = create_course_split()

    def loader(*args: object, **kwargs: object) -> InstrumentedSplit:
        del args
        if kwargs["split"] == "train":
            return InstrumentedSplit(6)
        return InstrumentedSplit(872, split.final_unseen)

    assert (
        main(
            [
                "--public-dir",
                str(public_dir),
                "--private-root",
                str(private_root),
                "--cache-dir",
                str(cache_dir),
            ],
            loader=loader,
            torch_version="2.1.2+cu121",
        )
        == 0
    )

    def forbidden_loader(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("verify-only must not load the dataset")

    assert (
        main(
            [
                "--verify-only",
                "--public-dir",
                str(public_dir),
                "--private-root",
                str(private_root),
            ],
            loader=forbidden_loader,
        )
        == 0
    )
    reports = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert reports[0]["valid"] is True
    assert reports[1] == reports[0]


def test_private_indices_cannot_be_written_inside_the_repository(
    tmp_path: Path,
) -> None:
    forbidden_root = Path.cwd() / ".must-not-write-private-split"
    assert not forbidden_root.exists()
    training = InstrumentedSplit(1, forbidden=[0])

    with pytest.raises(ValueError, match="outside the Git repository"):
        generate_split_artifacts(
            {
                "train": training,
                "validation": InstrumentedSplit(872),
            },
            public_dir=tmp_path / "public",
            private_root=forbidden_root,
            torch_version="2.1.2+cu121",
        )

    assert training.accessed == []
    assert not forbidden_root.exists()
