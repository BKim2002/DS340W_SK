from __future__ import annotations

from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path

import pytest

from scripts.fetch_pinned_inputs import run
from ds340w_repro.source_pins import (
    DATASET_MIRROR_PIN,
    DATASET_PIN,
    MODEL_PIN,
    OPENDELTA_REVISION,
    DatasetSourcePin,
    HubPin,
    PinnedFile,
    build_source_lock,
    fetch_dataset_inputs,
    fetch_model_snapshot,
    require_full_git_sha,
    verify_source_lock,
)


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _test_dataset_pin(payloads: dict[str, bytes]) -> DatasetSourcePin:
    return DatasetSourcePin(
        repository_url="https://github.com/example/datasets.git",
        revision="a" * 40,
        dataset_name="glue",
        configuration="sst2",
        loader_script=PinnedFile(
            url="https://example.test/glue.py",
            relative_path="dataset/glue.py",
            size=len(payloads["https://example.test/glue.py"]),
            sha256=_digest(payloads["https://example.test/glue.py"]),
        ),
        dataset_infos=PinnedFile(
            url="https://example.test/dataset_infos.json",
            relative_path="dataset/dataset_infos.json",
            size=len(payloads["https://example.test/dataset_infos.json"]),
            sha256=_digest(payloads["https://example.test/dataset_infos.json"]),
        ),
        raw_archive=PinnedFile(
            url="https://example.test/SST-2.zip",
            relative_path="dataset/raw/SST-2.zip",
            size=len(payloads["https://example.test/SST-2.zip"]),
            sha256=_digest(payloads["https://example.test/SST-2.zip"]),
        ),
        provenance_mirror=HubPin("example/glue", "b" * 40, repo_type="dataset"),
    )


def _fake_model_snapshot(tmp_path: Path, revision: str) -> Path:
    snapshot = tmp_path / "hub" / "models--example--model" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    for name in (
        "config.json",
        "generation_config.json",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    ):
        (snapshot / name).write_bytes(f"fixture:{name}".encode())
    return snapshot


def test_approved_inputs_are_full_immutable_revisions() -> None:
    assert OPENDELTA_REVISION == "20e426f716ea53618657cd34bbfc4b347b98cd74"
    assert MODEL_PIN == HubPin(
        "google/t5-base-lm-adapt",
        "bf48a6b7de15c6afe1e27e1a51aa1ea4626f96bf",
    )
    assert DATASET_PIN.repository_url == "https://github.com/huggingface/datasets.git"
    assert DATASET_PIN.revision == "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
    assert DATASET_PIN.dataset_name == "glue"
    assert DATASET_PIN.configuration == "sst2"
    assert DATASET_MIRROR_PIN == HubPin(
        "nyu-mll/glue",
        "bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
        repo_type="dataset",
    )

    for revision in (
        OPENDELTA_REVISION,
        MODEL_PIN.revision,
        DATASET_PIN.revision,
        DATASET_MIRROR_PIN.revision,
    ):
        require_full_git_sha(revision)


def test_canonical_sst2_files_have_fixed_urls_sizes_and_hashes() -> None:
    assert DATASET_PIN.loader_script == PinnedFile(
        url=(
            "https://raw.githubusercontent.com/huggingface/datasets/"
            "dff6c92c80fbfac456349a531c0a4521c2b0c8f8/datasets/glue/glue.py"
        ),
        relative_path="dataset/glue.py",
        size=28_751,
        sha256="367527feace8a5c5d1e4838c9395165726bc692fb2e0fc108e70e2c5f25f3be5",
    )
    assert DATASET_PIN.dataset_infos == PinnedFile(
        url=(
            "https://raw.githubusercontent.com/huggingface/datasets/"
            "dff6c92c80fbfac456349a531c0a4521c2b0c8f8/datasets/glue/"
            "dataset_infos.json"
        ),
        relative_path="dataset/dataset_infos.json",
        size=28_682,
        sha256="8619953890393d781426d5b6e3d31adf140ecf05d8dfd2244b5156cd91a6fc35",
    )
    assert DATASET_PIN.raw_archive == PinnedFile(
        url="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        relative_path="dataset/raw/SST-2.zip",
        size=7_439_277,
        sha256="d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa",
    )


@pytest.mark.parametrize("revision", ["main", "latest", "", "A" * 40, "a" * 39])
def test_mutable_or_malformed_revisions_are_rejected(revision: str) -> None:
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        require_full_git_sha(revision)


def test_model_transport_uses_snapshot_download_and_reports_file_hashes(
    tmp_path: Path,
) -> None:
    pin = HubPin("example/model", "c" * 40)
    snapshot = _fake_model_snapshot(tmp_path, pin.revision)
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot)

    report = fetch_model_snapshot(
        tmp_path / "hub",
        pin=pin,
        local_files_only=True,
        snapshot_download_fn=fake_snapshot_download,
    )

    assert calls == [
        {
            "repo_id": "example/model",
            "repo_type": "model",
            "revision": "c" * 40,
            "cache_dir": str(tmp_path / "hub"),
            "local_files_only": True,
            "allow_patterns": list(report.required_files),
        }
    ]
    assert report.requested_revision == pin.revision
    assert report.resolved_revision == pin.revision
    assert report.snapshot_path == snapshot.resolve()
    assert {record.relative_path for record in report.files} == set(
        report.required_files
    )
    assert all(len(record.sha256) == 64 for record in report.files)


def test_model_transport_rejects_a_different_resolved_revision(tmp_path: Path) -> None:
    pin = HubPin("example/model", "c" * 40)
    mismatched = _fake_model_snapshot(tmp_path, "d" * 40)

    with pytest.raises(ValueError, match="Model revision mismatch"):
        fetch_model_snapshot(
            tmp_path / "hub",
            pin=pin,
            snapshot_download_fn=lambda **_: str(mismatched),
        )


def test_dataset_materialization_is_hash_checked_and_reusable_offline(
    tmp_path: Path,
) -> None:
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    pin = _test_dataset_pin(payloads)
    calls: list[str] = []

    def fetch_bytes(url: str) -> bytes:
        calls.append(url)
        return payloads[url]

    first = fetch_dataset_inputs(tmp_path / "inputs", pin=pin, fetch_bytes=fetch_bytes)
    second = fetch_dataset_inputs(
        tmp_path / "inputs",
        pin=pin,
        fetch_bytes=lambda url: pytest.fail(f"unexpected network access: {url}"),
    )

    assert calls == list(payloads)
    assert first == second
    assert first.loader_path.read_bytes() == b"loader"
    assert first.raw_archive_path.read_bytes() == b"archive"


def test_dataset_materialization_rejects_wrong_content(tmp_path: Path) -> None:
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    pin = _test_dataset_pin(payloads)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        fetch_dataset_inputs(
            tmp_path / "inputs",
            pin=pin,
            fetch_bytes=lambda _: b"xxxxxx",
        )


def test_dataset_local_only_mode_never_fetches_a_missing_input(tmp_path: Path) -> None:
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    pin = _test_dataset_pin(payloads)

    with pytest.raises(FileNotFoundError, match="local-only mode"):
        fetch_dataset_inputs(
            tmp_path / "inputs",
            pin=pin,
            local_files_only=True,
            fetch_bytes=lambda url: pytest.fail(f"unexpected network access: {url}"),
        )


def test_source_lock_round_trip_detects_tampering(tmp_path: Path) -> None:
    model_pin = HubPin("example/model", "c" * 40)
    snapshot = _fake_model_snapshot(tmp_path, model_pin.revision)
    model = fetch_model_snapshot(
        tmp_path / "hub",
        pin=model_pin,
        snapshot_download_fn=lambda **_: str(snapshot),
    )
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    dataset_pin = _test_dataset_pin(payloads)
    dataset = fetch_dataset_inputs(
        tmp_path / "inputs",
        pin=dataset_pin,
        fetch_bytes=payloads.__getitem__,
    )
    manifest = build_source_lock(
        model,
        dataset,
        cache_root=tmp_path,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        generated_at_utc="2026-09-06T12:00:00Z",
        python_version="3.10.12",
        huggingface_hub_version="0.36.2",
        transformers_version="4.20.1",
        datasets_version="1.17.0",
    )
    manifest_path = tmp_path / "source-lock.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    verify_source_lock(
        manifest_path,
        model,
        dataset,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        python_version="3.10.12",
        huggingface_hub_version="0.36.2",
        transformers_version="4.20.1",
        datasets_version="1.17.0",
    )
    assert manifest["model"]["requested_revision"] == "c" * 40
    assert manifest["model"]["resolved_revision"] == "c" * 40
    assert manifest["tokenizer"]["requested_revision"] == "c" * 40
    assert manifest["tokenizer"]["resolved_revision"] == "c" * 40
    assert manifest["dataset"]["requested_revision"] == "a" * 40
    assert manifest["dataset"]["resolved_revision"] == "a" * 40
    assert manifest["tools"] == {
        "python": "3.10.12",
        "huggingface_hub": "0.36.2",
        "transformers": "4.20.1",
        "datasets": "1.17.0",
    }

    (snapshot / "config.json").write_bytes(b"tampered")
    changed_model = fetch_model_snapshot(
        tmp_path / "hub",
        pin=model_pin,
        snapshot_download_fn=lambda **_: str(snapshot),
    )
    with pytest.raises(ValueError, match="does not match source lock"):
        verify_source_lock(
            manifest_path,
            changed_model,
            dataset,
            model_pin=model_pin,
            dataset_pin=dataset_pin,
            python_version="3.10.12",
            huggingface_hub_version="0.36.2",
            transformers_version="4.20.1",
            datasets_version="1.17.0",
        )


def test_dataset_revision_mismatch_is_rejected_before_lock_creation(
    tmp_path: Path,
) -> None:
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    pin = _test_dataset_pin(payloads)

    with pytest.raises(ValueError, match="Dataset revision mismatch"):
        fetch_dataset_inputs(
            tmp_path / "inputs",
            pin=replace(pin, resolved_revision="d" * 40),
            fetch_bytes=payloads.__getitem__,
        )


def test_fetch_script_writes_then_verifies_a_lock_without_network(tmp_path: Path) -> None:
    model_pin = HubPin("example/model", "c" * 40)
    snapshot = _fake_model_snapshot(tmp_path, model_pin.revision)
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    dataset_pin = _test_dataset_pin(payloads)
    cache_root = tmp_path
    manifest_path = tmp_path / "source-lock.json"

    created = run(
        cache_root=cache_root,
        manifest_path=manifest_path,
        verify=False,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        snapshot_download_fn=lambda **_: str(snapshot),
        fetch_bytes=payloads.__getitem__,
        generated_at_utc="2026-09-06T12:00:00Z",
        python_version="3.10.12",
        huggingface_hub_version="0.36.2",
        transformers_version="4.20.1",
        datasets_version="1.17.0",
    )
    verified = run(
        cache_root=cache_root,
        manifest_path=manifest_path,
        verify=True,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        snapshot_download_fn=lambda **_: str(snapshot),
        fetch_bytes=lambda url: pytest.fail(f"unexpected network access: {url}"),
        python_version="3.10.12",
        huggingface_hub_version="0.36.2",
        transformers_version="4.20.1",
        datasets_version="1.17.0",
    )

    assert created["status"] == "locked"
    assert verified["status"] == "verified"
    assert created["model_path"] == str(snapshot.resolve())
    assert verified["dataset_loader_path"].endswith("inputs/dataset/glue.py")


def test_fetch_script_can_download_missing_bytes_then_verify_existing_lock(
    tmp_path: Path,
) -> None:
    model_pin = HubPin("example/model", "c" * 40)
    snapshot = _fake_model_snapshot(tmp_path, model_pin.revision)
    payloads = {
        "https://example.test/glue.py": b"loader",
        "https://example.test/dataset_infos.json": b"metadata",
        "https://example.test/SST-2.zip": b"archive",
    }
    dataset_pin = _test_dataset_pin(payloads)
    manifest_path = tmp_path / "source-lock.json"
    run(
        cache_root=tmp_path,
        manifest_path=manifest_path,
        verify=False,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        snapshot_download_fn=lambda **_: str(snapshot),
        fetch_bytes=payloads.__getitem__,
        generated_at_utc="2026-09-06T12:00:00Z",
        python_version="3.10.12",
        huggingface_hub_version="0.36.2",
        transformers_version="4.20.1",
        datasets_version="1.17.0",
    )
    for path in (tmp_path / "inputs").rglob("*"):
        if path.is_file():
            path.unlink()

    calls: list[str] = []
    synchronized = run(
        cache_root=tmp_path,
        manifest_path=manifest_path,
        verify=True,
        allow_download=True,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        snapshot_download_fn=lambda **_: str(snapshot),
        fetch_bytes=lambda url: calls.append(url) or payloads[url],
        python_version="3.10.12",
        huggingface_hub_version="0.36.2",
        transformers_version="4.20.1",
        datasets_version="1.17.0",
    )

    assert synchronized["status"] == "verified"
    assert calls == list(payloads)
