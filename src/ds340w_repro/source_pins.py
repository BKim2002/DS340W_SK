"""Immutable, hash-verified inputs for the OpenDelta SST-2 reproduction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from typing import Any, Callable
from urllib.request import Request, urlopen


OPENDELTA_REVISION = "20e426f716ea53618657cd34bbfc4b347b98cd74"
OPENDELTA_REPOSITORY_URL = "https://github.com/thunlp/OpenDelta.git"


@dataclass(frozen=True)
class HubPin:
    """An immutable Hugging Face repository reference."""

    repo_id: str
    revision: str
    repo_type: str = "model"


@dataclass(frozen=True)
class PinnedFile:
    """A remotely sourced file with an independently verified identity."""

    url: str
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class DatasetSourcePin:
    """Canonical Datasets loader source plus the raw SST-2 archive."""

    repository_url: str
    revision: str
    dataset_name: str
    configuration: str
    loader_script: PinnedFile
    dataset_infos: PinnedFile
    raw_archive: PinnedFile
    provenance_mirror: HubPin
    resolved_revision: str | None = None


@dataclass(frozen=True)
class FileRecord:
    relative_path: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ModelSnapshotReport:
    repo_id: str
    requested_revision: str
    resolved_revision: str
    snapshot_path: Path
    required_files: tuple[str, ...]
    files: tuple[FileRecord, ...]


@dataclass(frozen=True)
class DatasetInputReport:
    requested_revision: str
    resolved_revision: str
    root: Path
    files: tuple[FileRecord, ...]

    @property
    def loader_path(self) -> Path:
        return self.root / self.files[0].relative_path

    @property
    def dataset_infos_path(self) -> Path:
        return self.root / self.files[1].relative_path

    @property
    def raw_archive_path(self) -> Path:
        return self.root / self.files[2].relative_path


MODEL_PIN = HubPin(
    repo_id="google/t5-base-lm-adapt",
    revision="bf48a6b7de15c6afe1e27e1a51aa1ea4626f96bf",
)

# This immutable Hub revision documents the approved dataset mirror, but it is
# not the executable dataset source. The legacy runner loads the canonical
# GLUE script below, as it did before GLUE was converted to Parquet on the Hub.
DATASET_MIRROR_PIN = HubPin(
    repo_id="nyu-mll/glue",
    revision="bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c",
    repo_type="dataset",
)

_DATASETS_REVISION = "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
_DATASETS_RAW_ROOT = (
    "https://raw.githubusercontent.com/huggingface/datasets/"
    f"{_DATASETS_REVISION}/datasets/glue"
)
DATASET_PIN = DatasetSourcePin(
    repository_url="https://github.com/huggingface/datasets.git",
    revision=_DATASETS_REVISION,
    dataset_name="glue",
    configuration="sst2",
    loader_script=PinnedFile(
        url=f"{_DATASETS_RAW_ROOT}/glue.py",
        relative_path="dataset/glue.py",
        size=28_751,
        sha256="367527feace8a5c5d1e4838c9395165726bc692fb2e0fc108e70e2c5f25f3be5",
    ),
    dataset_infos=PinnedFile(
        url=f"{_DATASETS_RAW_ROOT}/dataset_infos.json",
        relative_path="dataset/dataset_infos.json",
        size=28_682,
        sha256="8619953890393d781426d5b6e3d31adf140ecf05d8dfd2244b5156cd91a6fc35",
    ),
    raw_archive=PinnedFile(
        url="https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
        relative_path="dataset/raw/SST-2.zip",
        size=7_439_277,
        sha256="d67e16fb55739c1b32cdce9877596db1c127dc322d93c082281f64057c16deaa",
    ),
    provenance_mirror=DATASET_MIRROR_PIN,
)

MODEL_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)
TOKENIZER_FILES = (
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
)


SnapshotDownload = Callable[..., str]
FetchBytes = Callable[[str], bytes]


def require_full_git_sha(revision: str) -> None:
    """Reject mutable names and non-canonical Git object identifiers."""

    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError(f"Revision must be a full lowercase Git SHA: {revision!r}")


def fetch_model_snapshot(
    cache_dir: Path,
    *,
    pin: HubPin = MODEL_PIN,
    local_files_only: bool = False,
    snapshot_download_fn: SnapshotDownload | None = None,
) -> ModelSnapshotReport:
    """Fetch and identify a Hub snapshot suitable for local ``from_pretrained``."""

    require_full_git_sha(pin.revision)
    if pin.repo_type != "model":
        raise ValueError(f"Expected a model Hub pin, found {pin.repo_type!r}")
    if snapshot_download_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_download_fn = snapshot_download

    snapshot = Path(
        snapshot_download_fn(
            repo_id=pin.repo_id,
            repo_type="model",
            revision=pin.revision,
            cache_dir=str(cache_dir),
            local_files_only=local_files_only,
            allow_patterns=list(MODEL_REQUIRED_FILES),
        )
    ).resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Model snapshot is missing: {snapshot}")

    resolved_revision = _snapshot_revision(snapshot)
    if resolved_revision != pin.revision:
        raise ValueError(
            f"Model revision mismatch: requested {pin.revision}, "
            f"resolved {resolved_revision}"
        )

    missing = [name for name in MODEL_REQUIRED_FILES if not (snapshot / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "Pinned model snapshot is incomplete: " + ", ".join(sorted(missing))
        )

    files = tuple(
        _file_record(snapshot / name, relative_path=name)
        for name in MODEL_REQUIRED_FILES
    )
    return ModelSnapshotReport(
        repo_id=pin.repo_id,
        requested_revision=pin.revision,
        resolved_revision=resolved_revision,
        snapshot_path=snapshot,
        required_files=MODEL_REQUIRED_FILES,
        files=files,
    )


def fetch_dataset_inputs(
    output_root: Path,
    *,
    pin: DatasetSourcePin = DATASET_PIN,
    local_files_only: bool = False,
    fetch_bytes: FetchBytes | None = None,
) -> DatasetInputReport:
    """Materialize the canonical loader and raw archive with strict hashes."""

    require_full_git_sha(pin.revision)
    require_full_git_sha(pin.provenance_mirror.revision)
    resolved_revision = pin.resolved_revision or pin.revision
    require_full_git_sha(resolved_revision)
    if resolved_revision != pin.revision:
        raise ValueError(
            f"Dataset revision mismatch: requested {pin.revision}, "
            f"resolved {resolved_revision}"
        )

    if fetch_bytes is None:
        fetch_bytes = _download_bytes
    output_root = output_root.resolve()
    records = tuple(
        _materialize_pinned_file(
            output_root,
            file_pin,
            fetch_bytes,
            local_files_only=local_files_only,
        )
        for file_pin in (pin.loader_script, pin.dataset_infos, pin.raw_archive)
    )
    return DatasetInputReport(
        requested_revision=pin.revision,
        resolved_revision=resolved_revision,
        root=output_root,
        files=records,
    )


def build_source_lock(
    model: ModelSnapshotReport,
    dataset: DatasetInputReport,
    *,
    cache_root: Path,
    model_pin: HubPin = MODEL_PIN,
    dataset_pin: DatasetSourcePin = DATASET_PIN,
    generated_at_utc: str,
    python_version: str,
    huggingface_hub_version: str,
    transformers_version: str,
    datasets_version: str,
) -> dict[str, Any]:
    """Build the portable lock record for already verified local inputs."""

    _validate_report_revisions(model, dataset, model_pin, dataset_pin)
    cache_root = cache_root.resolve()
    model_cache_path = _relative_to(model.snapshot_path, cache_root)
    dataset_cache_path = _relative_to(dataset.root, cache_root)
    model_files = [_record_dict(record) for record in model.files]
    tokenizer_files = [
        _record_dict(record)
        for record in model.files
        if record.relative_path in TOKENIZER_FILES
    ]
    dataset_records = {
        record.relative_path: record for record in dataset.files
    }

    return {
        "schema_version": 1,
        "generated_at_utc": generated_at_utc,
        "tools": {
            "python": python_version,
            "huggingface_hub": huggingface_hub_version,
            "transformers": transformers_version,
            "datasets": datasets_version,
        },
        "opendelta": {
            "repository_url": OPENDELTA_REPOSITORY_URL,
            "requested_revision": OPENDELTA_REVISION,
            "resolved_revision": OPENDELTA_REVISION,
        },
        "model": {
            "repo_id": model_pin.repo_id,
            "repository_url": f"https://huggingface.co/{model_pin.repo_id}",
            "repo_type": model_pin.repo_type,
            "requested_revision": model_pin.revision,
            "resolved_revision": model.resolved_revision,
            "transport": "huggingface_hub.snapshot_download",
            "cache_relative_path": model_cache_path,
            "offline_loader": "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            "allow_patterns": list(model.required_files),
            "files": model_files,
        },
        "tokenizer": {
            "repo_id": model_pin.repo_id,
            "repository_url": f"https://huggingface.co/{model_pin.repo_id}",
            "requested_revision": model_pin.revision,
            "resolved_revision": model.resolved_revision,
            "cache_relative_path": model_cache_path,
            "offline_loader": "transformers.AutoTokenizer.from_pretrained",
            "files": tokenizer_files,
        },
        "dataset": {
            "dataset_name": dataset_pin.dataset_name,
            "configuration": dataset_pin.configuration,
            "repository_url": dataset_pin.repository_url,
            "revision_kind": "peeled_git_commit",
            "requested_revision": dataset_pin.revision,
            "resolved_revision": dataset.resolved_revision,
            "cache_relative_path": dataset_cache_path,
            "offline_loader": "datasets.load_dataset",
            "loader_script": _dataset_file_dict(
                dataset_pin.loader_script,
                dataset_records[dataset_pin.loader_script.relative_path],
            ),
            "dataset_infos": _dataset_file_dict(
                dataset_pin.dataset_infos,
                dataset_records[dataset_pin.dataset_infos.relative_path],
            ),
            "raw_archive": _dataset_file_dict(
                dataset_pin.raw_archive,
                dataset_records[dataset_pin.raw_archive.relative_path],
            ),
            "provenance_mirror": {
                "repo_id": dataset_pin.provenance_mirror.repo_id,
                "repo_type": dataset_pin.provenance_mirror.repo_type,
                "revision": dataset_pin.provenance_mirror.revision,
                "executable_source": False,
            },
        },
    }


def verify_source_lock(
    manifest_path: Path,
    model: ModelSnapshotReport,
    dataset: DatasetInputReport,
    *,
    model_pin: HubPin = MODEL_PIN,
    dataset_pin: DatasetSourcePin = DATASET_PIN,
    python_version: str,
    huggingface_hub_version: str,
    transformers_version: str,
    datasets_version: str,
) -> dict[str, Any]:
    """Verify current local bytes, tool versions, and revisions against a lock."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cache_root = Path(
        os.path.commonpath((str(model.snapshot_path), str(dataset.root)))
    )
    expected = build_source_lock(
        model,
        dataset,
        cache_root=cache_root,
        model_pin=model_pin,
        dataset_pin=dataset_pin,
        generated_at_utc=manifest.get("generated_at_utc", ""),
        python_version=python_version,
        huggingface_hub_version=huggingface_hub_version,
        transformers_version=transformers_version,
        datasets_version=datasets_version,
    )
    if manifest != expected:
        raise ValueError("Current input state does not match source lock")
    return manifest


def _validate_report_revisions(
    model: ModelSnapshotReport,
    dataset: DatasetInputReport,
    model_pin: HubPin,
    dataset_pin: DatasetSourcePin,
) -> None:
    for revision in (
        OPENDELTA_REVISION,
        model_pin.revision,
        dataset_pin.revision,
        dataset_pin.provenance_mirror.revision,
    ):
        require_full_git_sha(revision)
    if model.requested_revision != model_pin.revision:
        raise ValueError("Model report requested revision does not match its pin")
    if model.resolved_revision != model_pin.revision:
        raise ValueError("Model report resolved revision does not match its pin")
    if dataset.requested_revision != dataset_pin.revision:
        raise ValueError("Dataset report requested revision does not match its pin")
    if dataset.resolved_revision != dataset_pin.revision:
        raise ValueError("Dataset report resolved revision does not match its pin")


def _snapshot_revision(snapshot: Path) -> str:
    if snapshot.parent.name != "snapshots":
        raise ValueError(
            "snapshot_download returned a path outside the Hub snapshots layout: "
            f"{snapshot}"
        )
    require_full_git_sha(snapshot.name)
    return snapshot.name


def _materialize_pinned_file(
    root: Path,
    pin: PinnedFile,
    fetch_bytes: FetchBytes,
    *,
    local_files_only: bool,
) -> FileRecord:
    _require_sha256(pin.sha256)
    if pin.size < 0:
        raise ValueError(f"Pinned size must be non-negative: {pin.size}")
    relative = Path(pin.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Pinned path must remain below its root: {pin.relative_path}")
    destination = (root / relative).resolve()
    if destination.is_file():
        return _verify_file(destination, pin)
    if local_files_only:
        raise FileNotFoundError(
            f"Pinned dataset input is missing in local-only mode: {destination}"
        )

    payload = fetch_bytes(pin.url)
    _verify_payload(payload, pin)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.partial")
    temporary.write_bytes(payload)
    temporary.replace(destination)
    return _verify_file(destination, pin)


def _download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "ds340w-reproduction/0.1"})
    with urlopen(request) as response:
        return response.read()


def _verify_payload(payload: bytes, pin: PinnedFile) -> None:
    if len(payload) != pin.size:
        raise ValueError(
            f"Size mismatch for {pin.url}: expected {pin.size}, found {len(payload)}"
        )
    digest = sha256(payload).hexdigest()
    if digest != pin.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {pin.url}: expected {pin.sha256}, found {digest}"
        )


def _verify_file(path: Path, pin: PinnedFile) -> FileRecord:
    record = _file_record(path, relative_path=pin.relative_path)
    if record.size != pin.size:
        raise ValueError(
            f"Size mismatch for {path}: expected {pin.size}, found {record.size}"
        )
    if record.sha256 != pin.sha256:
        raise ValueError(
            f"SHA-256 mismatch for {path}: expected {pin.sha256}, "
            f"found {record.sha256}"
        )
    return record


def _file_record(path: Path, *, relative_path: str) -> FileRecord:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return FileRecord(
        relative_path=relative_path,
        size=path.stat().st_size,
        sha256=digest.hexdigest(),
    )


def _require_sha256(digest: str) -> None:
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError(f"Expected a full lowercase SHA-256: {digest!r}")


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Input path {path} is outside cache root {root}") from error


def _record_dict(record: FileRecord) -> dict[str, Any]:
    return asdict(record)


def _dataset_file_dict(pin: PinnedFile, record: FileRecord) -> dict[str, Any]:
    return {
        "url": pin.url,
        **_record_dict(record),
    }
