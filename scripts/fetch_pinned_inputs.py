#!/usr/bin/env python3
"""Fetch or verify the immutable model, tokenizer, and SST-2 source bundle."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
from typing import Callable

from ds340w_repro.source_pins import (
    DATASET_PIN,
    MODEL_PIN,
    DatasetSourcePin,
    HubPin,
    build_source_lock,
    fetch_dataset_inputs,
    fetch_model_snapshot,
    verify_source_lock,
)


def run(
    *,
    cache_root: Path,
    manifest_path: Path,
    verify: bool,
    allow_download: bool = False,
    model_pin: HubPin = MODEL_PIN,
    dataset_pin: DatasetSourcePin = DATASET_PIN,
    snapshot_download_fn: Callable[..., str] | None = None,
    fetch_bytes: Callable[[str], bytes] | None = None,
    generated_at_utc: str | None = None,
    python_version: str | None = None,
    huggingface_hub_version: str | None = None,
    transformers_version: str | None = None,
    datasets_version: str | None = None,
) -> dict[str, str]:
    """Materialize inputs or verify that local bytes still equal the lock."""

    cache_root = cache_root.resolve()
    local_files_only = verify and not allow_download
    model = fetch_model_snapshot(
        cache_root / "hub",
        pin=model_pin,
        local_files_only=local_files_only,
        snapshot_download_fn=snapshot_download_fn,
    )
    dataset = fetch_dataset_inputs(
        cache_root / "inputs",
        pin=dataset_pin,
        local_files_only=local_files_only,
        fetch_bytes=fetch_bytes,
    )
    python_version = python_version or platform.python_version()
    if huggingface_hub_version is None:
        import huggingface_hub

        huggingface_hub_version = huggingface_hub.__version__
    if transformers_version is None:
        import transformers

        transformers_version = transformers.__version__
    if datasets_version is None:
        import datasets

        datasets_version = datasets.__version__

    if verify:
        verify_source_lock(
            manifest_path,
            model,
            dataset,
            model_pin=model_pin,
            dataset_pin=dataset_pin,
            python_version=python_version,
            huggingface_hub_version=huggingface_hub_version,
            transformers_version=transformers_version,
            datasets_version=datasets_version,
        )
        status = "verified"
    else:
        generated_at_utc = generated_at_utc or _utc_now()
        manifest = build_source_lock(
            model,
            dataset,
            cache_root=cache_root,
            model_pin=model_pin,
            dataset_pin=dataset_pin,
            generated_at_utc=generated_at_utc,
            python_version=python_version,
            huggingface_hub_version=huggingface_hub_version,
            transformers_version=transformers_version,
            datasets_version=datasets_version,
        )
        _write_json_atomically(manifest_path, manifest)
        status = "locked"

    return {
        "status": status,
        "manifest_path": str(manifest_path.resolve()),
        "model_path": str(model.snapshot_path),
        "dataset_loader_path": str(dataset.loader_path),
        "dataset_infos_path": str(dataset.dataset_infos_path),
        "dataset_archive_path": str(dataset.raw_archive_path),
    }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _write_json_atomically(path: Path, value: object) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _main() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    default_cache = Path(
        os.environ.get("DS340W_CACHE_ROOT", Path.home() / ".cache" / "ds340w")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=default_cache)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repository_root / "data" / "manifests" / "source-lock.json",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--verify",
        action="store_true",
        help="use local-only mode and compare every input with the existing lock",
    )
    mode.add_argument(
        "--sync",
        action="store_true",
        help="download missing inputs, then compare them with the existing lock",
    )
    arguments = parser.parse_args()
    result = run(
        cache_root=arguments.cache_root,
        manifest_path=arguments.manifest,
        verify=arguments.verify or arguments.sync,
        allow_download=arguments.sync,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
