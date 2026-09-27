#!/usr/bin/env python3
"""Capture the reproducibility-critical software and GPU environment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter_ns
from typing import Any


CUBLAS_WORKSPACE_CONFIG = ":4096:8"
EXPECTED_OPENDELTA_COMMIT = "20e426f716ea53618657cd34bbfc4b347b98cd74"
EXPECTED_OPENDELTA_REMOTE = "https://github.com/thunlp/OpenDelta.git"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def portable_path(path: Path, *, home: Path | None = None) -> str:
    """Represent a user-home path without publishing the account name."""

    absolute = Path(os.path.abspath(path.expanduser()))
    absolute_home = Path(os.path.abspath((home or Path.home()).expanduser()))
    if absolute == absolute_home:
        return "$HOME"
    if absolute.is_relative_to(absolute_home):
        return f"$HOME/{absolute.relative_to(absolute_home).as_posix()}"
    return f"$EXTERNAL/{absolute.name}"


def _run(*command: str, cwd: Path | None = None) -> str:
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_distribution_name(name: str) -> str:
    return name.lower().replace("_", "-").replace(".", "-")


def _installed_packages() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            continue
        name = _normalized_distribution_name(raw_name)
        version = distribution.version
        previous = packages.setdefault(name, version)
        if previous != version:
            raise RuntimeError(
                f"multiple installed versions detected for {name}: "
                f"{previous} and {version}"
            )
    return dict(sorted(packages.items()))


def _locked_packages(path: Path) -> dict[str, str]:
    packages: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", "--")):
            continue
        name, separator, version = line.partition("==")
        if not separator or not name or not version:
            raise RuntimeError(f"unsupported requirement in {path.name}: {line}")
        packages[_normalized_distribution_name(name)] = version
    return dict(sorted(packages.items()))


def _os_release() -> dict[str, str]:
    values: dict[str, str] = {}
    path = Path("/etc/os-release")
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value.strip().strip('"')
    return values


def _git(repository: Path, *arguments: str) -> str:
    return _run("git", "-C", str(repository), *arguments)


def _git_report(project_root: Path) -> dict[str, Any]:
    upstream = project_root / "vendor" / "OpenDelta"
    upstream_commit = _git(upstream, "rev-parse", "HEAD")
    upstream_remote = _git(upstream, "remote", "get-url", "origin")
    if upstream_commit != EXPECTED_OPENDELTA_COMMIT:
        raise RuntimeError(
            "OpenDelta source mismatch: "
            f"expected {EXPECTED_OPENDELTA_COMMIT}, found {upstream_commit}"
        )
    if upstream_remote != EXPECTED_OPENDELTA_REMOTE:
        raise RuntimeError(
            "OpenDelta remote mismatch: "
            f"expected {EXPECTED_OPENDELTA_REMOTE}, found {upstream_remote}"
        )

    return {
        "client_version": _run("git", "--version"),
        "project": {
            "commit": _git(project_root, "rev-parse", "HEAD"),
            "remote_url": _git(project_root, "remote", "get-url", "origin"),
        },
        "opendelta": {
            "commit": upstream_commit,
            "remote_url": upstream_remote,
            "worktree_clean": not bool(_git(upstream, "status", "--porcelain")),
        },
    }


def _gpu_report(torch: Any) -> tuple[list[dict[str, Any]], str]:
    rows = _run(
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ).splitlines()
    devices: list[dict[str, Any]] = []
    drivers: set[str] = set()
    for row in rows:
        index_text, name, driver, memory_text = (
            value.strip() for value in row.split(",", maxsplit=3)
        )
        index = int(index_text)
        drivers.add(driver)
        capability = torch.cuda.get_device_capability(index)
        devices.append(
            {
                "compute_capability": f"{capability[0]}.{capability[1]}",
                "index": index,
                "memory_total_mib": int(memory_text),
                "name": name,
            }
        )
    if len(drivers) != 1:
        raise RuntimeError(f"expected one NVIDIA driver version, found {drivers}")
    return devices, drivers.pop()


def _determinism_report(torch: Any) -> dict[str, Any]:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    matmul_backend = torch.backends.cuda.matmul
    if hasattr(matmul_backend, "allow_fp16_reduced_precision_reduction"):
        matmul_backend.allow_fp16_reduced_precision_reduction = False
    if hasattr(matmul_backend, "allow_bf16_reduced_precision_reduction"):
        matmul_backend.allow_bf16_reduced_precision_reduction = False

    left = torch.tensor([[1.0, 2.0]], device="cuda", dtype=torch.float32)
    right = torch.tensor([[3.0], [4.0]], device="cuda", dtype=torch.float32)
    product = left @ right
    torch.cuda.synchronize()
    capability_check_passed = product.item() == 11.0
    del left, right, product
    torch.cuda.empty_cache()
    if not capability_check_passed:
        raise RuntimeError("deterministic CUDA matrix multiplication check failed")

    return {
        "algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "bf16_reduced_precision_reduction": getattr(
            matmul_backend, "allow_bf16_reduced_precision_reduction", None
        ),
        "capability_check_passed": capability_check_passed,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "fp16_reduced_precision_reduction": getattr(
            matmul_backend, "allow_fp16_reduced_precision_reduction", None
        ),
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "tf32_matmul": matmul_backend.allow_tf32,
    }


def probe_environment(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Return a JSON-serializable report for the current experiment host."""

    probe_started_ns = perf_counter_ns()
    project_root = project_root.resolve(strict=True)
    lockfile = project_root / "requirements.lock"
    if not lockfile.is_file():
        raise FileNotFoundError(f"dependency lockfile is missing: {lockfile}")

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable to PyTorch")

    installed_packages = _installed_packages()
    locked_packages = _locked_packages(lockfile)
    locked_environment = {
        name: version
        for name, version in installed_packages.items()
        if name != "ds340w-repro"
    }
    if locked_environment != locked_packages:
        missing = sorted(locked_environment.keys() - locked_packages.keys())
        extra = sorted(locked_packages.keys() - locked_environment.keys())
        mismatched = {
            name: {
                "installed": locked_environment[name],
                "locked": locked_packages[name],
            }
            for name in locked_environment.keys() & locked_packages.keys()
            if locked_environment[name] != locked_packages[name]
        }
        raise RuntimeError(
            "installed packages do not exactly match requirements.lock: "
            f"missing={missing}, extra={extra}, mismatched={mismatched}"
        )

    determinism = _determinism_report(torch)
    devices, driver_version = _gpu_report(torch)
    release = _os_release()
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "operating_system": {
            "id": release.get("ID"),
            "is_wsl": "microsoft" in platform.release().lower(),
            "kernel": platform.release(),
            "pretty_name": release.get("PRETTY_NAME"),
        },
        "python": {
            "executable": portable_path(Path(sys.executable)),
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
        },
        "packages": installed_packages,
        "pytorch": {
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "version": torch.__version__,
        },
        "nvidia": {
            "devices": devices,
            "driver_version": driver_version,
        },
        "determinism": determinism,
        "precision_policy": {
            "bf16": False,
            "fp16": False,
            "tf32": False,
            "torch_dtype": "float32",
        },
        "git": _git_report(project_root),
        "dependency_lock": {
            "matches_installed": True,
            "package_count": len(locked_packages),
            "path": "requirements.lock",
            "sha256": _sha256_file(lockfile),
        },
    }
    report["probe_wall_seconds"] = max(
        perf_counter_ns() - probe_started_ns, 1
    ) / 1_000_000_000
    return report


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe and optionally write the locked GPU environment."
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        metavar="PATH",
        help="write the JSON report to PATH in addition to stdout",
    )
    arguments = parser.parse_args()
    report = probe_environment()
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if arguments.write_manifest is not None:
        destination = arguments.write_manifest.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized, encoding="utf-8", newline="\n")
    print(serialized, end="")


if __name__ == "__main__":
    _main()
