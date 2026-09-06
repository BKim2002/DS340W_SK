"""Verify that an experiment uses the approved authors' source checkout."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import subprocess


PINNED_OPENDELTA_COMMIT = "20e426f716ea53618657cd34bbfc4b347b98cd74"
OPENDELTA_REMOTE_URL = "https://github.com/thunlp/OpenDelta.git"
OPENDELTA_RUNNER = Path("examples/legacies/examples_seq2seq/run_seq2seq.py")


@dataclass(frozen=True)
class UpstreamReport:
    """Verified source identity exposed to experiment launchers."""

    commit: str
    remote_url: str
    runner_relative_path: str
    runner_sha256: str
    is_clean: bool


def verify_upstream(repository: Path) -> UpstreamReport:
    """Return source provenance or raise when the checkout is not approved."""

    repository = repository.resolve(strict=True)
    commit = _git(repository, "rev-parse", "HEAD")
    if commit != PINNED_OPENDELTA_COMMIT:
        raise ValueError(
            f"OpenDelta commit mismatch: expected {PINNED_OPENDELTA_COMMIT}, "
            f"found {commit}"
        )

    remote_url = _git(repository, "remote", "get-url", "origin")
    if remote_url != OPENDELTA_REMOTE_URL:
        raise ValueError(
            f"OpenDelta remote mismatch: expected {OPENDELTA_REMOTE_URL}, "
            f"found {remote_url}"
        )

    runner = repository / OPENDELTA_RUNNER
    if not runner.is_file():
        raise FileNotFoundError(f"OpenDelta runner is missing: {runner}")

    if _git(repository, "status", "--porcelain"):
        raise ValueError(f"OpenDelta checkout has local modifications: {repository}")

    return UpstreamReport(
        commit=commit,
        remote_url=remote_url,
        runner_relative_path=OPENDELTA_RUNNER.as_posix(),
        runner_sha256=_sha256_file(runner),
        is_clean=True,
    )


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
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


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the pinned OpenDelta source checkout."
    )
    parser.add_argument("repository", type=Path)
    arguments = parser.parse_args()
    print(json.dumps(asdict(verify_upstream(arguments.repository)), sort_keys=True))


if __name__ == "__main__":
    _main()
