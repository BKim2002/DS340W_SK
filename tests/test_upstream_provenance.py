import json
from pathlib import Path
import subprocess
import sys

from ds340w_repro.provenance import verify_upstream


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PINNED_COMMIT = "20e426f716ea53618657cd34bbfc4b347b98cd74"
UPSTREAM_URL = "https://github.com/thunlp/OpenDelta.git"
RUNNER_PATH = "examples/legacies/examples_seq2seq/run_seq2seq.py"


def test_approved_opendelta_checkout_is_verified() -> None:
    report = verify_upstream(PROJECT_ROOT / "vendor" / "OpenDelta")

    assert report.commit == PINNED_COMMIT
    assert report.remote_url == UPSTREAM_URL
    assert report.runner_relative_path == RUNNER_PATH
    assert len(report.runner_sha256) == 64
    assert report.is_clean is True


def test_provenance_cli_emits_machine_readable_report() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ds340w_repro.provenance",
            str(PROJECT_ROOT / "vendor" / "OpenDelta"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report["commit"] == PINNED_COMMIT
    assert report["remote_url"] == UPSTREAM_URL
    assert report["runner_relative_path"] == RUNNER_PATH
    assert len(report["runner_sha256"]) == 64
    assert report["is_clean"] is True
