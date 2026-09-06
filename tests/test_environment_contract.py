from importlib import metadata
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from scripts.probe_environment import portable_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_environment_paths_are_sanitized_for_public_manifests(tmp_path: Path) -> None:
    home = tmp_path / "student-home"
    executable = home / ".venvs/ds340w/bin/python"

    assert portable_path(executable, home=home) == "$HOME/.venvs/ds340w/bin/python"


def test_bootstrap_initializes_the_pinned_author_submodule() -> None:
    bootstrap = (PROJECT_ROOT / "scripts" / "bootstrap_wsl.sh").read_text(
        encoding="utf-8"
    )

    assert "submodule update --init --recursive -- vendor/OpenDelta" in bootstrap
    assert "20e426f716ea53618657cd34bbfc4b347b98cd74" in bootstrap
    assert 'fetch_pinned_inputs.py" --sync' in bootstrap
    assert "-m ds340w_repro.data.materialize_sst2" in bootstrap


def test_build_backend_is_immutably_pinned() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires = ["setuptools==80.9.0"]' in pyproject
    assert "setuptools>=" not in pyproject


def test_locked_python_and_cuda_runtime_are_available() -> None:
    assert platform.python_version_tuple()[:2] == ("3", "10")
    assert torch.__version__ == "2.1.2+cu121"
    assert torch.version.cuda == "12.1"
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == "NVIDIA GeForce RTX 4060"

    left = torch.tensor([[1.0, 2.0]], device="cuda")
    right = torch.tensor([[3.0], [4.0]], device="cuda")
    product = left @ right
    torch.cuda.synchronize()

    assert product.item() == 11.0


def test_legacy_runner_dependencies_match_the_compatibility_lock() -> None:
    expected_versions = {
        "bigmodelvis": "0.0.1",
        "datasets": "1.17.0",
        "fsspec": "2022.1.0",
        "huggingface-hub": "0.10.1",
        "numpy": "1.23.5",
        "protobuf": "3.20.3",
        "pyarrow": "7.0.0",
        "scikit-learn": "1.1.3",
        "scipy": "1.9.3",
        "sentencepiece": "0.1.99",
        "tensorboardX": "2.6.2.2",
        "transformers": "4.20.1",
    }

    installed_versions = {
        package: metadata.version(package) for package in expected_versions
    }
    assert installed_versions == expected_versions

    from datasets import load_metric
    from transformers.trainer_utils import is_main_process

    assert callable(load_metric)
    assert callable(is_main_process)


def test_author_packages_import_from_the_pinned_submodule() -> None:
    import examples_seq2seq
    import opendelta

    upstream = (PROJECT_ROOT / "vendor" / "OpenDelta").resolve()
    assert Path(opendelta.__file__).resolve().is_relative_to(upstream)
    assert Path(examples_seq2seq.__file__).resolve().is_relative_to(upstream)


def test_original_runner_module_imports_through_explicit_source_paths(
    tmp_path: Path,
) -> None:
    upstream = PROJECT_ROOT / "vendor" / "OpenDelta"
    source_paths = [upstream, upstream / "examples" / "legacies"]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(str(path) for path in source_paths)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from examples_seq2seq.run_seq2seq import main; "
                "assert callable(main)"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
