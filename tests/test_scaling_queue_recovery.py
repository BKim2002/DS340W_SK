"""Guard against retraining or silently accepting mismatched completed runs."""

import json
from pathlib import Path

import pytest

from scripts import run_scaling_sweep as sweep
from ds340w_repro.scaling_experiment import build_scaling_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("case", ("absent", "complete", "wrong_hash", "failed", "partial"))
def test_resume_accepts_only_completed_matching_training(tmp_path, monkeypatch, case):
    config = build_scaling_config(ROOT, 2)
    config.write(tmp_path / f"configs/resolved/scaling/{config.experiment_id}.json")
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    if case == "absent":
        assert sweep.completed_training(config.experiment_id) is False
        return
    directory = tmp_path / f"artifacts/runs/{config.experiment_id}"
    directory.mkdir(parents=True)
    (directory / "process-result.json").write_text(json.dumps({
        "exit_code": 1 if case == "failed" else 0,
        "resolved_config_sha256": "f" * 64 if case == "wrong_hash" else config.sha256,
    }))
    (directory / "trainer_state.json").write_text(json.dumps({
        "global_step": 19999 if case == "partial" else 20000,
    }))
    if case == "complete":
        assert sweep.completed_training(config.experiment_id) is True
    else:
        with pytest.raises(ValueError, match="incomplete or its configuration differs"):
            sweep.completed_training(config.experiment_id)
