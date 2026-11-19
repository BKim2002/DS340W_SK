import json
from pathlib import Path

import pytest

from scripts import run_confirmation as confirmation
from ds340w_repro.config_schema import ResolvedConfig

ROOT = Path(__file__).resolve().parents[1]


def test_confirmation_plan_is_original_mode_and_exactly_four_reserved_settings():
    confirmation.validate_plan()
    settings = []
    for name in confirmation.EXPERIMENTS:
        config = ResolvedConfig.from_json(confirmation.config_path(name).read_text())
        v = config.values
        assert v.get('lora_scaling_mode', 'original') == 'original'
        assert v.get('patch_series', 'patches/series.json') == 'patches/series.json'
        assert v['do_test'] is False and v['max_val_samples'] == 436
        assert v['max_steps'] == 20000 and v['eval_steps'] == v['save_steps'] == 100
        assert v['lora_alpha'] == 2*v['lora_r']
        assert v['data_sample_seed'] == 42
        settings.append((v['lora_r'], v['seed']))
    assert settings == [(8,17),(8,23),(16,17),(16,23)]


def test_budget_replaces_reserved_runs_with_actual_cost_instead_of_double_counting():
    base = {'projection': {'projected_seconds': 100000, 'original_seconds_per_future_run': 10000,
                            'reserve_multiplier': 1.2}}
    assert confirmation.adjusted_projection(base, []) == 100120
    assert confirmation.adjusted_projection(base, [9000], 20) == 98910
    # A slower completion also increases the estimate for the three future runs.
    assert confirmation.adjusted_projection(base, [12000], 20) == 109710
    assert confirmation.adjusted_projection(base, [9000]*4, 80) == 95280


@pytest.mark.parametrize('costs', ([0], [-1], [float('nan')], [1000]*5))
def test_budget_rejects_invalid_confirmation_costs(costs):
    with pytest.raises(ValueError, match='Invalid'):
        confirmation.adjusted_projection({'projection': {}}, costs)


@pytest.mark.parametrize('case', ['fresh', 'partial', 'complete', 'wrong_hash', 'failed'])
def test_queue_never_overwrites_existing_training(tmp_path, monkeypatch, case):
    name = confirmation.EXPERIMENTS[0]
    config = ResolvedConfig.from_json(confirmation.config_path(name).read_text())
    config.write(tmp_path / f'configs/resolved/confirmation/{name}.json')
    monkeypatch.setattr(confirmation, 'ROOT', tmp_path)
    directory = tmp_path / f'artifacts/runs/{name}'
    if case != 'fresh':
        directory.mkdir(parents=True)
    if case in ('complete', 'wrong_hash', 'failed'):
        (directory/'process-result.json').write_text(json.dumps({
            'exit_code': 1 if case == 'failed' else 0,
            'resolved_config_sha256': 'f'*64 if case == 'wrong_hash' else config.sha256}))
        (directory/'trainer_state.json').write_text(json.dumps({'global_step': 20000}))
    if case == 'fresh':
        assert confirmation.completed_training(name) is False
    elif case == 'complete':
        assert confirmation.completed_training(name) is True
    else:
        with pytest.raises(ValueError, match='overwrite|mismatched'):
            confirmation.completed_training(name)
