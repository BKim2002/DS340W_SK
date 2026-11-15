"""Check default equivalence, saved adapter semantics, and the expanded gate."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ds340w_repro.config_schema import validate_nonfinal_config
from ds340w_repro.preflight import ComputeGateError
from ds340w_repro.run_opendelta import prepare_execution_tree
from ds340w_repro.scaling_experiment import build_scaling_config, enforce_scaling_gate

ROOT = Path(__file__).resolve().parents[1]


def test_scaling_configs_preserve_original_recipe():
    for rank in (2, 4, 8, 16):
        config = build_scaling_config(ROOT, rank)
        validate_nonfinal_config(config)
        assert config.values["lora_alpha"] == 2 * rank
        assert config.values["seed"] == 42
        assert config.values["effective_batch_size"] == 16
        assert config.values["max_steps"] == 20000
        assert config.values["lora_scaling_mode"] == "sqrt_rank"
        assert config.runner_payload()["lora_scaling_mode"] == "sqrt_rank"
        assert "patch_series" not in config.runner_payload()


def test_full_scaling_run_cannot_use_the_old_eight_run_gate(tmp_path):
    config = build_scaling_config(ROOT, 16)
    with pytest.raises((FileNotFoundError, ComputeGateError)):
        enforce_scaling_gate(config, ROOT / "data/manifests/nonexistent-scaling-gate.json")
    invalid = build_scaling_config(ROOT, 16, probe="pilot").with_value("max_steps", 201, "test")
    with pytest.raises(ComputeGateError, match="bounded"):
        enforce_scaling_gate(invalid, ROOT / "data/manifests/scaling-compute-gate.json")


def test_optional_block_default_gradients_and_checkpoint_roundtrip(tmp_path):
    tree = prepare_execution_tree(ROOT, tmp_path / "runtime", "patches/scaling-series.json")
    program = r'''
import ast, copy, json, math, tempfile
from pathlib import Path
import torch
from torch import nn
from opendelta import AutoDeltaConfig, AutoDeltaModel
from opendelta.delta_models.lora import LowRankLinear
from transformers import T5Config, T5ForConditionalGeneration
from examples_seq2seq.scaling_diagnostics import adapter_gradient_rms
torch.set_num_threads(1)
original_source = Path(__ROOT__ + '/vendor/OpenDelta/opendelta/delta_models/lora.py').read_text()
node = next(n for n in ast.parse(original_source).body if isinstance(n, ast.ClassDef) and n.name == 'LowRankLinear')
scope = {'nn': nn, 'math': math}
exec(compile(ast.Module(body=[node], type_ignores=[]), '<pinned-original>', 'exec'), scope)
for r in (2,4,8,16):
    torch.manual_seed(42)
    old = scope['LowRankLinear'](6, 5, torch.zeros(5,6), r=r, lora_alpha=2*r)
    new = LowRankLinear(6,5,torch.zeros(5,6),r=r,lora_alpha=2*r)
    with torch.no_grad(): old.lora_B.normal_()
    new.load_state_dict(old.state_dict())
    x = torch.randn(3,6)
    assert torch.equal(old(x), new(x))
    old(x).square().mean().backward(); new(x).square().mean().backward()
    assert all(torch.equal(a.grad,b.grad) for a,b in zip(old.parameters(),new.parameters()))
    modified = LowRankLinear(6,5,torch.zeros(5,6),r=r,lora_alpha=2*r,lora_scaling_mode='sqrt_rank')
    modified.load_state_dict(old.state_dict())
    assert math.isclose(modified.scaling, 2*math.sqrt(r), rel_tol=1e-12)
    assert torch.allclose(modified(x), old(x)*math.sqrt(r), atol=1e-6)

config = T5Config(vocab_size=32,d_model=16,d_kv=8,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,dropout_rate=0.,decoder_start_token_id=0,eos_token_id=1,pad_token_id=0)
torch.manual_seed(4)
base = T5ForConditionalGeneration(config)
base_state = copy.deepcopy(base.state_dict())
inputs = torch.tensor([[3,4,1],[6,7,1]])
labels = torch.tensor([[8,1],[9,1]])
for mode in ('original','sqrt_rank'):
    model = T5ForConditionalGeneration(config); model.load_state_dict(base_state)
    dc = AutoDeltaConfig.from_dict({'delta_type':'lora','lora_r':2,'lora_alpha':4,'lora_dropout':0.,'lora_scaling_mode':mode,'modified_modules':['q','v'],'common_structure':False,'unfrozen_modules':['deltas']})
    delta = AutoDeltaModel.from_config(dc, backbone_model=model)
    delta.freeze_module(set_state_dict=True)
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1)
    # Two microbatches, each divided by accumulation factor two.
    for index in range(2):
        (model(input_ids=inputs[index:index+1],labels=labels[index:index+1]).loss/2).backward()
    observed = adapter_gradient_rms(list(model.named_parameters()))
    expected_B = torch.cat([p.grad.flatten() for n,p in model.named_parameters() if n.endswith('.lora_B')]).square().mean().sqrt().item()
    assert math.isclose(observed['B_rms'],expected_B,rel_tol=1e-6)
    assert observed['A_rms'] == 0.0 and observed['B_rms'] > 0.0
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    opt.step()
    model.eval()
    before = model.generate(inputs,max_length=5)
    with tempfile.TemporaryDirectory() as path:
        delta.save_finetuned(finetuned_delta_path=path,push_to_dc=False)
        assert json.loads((Path(path)/'config.json').read_text())['lora_scaling_mode'] == mode
        fresh = T5ForConditionalGeneration(config); fresh.load_state_dict(base_state)
        restored = AutoDeltaModel.from_finetuned(path,backbone_model=fresh)
        fresh.eval()
        assert restored.lora_scaling_mode == mode
        assert torch.equal(before,fresh.generate(inputs,max_length=5))
        assert all(math.isclose(m.scaling,4/(math.sqrt(2) if mode=='sqrt_rank' else 2)) for m in restored.delta_modules)
print('Default outputs/gradients exact; four multipliers correct; both modes save/reload with identical predictions; accumulated A/B RMS verified.')
'''.replace("__ROOT__", repr(str(ROOT)))
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(tree.root), str(tree.root / "examples/legacies"))))
    result = subprocess.run([sys.executable, "-c", program], env=environment,
                            text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
