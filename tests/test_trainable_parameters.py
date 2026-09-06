from ds340w_repro.config_schema import (
    LORA_MODIFIED_MODULES,
    UNFROZEN_MODULES,
    expected_trainable_counts,
)


def test_paper_consistent_matchers_are_explicit() -> None:
    assert LORA_MODIFIED_MODULES == (
        "attn@.q@",
        "attn@.v@",
        "crossattn@.q@",
        "crossattn@.v@",
    )
    assert UNFROZEN_MODULES == ("deltas", "layer_norm@")


def test_trainable_parameter_formula_matches_t5_base() -> None:
    assert expected_trainable_counts(2) == {
        "injected_projections": 72,
        "lora_parameters": 221_184,
        "layer_norm_parameters": 47_616,
        "trainable_parameters": 268_800,
    }
    assert expected_trainable_counts(4)["lora_parameters"] == 442_368
    assert expected_trainable_counts(8) == {
        "injected_projections": 72,
        "lora_parameters": 884_736,
        "layer_norm_parameters": 47_616,
        "trainable_parameters": 932_352,
    }
    assert expected_trainable_counts(16)["lora_parameters"] == 1_769_472
