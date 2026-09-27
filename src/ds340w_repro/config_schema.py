"""Normalized experiment configuration and invariant checks."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping


MODEL_REPOSITORY = "google/t5-base-lm-adapt"
MODEL_REVISION = "bf48a6b7de15c6afe1e27e1a51aa1ea4626f96bf"
DATASET_LOADER_REPOSITORY = "https://github.com/huggingface/datasets.git"
DATASET_LOADER_REVISION = "dff6c92c80fbfac456349a531c0a4521c2b0c8f8"
SOURCE_LOCK_SHA256 = (
    "4aae9bca3d409d7b075f6440bf4f7b3ce3023f8fbd24391ec1fa514148cb6b0d"
)
TEST_INDICES_SHA256 = (
    "83d6bdad006418cd841167d938f835561dc13b9ccd464ad21ebe7dabeb0b2394"
)
COURSE_DATA_MANIFEST_SHA256 = (
    "400143658894e4460f67d8e96dc2e69225150d1ed9ec15ec3da3cef0f37bff66"
)

# OpenDelta's common-structure vocabulary distinguishes decoder cross-attention
# from self-attention. Both must be named to reproduce the paper's 72 q/v
# projections. The trailing @ is also required by the pinned matcher's grammar.
LORA_MODIFIED_MODULES = (
    "attn@.q@",
    "attn@.v@",
    "crossattn@.q@",
    "crossattn@.v@",
)
UNFROZEN_MODULES = ("deltas", "layer_norm@")


@dataclass(frozen=True)
class ResolvedConfig:
    """A runner configuration with per-field provenance and a stable digest."""

    experiment_id: str
    values: dict[str, Any]
    authority_by_field: dict[str, str]
    schema_version: int = 1

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "values": self.values,
            "authority_by_field": self.authority_by_field,
        }

    def to_json(self) -> str:
        return json.dumps(
            self._payload(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ) + "\n"

    @property
    def sha256(self) -> str:
        return sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_json(cls, text: str) -> "ResolvedConfig":
        payload = json.loads(text)
        return cls(
            schema_version=payload["schema_version"],
            experiment_id=payload["experiment_id"],
            values=payload["values"],
            authority_by_field=payload["authority_by_field"],
        )

    def with_value(self, key: str, value: Any, authority: str) -> "ResolvedConfig":
        values = dict(self.values)
        values[key] = value
        authorities = dict(self.authority_by_field)
        authorities[key] = authority
        return ResolvedConfig(
            experiment_id=self.experiment_id,
            values=values,
            authority_by_field=authorities,
            schema_version=self.schema_version,
        )

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")

    def runner_payload(self) -> dict[str, Any]:
        """Return only arguments consumed by the authors' parser or OpenDelta."""

        wrapper_only = {
            "dataset_archive_sha256",
            "dataset_loader_repository",
            "dataset_loader_revision",
            "course_data_manifest_sha256",
            "effective_batch_size",
            "effective_model_dropout",
            "model_source_repository",
            "model_source_revision",
            "reuse_checkpoint",
            "source_lock_sha256",
            "test_indices_sha256",
        }
        return {key: value for key, value in self.values.items() if key not in wrapper_only}


def expected_trainable_counts(rank: int) -> dict[str, int]:
    """Return structural T5-base counts for the declared q/v LoRA scope."""

    if not isinstance(rank, int) or rank <= 0:
        raise ValueError("LoRA rank must be a positive integer")
    # 72 projections, each with A(rank x 768) and B(768 x rank).
    lora_parameters = 72 * 2 * 768 * rank
    layer_norm_parameters = 47_616
    return {
        "injected_projections": 72,
        "lora_parameters": lora_parameters,
        "layer_norm_parameters": layer_norm_parameters,
        "trainable_parameters": lora_parameters + layer_norm_parameters,
    }


def validate_nonfinal_config(config: ResolvedConfig) -> None:
    """Reject configurations that can touch final-unseen data or drift scientifically."""

    values = config.values
    if set(config.authority_by_field) != set(values):
        raise ValueError("Every resolved value must have exactly one authority")

    required_equalities: Mapping[str, Any] = {
        "model_source_repository": MODEL_REPOSITORY,
        "model_source_revision": MODEL_REVISION,
        "model_revision": MODEL_REVISION,
        "dataset_loader_repository": DATASET_LOADER_REPOSITORY,
        "dataset_loader_revision": DATASET_LOADER_REVISION,
        "course_data_manifest_sha256": COURSE_DATA_MANIFEST_SHA256,
        "source_lock_sha256": SOURCE_LOCK_SHA256,
        "test_indices_sha256": TEST_INDICES_SHA256,
        "task_name": "sst2",
        "eval_dataset_name": "sst2",
        "do_test": False,
        "split_validation_test": False,
        "max_source_length": 128,
        "max_target_length": 128,
        "val_max_target_length": 128,
        "test_max_target_length": 128,
        "pad_to_max_length": False,
        "ignore_pad_token_for_loss": True,
        "predict_with_generate": True,
        "num_beams": 1,
        "delta_type": "lora",
        "lora_dropout": 0.0,
        "modified_modules": list(LORA_MODIFIED_MODULES),
        "unfrozen_modules": list(UNFROZEN_MODULES),
        "common_structure": True,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "lr_scheduler_type": "linear",
        "warmup_steps": 0,
        "seed": 42,
        "data_seed": 42,
        "data_sample_seed": 42,
        "overwrite_cache": True,
        "fp16": False,
        "bf16": False,
        "tf32": False,
        "full_determinism": True,
        "effective_model_dropout": 0.0,
        "dataloader_num_workers": 0,
        "push_to_hub": False,
    }
    for field, expected in required_equalities.items():
        if values.get(field) != expected:
            raise ValueError(
                f"Unsafe or unresolved {field}: expected {expected!r}, "
                f"found {values.get(field)!r}"
            )

    if any("unseen" in key.lower() for key in values):
        raise ValueError("A non-final configuration may not contain unseen fields")
    if not 1 <= values["max_val_samples"] <= 436:
        raise ValueError("Evaluation must be a non-empty prefix of the 436-row Test split")
    if values["lora_alpha"] != 2 * values["lora_r"]:
        raise ValueError("LoRA alpha must remain exactly twice the rank")
    if (
        values["per_device_train_batch_size"]
        * values["gradient_accumulation_steps"]
        != values["effective_batch_size"]
    ):
        raise ValueError("Physical batch times accumulation must equal effective batch")
    if values["per_device_eval_batch_size"] < 1:
        raise ValueError("Evaluation batch size must be positive")
    if values["max_steps"] < 1:
        raise ValueError("At least one optimizer step is required")
