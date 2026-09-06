"""Serializable fresh-process runtime measurements for retained experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Mapping


@dataclass(frozen=True)
class FreshProcessMeasurement:
    """Resource and process evidence for one explicitly fresh execution."""

    phase: str
    fresh_process: bool
    exit_status: int
    wall_time_seconds: float
    cuda_peak_allocated_bytes: int
    cuda_peak_reserved_bytes: int
    config_sha256: str
    software_versions: Mapping[str, str]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.phase:
            raise ValueError("Measurement phase must be non-empty")
        if self.fresh_process is not True:
            raise ValueError("Retained measurements must come from a fresh process")
        if type(self.exit_status) is not int:
            raise ValueError("Process exit status must be an integer")
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0:
            raise ValueError("Wall time must be a finite non-negative number")
        cuda_values = (
            self.cuda_peak_allocated_bytes,
            self.cuda_peak_reserved_bytes,
        )
        if any(type(value) is not int or value < 0 for value in cuda_values):
            raise ValueError("CUDA peak byte counts must be non-negative integers")
        if self.cuda_peak_reserved_bytes < self.cuda_peak_allocated_bytes:
            raise ValueError("CUDA peak reserved bytes cannot be below allocated bytes")
        if not re.fullmatch(r"[0-9a-f]{64}", self.config_sha256):
            raise ValueError("Configuration checksum must be a lowercase SHA-256")
        if not self.software_versions or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in self.software_versions.items()
        ):
            raise ValueError("Software versions must contain non-empty strings")
        object.__setattr__(self, "software_versions", dict(self.software_versions))

    def to_dict(self) -> dict[str, object]:
        return {
            "config_sha256": self.config_sha256,
            "cuda_peak_allocated_bytes": self.cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": self.cuda_peak_reserved_bytes,
            "exit_status": self.exit_status,
            "fresh_process": self.fresh_process,
            "phase": self.phase,
            "schema_version": self.schema_version,
            "software_versions": dict(sorted(self.software_versions.items())),
            "wall_time_seconds": self.wall_time_seconds,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True
        ) + "\n"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8", newline="\n")
