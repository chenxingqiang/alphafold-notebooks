"""AlphaFold 3 fine-tuning entry point with compliance guards."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from . import param_groups, record_io, schema
from .lora import AF3LoRA, LoRAConfig
from .param_groups import ParamGroup, select_lora_targets


class WeightsComplianceError(RuntimeError):
    """Raised when an action would violate AF3 weight terms of use."""


@dataclass
class AF3FineTuneConfig:
    strategy: Literal["lora", "head_only", "full"] = "lora"
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    trainable_groups: tuple[ParamGroup, ...] = (
        ParamGroup.DIFFUSION,
        ParamGroup.CONFIDENCE,
    )
    lora_groups: tuple[ParamGroup, ...] = (
        ParamGroup.PAIRFORMER,
        ParamGroup.DIFFUSION,
    )
    lora_target_patterns: tuple[str, ...] = param_groups.DEFAULT_LORA_TARGET_PATTERNS


@dataclass(frozen=True)
class ParameterSummary:
    total: int
    trainable: int
    frozen: int
    trainable_ratio: float
    by_group: dict[str, int]

    def describe(self) -> str:
        pct = self.trainable_ratio * 100
        lines = [
            f"Total parameters: {self.total:,}",
            f"Trainable parameters: {self.trainable:,} ({pct:.4f}%)",
            f"Frozen parameters: {self.frozen:,}",
        ]
        if self.by_group:
            lines.append("Trainable by group:")
            for group, count in sorted(self.by_group.items()):
                lines.append(f"  {group}: {count:,}")
        return "\n".join(lines)


class AlphaFold3FineTuner:
    """Prepare AlphaFold 3 checkpoints for parameter-efficient fine-tuning."""

    def __init__(
        self,
        params: dict[str, dict[str, np.ndarray]],
        config: AF3FineTuneConfig,
        *,
        schema_specs: tuple[schema.ParamSpec, ...] | None = None,
    ):
        self.config = config
        self.params = params
        self._schema = schema_specs
        self._flat = record_io.flatten(params)
        self.lora: AF3LoRA | None = None
        self._trainable: tuple[str, ...] = ()

        if config.strategy == "lora":
            targets = select_lora_targets(
                self._schema,
                groups=config.lora_groups,
                patterns=config.lora_target_patterns,
            )
            # Restrict to parameters present in this checkpoint (subset schemas in tests).
            targets = tuple(name for name in targets if name in self._flat)
            self.lora = AF3LoRA(params, targets, config.lora)
            self._trainable = self.lora.targets
        elif config.strategy == "head_only":
            allowed = set(config.trainable_groups)
            self._trainable = tuple(
                name
                for name in self._flat
                if not name.startswith("__meta__")
                and param_groups.classify(name) in allowed
            )
        elif config.strategy == "full":
            self._trainable = tuple(
                name for name in self._flat if not name.startswith("__meta__")
            )
        else:
            raise ValueError(f"Unknown strategy: {config.strategy}")

    @classmethod
    def from_pretrained(
        cls,
        model_dir: Path | str,
        config: AF3FineTuneConfig | None = None,
    ) -> "AlphaFold3FineTuner":
        params = record_io.read_params(model_dir)
        return cls(params, config or AF3FineTuneConfig())

    @classmethod
    def from_random(
        cls,
        config: AF3FineTuneConfig | None = None,
        *,
        seed: int = 0,
        schema_specs: tuple[schema.ParamSpec, ...] | None = None,
    ) -> "AlphaFold3FineTuner":
        specs = schema_specs or schema.load_schema()
        params = schema.generate_random_params(specs, seed=seed)
        return cls(params, config or AF3FineTuneConfig(), schema_specs=specs)

    def trainable_param_names(self) -> tuple[str, ...]:
        return self._trainable

    def frozen_param_names(self) -> tuple[str, ...]:
        trainable = set(self._trainable)
        return tuple(name for name in self._flat if name not in trainable)

    def parameter_summary(self) -> ParameterSummary:
        trainable = 0
        by_group: dict[str, int] = {}
        if self.config.strategy == "lora" and self.lora is not None:
            trainable = self.lora.num_lora_params()
            for name in self.lora.targets:
                group = param_groups.classify(name).value
                layout = self.lora.layout(name)
                effective_rank = self.lora._effective_ranks[name]
                stack = int(np.prod(layout.stack)) if layout.stack else 1
                count = stack * effective_rank * (layout.in_dim + layout.out_dim)
                by_group[group] = by_group.get(group, 0) + count
        else:
            for name in self._trainable:
                trainable += self._flat[name].size
                group = param_groups.classify(name).value
                by_group[group] = by_group.get(group, 0) + self._flat[name].size

        total = sum(arr.size for arr in self._flat.values())
        return ParameterSummary(
            total=total,
            trainable=trainable,
            frozen=total - trainable,
            trainable_ratio=trainable / total if total else 0.0,
            by_group=by_group,
        )

    def save_adapter(self, path: Path | str) -> None:
        if self.lora is None:
            raise ValueError("save_adapter requires strategy='lora'")
        self.lora.save(path)

    def load_adapter(self, path: Path | str) -> None:
        if self.lora is None:
            raise ValueError("load_adapter requires strategy='lora'")
        restored = AF3LoRA.load(path, self.params)
        self.lora = restored
        self._trainable = restored.targets

    def export_merged_weights(
        self,
        path: Path | str,
        *,
        acknowledge_weights_terms: bool = False,
    ) -> None:
        if not acknowledge_weights_terms:
            raise WeightsComplianceError(
                "Exporting merged AlphaFold 3 weights incorporates Model Parameters "
                "and is restricted by the AlphaFold 3 Model Parameters Terms of Use. "
                "Pass acknowledge_weights_terms=True only if you are permitted to do so."
            )
        if self.lora is not None:
            params = self.lora.apply()
        else:
            params = self.params
        wire_dtypes = None
        if self._schema is not None:
            wire_dtypes = {spec.full_name: spec.dtype for spec in self._schema}
        record_io.write_params(path, params, wire_dtypes=wire_dtypes)
