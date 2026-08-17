"""Parameter-space LoRA for AlphaFold 3 Haiku checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import param_groups, record_io


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 16.0
    seed: int = 0


def count_lora_params(
    shapes: dict[str, tuple[int, ...]],
    config: LoRAConfig,
) -> int:
    total = 0
    for full_name, shape in shapes.items():
        layout = param_groups.matrix_layout(full_name, shape)
        effective_rank = min(config.rank, layout.in_dim, layout.out_dim)
        stack_count = int(np.prod(layout.stack)) if layout.stack else 1
        total += stack_count * effective_rank * (layout.in_dim + layout.out_dim)
    return total


class AF3LoRA:
    """Low-rank adapters stored alongside a frozen Haiku parameter tree."""

    def __init__(
        self,
        base_params: dict[str, dict[str, np.ndarray]],
        target_names: tuple[str, ...],
        config: LoRAConfig = LoRAConfig(),
    ):
        self.base_params = base_params
        self.config = config
        self._flat = record_io.flatten(base_params)
        self._targets = tuple(target_names)
        self._layouts: dict[str, param_groups.MatrixLayout] = {}
        self._lora_a: dict[str, np.ndarray] = {}
        self._lora_b: dict[str, np.ndarray] = {}
        self._effective_ranks: dict[str, int] = {}

        rng = np.random.default_rng(config.seed)
        for name in self._targets:
            if name not in self._flat:
                raise KeyError(f"Target not found in base parameters: {name}")
            layout = param_groups.matrix_layout(name, self._flat[name].shape)
            effective_rank = min(config.rank, layout.in_dim, layout.out_dim)
            if effective_rank <= 0:
                raise ValueError(
                    f"Cannot apply LoRA to {name} (in={layout.in_dim}, out={layout.out_dim})"
                )
            self._layouts[name] = layout
            stack_shape = layout.stack if layout.stack else ()
            a_shape = (*stack_shape, layout.in_dim, effective_rank)
            b_shape = (*stack_shape, effective_rank, layout.out_dim)
            self._lora_a[name] = rng.standard_normal(a_shape).astype(np.float32) / np.sqrt(
                layout.in_dim
            )
            self._lora_b[name] = np.zeros(b_shape, dtype=np.float32)
            self._effective_ranks[name] = effective_rank

    @property
    def targets(self) -> tuple[str, ...]:
        return self._targets

    def layout(self, full_name: str) -> param_groups.MatrixLayout:
        return self._layouts[full_name]

    def num_lora_params(self) -> int:
        total = 0
        for name in self._targets:
            total += self._lora_a[name].size + self._lora_b[name].size
        return total

    def num_base_params(self) -> int:
        return sum(arr.size for arr in self._flat.values())

    def delta(self, full_name: str) -> np.ndarray:
        if full_name not in self._targets:
            raise KeyError(full_name)
        layout = self._layouts[full_name]
        a = self._lora_a[full_name]
        b = self._lora_b[full_name]
        scaling = self.config.alpha / self._effective_ranks[full_name]

        if not layout.stack:
            low_rank = a @ b * scaling
            if layout.transposed:
                low_rank = low_rank.T
            return low_rank.reshape(layout.core_shape)

        delta = np.empty((*layout.stack, *layout.core_shape), dtype=np.float32)
        for idx in np.ndindex(layout.stack):
            low_rank = a[idx] @ b[idx] * scaling
            if layout.transposed:
                low_rank = low_rank.T
            delta[idx] = low_rank.reshape(layout.core_shape)
        return delta

    def apply(self) -> dict[str, dict[str, np.ndarray]]:
        merged = {
            scope: {name: arr.copy() for name, arr in entries.items()}
            for scope, entries in self.base_params.items()
        }
        flat = record_io.flatten(merged)
        for name in self._targets:
            delta = self.delta(name)
            if np.any(delta):
                flat[name] = (flat[name].astype(np.float32) + delta).astype(flat[name].dtype)
        return record_io.unflatten(flat)

    def state_dict(self) -> dict[str, dict[str, np.ndarray]]:
        return {
            name: {"lora_a": self._lora_a[name].copy(), "lora_b": self._lora_b[name].copy()}
            for name in self._targets
        }

    def load_state_dict(self, state: dict[str, dict[str, np.ndarray]]) -> None:
        for name in self._targets:
            if name not in state:
                raise KeyError(f"Missing LoRA state for {name}")
            self._lora_a[name] = np.asarray(state[name]["lora_a"], dtype=np.float32)
            self._lora_b[name] = np.asarray(state[name]["lora_b"], dtype=np.float32)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        payload = {
            "config_rank": np.array(self.config.rank, dtype=np.int32),
            "config_alpha": np.array(self.config.alpha, dtype=np.float32),
            "config_seed": np.array(self.config.seed, dtype=np.int32),
        }
        for name in self._targets:
            payload[f"{name}/lora_a"] = self._lora_a[name]
            payload[f"{name}/lora_b"] = self._lora_b[name]
        np.savez(path, **payload)

    @classmethod
    def load(cls, path: Path | str, base_params: dict[str, dict[str, np.ndarray]]) -> "AF3LoRA":
        data = np.load(path)
        config = LoRAConfig(
            rank=int(data["config_rank"]),
            alpha=float(data["config_alpha"]),
            seed=int(data["config_seed"]),
        )
        targets = tuple(
            key[:-len("/lora_a")]
            for key in data.files
            if key.endswith("/lora_a")
        )
        lora = cls(base_params, targets, config)
        state = {
            name: {
                "lora_a": data[f"{name}/lora_a"],
                "lora_b": data[f"{name}/lora_b"],
            }
            for name in targets
        }
        lora.load_state_dict(state)
        return lora
