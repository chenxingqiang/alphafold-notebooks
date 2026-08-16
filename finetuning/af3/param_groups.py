"""AlphaFold 3 parameter grouping and LoRA target selection."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

from .schema import ParamSpec, load_schema

STACK_MARKERS = ("__layer_stack_with_per_layer", "__layer_stack_no_per_layer")

DEFAULT_LORA_TARGET_PATTERNS: tuple[str, ...] = (
    "q_projection",
    "k_projection",
    "v_projection",
    "output_projection",
    "gating_query",
    "transition1",
    "transition2",
    "left_projection",
    "right_projection",
    "projection",
)

LORA_TARGET_EXCLUSIONS: tuple[str, ...] = (
    "pair_bias_projection",
    "pair_logits_projection",
    "single_pair_logits_projection",
)


class ParamGroup(str, Enum):
    EMBEDDING = "embedding"
    TEMPLATE = "template"
    MSA = "msa"
    PAIRFORMER = "pairformer"
    DIFFUSION = "diffusion"
    CONFIDENCE = "confidence"
    META = "meta"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class MatrixLayout:
    stack: tuple[int, ...]
    in_dim: int
    out_dim: int
    transposed: bool
    core_shape: tuple[int, ...]


def classify(full_name: str) -> ParamGroup:
    if full_name.startswith("__meta__:"):
        return ParamGroup.META
    if "/confidence_head/" in full_name or full_name.startswith("diffuser/confidence_head"):
        return ParamGroup.CONFIDENCE
    if "/~/diffusion_head/" in full_name or "/diffusion_head/" in full_name:
        return ParamGroup.DIFFUSION
    if "/distogram_head/" in full_name or full_name.startswith("diffuser/distogram_head"):
        return ParamGroup.DIFFUSION
    if "/template_embedding/" in full_name:
        return ParamGroup.TEMPLATE
    if "/__layer_stack_no_per_layer_1/" in full_name:
        return ParamGroup.PAIRFORMER
    if "/__layer_stack_no_per_layer/" in full_name:
        return ParamGroup.MSA
    if full_name.startswith("diffuser/evoformer_conditioning"):
        return ParamGroup.EMBEDDING
    if full_name.startswith("diffuser/evoformer/"):
        return ParamGroup.EMBEDDING
    return ParamGroup.UNKNOWN


def num_stack_dims(full_name: str) -> int:
    return sum(full_name.count(marker) for marker in STACK_MARKERS)


def stack_dims(full_name: str, shape: tuple[int, ...]) -> tuple[int, ...]:
    n = num_stack_dims(full_name)
    return tuple(shape[:n]) if n else ()


def is_linear_weight(full_name: str) -> bool:
    if not full_name.endswith(":weights") and not full_name.endswith(":output_w"):
        return False
    if any(
        token in full_name
        for token in (
            ":scale",
            ":offset",
            ":bias",
            "layer_norm",
            "act_norm",
            "pair_norm",
            "single_pair_logits_norm",
        )
    ):
        return False
    return True


def is_transposed_weight(full_name: str) -> bool:
    """GridSelfAttention q/k/gating use ``transpose_weights=True`` in AF3."""
    if not is_linear_weight(full_name):
        return False
    leaf = full_name.rsplit("/", 1)[-1]
    if not (
        leaf.startswith("q_projection:")
        or leaf.startswith("k_projection:")
        or leaf.startswith("gating_query:")
    ):
        return False
    # Only GridSelfAttention pair-attention blocks use transpose_weights=True.
    return "pair_attention" in full_name or "template_embedding_iteration" in full_name


def matrix_layout(full_name: str, shape: tuple[int, ...]) -> MatrixLayout:
    if not is_linear_weight(full_name):
        raise ValueError(f"Not a linear weight parameter: {full_name}")

    stack = stack_dims(full_name, shape)
    core = shape[len(stack):]

    if full_name.endswith(":output_w"):
        if len(core) != 3:
            raise ValueError(f"Unexpected output_w shape for {full_name}: {shape}")
        in_dim = core[0] * core[1]
        out_dim = core[2]
        transposed = False
    elif is_transposed_weight(full_name):
        if len(core) < 2:
            raise ValueError(f"Cannot infer matrix layout for {full_name}: {shape}")
        in_dim = core[-1]
        out_dim = int(np.prod(core[:-1]))
        transposed = True
    else:
        if len(core) < 2:
            raise ValueError(f"Cannot infer matrix layout for {full_name}: {shape}")
        in_dim = core[0]
        out_dim = int(np.prod(core[1:]))
        transposed = False

    if in_dim <= 1 or out_dim <= 1:
        raise ValueError(
            f"Degenerate matrix layout for {full_name}: in={in_dim}, out={out_dim}"
        )

    return MatrixLayout(
        stack=stack,
        in_dim=in_dim,
        out_dim=out_dim,
        transposed=transposed,
        core_shape=core,
    )


def group_param_counts(schema: tuple[ParamSpec, ...] | None = None) -> dict[ParamGroup, int]:
    specs = schema or load_schema()
    counts: dict[ParamGroup, int] = {}
    for spec in specs:
        group = classify(spec.full_name)
        counts[group] = counts.get(group, 0) + spec.num_params
    return counts


def select_lora_targets(
    schema: tuple[ParamSpec, ...] | None = None,
    *,
    groups: Sequence[ParamGroup] | None = None,
    patterns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    specs = schema or load_schema()
    allowed_groups = set(groups) if groups is not None else {
        ParamGroup.PAIRFORMER,
        ParamGroup.DIFFUSION,
    }
    pattern_list = tuple(patterns) if patterns is not None else DEFAULT_LORA_TARGET_PATTERNS

    targets: list[str] = []
    for spec in specs:
        name = spec.full_name
        if classify(name) not in allowed_groups:
            continue
        if not is_linear_weight(name):
            continue
        if not any(pattern in name for pattern in pattern_list):
            continue
        if any(excluded in name for excluded in LORA_TARGET_EXCLUSIONS):
            continue
        try:
            matrix_layout(name, spec.shape)
        except ValueError:
            continue
        targets.append(name)
    return tuple(targets)
