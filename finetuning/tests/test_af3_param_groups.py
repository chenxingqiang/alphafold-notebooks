"""Tests for AlphaFold 3 parameter grouping and LoRA target selection.

Test plan: alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md section 5.3.
"""

import pytest

from finetuning.af3 import schema
from finetuning.af3.param_groups import (
    DEFAULT_LORA_TARGET_PATTERNS,
    ParamGroup,
    classify,
    group_param_counts,
    is_linear_weight,
    is_transposed_weight,
    matrix_layout,
    num_stack_dims,
    select_lora_targets,
    stack_dims,
)

TRUNK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer"
MSA = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack"
DIFF_TRANSFORMER = (
    "diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer"
    "/__layer_stack_with_per_layer"
)


def test_every_schema_entry_is_classified(full_schema):
    unknown = [s.full_name for s in full_schema if classify(s.full_name) is ParamGroup.UNKNOWN]
    assert unknown == []


@pytest.mark.parametrize(
    "full_name, expected",
    [
        ("__meta__:__identifier__", ParamGroup.META),
        (f"{TRUNK}/pair_attention1/q_projection:weights", ParamGroup.PAIRFORMER),
        (f"{MSA}/outer_product_mean:output_w", ParamGroup.MSA),
        (f"{DIFF_TRANSFORMER}/transformerq_projection:weights", ParamGroup.DIFFUSION),
        ("diffuser/confidence_head/plddt_logits:weights", ParamGroup.CONFIDENCE),
        (
            "diffuser/evoformer/template_embedding/single_template_embedding"
            "/template_pair_embedding_8:weights",
            ParamGroup.TEMPLATE,
        ),
        ("diffuser/evoformer/left_single:weights", ParamGroup.EMBEDDING),
        (
            "diffuser/evoformer_conditioning_atom_transformer_encoder"
            "/__layer_stack_with_per_layer/evoformer_conditioning_atom_transformer_encoder"
            "q_projection:weights",
            ParamGroup.EMBEDDING,
        ),
    ],
)
def test_classification_of_representative_names(full_name, expected):
    assert classify(full_name) is expected


def test_group_counts_partition_the_schema(full_schema):
    counts = group_param_counts(full_schema)
    assert sum(counts.values()) == sum(spec.num_params for spec in full_schema)

    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    assert ordered[0][0] is ParamGroup.DIFFUSION
    assert ordered[1][0] is ParamGroup.PAIRFORMER


def test_num_stack_dims_counts_nested_layer_stacks():
    assert num_stack_dims("__meta__:__identifier__") == 0
    assert num_stack_dims(f"{TRUNK}/pair_attention1/q_projection:weights") == 1
    assert num_stack_dims(f"{DIFF_TRANSFORMER}/transformerq_projection:weights") == 2


def test_stack_dims_returns_leading_dimensions():
    assert stack_dims("diffuser/evoformer/left_single:weights", (447, 128)) == ()
    assert stack_dims(f"{TRUNK}/pair_attention1/gating_query:weights", (48, 128, 128)) == (48,)
    assert stack_dims(
        f"{DIFF_TRANSFORMER}/transformerq_projection:weights", (6, 4, 768, 16, 48)
    ) == (6, 4)


@pytest.mark.parametrize(
    "full_name, expected",
    [
        (f"{TRUNK}/pair_attention1/q_projection:weights", True),
        (f"{MSA}/outer_product_mean:output_w", True),
        (f"{TRUNK}/pair_attention1/act_norm:scale", False),
        (f"{TRUNK}/pair_attention1/q_projection:bias", False),
        (f"{MSA}/outer_product_mean/layer_norm_input:offset", False),
        ("__meta__:__identifier__", False),
    ],
)
def test_is_linear_weight(full_name, expected):
    assert is_linear_weight(full_name) is expected


@pytest.mark.parametrize(
    "full_name, expected",
    [
        (f"{TRUNK}/pair_attention1/q_projection:weights", True),
        (f"{TRUNK}/pair_attention2/k_projection:weights", True),
        (f"{TRUNK}/pair_attention1/gating_query:weights", True),
        (f"{TRUNK}/pair_attention1/v_projection:weights", False),
        (f"{TRUNK}/pair_attention1/output_projection:weights", False),
        (f"{TRUNK}/single_attention_q_projection:weights", False),
        (f"{MSA}/msa_attention1/gating_query:weights", False),
    ],
)
def test_is_transposed_weight(full_name, expected):
    assert is_transposed_weight(full_name) is expected


@pytest.mark.parametrize(
    "full_name, shape, stack, in_dim, out_dim, transposed",
    [
        # hm.Linear default layout: leading dim is the input channel.
        (f"{TRUNK}/pair_attention1/v_projection:weights", (48, 128, 4, 32), (48,), 128, 128, False),
        (f"{TRUNK}/single_attention_q_projection:weights", (48, 384, 16, 24), (48,), 384, 384, False),
        # transpose_weights=True in GridSelfAttention: trailing dim is the input.
        (f"{TRUNK}/pair_attention1/q_projection:weights", (48, 4, 32, 128), (48,), 128, 128, True),
        # OuterProductMean's einsum weight consumes two input axes.
        (f"{MSA}/outer_product_mean:output_w", (4, 32, 32, 128), (4,), 1024, 128, False),
        # Doubly stacked diffusion transformer.
        (
            f"{DIFF_TRANSFORMER}/transformerq_projection:weights",
            (6, 4, 768, 16, 48),
            (6, 4),
            768,
            768,
            False,
        ),
        ("diffuser/evoformer/left_single:weights", (447, 128), (), 447, 128, False),
    ],
)
def test_matrix_layout(full_name, shape, stack, in_dim, out_dim, transposed):
    layout = matrix_layout(full_name, shape)
    assert layout.stack == stack
    assert layout.in_dim == in_dim
    assert layout.out_dim == out_dim
    assert layout.transposed is transposed
    assert layout.core_shape == shape[len(stack):]


def test_matrix_layout_rejects_parameters_without_input_axis():
    with pytest.raises(ValueError):
        matrix_layout(
            "diffuser/evoformer/template_embedding/single_template_embedding"
            "/template_pair_embedding_7:weights",
            (64,),
        )


def test_default_lora_targets(full_schema):
    targets = select_lora_targets(full_schema)
    assert targets
    assert len(set(targets)) == len(targets)
    for name in targets:
        assert is_linear_weight(name)
        assert classify(name) in (ParamGroup.PAIRFORMER, ParamGroup.DIFFUSION)


def test_lora_targets_restricted_to_groups(full_schema):
    default = set(select_lora_targets(full_schema))
    pairformer_only = select_lora_targets(full_schema, groups=(ParamGroup.PAIRFORMER,))

    assert pairformer_only
    assert set(pairformer_only) <= default
    assert all(classify(name) is ParamGroup.PAIRFORMER for name in pairformer_only)


def test_lora_targets_filtered_by_pattern(full_schema):
    targets = select_lora_targets(full_schema, patterns=("q_projection",))
    assert targets
    assert all("q_projection" in name for name in targets)


def test_default_patterns_cover_attention_and_transitions():
    joined = " ".join(DEFAULT_LORA_TARGET_PATTERNS)
    for expected in ("q_projection", "k_projection", "v_projection", "output_projection"):
        assert expected in joined


def test_selected_targets_have_usable_matrix_layout(full_schema):
    index = schema.schema_index(full_schema)
    for name in select_lora_targets(full_schema):
        layout = matrix_layout(name, index[name].shape)
        assert layout.in_dim > 1 and layout.out_dim > 1
