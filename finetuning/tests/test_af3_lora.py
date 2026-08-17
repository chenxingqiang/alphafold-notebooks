"""Tests for parameter-space LoRA on AlphaFold 3 Haiku parameters.

Test plan: alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md section 5.4.

The parameter *names* below are real AlphaFold 3 names (so that the stack-depth
and transposition rules are exercised), while the shapes are deliberately small:
the layout logic is driven by the name plus the actual array shape, never by the
schema, so shrinking the arrays keeps the tests fast without weakening them.
"""

import copy

import numpy as np
import pytest

from finetuning.af3 import record_io
from finetuning.af3.lora import AF3LoRA, LoRAConfig

TRUNK = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer"
DIFF_TRANSFORMER = (
    "diffuser/~/diffusion_head/transformer/__layer_stack_with_per_layer"
    "/__layer_stack_with_per_layer"
)

PLAIN = "diffuser/evoformer/left_single:weights"
STACKED = f"{TRUNK}/pair_attention1/v_projection:weights"
TRANSPOSED = f"{TRUNK}/pair_attention1/q_projection:weights"
DOUBLE_STACKED = f"{DIFF_TRANSFORMER}/transformerq_projection:weights"
LAYER_NORM = f"{TRUNK}/pair_attention1/act_norm:scale"


@pytest.fixture
def base_params():
    rng = np.random.default_rng(0)

    def arr(*shape):
        return rng.standard_normal(shape).astype(np.float32)

    return record_io.unflatten(
        {
            PLAIN: arr(6, 4),
            STACKED: arr(2, 8, 2, 3),
            TRANSPOSED: arr(2, 2, 3, 8),
            DOUBLE_STACKED: arr(2, 3, 8, 2, 2),
            LAYER_NORM: arr(2, 8),
        }
    )


@pytest.fixture
def targets():
    return (PLAIN, STACKED, TRANSPOSED, DOUBLE_STACKED)


def test_initial_delta_is_zero(base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2, alpha=4.0, seed=0))
    for name in targets:
        assert not np.any(lora.delta(name))


def test_apply_is_identity_before_training(base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2))
    applied = record_io.flatten(lora.apply())
    original = record_io.flatten(base_params)

    assert set(applied) == set(original)
    for name, arr in original.items():
        np.testing.assert_array_equal(applied[name], arr)


def test_delta_shape_matches_base_weight(base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2))
    flat = record_io.flatten(base_params)
    for name in targets:
        assert lora.delta(name).shape == flat[name].shape


def test_nonzero_delta_only_affects_its_target(base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2))
    state = lora.state_dict()
    state[STACKED]["lora_b"] += 1.0
    lora.load_state_dict(state)

    applied = record_io.flatten(lora.apply())
    original = record_io.flatten(base_params)
    for name, arr in original.items():
        if name == STACKED:
            assert not np.array_equal(applied[name], arr)
        else:
            np.testing.assert_array_equal(applied[name], arr)


def test_delta_math_without_stack(base_params):
    config = LoRAConfig(rank=2, alpha=6.0, seed=3)
    lora = AF3LoRA(base_params, (PLAIN,), config)

    state = lora.state_dict()
    rng = np.random.default_rng(11)
    state[PLAIN]["lora_b"] = rng.standard_normal(state[PLAIN]["lora_b"].shape).astype(np.float32)
    lora.load_state_dict(state)

    expected = state[PLAIN]["lora_a"] @ state[PLAIN]["lora_b"] * (config.alpha / config.rank)
    np.testing.assert_allclose(lora.delta(PLAIN), expected, rtol=1e-6, atol=1e-6)


def test_delta_math_with_stack_and_multihead_output(base_params):
    config = LoRAConfig(rank=2, alpha=4.0, seed=5)
    lora = AF3LoRA(base_params, (STACKED,), config)

    state = lora.state_dict()
    rng = np.random.default_rng(13)
    state[STACKED]["lora_b"] = rng.standard_normal(state[STACKED]["lora_b"].shape).astype(np.float32)
    lora.load_state_dict(state)

    a, b = state[STACKED]["lora_a"], state[STACKED]["lora_b"]
    delta = lora.delta(STACKED)
    assert delta.shape == (2, 8, 2, 3)
    scaling = config.alpha / config.rank
    for layer in range(2):
        expected = (a[layer] @ b[layer] * scaling).reshape(8, 2, 3)
        np.testing.assert_allclose(delta[layer], expected, rtol=1e-6, atol=1e-6)


def test_delta_math_with_transposed_weights(base_params):
    config = LoRAConfig(rank=2, alpha=4.0, seed=5)
    lora = AF3LoRA(base_params, (TRANSPOSED,), config)

    state = lora.state_dict()
    rng = np.random.default_rng(17)
    state[TRANSPOSED]["lora_b"] = rng.standard_normal(
        state[TRANSPOSED]["lora_b"].shape
    ).astype(np.float32)
    lora.load_state_dict(state)

    a, b = state[TRANSPOSED]["lora_a"], state[TRANSPOSED]["lora_b"]
    delta = lora.delta(TRANSPOSED)
    assert delta.shape == (2, 2, 3, 8)
    scaling = config.alpha / config.rank
    for layer in range(2):
        # Stored layout is (out_head, out_dim, in); the low-rank product is (in, out).
        expected = (a[layer] @ b[layer] * scaling).T.reshape(2, 3, 8)
        np.testing.assert_allclose(delta[layer], expected, rtol=1e-6, atol=1e-6)


def test_delta_math_with_two_stack_dimensions(base_params):
    config = LoRAConfig(rank=2, alpha=4.0, seed=5)
    lora = AF3LoRA(base_params, (DOUBLE_STACKED,), config)

    state = lora.state_dict()
    assert state[DOUBLE_STACKED]["lora_a"].shape == (2, 3, 8, 2)
    assert state[DOUBLE_STACKED]["lora_b"].shape == (2, 3, 2, 4)

    rng = np.random.default_rng(19)
    state[DOUBLE_STACKED]["lora_b"] = rng.standard_normal((2, 3, 2, 4)).astype(np.float32)
    lora.load_state_dict(state)

    a, b = state[DOUBLE_STACKED]["lora_a"], state[DOUBLE_STACKED]["lora_b"]
    delta = lora.delta(DOUBLE_STACKED)
    assert delta.shape == (2, 3, 8, 2, 2)
    scaling = config.alpha / config.rank
    for i in range(2):
        for j in range(3):
            expected = (a[i, j] @ b[i, j] * scaling).reshape(8, 2, 2)
            np.testing.assert_allclose(delta[i, j], expected, rtol=1e-6, atol=1e-6)


def test_parameter_counts(base_params, targets):
    config = LoRAConfig(rank=2)
    lora = AF3LoRA(base_params, targets, config)

    expected = 0
    for name in targets:
        layout = lora.layout(name)
        effective_rank = lora._effective_ranks[name]
        stack = int(np.prod(layout.stack)) if layout.stack else 1
        expected += stack * effective_rank * (layout.in_dim + layout.out_dim)
    assert lora.num_lora_params() == expected
    assert lora.num_base_params() == sum(
        arr.size for arr in record_io.flatten(base_params).values()
    )


def test_lora_is_a_small_fraction_of_the_real_model(full_schema):
    """Metadata-only check against the real 368M-parameter checkpoint."""
    from finetuning.af3.lora import count_lora_params
    from finetuning.af3.param_groups import select_lora_targets

    targets = select_lora_targets(full_schema)
    shapes = {spec.full_name: spec.shape for spec in full_schema}
    adapter_params = count_lora_params(
        {name: shapes[name] for name in targets}, LoRAConfig(rank=8)
    )
    total = sum(spec.num_params for spec in full_schema)

    assert 0 < adapter_params / total < 0.05


def test_state_dict_contains_only_adapters(base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2))
    state = lora.state_dict()

    assert set(state) == set(targets)
    for entry in state.values():
        assert set(entry) == {"lora_a", "lora_b"}


def test_save_and_load_roundtrip(tmp_path, base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2, alpha=8.0, seed=1))
    state = lora.state_dict()
    rng = np.random.default_rng(23)
    for entry in state.values():
        entry["lora_b"] = rng.standard_normal(entry["lora_b"].shape).astype(np.float32)
    lora.load_state_dict(state)

    path = tmp_path / "adapter.npz"
    lora.save(path)

    restored = AF3LoRA.load(path, base_params)
    assert restored.config == lora.config
    assert restored.targets == lora.targets
    for name in targets:
        np.testing.assert_array_equal(restored.delta(name), lora.delta(name))


def test_saved_adapter_holds_no_base_weight_values(tmp_path, base_params, targets):
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2))
    path = tmp_path / "adapter.npz"
    lora.save(path)

    stored = [np.asarray(v) for v in np.load(path).values()]
    base_bytes = {arr.tobytes() for arr in record_io.flatten(base_params).values()}
    assert all(arr.tobytes() not in base_bytes for arr in stored if arr.size)


def test_unknown_target_raises(base_params):
    with pytest.raises(KeyError):
        AF3LoRA(base_params, ("scope/does_not_exist:weights",))


def test_non_linear_target_raises(base_params):
    with pytest.raises(ValueError):
        AF3LoRA(base_params, (LAYER_NORM,))


@pytest.mark.parametrize("rank", [0, -1])
def test_invalid_rank_raises(base_params, rank):
    with pytest.raises(ValueError):
        AF3LoRA(base_params, (PLAIN,), LoRAConfig(rank=rank))


def test_apply_does_not_mutate_base_params(base_params, targets):
    snapshot = copy.deepcopy(base_params)
    lora = AF3LoRA(base_params, targets, LoRAConfig(rank=2))
    state = lora.state_dict()
    for entry in state.values():
        entry["lora_b"] += 0.5
    lora.load_state_dict(state)
    lora.apply()

    for scope, entries in snapshot.items():
        for name, arr in entries.items():
            np.testing.assert_array_equal(base_params[scope][name], arr)
