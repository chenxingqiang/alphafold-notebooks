"""Tests for the AlphaFold 3 parameter schema utilities.

Test plan: alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md section 5.2.
"""

import numpy as np
import pytest

from finetuning.af3 import record_io, schema

EXPECTED_NUM_ENTRIES = 405
EXPECTED_NUM_PARAMS = 368_384_602
TRUNK_PAIRFORMER_PREFIX = "diffuser/evoformer/__layer_stack_no_per_layer_1/trunk_pairformer"
MSA_STACK_PREFIX = "diffuser/evoformer/__layer_stack_no_per_layer/msa_stack"


def test_schema_entry_count(full_schema):
    assert len(full_schema) == EXPECTED_NUM_ENTRIES


def test_schema_total_parameter_count(full_schema):
    assert sum(spec.num_params for spec in full_schema) == EXPECTED_NUM_PARAMS


def test_schema_dtypes(full_schema):
    assert {spec.dtype for spec in full_schema} == {"float32", "bfloat16", "uint8"}


def test_identifier_entry(full_schema):
    index = schema.schema_index(full_schema)
    spec = index["__meta__:__identifier__"]
    assert spec.scope == "__meta__"
    assert spec.name == "__identifier__"
    assert spec.shape == (64,)
    assert spec.dtype == "uint8"


def test_layer_stack_leading_dimensions(full_schema):
    index = schema.schema_index(full_schema)
    trunk = index[f"{TRUNK_PAIRFORMER_PREFIX}/pair_attention1/q_projection:weights"]
    assert trunk.shape[0] == 48, "trunk Pairformer has 48 stacked blocks"

    msa = index[f"{MSA_STACK_PREFIX}/pair_attention1/q_projection:weights"]
    assert msa.shape[0] == 4, "MSA stack has 4 stacked blocks"


def test_param_spec_derived_fields():
    spec = schema.ParamSpec(scope="scope/a", name="weights", shape=(3, 4, 5), dtype="float32")
    assert spec.full_name == "scope/a:weights"
    assert spec.num_params == 60
    assert spec.numpy_dtype == np.dtype(np.float32)


def test_parse_line_rejects_malformed_input():
    with pytest.raises(ValueError):
        schema.parse_schema_line("name=scope:weights dtype=float32")


def test_generate_random_params_matches_schema(small_schema):
    params = schema.generate_random_params(small_schema, seed=0)
    flat = record_io.flatten(params)

    assert set(flat) == {spec.full_name for spec in small_schema}
    for spec in small_schema:
        arr = flat[spec.full_name]
        assert arr.shape == spec.shape
        assert arr.dtype == spec.numpy_dtype


def test_generate_random_params_identifier_is_zero_and_rest_is_not(small_schema):
    full = schema.load_schema()
    index = schema.schema_index(full)
    specs = (index["__meta__:__identifier__"],) + tuple(small_schema)
    flat = record_io.flatten(schema.generate_random_params(specs, seed=0))

    assert not np.any(flat["__meta__:__identifier__"])
    non_meta = [
        arr
        for name, arr in flat.items()
        if not name.startswith("__meta__") and arr.size > 100
    ]
    assert non_meta, "expected at least one sizeable parameter in the subset"
    assert all(np.any(arr.view(np.uint8)) for arr in non_meta)


def test_generate_random_params_is_reproducible(small_schema):
    a = record_io.flatten(schema.generate_random_params(small_schema, seed=7))
    b = record_io.flatten(schema.generate_random_params(small_schema, seed=7))
    c = record_io.flatten(schema.generate_random_params(small_schema, seed=8))

    for name in a:
        assert a[name].tobytes() == b[name].tobytes()
    assert any(a[name].tobytes() != c[name].tobytes() for name in a)


def test_validate_params_accepts_generated_params(small_schema, small_params):
    report = schema.validate_params(small_params, small_schema)
    assert report.ok
    assert report.missing == ()
    assert report.unexpected == ()
    assert report.shape_mismatch == ()
    assert report.dtype_mismatch == ()
    assert "OK" in report.describe()


def test_validate_params_detects_missing(small_schema, small_params):
    victim = small_schema[0]
    del small_params[victim.scope][victim.name]
    if not small_params[victim.scope]:
        del small_params[victim.scope]

    report = schema.validate_params(small_params, small_schema)
    assert not report.ok
    assert victim.full_name in report.missing


def test_validate_params_detects_unexpected(small_schema, small_params):
    small_params.setdefault("scope/bogus", {})["weights"] = np.zeros((2,), dtype=np.float32)

    report = schema.validate_params(small_params, small_schema)
    assert not report.ok
    assert "scope/bogus:weights" in report.unexpected


def test_validate_params_detects_shape_mismatch(small_schema, small_params):
    victim = small_schema[0]
    small_params[victim.scope][victim.name] = np.zeros((1, 1), dtype=victim.numpy_dtype)

    report = schema.validate_params(small_params, small_schema)
    assert not report.ok
    assert (victim.full_name, victim.shape, (1, 1)) in report.shape_mismatch


def test_validate_params_detects_dtype_mismatch(small_schema, small_params):
    victim = next(s for s in small_schema if s.dtype != "int8")
    small_params[victim.scope][victim.name] = np.zeros(victim.shape, dtype=np.int8)

    report = schema.validate_params(small_params, small_schema)
    assert not report.ok
    assert any(entry[0] == victim.full_name for entry in report.dtype_mismatch)


def test_summarize(full_schema):
    summary = schema.summarize(full_schema)
    assert summary.num_entries == EXPECTED_NUM_ENTRIES
    assert summary.num_params == EXPECTED_NUM_PARAMS
    assert sum(summary.params_by_dtype.values()) == EXPECTED_NUM_PARAMS
    assert summary.params_by_dtype["uint8"] == 64


def test_end_to_end_write_read_validate(tmp_path, small_schema, small_params):
    path = tmp_path / "af3.bin"
    record_io.write_params(path, small_params)

    reloaded = record_io.read_params(path)
    report = schema.validate_params(reloaded, small_schema)
    assert report.ok, report.describe()
