"""Tests for the AlphaFold 3 fine-tuning entry point and weight utilities.

Test plan: alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md section 5.5.
"""

import numpy as np
import pytest

from finetuning.af3 import record_io, schema, weights
from finetuning.af3.finetuner import (
    AF3FineTuneConfig,
    AlphaFold3FineTuner,
    WeightsComplianceError,
)
from finetuning.af3.lora import LoRAConfig
from finetuning.af3.param_groups import ParamGroup, classify


@pytest.fixture
def tuner(small_schema):
    return AlphaFold3FineTuner.from_random(schema_specs=small_schema, seed=3)


def test_from_random_produces_valid_params(small_schema, tuner):
    report = schema.validate_params(tuner.params, small_schema)
    assert report.ok, report.describe()


def test_lora_strategy_trains_only_adapters(tuner):
    assert tuner.config.strategy == "lora"
    assert tuner.lora is not None
    assert tuner.trainable_param_names() == tuner.lora.targets

    frozen = set(tuner.frozen_param_names())
    all_names = set(record_io.flatten(tuner.params))
    assert frozen == all_names - set(tuner.lora.targets)


def test_head_only_strategy_excludes_the_trunk(small_schema):
    config = AF3FineTuneConfig(
        strategy="head_only",
        trainable_groups=(ParamGroup.CONFIDENCE,),
    )
    tuner = AlphaFold3FineTuner.from_random(config=config, schema_specs=small_schema, seed=1)

    trainable = tuner.trainable_param_names()
    assert trainable
    assert all(classify(name) is ParamGroup.CONFIDENCE for name in trainable)
    assert tuner.lora is None


def test_full_strategy_trains_every_non_meta_parameter(small_schema):
    config = AF3FineTuneConfig(strategy="full")
    tuner = AlphaFold3FineTuner.from_random(config=config, schema_specs=small_schema, seed=1)

    expected = {
        name for name in record_io.flatten(tuner.params) if not name.startswith("__meta__")
    }
    assert set(tuner.trainable_param_names()) == expected
    assert set(tuner.frozen_param_names()) == {"__meta__:__identifier__"}


def test_unknown_strategy_raises(small_schema):
    with pytest.raises(ValueError):
        AlphaFold3FineTuner.from_random(
            config=AF3FineTuneConfig(strategy="magic"), schema_specs=small_schema
        )


def test_parameter_summary_ratios(small_schema, tuner):
    lora_summary = tuner.parameter_summary()
    assert lora_summary.total > 0
    assert lora_summary.trainable < lora_summary.frozen
    assert lora_summary.trainable_ratio < 0.15
    assert sum(lora_summary.by_group.values()) == lora_summary.trainable
    assert "Trainable" in lora_summary.describe()

    full = AlphaFold3FineTuner.from_random(
        config=AF3FineTuneConfig(strategy="full"), schema_specs=small_schema, seed=3
    ).parameter_summary()
    assert full.trainable_ratio > 0.99


def test_save_and_load_adapter_roundtrip(tmp_path, tuner):
    state = tuner.lora.state_dict()
    rng = np.random.default_rng(5)
    for entry in state.values():
        entry["lora_b"] = rng.standard_normal(entry["lora_b"].shape).astype(np.float32)
    tuner.lora.load_state_dict(state)

    path = tmp_path / "adapter.npz"
    tuner.save_adapter(path)
    assert path.exists()

    reloaded = AlphaFold3FineTuner.from_random(
        schema_specs=tuple(
            spec for spec in schema.load_schema() if spec.full_name in record_io.flatten(tuner.params)
        ),
        seed=99,
    )
    reloaded.load_adapter(path)
    for name in tuner.lora.targets:
        np.testing.assert_array_equal(reloaded.lora.delta(name), tuner.lora.delta(name))


def test_adapter_file_holds_no_base_weights(tmp_path, tuner):
    path = tmp_path / "adapter.npz"
    tuner.save_adapter(path)

    base_bytes = {arr.tobytes() for arr in record_io.flatten(tuner.params).values() if arr.size}
    stored = [np.asarray(v) for v in np.load(path).values()]
    assert all(arr.tobytes() not in base_bytes for arr in stored if arr.size)


def test_export_merged_weights_requires_acknowledgement(tmp_path, tuner):
    path = tmp_path / "merged.bin"
    with pytest.raises(WeightsComplianceError):
        tuner.export_merged_weights(path)
    assert not path.exists()


def test_export_merged_weights_after_acknowledgement(tmp_path, small_schema, tuner):
    path = tmp_path / "merged.bin"
    tuner.export_merged_weights(path, acknowledge_weights_terms=True)

    report = schema.validate_params(record_io.read_params(path), small_schema)
    assert report.ok, report.describe()


def test_download_requires_terms_acceptance(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("no network access must happen before terms are accepted")

    monkeypatch.setattr(weights.urllib.request, "urlopen", explode)
    with pytest.raises(WeightsComplianceError):
        weights.download_weights(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_weights_url_points_at_the_public_bucket():
    assert weights.AF3_WEIGHTS_URL == "https://storage.googleapis.com/alphafold3/af3.bin.zst"
    assert "WEIGHTS_TERMS_OF_USE.md" in weights.WEIGHTS_TERMS_URL


def test_check_weights_reports_missing_entries(tmp_path, small_schema, small_params):
    path = tmp_path / "af3.bin"
    record_io.write_params(path, small_params)

    report = weights.check_weights(tmp_path, schema_specs=small_schema)
    assert report.ok

    full_report = weights.check_weights(tmp_path)
    assert not full_report.ok
    assert len(full_report.missing) > 300


def test_cli_info(capsys):
    assert weights.main(["info"]) == 0
    out = capsys.readouterr().out
    assert "405" in out
    assert "368,384,602" in out
    assert weights.AF3_WEIGHTS_URL in out


def test_cli_check(tmp_path, capsys, small_params):
    record_io.write_params(tmp_path / "af3.bin", small_params)
    exit_code = weights.main(["check", str(tmp_path)])
    out = capsys.readouterr().out

    assert exit_code == 1, "the subset is not a complete checkpoint"
    assert "missing" in out.lower()
