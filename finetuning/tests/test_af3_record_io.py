"""Tests for the AlphaFold 3 parameter container codec.

Test plan: alphafold3/AF3_WEIGHTS_FINETUNING_DESIGN.md section 5.1.
"""

import io
import struct

import numpy as np
import pytest

from finetuning.af3 import record_io


def _bf16_zeros(shape):
    return np.zeros(shape, dtype=record_io.numpy_dtype_for("bfloat16"))


def test_encode_record_header_layout():
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    blob = record_io.encode_record("scope/a", "weights", arr)

    header_size = struct.calcsize("<5i")
    scope_len, name_len, dtype_len, shape_len, buffer_len = struct.unpack(
        "<5i", blob[:header_size]
    )
    assert scope_len == len(b"scope/a")
    assert name_len == len(b"weights")
    assert dtype_len == len(b"float32")
    assert shape_len == 2
    assert buffer_len == arr.nbytes
    assert len(blob) == header_size + scope_len + name_len + dtype_len + 4 * shape_len + buffer_len


def test_single_record_roundtrip():
    arr = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(3, 4)
    blob = record_io.encode_record("diffuser/evoformer", "weights", arr)

    (scope, name, decoded), = record_io.read_records(io.BytesIO(blob))
    assert scope == "diffuser/evoformer"
    assert name == "weights"
    assert decoded.shape == arr.shape
    assert decoded.dtype == arr.dtype
    np.testing.assert_array_equal(decoded, arr)


@pytest.mark.parametrize("wire_dtype", ["float32", "bfloat16", "uint8", "int32"])
def test_roundtrip_preserves_bytes_for_every_dtype(wire_dtype):
    np_dtype = record_io.numpy_dtype_for(wire_dtype)
    arr = np.arange(8, dtype=np.int64).astype(np_dtype).reshape(2, 4)

    blob = record_io.encode_record("s", "weights", arr, wire_dtype=wire_dtype)
    (_, _, decoded), = record_io.read_records(io.BytesIO(blob))

    assert decoded.tobytes() == arr.tobytes()
    assert decoded.shape == arr.shape
    assert wire_dtype.encode() in blob


def test_params_roundtrip_uncompressed(tmp_path):
    params = {
        "diffuser/evoformer": {"weights": np.ones((2, 3), dtype=np.float32)},
        "diffuser/confidence_head": {
            "weights": _bf16_zeros((4,)),
            "bias": np.arange(4, dtype=np.float32),
        },
    }
    path = tmp_path / "af3.bin"
    record_io.write_params(path, params)

    loaded = record_io.read_params(path)
    assert set(loaded) == set(params)
    for scope, entries in params.items():
        assert set(loaded[scope]) == set(entries)
        for name, arr in entries.items():
            assert loaded[scope][name].tobytes() == arr.tobytes()
            assert loaded[scope][name].shape == arr.shape


@pytest.mark.skipif(not record_io.HAS_ZSTANDARD, reason="zstandard not installed")
def test_params_roundtrip_compressed(tmp_path):
    params = {"scope": {"weights": np.arange(24, dtype=np.float32).reshape(2, 3, 4)}}
    path = tmp_path / "af3.bin.zst"
    record_io.write_params(path, params)

    assert path.read_bytes()[:4] == b"\x28\xb5\x2f\xfd"  # zstd magic
    loaded = record_io.read_params(path)
    np.testing.assert_array_equal(loaded["scope"]["weights"], params["scope"]["weights"])


def test_read_params_from_sharded_directory(tmp_path):
    first = {"scope_a": {"weights": np.ones((2,), dtype=np.float32)}}
    second = {"scope_b": {"weights": np.full((3,), 2.0, dtype=np.float32)}}
    record_io.write_params(tmp_path / "af3.0.bin", first)
    record_io.write_params(tmp_path / "af3.1.bin", second)

    loaded = record_io.read_params(tmp_path)
    assert set(loaded) == {"scope_a", "scope_b"}
    np.testing.assert_array_equal(loaded["scope_b"]["weights"], second["scope_b"]["weights"])


def test_select_model_files_prefers_compressed_shards(tmp_path):
    (tmp_path / "af3.bin").write_bytes(b"")
    (tmp_path / "af3.0.bin.zst").write_bytes(b"")
    (tmp_path / "af3.1.bin.zst").write_bytes(b"")

    files, is_compressed = record_io.select_model_files(tmp_path)
    assert is_compressed is True
    assert [f.name for f in files] == ["af3.0.bin.zst", "af3.1.bin.zst"]

    files, is_compressed = record_io.select_model_files(tmp_path, model_name="af3")
    assert [f.name for f in files] == ["af3.0.bin.zst", "af3.1.bin.zst"]


def test_select_model_files_single_uncompressed(tmp_path):
    (tmp_path / "af3.bin").write_bytes(b"")
    files, is_compressed = record_io.select_model_files(tmp_path)
    assert is_compressed is False
    assert [f.name for f in files] == ["af3.bin"]


def test_truncated_record_raises(tmp_path):
    params = {"scope": {"weights": np.ones((4,), dtype=np.float32)}}
    path = tmp_path / "af3.bin"
    record_io.write_params(path, params)
    path.write_bytes(path.read_bytes()[:-4])

    with pytest.raises(record_io.RecordError):
        record_io.read_params(path)


def test_empty_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        record_io.select_model_files(tmp_path)
    with pytest.raises(FileNotFoundError):
        record_io.read_params(tmp_path)


def test_flatten_unflatten_roundtrip():
    params = {
        "scope/a": {"weights": np.ones((2,), dtype=np.float32)},
        "scope/b": {"scale": np.zeros((3,), dtype=np.float32)},
    }
    flat = record_io.flatten(params)
    assert set(flat) == {"scope/a:weights", "scope/b:scale"}

    restored = record_io.unflatten(flat)
    assert set(restored) == set(params)
    np.testing.assert_array_equal(restored["scope/a"]["weights"], params["scope/a"]["weights"])
