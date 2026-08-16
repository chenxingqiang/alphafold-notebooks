"""AlphaFold 3 parameter container codec.

Binary layout matches ``alphafold3.model.params`` in the official repository:
little-endian header ``<5i`` followed by scope, name, dtype string, shape ints,
and raw array bytes.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import struct
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np

try:
    import zstandard

    HAS_ZSTANDARD = True
except ImportError:
    HAS_ZSTANDARD = False


class RecordError(Exception):
    """Raised when a parameter record cannot be read."""


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency (e.g. zstandard) is required but missing."""


def numpy_dtype_for(dtype_name: str) -> np.dtype:
    """Map AlphaFold 3 wire dtype names to numpy dtypes."""
    if dtype_name == "bfloat16":
        return np.dtype(np.uint16)
    return np.dtype(dtype_name)


def encode_record(
    scope: str,
    name: str,
    arr: np.ndarray,
    *,
    wire_dtype: str | None = None,
) -> bytes:
    """Encode one Haiku parameter as an AlphaFold 3 binary record."""
    scope_b = scope.encode("utf-8")
    name_b = name.encode("utf-8")
    dtype_b = (wire_dtype or str(arr.dtype)).encode("utf-8")
    arr = np.ascontiguousarray(arr)
    if sys.byteorder == "big":
        arr = arr.byteswap()
    arr_buffer = arr.tobytes("C")
    shape = arr.shape
    header = struct.pack(
        "<5i",
        len(scope_b),
        len(name_b),
        len(dtype_b),
        len(shape),
        len(arr_buffer),
    )
    return header + b"".join(
        (
            scope_b,
            name_b,
            dtype_b,
            struct.pack(f"{len(shape)}i", *shape),
            arr_buffer,
        )
    )


def _read_record(stream: BinaryIO) -> tuple[str, str, np.ndarray] | None:
    header_size = struct.calcsize("<5i")
    header = stream.read(header_size)
    if not header:
        return None
    if len(header) < header_size:
        raise RecordError(f"Incomplete header: {len(header)} < {header_size}")

    scope_len, name_len, dtype_len, shape_len, arr_buffer_len = struct.unpack(
        "<5i", header
    )
    fmt = f"<{scope_len}s{name_len}s{dtype_len}s{shape_len}i"
    payload_size = struct.calcsize(fmt) + arr_buffer_len
    payload = stream.read(payload_size)
    if len(payload) < payload_size:
        raise RecordError(f"Incomplete payload: {len(payload)} < {payload_size}")

    scope, name, dtype, *shape = struct.unpack_from(fmt, payload)
    scope = scope.decode("utf-8")
    name = name.decode("utf-8")
    dtype = dtype.decode("utf-8")
    arr = np.frombuffer(payload[-arr_buffer_len:], dtype=numpy_dtype_for(dtype))
    arr = np.reshape(arr, shape)
    if sys.byteorder == "big":
        arr = arr.byteswap()
    return scope, name, arr


def read_records(stream: BinaryIO) -> Iterator[tuple[str, str, np.ndarray]]:
    """Yield ``(scope, name, array)`` tuples until EOF."""
    while record := _read_record(stream):
        yield record


def flatten(params: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """``{scope: {name: arr}}`` → ``{scope:name: arr}``."""
    flat: dict[str, np.ndarray] = {}
    for scope, entries in params.items():
        for name, arr in entries.items():
            flat[f"{scope}:{name}"] = arr
    return flat


def unflatten(flat: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    """``{scope:name: arr}`` → ``{scope: {name: arr}}``."""
    params: dict[str, dict[str, np.ndarray]] = {}
    for full_name, arr in flat.items():
        scope, name = full_name.split(":", 1)
        params.setdefault(scope, {})[name] = arr
    return params


def _match_model(paths: list[Path], pattern: re.Pattern[str]) -> dict[str, list[Path]]:
    models: dict[str, list[Path]] = {}
    for path in paths:
        match = pattern.fullmatch(path.name)
        if match:
            models.setdefault(match.group("model_name"), []).append(path)
    return {k: sorted(v) for k, v in models.items()}


def select_model_files(
    model_dir: os.PathLike[str] | str,
    model_name: str | None = None,
) -> tuple[list[Path], bool]:
    """Select parameter files from a model directory (mirrors upstream logic)."""
    model_path = Path(model_dir)
    files = [f for f in model_path.iterdir() if f.is_file()] if model_path.exists() else []

    patterns = (
        (r"(?P<model_name>.*)\.[0-9]+\.bin\.zst$", True),
        (r"(?P<model_name>.*)\.bin\.zst\.[0-9]+$", True),
        (r"(?P<model_name>.*)\.[0-9]+\.bin$", False),
        (r"(?P<model_name>.*)\.bin\.[0-9]+$", False),
        (r"(?P<model_name>.*)\.bin\.zst$", True),
        (r"(?P<model_name>.*)\.bin$", False),
    )
    for pattern, is_compressed in patterns:
        models = _match_model(files, re.compile(pattern))
        if model_name is not None:
            if model_name in models:
                return models[model_name], is_compressed
        elif models:
            if len(models) > 1:
                raise RuntimeError(f"Multiple models matched in {model_path}")
            _, model_files = next(iter(models.items()))
            return model_files, is_compressed

    raise FileNotFoundError(f"No models matched in {model_path}")


@contextlib.contextmanager
def _open_for_reading(model_files: list[Path], is_compressed: bool) -> Iterator[BinaryIO]:
    with contextlib.ExitStack() as stack:
        handles = [stack.enter_context(path.open("rb")) for path in model_files]
        stream: BinaryIO = _MultiFileIO(handles)
        if is_compressed:
            if not HAS_ZSTANDARD:
                raise MissingDependencyError(
                    "Reading .zst parameter files requires the zstandard package."
                )
            buffered = io.BufferedReader(stream)
            yield zstandard.ZstdDecompressor().stream_reader(buffered)
        else:
            yield stream


class _MultiFileIO(io.RawIOBase):
  """Concatenated read-only view of multiple files."""

  def __init__(self, handles: list[BinaryIO]):
    self._handles = handles
    self._sizes: list[int] = []
    for handle in self._handles:
      handle.seek(0, os.SEEK_END)
      self._sizes.append(handle.tell())
    self._length = sum(self._sizes)
    self._offsets = [0]
    for size in self._sizes[:-1]:
      self._offsets.append(self._offsets[-1] + size)
    self._abspos = 0
    self._relpos = (0, 0)

  def _abs_to_rel(self, pos: int) -> tuple[int, int]:
    import bisect

    idx = bisect.bisect_right(self._offsets, pos) - 1
    return idx, pos - self._offsets[idx]

  def close(self) -> None:
    for handle in self._handles:
      handle.close()

  @property
  def closed(self) -> bool:
    return all(handle.closed for handle in self._handles)

  def fileno(self) -> int:
    return -1

  def readable(self) -> bool:
    return True

  def tell(self) -> int:
    return self._abspos

  def seek(self, pos: int, whence: int = os.SEEK_SET, /) -> int:
    if whence == os.SEEK_SET:
      pass
    elif whence == os.SEEK_CUR:
      pos += self._abspos
    elif whence == os.SEEK_END:
      pos = self._length - pos
    else:
      raise ValueError(f"Invalid whence: {whence}")
    self._abspos = pos
    self._relpos = self._abs_to_rel(pos)
    return self._abspos

  def readinto(self, b: bytearray | memoryview) -> int:
    result = 0
    mem = memoryview(b)
    while mem:
      handle = self._handles[self._relpos[0]]
      handle.seek(self._relpos[1])
      if hasattr(handle, "readinto"):
        count = handle.readinto(mem)
      else:
        data = handle.read(len(mem))
        count = len(data)
        mem[:count] = data
      result += count
      self._abspos += count
      self._relpos = self._abs_to_rel(self._abspos)
      mem = mem[count:]
      if self._abspos == self._length:
        break
    return result


def read_params(path_or_dir: os.PathLike[str] | str) -> dict[str, dict[str, np.ndarray]]:
    """Read parameters from a file or directory of shard files."""
    path = Path(path_or_dir)
    if path.is_dir():
        model_files, is_compressed = select_model_files(path)
    else:
        model_files = [path]
        is_compressed = path.suffix == ".zst" or path.name.endswith(".bin.zst")

    params: dict[str, dict[str, np.ndarray]] = {}
    with _open_for_reading(model_files, is_compressed) as stream:
        for scope, name, arr in read_records(stream):
            params.setdefault(scope, {})[name] = arr
    if not params:
        raise FileNotFoundError(f"Model missing from {path_or_dir}")
    return params


def write_params(
    path: os.PathLike[str] | str,
    params: dict[str, dict[str, np.ndarray]],
    *,
    compress: bool | None = None,
    wire_dtypes: dict[str, str] | None = None,
) -> None:
    """Write parameters to a binary file (optionally zstd-compressed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if compress is None:
        compress = path.suffix == ".zst" or path.name.endswith(".bin.zst")

    flat = flatten(params)

    def write_records(stream: IO[bytes]) -> None:
        for scope, entries in params.items():
            for name, arr in entries.items():
                full_name = f"{scope}:{name}"
                wire = wire_dtypes.get(full_name) if wire_dtypes else None
                stream.write(encode_record(scope, name, arr, wire_dtype=wire))

    if compress:
        if not HAS_ZSTANDARD:
            raise MissingDependencyError(
                "Writing .zst parameter files requires the zstandard package."
            )
        with zstandard.open(path, "wb") as compressed:
            write_records(compressed)
    else:
        with path.open("wb") as stream:
            write_records(stream)
