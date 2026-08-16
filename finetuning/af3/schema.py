"""AlphaFold 3 parameter schema parsing, validation, and random weight generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np

from . import record_io

DEFAULT_SCHEMA_PATH = Path(__file__).with_name("param_schema.txt")
_SCHEMA_LINE_RE = re.compile(
    r"^name=(?P<full>[^\s]+)\s+dtype=(?P<dtype>\S+)\s+shape=\((?P<shape>[^)]*)\)$"
)


@dataclass(frozen=True)
class ParamSpec:
    scope: str
    name: str
    shape: tuple[int, ...]
    dtype: str

    @property
    def full_name(self) -> str:
        return f"{self.scope}:{self.name}"

    @property
    def numpy_dtype(self) -> np.dtype:
        return record_io.numpy_dtype_for(self.dtype)

    @property
    def num_params(self) -> int:
        n = 1
        for dim in self.shape:
            n *= dim
        return n


@dataclass(frozen=True)
class ValidationReport:
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatch: tuple[tuple[str, tuple[int, ...], tuple[int, ...]], ...]
    dtype_mismatch: tuple[tuple[str, str, str], ...]

    @property
    def ok(self) -> bool:
        return not (
            self.missing
            or self.unexpected
            or self.shape_mismatch
            or self.dtype_mismatch
        )

    def describe(self) -> str:
        if self.ok:
            return "OK: parameters match the schema."
        parts = ["Parameter validation failed:"]
        if self.missing:
            parts.append(f"  missing ({len(self.missing)}): {self.missing[:5]}")
        if self.unexpected:
            parts.append(f"  unexpected ({len(self.unexpected)}): {self.unexpected[:5]}")
        if self.shape_mismatch:
            parts.append(f"  shape mismatch ({len(self.shape_mismatch)})")
        if self.dtype_mismatch:
            parts.append(f"  dtype mismatch ({len(self.dtype_mismatch)})")
        return "\n".join(parts)


@dataclass(frozen=True)
class SchemaSummary:
    num_entries: int
    num_params: int
    params_by_dtype: dict[str, int]


def parse_schema_line(line: str) -> ParamSpec:
    match = _SCHEMA_LINE_RE.match(line.strip())
    if not match:
        raise ValueError(f"Invalid schema line: {line!r}")
    full = match.group("full")
    scope, name = full.split(":", 1)
    shape_text = match.group("shape").replace(" ", "")
    shape = tuple(int(x) for x in shape_text.split(",") if x)
    return ParamSpec(scope=scope, name=name, shape=shape, dtype=match.group("dtype"))


def load_schema(path: PathLike[str] | str | None = None) -> tuple[ParamSpec, ...]:
    schema_path = Path(path) if path is not None else DEFAULT_SCHEMA_PATH
    specs: list[ParamSpec] = []
    with schema_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            specs.append(parse_schema_line(line))
    return tuple(specs)


def schema_index(schema: tuple[ParamSpec, ...]) -> dict[str, ParamSpec]:
    return {spec.full_name: spec for spec in schema}


def validate_params(
    params: dict[str, dict[str, np.ndarray]],
    schema: tuple[ParamSpec, ...] | None = None,
) -> ValidationReport:
    schema = schema or load_schema()
    expected = schema_index(schema)
    actual = record_io.flatten(params)

    missing = tuple(name for name in expected if name not in actual)
    unexpected = tuple(name for name in actual if name not in expected)

    shape_mismatch: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    dtype_mismatch: list[tuple[str, str, str]] = []
    for name, spec in expected.items():
        if name not in actual:
            continue
        arr = actual[name]
        if tuple(arr.shape) != spec.shape:
            shape_mismatch.append((name, spec.shape, tuple(arr.shape)))
        if str(arr.dtype) != str(spec.numpy_dtype):
            dtype_mismatch.append((name, spec.dtype, str(arr.dtype)))

    return ValidationReport(
        missing=missing,
        unexpected=unexpected,
        shape_mismatch=tuple(shape_mismatch),
        dtype_mismatch=tuple(dtype_mismatch),
    )


def generate_random_params(
    schema: tuple[ParamSpec, ...] | None = None,
    *,
    seed: int = 0,
    scopes: tuple[str, ...] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Generate random parameters matching the schema (for testing / dry runs)."""
    schema = schema or load_schema()
    rng = np.random.default_rng(seed)
    params: dict[str, dict[str, np.ndarray]] = {}

    for spec in schema:
        if scopes is not None and spec.scope not in scopes:
            continue
        if spec.full_name == "__meta__:__identifier__":
            arr = np.zeros(spec.shape, dtype=spec.numpy_dtype)
        elif spec.dtype == "bfloat16":
            # Store as uint16 wire bytes; avoid all-zero payloads for accelerator tests.
            arr = rng.integers(1, 65535, size=spec.shape, dtype=np.uint16)
        else:
            arr = rng.uniform(-1.0, 1.0, size=spec.shape).astype(spec.numpy_dtype)
        params.setdefault(spec.scope, {})[spec.name] = arr
    return params


def summarize(schema: tuple[ParamSpec, ...] | None = None) -> SchemaSummary:
    schema = schema or load_schema()
    by_dtype: dict[str, int] = {}
    total = 0
    for spec in schema:
        by_dtype[spec.dtype] = by_dtype.get(spec.dtype, 0) + spec.num_params
        total += spec.num_params
    return SchemaSummary(
        num_entries=len(schema),
        num_params=total,
        params_by_dtype=by_dtype,
    )
