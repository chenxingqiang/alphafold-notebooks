"""Pytest configuration for the fine-tuning test-suite.

Makes the repository root importable so the tests can be run from anywhere via
``pytest finetuning/tests``.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finetuning.af3 import param_groups, schema  # noqa: E402

# Keep generated fixtures small: the real checkpoint holds 368M parameters, which
# is far too large to materialise in a unit test.
_MAX_PARAMS_PER_SPEC = 1_000_000
_SPECS_PER_GROUP = 6


@pytest.fixture(scope="session")
def full_schema():
    """The complete vendored AlphaFold 3 parameter schema (metadata only)."""
    return schema.load_schema()


@pytest.fixture(scope="session")
def small_schema(full_schema):
    """A small schema subset covering every parameter group.

    Deterministic: entries keep their order from ``param_schema.txt``.
    """
    by_group: dict[param_groups.ParamGroup, list[schema.ParamSpec]] = {}
    for spec in full_schema:
        if spec.num_params > _MAX_PARAMS_PER_SPEC:
            continue
        group = param_groups.classify(spec.full_name)
        by_group.setdefault(group, []).append(spec)

    picked: list[schema.ParamSpec] = []
    for group in sorted(by_group, key=lambda g: g.value):
        picked.extend(by_group[group][:_SPECS_PER_GROUP])
    return tuple(picked)


@pytest.fixture
def small_params(small_schema):
    """Random parameters matching ``small_schema``."""
    return schema.generate_random_params(small_schema, seed=1234)
