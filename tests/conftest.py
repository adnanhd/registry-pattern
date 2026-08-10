"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from registry import FunctionalRegistry, TypeRegistry

# Make the example modules (e.g. examples/torch_compat.py) importable by name so
# tests can exercise them the same way a user running the scripts would.
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
if _EXAMPLES_DIR.is_dir() and str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))


@pytest.fixture
def fresh_type_registry():
    """A fresh TypeRegistry subclass scoped to one test."""

    class _TestTypeRegistry(TypeRegistry[Any]):
        pass

    yield _TestTypeRegistry
    _TestTypeRegistry._repository.clear()


@pytest.fixture
def fresh_func_registry():
    """A fresh FunctionalRegistry subclass scoped to one test."""

    class _TestFuncRegistry(FunctionalRegistry[[Any], Any]):
        pass

    yield _TestFuncRegistry
    _TestFuncRegistry._repository.clear()


@pytest.fixture
def sample_class():
    class SampleClass:
        def __init__(self, x: int, y: str = "default"):
            self.x = x
            self.y = y

    return SampleClass


@pytest.fixture
def sample_function():
    def sample_func(a: int, b: str = "hello") -> str:
        return f"{b}: {a}"

    return sample_func


@pytest.fixture
def sample_params_model():
    class SampleParams(BaseModel):
        x: int
        y: str = "default"

    return SampleParams
