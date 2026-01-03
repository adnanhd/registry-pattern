"""Pytest configuration and shared fixtures."""

from __future__ import annotations

from typing import Any, Dict, Optional

import pytest
from pydantic import BaseModel

from registry import (
    BuildCfg,
    ContainerMixin,
    FunctionalRegistry,
    TypeRegistry,
    ValidationError,
)

# =============================================================================
# Fixtures: Fresh Registry Classes
# =============================================================================


@pytest.fixture
def fresh_type_registry():
    """Create a fresh TypeRegistry subclass for each test."""

    class TestTypeRegistry(TypeRegistry[object]):
        pass

    yield TestTypeRegistry
    # Cleanup
    TestTypeRegistry._repository.clear()
    TestTypeRegistry._schemetry.clear()


@pytest.fixture
def fresh_func_registry():
    """Create a fresh FunctionalRegistry subclass for each test."""

    class TestFuncRegistry(FunctionalRegistry[[Any], Any]):
        pass

    yield TestFuncRegistry
    # Cleanup
    TestFuncRegistry._repository.clear()
    TestFuncRegistry._schemetry.clear()


@pytest.fixture
def container_cleanup():
    """Clean up ContainerMixin state after each test."""
    yield
    ContainerMixin._repos.clear()
    ContainerMixin._ctx.clear()


# =============================================================================
# Fixtures: Sample Classes and Functions
# =============================================================================


@pytest.fixture
def sample_class():
    """A simple sample class for testing."""

    class SampleClass:
        def __init__(self, x: int, y: str = "default"):
            self.x = x
            self.y = y

        def __repr__(self):
            return f"SampleClass(x={self.x}, y={self.y!r})"

    return SampleClass


@pytest.fixture
def sample_function():
    """A simple sample function for testing."""

    def sample_func(a: int, b: str = "hello") -> str:
        return f"{b}: {a}"

    return sample_func


@pytest.fixture
def sample_params_model():
    """A sample Pydantic params model."""

    class SampleParams(BaseModel):
        x: int
        y: str = "default"

    return SampleParams


# =============================================================================
# Fixtures: Multi-Registry Setup
# =============================================================================


@pytest.fixture
def multi_registry_setup(container_cleanup):
    """Set up multiple registries for DI container testing."""

    class ModelRegistry(TypeRegistry[object]):
        pass

    class TransformRegistry(FunctionalRegistry[[Any], Any]):
        pass

    class UtilsRegistry(FunctionalRegistry[[Any], Any]):
        pass

    # Register some artifacts
    @ModelRegistry.register_artifact
    class SimpleModel:
        def __init__(self, num_classes: int = 10):
            self.num_classes = num_classes

        def __repr__(self):
            return f"SimpleModel(num_classes={self.num_classes})"

    @TransformRegistry.register_artifact
    def normalize(mean: tuple, std: tuple):
        return {"type": "Normalize", "mean": mean, "std": std}

    @TransformRegistry.register_artifact
    def to_tensor():
        return {"type": "ToTensor"}

    @UtilsRegistry.register_artifact
    def ref(key: str, ctx: Dict[str, Any]):
        return ctx[key]

    # Configure repos
    ContainerMixin.configure_repos(
        {
            "models": ModelRegistry,
            "transforms": TransformRegistry,
            "utils": UtilsRegistry,
            "default": UtilsRegistry,
        }
    )

    return {
        "ModelRegistry": ModelRegistry,
        "TransformRegistry": TransformRegistry,
        "UtilsRegistry": UtilsRegistry,
        "SimpleModel": SimpleModel,
    }
