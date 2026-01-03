"""Tests for the Factory Pattern.

The Factory Pattern allows creating objects from registered artifacts
using configuration data. It builds on the Registry Pattern by adding
instantiation capabilities.

Tests cover:
- Basic factory instantiation
- Parameter validation via Pydantic
- Auto-extraction of params_model from signatures
- Explicit params_model specification
- Type coercion
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from pydantic import BaseModel

from registry import (
    BuildCfg,
    ContainerMixin,
    FunctionalRegistry,
    TypeRegistry,
    ValidationError,
)


class TestFactoryBasics:
    """Tests for basic factory functionality."""

    def test_simple_instantiation(self, fresh_type_registry, container_cleanup):
        """Test simple object instantiation from config."""

        @fresh_type_registry.register_artifact
        class SimpleClass:
            def __init__(self, value: int):
                self.value = value

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        cfg = BuildCfg(type="SimpleClass", data={"value": 42})
        obj = ContainerMixin.build_cfg(cfg)

        assert isinstance(obj, SimpleClass)
        assert obj.value == 42

    def test_default_values(self, fresh_type_registry, container_cleanup):
        """Test that default parameter values are used."""

        @fresh_type_registry.register_artifact
        class WithDefaults:
            def __init__(self, required: int, optional: str = "default"):
                self.required = required
                self.optional = optional

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        cfg = BuildCfg(type="WithDefaults", data={"required": 10})
        obj = ContainerMixin.build_cfg(cfg)

        assert obj.required == 10
        assert obj.optional == "default"

    def test_function_factory(self, fresh_func_registry, container_cleanup):
        """Test factory with function artifacts."""

        @fresh_func_registry.register_artifact
        def create_dict(a: int, b: str) -> dict:
            return {"a": a, "b": b}

        ContainerMixin.configure_repos({"default": fresh_func_registry})

        cfg = BuildCfg(type="create_dict", data={"a": 1, "b": "hello"})
        result = ContainerMixin.build_cfg(cfg)

        assert result == {"a": 1, "b": "hello"}


class TestParamsModel:
    """Tests for params_model (Pydantic schema) functionality."""

    def test_auto_extraction(self, fresh_type_registry, container_cleanup):
        """Test auto-extraction of params_model from signature."""

        @fresh_type_registry.register_artifact
        class AutoExtracted:
            def __init__(self, x: int, y: str = "default"):
                self.x = x
                self.y = y

        # Verify scheme was auto-extracted
        scheme = fresh_type_registry._get_scheme("AutoExtracted")
        assert scheme is not None
        assert "x" in scheme.model_fields
        assert "y" in scheme.model_fields

    def test_explicit_params_model(self, fresh_type_registry, container_cleanup):
        """Test explicit params_model specification."""

        class MyParams(BaseModel):
            value: int
            name: str = "unnamed"

        @fresh_type_registry.register_artifact(params_model=MyParams)
        class WithExplicitParams:
            def __init__(self, value: int, name: str = "unnamed"):
                self.value = value
                self.name = name

        scheme = fresh_type_registry._get_scheme("WithExplicitParams")
        assert scheme is MyParams

    def test_type_coercion(self, fresh_type_registry, container_cleanup):
        """Test that Pydantic coerces types."""

        class StrictParams(BaseModel):
            x: int
            y: float

        @fresh_type_registry.register_artifact(params_model=StrictParams)
        class CoercedClass:
            def __init__(self, x: int, y: float):
                self.x = x
                self.y = y

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        # Pass strings - should be coerced to int/float
        cfg = BuildCfg(type="CoercedClass", data={"x": "42", "y": "3.14"})
        obj = ContainerMixin.build_cfg(cfg)

        assert obj.x == 42
        assert isinstance(obj.x, int)
        assert obj.y == 3.14
        assert isinstance(obj.y, float)

    def test_missing_required_field(self, fresh_type_registry, container_cleanup):
        """Test that missing required fields raise an error."""

        class StrictParams(BaseModel):
            x: int  # required, no default

        @fresh_type_registry.register_artifact(params_model=StrictParams)
        class StrictClass:
            def __init__(self, x: int):
                self.x = x

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        # Missing required field 'x'
        cfg = BuildCfg(type="StrictClass", data={})

        # Should fail because x is required
        with pytest.raises((ValidationError, TypeError)):
            ContainerMixin.build_cfg(cfg)


class TestExtrasHandling:
    """Tests for handling of extra/unknown fields."""

    def test_extras_go_to_meta(self, fresh_type_registry, container_cleanup):
        """Test that unknown fields go to meta._unused_data."""

        class KnownParams(BaseModel):
            x: int

        @fresh_type_registry.register_artifact(params_model=KnownParams)
        class OnlyX:
            def __init__(self, x: int):
                self.x = x

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        cfg = BuildCfg(
            type="OnlyX",
            data={"x": 10, "unknown1": "extra", "unknown2": 123},
            meta={"original": "preserved"},
        )
        obj = ContainerMixin.build_cfg(cfg)

        assert obj.x == 10
        meta = getattr(obj, "__meta__", {})
        assert meta["original"] == "preserved"
        assert "_unused_data" in meta
        assert meta["_unused_data"]["unknown1"] == "extra"
        assert meta["_unused_data"]["unknown2"] == 123

    def test_buildcfg_extra_keys_to_meta(self):
        """Test that extra top-level keys in BuildCfg go to meta."""
        cfg = BuildCfg(
            type="test",
            data={"x": 1},
            extra_top_level="should_move",
            another_extra=42,
        )

        assert "_extra_cfg_keys" in cfg.meta
        assert cfg.meta["_extra_cfg_keys"]["extra_top_level"] == "should_move"
        assert cfg.meta["_extra_cfg_keys"]["another_extra"] == 42


class TestMultiRepo:
    """Tests for multi-repository factory support."""

    def test_repo_selection(self, multi_registry_setup):
        """Test that repo parameter selects correct registry."""
        cfg_model = BuildCfg(type="SimpleModel", repo="models", data={"num_classes": 5})
        model = ContainerMixin.build_cfg(cfg_model)
        assert model.num_classes == 5

        cfg_transform = BuildCfg(type="to_tensor", repo="transforms", data={})
        transform = ContainerMixin.build_cfg(cfg_transform)
        assert transform["type"] == "ToTensor"

    def test_unknown_repo_fails(self, multi_registry_setup):
        """Test that unknown repo raises error."""
        cfg = BuildCfg(type="anything", repo="nonexistent", data={})
        with pytest.raises(ValidationError):
            ContainerMixin.build_cfg(cfg)
