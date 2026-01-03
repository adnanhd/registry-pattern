"""Tests for the Dependency Injection (DI) Container Pattern.

The DI Container extends the Factory Pattern with:
- Recursive object graph construction from nested configs
- Context injection for cross-references between objects
- Full IoC (Inversion of Control) capabilities

Tests cover:
- Nested config resolution
- Context injection (ctx parameter)
- Cross-references between built objects
- Complex object graphs
- BuildCfg normalization
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel

from registry import (
    BuildCfg,
    ContainerMixin,
    FunctionalRegistry,
    TypeRegistry,
    ValidationError,
    is_build_cfg,
    normalize_cfg,
)


class TestNestedConfigs:
    """Tests for nested configuration resolution."""

    def test_nested_dict_config(self, multi_registry_setup):
        """Test that nested dict configs are recursively built."""
        cfg = {
            "type": "normalize",
            "repo": "transforms",
            "data": {"mean": (0.5,), "std": (0.5,)},
        }
        result = ContainerMixin.build_cfg(cfg)
        assert result["type"] == "Normalize"
        assert result["mean"] == (0.5,)

    def test_deeply_nested_configs(self, container_cleanup):
        """Test deeply nested object graph construction."""

        class OuterRegistry(TypeRegistry[object]):
            pass

        class InnerRegistry(TypeRegistry[object]):
            pass

        @InnerRegistry.register_artifact
        class Inner:
            def __init__(self, value: int):
                self.value = value

        @OuterRegistry.register_artifact
        class Outer:
            def __init__(self, inner: Any, name: str):
                self.inner = inner
                self.name = name

        ContainerMixin.configure_repos(
            {"outer": OuterRegistry, "inner": InnerRegistry, "default": OuterRegistry}
        )

        cfg = BuildCfg(
            type="Outer",
            repo="outer",
            data={
                "name": "outer_instance",
                "inner": {
                    "type": "Inner",
                    "repo": "inner",
                    "data": {"value": 42},
                },
            },
        )

        result = ContainerMixin.build_cfg(cfg)
        assert result.name == "outer_instance"
        assert isinstance(result.inner, Inner)
        assert result.inner.value == 42

    def test_list_of_nested_configs(self, container_cleanup):
        """Test list containing nested configs."""

        class ListRegistry(FunctionalRegistry[[Any], Any]):
            pass

        @ListRegistry.register_artifact
        def item(value: int):
            return {"item": value}

        @ListRegistry.register_artifact
        def container(items: List[Any]):
            return {"container": items}

        ContainerMixin.configure_repos({"default": ListRegistry})

        cfg = BuildCfg(
            type="container",
            data={
                "items": [
                    {"type": "item", "data": {"value": 1}},
                    {"type": "item", "data": {"value": 2}},
                    {"type": "item", "data": {"value": 3}},
                ]
            },
        )

        result = ContainerMixin.build_cfg(cfg)
        assert result["container"] == [
            {"item": 1},
            {"item": 2},
            {"item": 3},
        ]


class TestContextInjection:
    """Tests for context injection (ctx parameter)."""

    def test_build_named_stores_in_context(self, multi_registry_setup):
        """Test that build_named stores objects in context."""
        ContainerMixin.clear_context()

        cfg = BuildCfg(type="SimpleModel", repo="models", data={"num_classes": 10})
        model = ContainerMixin.build_named("my_model", cfg)

        ctx = ContainerMixin.get_context()
        assert "my_model" in ctx
        assert ctx["my_model"] is model

    def test_ctx_injection(self, multi_registry_setup):
        """Test that ctx is injected into builders that accept it."""
        ContainerMixin.clear_context()

        model_cfg = BuildCfg(type="SimpleModel", repo="models", data={"num_classes": 5})
        model = ContainerMixin.build_named("stored_model", model_cfg)

        # Now use ref to retrieve it via ctx
        ref_cfg = BuildCfg(type="ref", repo="utils", data={"key": "stored_model"})
        retrieved = ContainerMixin.build_cfg(ref_cfg)

        assert retrieved is model

    def test_cross_reference_objects(self, container_cleanup):
        """Test cross-references between built objects."""

        class ModelsRegistry(TypeRegistry[object]):
            pass

        class OptimizersRegistry(TypeRegistry[object]):
            pass

        class UtilsRegistry(FunctionalRegistry[[Any], Any]):
            pass

        @ModelsRegistry.register_artifact
        class Model:
            def __init__(self, hidden_size: int):
                self.hidden_size = hidden_size

            def parameters(self):
                return [f"param_{i}" for i in range(self.hidden_size)]

        @UtilsRegistry.register_artifact
        def model_parameters(model_key: str, ctx: Dict[str, Any]):
            return ctx[model_key].parameters()

        @OptimizersRegistry.register_artifact
        class Optimizer:
            def __init__(self, params: Any, lr: float):
                self.params = params
                self.lr = lr

        ContainerMixin.configure_repos(
            {
                "models": ModelsRegistry,
                "optimizers": OptimizersRegistry,
                "utils": UtilsRegistry,
                "default": UtilsRegistry,
            }
        )
        ContainerMixin.clear_context()

        # Build model first
        model = ContainerMixin.build_named(
            "model", {"type": "Model", "repo": "models", "data": {"hidden_size": 3}}
        )

        # Build optimizer with reference to model's parameters
        optimizer = ContainerMixin.build_cfg(
            {
                "type": "Optimizer",
                "repo": "optimizers",
                "data": {
                    "lr": 0.01,
                    "params": {
                        "type": "model_parameters",
                        "repo": "utils",
                        "data": {"model_key": "model"},
                    },
                },
            }
        )

        assert optimizer.lr == 0.01
        assert optimizer.params == ["param_0", "param_1", "param_2"]

    def test_clear_context(self, multi_registry_setup):
        """Test that clear_context removes all stored objects."""
        ContainerMixin.clear_context()

        ContainerMixin.build_named(
            "obj1", {"type": "SimpleModel", "repo": "models", "data": {}}
        )
        ContainerMixin.build_named(
            "obj2", {"type": "SimpleModel", "repo": "models", "data": {}}
        )

        assert len(ContainerMixin.get_context()) == 2

        ContainerMixin.clear_context()
        assert len(ContainerMixin.get_context()) == 0


class TestBuildCfgHelpers:
    """Tests for BuildCfg helper functions."""

    def test_is_build_cfg_with_dict(self):
        """Test is_build_cfg with dict inputs."""
        assert is_build_cfg({"type": "something"})
        assert is_build_cfg({"type": "something", "repo": "default", "data": {}})
        assert not is_build_cfg({"not_type": "something"})
        assert not is_build_cfg({"type": 123})  # type must be str
        assert not is_build_cfg("not a dict")
        assert not is_build_cfg(None)

    def test_is_build_cfg_with_buildcfg(self):
        """Test is_build_cfg with BuildCfg inputs."""
        cfg = BuildCfg(type="test", data={})
        assert is_build_cfg(cfg)

    def test_normalize_cfg_from_dict(self):
        """Test normalize_cfg converts dict to BuildCfg."""
        raw = {"type": "test", "data": {"x": 1}}
        cfg = normalize_cfg(raw)

        assert isinstance(cfg, BuildCfg)
        assert cfg.type == "test"
        assert cfg.data == {"x": 1}
        assert cfg.repo == "default"  # default value
        assert cfg.meta == {}  # default value

    def test_normalize_cfg_passthrough(self):
        """Test normalize_cfg passes through BuildCfg unchanged."""
        original = BuildCfg(type="test", data={"x": 1})
        result = normalize_cfg(original)
        assert result is original

    def test_buildcfg_with_unused_data(self):
        """Test BuildCfg.with_unused_data method."""
        cfg = BuildCfg(type="test", meta={"existing": "value"})
        new_cfg = cfg.with_unused_data({"extra1": 1, "extra2": 2})

        assert cfg.meta == {"existing": "value"}  # original unchanged
        assert new_cfg.meta["existing"] == "value"
        assert new_cfg.meta["_unused_data"] == {"extra1": 1, "extra2": 2}


class TestMetaAttachment:
    """Tests for meta attachment to built objects."""

    def test_meta_attached_to_object(self, fresh_type_registry, container_cleanup):
        """Test that meta is attached to built objects."""

        @fresh_type_registry.register_artifact
        class MetaClass:
            def __init__(self, x: int):
                self.x = x

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        cfg = BuildCfg(
            type="MetaClass",
            data={"x": 10},
            meta={"tag": "test", "version": "1.0"},
        )
        obj = ContainerMixin.build_cfg(cfg)

        assert hasattr(obj, "__meta__")
        assert obj.__meta__["tag"] == "test"
        assert obj.__meta__["version"] == "1.0"

    def test_meta_preserved_with_extras(self, fresh_type_registry, container_cleanup):
        """Test that original meta is preserved when extras are added."""

        class Params(BaseModel):
            x: int

        @fresh_type_registry.register_artifact(params_model=Params)
        class SimpleClass:
            def __init__(self, x: int):
                self.x = x

        ContainerMixin.configure_repos({"default": fresh_type_registry})

        cfg = BuildCfg(
            type="SimpleClass",
            data={"x": 10, "extra_field": "unused"},
            meta={"original": "meta"},
        )
        obj = ContainerMixin.build_cfg(cfg)

        assert obj.__meta__["original"] == "meta"
        assert obj.__meta__["_unused_data"]["extra_field"] == "unused"
