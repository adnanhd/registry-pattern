#!/usr/bin/env python
"""Tests for the Pydantic type guard (Buildable[T]).

This module tests the Buildable type annotation that allows Pydantic models
to accept either an instance of T or a BuildCfg that gets built into T.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from registry import Buildable, BuildCfg, TypeRegistry

# =============================================================================
# Test Registries and Components
# =============================================================================


class ComponentRegistry(TypeRegistry[Any], repo="components"):
    pass


class ServiceRegistry(TypeRegistry[Any], repo="services"):
    pass


@pytest.fixture(autouse=True)
def setup_registries():
    """Register a fresh set of test components for each test."""
    ComponentRegistry.clear_artifacts()
    ServiceRegistry.clear_artifacts()

    @ComponentRegistry.register_artifact
    class SimpleComponent:
        def __init__(self, value: int, name: str = "default"):
            self.value = value
            self.name = name

    @ComponentRegistry.register_artifact
    class NestedComponent:
        def __init__(self, child: object, multiplier: int = 1):
            self.child = child
            self.multiplier = multiplier

    @ServiceRegistry.register_artifact
    class SimpleService:
        def __init__(self, endpoint: str, timeout: int = 30):
            self.endpoint = endpoint
            self.timeout = timeout

    yield


# =============================================================================
# Test: Basic Buildable Usage
# =============================================================================


class TestBuildableBasics:
    """Basic tests for Buildable type annotation."""

    def test_accepts_direct_instance(self):
        """Buildable[T] accepts an already-constructed instance of T."""
        # Get the component class from registry
        SimpleComponent = ComponentRegistry.get_artifact("SimpleComponent")

        class Config(BaseModel):
            component: Buildable[object]

        # Pass a direct instance
        instance = SimpleComponent(value=42, name="direct")
        config = Config(component=instance)

        assert config.component is instance
        assert config.component.value == 42
        assert config.component.name == "direct"

    def test_accepts_buildcfg_instance(self):
        """Buildable[T] accepts a BuildCfg and builds it."""

        class Config(BaseModel):
            component: Buildable[object]

        cfg = BuildCfg(
            type="SimpleComponent",
            repo="components",
            data={"value": 100, "name": "from_cfg"},
        )

        config = Config(component=cfg)

        assert config.component.value == 100
        assert config.component.name == "from_cfg"

    def test_accepts_dict_config(self):
        """Buildable[T] accepts a dict that looks like BuildCfg."""

        class Config(BaseModel):
            component: Buildable[object]

        config = Config(
            component={
                "type": "SimpleComponent",
                "repo": "components",
                "data": {"value": 200},
            }
        )

        assert config.component.value == 200
        assert config.component.name == "default"

    def test_rejects_invalid_value(self):
        """Buildable[T] rejects values that aren't instances or configs."""
        # Get a specific type to test rejection
        SimpleComponent = ComponentRegistry.get_artifact("SimpleComponent")

        class Config(BaseModel):
            component: Buildable[SimpleComponent]

        # String is not a SimpleComponent or config
        with pytest.raises(PydanticValidationError):
            Config(component="not a valid value")

        # Integer is not a SimpleComponent or config
        with pytest.raises(PydanticValidationError):
            Config(component=123)

        # List is not a SimpleComponent or config
        with pytest.raises(PydanticValidationError):
            Config(component=["list", "of", "values"])


class TestBuildableWithTypeChecking:
    """Tests for Buildable with specific type constraints."""

    def test_specific_type_accepts_matching_instance(self):
        """Buildable[SpecificType] accepts instances of that type."""
        SimpleComponent = ComponentRegistry.get_artifact("SimpleComponent")

        class Config(BaseModel):
            component: Buildable[SimpleComponent]

        instance = SimpleComponent(value=10)
        config = Config(component=instance)

        assert isinstance(config.component, SimpleComponent)

    def test_specific_type_builds_from_config(self):
        """Buildable[SpecificType] builds from config and returns that type."""
        SimpleComponent = ComponentRegistry.get_artifact("SimpleComponent")

        class Config(BaseModel):
            component: Buildable[SimpleComponent]

        config = Config(component={"type": "SimpleComponent", "data": {"value": 50}})

        assert isinstance(config.component, SimpleComponent)
        assert config.component.value == 50


class TestBuildableNested:
    """Tests for nested Buildable fields."""

    def test_nested_buildable_fields(self):
        """Model with multiple Buildable fields, some nested."""

        class AppConfig(BaseModel):
            main_component: Buildable[object]
            backup_component: Optional[Buildable[object]] = None

        config = AppConfig(
            main_component={
                "type": "NestedComponent",
                "repo": "components",
                "data": {
                    "child": {"type": "SimpleComponent", "data": {"value": 1}},
                    "multiplier": 5,
                },
            },
            backup_component={"type": "SimpleComponent", "data": {"value": 999}},
        )

        # Main component is NestedComponent with a child
        assert config.main_component.multiplier == 5
        assert config.main_component.child.value == 1

        # Backup is a simple component
        assert config.backup_component.value == 999

    def test_optional_buildable_none(self):
        """Optional Buildable field can be None."""

        class Config(BaseModel):
            required: Buildable[object]
            optional: Optional[Buildable[object]] = None

        config = Config(required={"type": "SimpleComponent", "data": {"value": 1}})

        assert config.required.value == 1
        assert config.optional is None

    def test_list_of_buildable(self):
        """List of Buildable items."""
        from typing import List

        class Config(BaseModel):
            components: List[Buildable[object]]

        config = Config(
            components=[
                {"type": "SimpleComponent", "data": {"value": 1}},
                {"type": "SimpleComponent", "data": {"value": 2}},
                {"type": "SimpleComponent", "data": {"value": 3}},
            ]
        )

        assert len(config.components) == 3
        for i, comp in enumerate(config.components):
            assert comp.value == i + 1


class TestBuildableWithMeta:
    """Tests for Buildable preserving meta information."""

    def test_meta_preserved_on_built_object(self):
        """Meta from BuildCfg is attached to built object."""

        class Config(BaseModel):
            component: Buildable[object]

        config = Config(
            component={
                "type": "SimpleComponent",
                "data": {"value": 42},
                "meta": {"version": "1.0", "author": "test"},
            }
        )

        meta = getattr(config.component, "__meta__", {})
        assert meta.get("version") == "1.0"
        assert meta.get("author") == "test"

class TestBuildableTypeCoercion:
    """Tests for type coercion through Buildable."""

    def test_string_to_int_coercion(self):
        """String values are coerced to appropriate types."""

        class Config(BaseModel):
            component: Buildable[object]

        config = Config(
            component={
                "type": "SimpleComponent",
                "data": {"value": "42"},  # String, should be coerced to int
            }
        )

        assert config.component.value == 42
        assert isinstance(config.component.value, int)


class TestBuildableWithMultipleRepos:
    """Tests for Buildable with different repository namespaces."""

    def test_different_repos(self):
        """Buildable works with different repository namespaces."""

        class Config(BaseModel):
            component: Buildable[object]
            service: Buildable[object]

        config = Config(
            component={
                "type": "SimpleComponent",
                "repo": "components",
                "data": {"value": 1},
            },
            service={
                "type": "SimpleService",
                "repo": "services",
                "data": {"endpoint": "/api/v1"},
            },
        )

        assert config.component.value == 1
        assert config.service.endpoint == "/api/v1"


class TestBuildableContext:
    """Tests for Buildable with context injection."""

class TestBuildableErrorHandling:
    """Tests for error handling in Buildable."""

    def test_invalid_type_name_raises(self):
        """Building with unknown type name raises error."""

        class Config(BaseModel):
            component: Buildable[object]

        with pytest.raises(PydanticValidationError) as exc_info:
            Config(component={"type": "NonExistentComponent", "data": {}})

        # Should contain information about the failure
        assert "NonExistentComponent" in str(
            exc_info.value
        ) or "Failed to build" in str(exc_info.value)

    def test_invalid_repo_raises(self):
        """Building with unknown repo raises error."""

        class Config(BaseModel):
            component: Buildable[object]

        with pytest.raises(PydanticValidationError):
            Config(
                component={
                    "type": "SimpleComponent",
                    "repo": "nonexistent_repo",
                    "data": {"value": 1},
                }
            )

    def test_missing_required_field_raises(self):
        """Building without required field raises error."""

        class Config(BaseModel):
            component: Buildable[object]

        with pytest.raises(PydanticValidationError):
            Config(
                component={
                    "type": "SimpleComponent",
                    "data": {},  # Missing required 'value' field
                }
            )


class TestBuildableMixedInput:
    """Tests for mixing instances and configs in same model."""

    def test_mix_instance_and_config(self):
        """Model can have some fields as instances, others as configs."""
        SimpleComponent = ComponentRegistry.get_artifact("SimpleComponent")

        class Config(BaseModel):
            instance_field: Buildable[object]
            config_field: Buildable[object]

        direct_instance = SimpleComponent(value=1, name="instance")

        config = Config(
            instance_field=direct_instance,
            config_field={
                "type": "SimpleComponent",
                "data": {"value": 2, "name": "from_config"},
            },
        )

        assert config.instance_field is direct_instance
        assert config.instance_field.value == 1
        assert config.config_field.value == 2
        assert config.config_field.name == "from_config"
