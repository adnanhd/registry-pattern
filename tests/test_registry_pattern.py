"""Tests for the Registry Pattern.

The Registry Pattern provides a central location for storing and retrieving
artifacts (classes, functions, objects) by name/key.

Tests cover:
- Basic registration and retrieval
- Decorator-based registration
- Module scanning
- Strict mode (inheritance/protocol checking)
- Error handling
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pytest

from registry import (
    ConformanceError,
    FunctionalRegistry,
    InheritanceError,
    RegistryError,
    TypeRegistry,
    ValidationError,
)


class TestTypeRegistry:
    """Tests for TypeRegistry (class registry)."""

    def test_register_and_retrieve_class(self, fresh_type_registry):
        """Test basic class registration and retrieval."""

        @fresh_type_registry.register_artifact
        class MyClass:
            pass

        assert fresh_type_registry.has_identifier("MyClass")
        retrieved = fresh_type_registry.get_artifact("MyClass")
        assert retrieved is MyClass

    def test_register_with_decorator(self, fresh_type_registry):
        """Test decorator-based registration."""

        @fresh_type_registry.register_artifact
        class MyClass:
            pass

        assert fresh_type_registry.has_identifier("MyClass")
        assert fresh_type_registry.get_artifact("MyClass") is MyClass

    def test_duplicate_registration_fails(self, fresh_type_registry):
        """Test that duplicate registration raises error."""

        @fresh_type_registry.register_artifact
        class MyClass:
            pass

        with pytest.raises(RegistryError):

            @fresh_type_registry.register_artifact
            class MyClass:  # noqa: F811
                pass

    def test_retrieve_nonexistent_fails(self, fresh_type_registry):
        """Test that retrieving non-existent key raises error."""
        with pytest.raises(RegistryError):
            fresh_type_registry.get_artifact("nonexistent")

    def test_has_identifier(self, fresh_type_registry):
        """Test has_identifier method."""
        assert not fresh_type_registry.has_identifier("MyClass")

        @fresh_type_registry.register_artifact
        class MyClass:
            pass

        assert fresh_type_registry.has_identifier("MyClass")

    def test_iter_identifiers(self, fresh_type_registry):
        """Test iteration over registered identifiers."""

        @fresh_type_registry.register_artifact
        class ClassA:
            pass

        @fresh_type_registry.register_artifact
        class ClassB:
            pass

        identifiers = list(fresh_type_registry.iter_identifiers())
        assert set(identifiers) == {"ClassA", "ClassB"}

    def test_unregister_artifact(self, fresh_type_registry):
        """Test unregistration of artifacts."""

        @fresh_type_registry.register_artifact
        class MyClass:
            pass

        assert fresh_type_registry.has_identifier("MyClass")
        fresh_type_registry.unregister_identifier("MyClass")
        assert not fresh_type_registry.has_identifier("MyClass")


class TestTypeRegistryStrict:
    """Tests for TypeRegistry with strict mode (inheritance checking)."""

    def test_abstract_mode_requires_inheritance(self):
        """Test that abstract mode requires classes to inherit from registry."""

        class BaseRegistry(TypeRegistry[object], abstract=True):
            pass

        # This should fail - doesn't inherit from BaseRegistry
        class NotASubclass:
            pass

        with pytest.raises(InheritanceError):
            BaseRegistry.register_artifact(NotASubclass)

    def test_strict_mode_protocol_checking(self):
        """Test that strict mode checks protocol conformance."""

        @runtime_checkable
        class MyProtocol(Protocol):
            def do_something(self) -> str: ...

        class StrictRegistry(TypeRegistry[MyProtocol], strict=True):
            pass

        # This should fail - doesn't implement protocol
        class NonConforming:
            pass

        with pytest.raises(ConformanceError):
            StrictRegistry.register_artifact(NonConforming)


class TestFunctionalRegistry:
    """Tests for FunctionalRegistry (function registry)."""

    def test_register_and_retrieve_function(self, fresh_func_registry):
        """Test basic function registration and retrieval."""

        @fresh_func_registry.register_artifact
        def my_func(x: int) -> int:
            return x * 2

        assert fresh_func_registry.has_identifier("my_func")
        retrieved = fresh_func_registry.get_artifact("my_func")
        assert retrieved is my_func
        assert retrieved(5) == 10

    def test_register_named_function(self, fresh_func_registry):
        """Test registration of named functions."""

        @fresh_func_registry.register_artifact
        def named_func(x: int) -> int:
            return x + 1

        assert fresh_func_registry.has_identifier("named_func")
        assert fresh_func_registry.get_artifact("named_func")(5) == 6

    def test_register_non_function_fails(self, fresh_func_registry):
        """Test that non-function objects fail registration."""
        with pytest.raises(ValidationError):
            fresh_func_registry.register_artifact(42)  # not a function

    def test_duplicate_registration_fails(self, fresh_func_registry):
        """Test that duplicate registration raises error."""

        @fresh_func_registry.register_artifact
        def my_func():
            pass

        with pytest.raises(RegistryError):

            @fresh_func_registry.register_artifact
            def my_func():  # noqa: F811
                pass


class TestRegistryValidation:
    """Tests for registry validation features."""

    def test_validate_artifact(self, fresh_type_registry, sample_class):
        """Test artifact validation without registration."""
        validated = fresh_type_registry.validate_artifact(sample_class)
        assert validated is sample_class

    def test_batch_validate(self, fresh_type_registry):
        """Test batch validation of multiple items."""

        class ClassA:
            pass

        class ClassB:
            pass

        results = fresh_type_registry.batch_validate(
            {"a": ClassA, "b": ClassB, "c": "not_a_class"}
        )

        assert results["a"] is True
        assert results["b"] is True
        assert isinstance(results["c"], ValidationError)

    def test_register_multiple_classes(self, fresh_type_registry):
        """Test registering multiple classes."""

        @fresh_type_registry.register_artifact
        class ClassA:
            pass

        @fresh_type_registry.register_artifact
        class ClassB:
            pass

        assert fresh_type_registry.has_identifier("ClassA")
        assert fresh_type_registry.has_identifier("ClassB")
        assert len(list(fresh_type_registry.iter_identifiers())) == 2
