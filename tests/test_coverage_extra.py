# test_registry_edge_cases.py

from collections.abc import Sequence
import pytest
from typing import Type
from registry import (
    TypeRegistry,
    FunctionalRegistry,
    ConformanceError,
    InheritanceError,
    RegistryError,
    ValidationError,
)
from typing import Protocol, runtime_checkable, Iterable, SupportsIndex

from registry.api import (
    functional_registry_factory,
    type_registry_decorator,
    type_registry_factory,
)


# Mock classes and functions for testing
@runtime_checkable
class MockProtocol(Protocol):
    def method(self, x: int) -> int:
        ...


class ValidClass:
    def method(self, x: int) -> int:
        return x


class InvalidClass(ValidClass):
    def method2(self, y: int) -> int:
        return y


class StructurallyInvalidClass:
    def some_other_method(self, x: int) -> int:
        return x


def valid_function(x: int) -> int:
    return x


def invalid_function(y: str) -> str:
    return y


def structurally_invalid_function(x: int, y: int) -> int:
    return x + y


def edge_case_function(x: int) -> None:
    if x < 0:
        raise ValueError("Negative value")
    return None


class IncompleteSubclass:
    def method(self, x: int) -> int:
        ...


@pytest.fixture
def type_registry():
    return type_registry_factory("Mock", protocol=MockProtocol)


@pytest.fixture
def type_registry_with_hierachy():
    return type_registry_factory("Mock", ValidClass, protocol=MockProtocol)


@pytest.fixture
def functional_registry():
    return functional_registry_factory("Mock", args=[int], ret=int)


@runtime_checkable
class SequenceProtocol(Protocol):
    __slots__ = ()

    def __getitem__(self, index: SupportsIndex):
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, item) -> bool:
        ...

    def __iter__(self) -> Iterable:
        ...

    def index(self, value, start: int = 0, stop: int = 9223372036854775807) -> int:
        ...

    def count(self, value) -> int:
        ...


@pytest.fixture
def sequence_registry():
    return type_registry_decorator(strict=True, abstract=False)(SequenceProtocol)


# Test cases for edge cases and additional scenarios


def test_unregister_nonexistent_class(type_registry):
    with pytest.raises(RegistryError):
        type_registry.unregister_class(ValidClass)  # Never registered


def test_unregister_nonexistent_function(functional_registry):
    with pytest.raises(RegistryError):
        functional_registry.unregister_function(valid_function)  # Never registered


def test_register_duplicate_class(type_registry):
    type_registry.register_class(ValidClass)
    with pytest.raises(RegistryError):
        type_registry.register_class(ValidClass)  # Duplicate registration


def test_register_duplicate_function(functional_registry):
    functional_registry.register_function(valid_function)
    with pytest.raises(RegistryError):
        functional_registry.register_function(valid_function)  # Duplicate registration


def test_register_structurally_invalid_class(type_registry):
    with pytest.raises(ConformanceError):
        type_registry.register_class(StructurallyInvalidClass)


def test_register_structurally_invalid_function(functional_registry):
    with pytest.raises(ConformanceError):
        functional_registry.register_function(structurally_invalid_function)


def test_clear_registry(type_registry):
    type_registry.register_class(ValidClass)
    type_registry.clear_registry()  # Assuming this method exists
    with pytest.raises(RegistryError):
        type_registry.get_registry_item("ValidClass")


def test_clear_function_registry(functional_registry):
    functional_registry.register_function(valid_function)
    functional_registry.clear_registry()  # Assuming this method exists
    with pytest.raises(RegistryError):
        functional_registry.get_registry_item("valid_function")


def test_register_incomplete_subclass(type_registry_with_hierachy):
    with pytest.raises(InheritanceError):
        type_registry_with_hierachy.register_class(IncompleteSubclass)


def test_register_module_subclasses_with_errors(
    sequence_registry: Type[TypeRegistry[MockProtocol]],
):
    import collections.abc

    # Attempt to register all classes in collections.abc
    with pytest.raises(ValidationError):
        sequence_registry.register_module_subclasses(collections.abc, raise_error=True)

    # Check that valid classes are registered
    assert sequence_registry.len() == 0

    sequence_registry.register_module_subclasses(collections.abc, raise_error=False)
    assert sequence_registry.len() == 3


def test_register_module_functions_with_errors(functional_registry):
    import math

    # Attempt to register all functions in the math module
    with pytest.raises(ValidationError):
        functional_registry.register_module_functions(math)

    # Check that valid functions are registered
    assert functional_registry.get_registry_item("acos") is math.acos
    assert functional_registry.get_registry_item("acosh") is math.acosh
    assert functional_registry.get_registry_item("asin") is math.asin
    assert functional_registry.get_registry_item("asinh") is math.asinh


def test_register_edge_case_function(functional_registry):
    functional_registry.register_function(edge_case_function)
    assert (
        functional_registry.get_registry_item("edge_case_function")
        is edge_case_function
    )

    # Test that the function raises the expected error
    with pytest.raises(ValueError):
        edge_case_function(-1)
