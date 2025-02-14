import inspect
import pytest
import logging

# Import functions and exceptions from your validator module.
# Adjust the import if your module is organized differently.
from registry.utils._validator import (
    validate_class,
    validate_class_structure,
    validate_class_hierarchy,
    validate_init,
    validate_function,
    validate_function_signature,
    validate_function_parameters,
    validate_instance_hierarchy,
    validate_instance_structure,
    ConformanceError,
    InheritanceError,
    ValidationError,
    get_func_name,
)

# -----------------------------------------------------------------------------
# Dummy Protocol and Classes for Testing
# -----------------------------------------------------------------------------


class DummyProtocol:
    def foo(self, x: int) -> int: ...
    def bar(self, y: str) -> str: ...


# A compliant class: implements both foo and bar with matching signatures.
class GoodClass:
    def foo(self, x: int) -> int:
        return x * 2

    def bar(self, y: str) -> str:
        return y.upper()


# A noncompliant class: missing 'bar'
class BadClass:
    def foo(self, x: int) -> int:
        return x * 2


# For inheritance testing, use an abstract base class.
class BaseABC:
    pass


class Derived(BaseABC):
    pass


class NotDerived:
    pass


# Dummy functions for signature testing
def expected_func(a: int, b: str) -> bool:
    return True


def matching_func(a: int, b: str) -> bool:
    return False


def mismatching_func(a: int) -> bool:
    return True


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------


def test_validate_class_success():
    # Should return the class if it's a class.
    cls = validate_class(GoodClass)
    assert inspect.isclass(cls)
    assert cls is GoodClass


def test_validate_class_failure():
    with pytest.raises(ValidationError):
        validate_class(42)  # 42 is not a class


def test_validate_class_structure_success():
    # GoodClass implements both methods required by DummyProtocol.
    cls = validate_class_structure(GoodClass, DummyProtocol)
    assert cls is GoodClass


def test_validate_class_structure_failure():
    with pytest.raises(ConformanceError) as exc_info:
        validate_class_structure(BadClass, DummyProtocol)
    assert "does not implement method 'bar'" in str(exc_info.value)


def test_validate_class_hierarchy_success():
    # Derived is a subclass of BaseABC.
    cls = validate_class_hierarchy(Derived, BaseABC)
    assert cls is Derived


def test_validate_class_hierarchy_failure():
    with pytest.raises(InheritanceError):
        validate_class_hierarchy(NotDerived, BaseABC)


def test_validate_init():
    # validate_init is a pass-through function.
    result = validate_init(GoodClass)
    assert result is GoodClass


def test_validate_function_success():
    func = validate_function(matching_func)
    assert inspect.isfunction(func)
    assert func is matching_func


def test_validate_function_failure():
    with pytest.raises(ValidationError):
        validate_function(123)  # Not a function


def test_validate_function_signature_success():
    # Both expected_func and matching_func share the same signature.
    func = validate_function_signature(matching_func, expected_func)
    assert func is matching_func


def test_validate_function_signature_failure():
    with pytest.raises(ConformanceError) as exc_info:
        validate_function_signature(mismatching_func, expected_func)
    assert "does not match argument count" in str(exc_info.value)


def test_validate_function_parameters_success():
    from typing import Callable

    ExpectedType = Callable[[int, str], bool]
    func = validate_function_parameters(matching_func, ExpectedType)
    assert func is matching_func


def test_validate_function_parameters_failure():
    from typing import Callable

    ExpectedType = Callable[[int, str], bool]
    with pytest.raises(ConformanceError) as exc_info:
        validate_function_parameters(mismatching_func, ExpectedType)
    assert "has 1 parameters" in str(exc_info.value)


def test_validate_instance_hierarchy_success():
    instance = GoodClass()
    validated = validate_instance_hierarchy(instance, GoodClass)
    assert validated is instance


def test_validate_instance_hierarchy_failure():
    instance = 42
    with pytest.raises(ValidationError):
        validate_instance_hierarchy(instance, str)


# -----------------------------------------------------------------------------
# New Tests for validate_instance_structure (without typeguard)
# -----------------------------------------------------------------------------


def test_validate_instance_structure_success():
    # With the new implementation, validate_instance_structure uses isinstance().
    instance = GoodClass()
    validated = validate_instance_structure(instance, GoodClass)
    assert validated is instance


def test_validate_instance_structure_failure():
    # In this test, GoodClass instance is not an instance of int.
    instance = GoodClass()
    with pytest.raises(ConformanceError) as exc_info:
        validate_instance_structure(instance, int)
    assert "does not implement attribute" in str(exc_info.value)
