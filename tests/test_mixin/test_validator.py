import pytest
from typing import Protocol

from registry.core._validator import (
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


# -------------------------------------------------------------------
# Test classes and protocols
# -------------------------------------------------------------------


class TestProtocol(Protocol):
    __test__ = False

    def method1(self, x: int) -> str: ...
    def method2(self, y: float, z: bool) -> list: ...


class CompliantClass:
    def method1(self, x: int) -> str:
        return str(x)

    def method2(self, y: float, z: bool) -> list:
        return [y] if z else []


class NonCompliantClass:
    def method1(self, x: int) -> str:
        return str(x)

    # Missing method2


class WrongSignatureClass:
    def method1(self, x: float) -> str:  # Wrong parameter type
        return str(x)

    def method2(self, y: float, z: bool) -> list:
        return [y] if z else []


class AbstractBaseClass:
    pass


class DerivedClass(AbstractBaseClass):
    pass


class NonDerivedClass:
    pass


# -------------------------------------------------------------------
# Test functions
# -------------------------------------------------------------------


def test_get_func_name():
    """Test that get_func_name returns the correct function name."""

    def regular_func():
        pass

    from functools import partial

    assert get_func_name(regular_func, qualname=False) == "regular_func"
    assert (
        get_func_name(regular_func, qualname=True)
        == "test_get_func_name.<locals>.regular_func"
    )

    partial_func = partial(regular_func)

    assert get_func_name(partial_func, qualname=False) == "regular_func"
    assert (
        get_func_name(partial_func, qualname=True)
        == "test_get_func_name.<locals>.regular_func"
    )


# -------------------------------------------------------------------
# Tests for class validation
# -------------------------------------------------------------------


def test_validate_class():
    """Test that validate_class accepts classes and rejects non-classes."""
    # Classes should pass
    assert validate_class(CompliantClass) is CompliantClass

    # Non-classes should be rejected
    with pytest.raises(ValidationError):
        validate_class(42)
    with pytest.raises(ValidationError):
        validate_class("not a class")
    with pytest.raises(ValidationError):
        validate_class(CompliantClass())  # Instance, not a class


def test_validate_class_structure():
    """Test that validate_class_structure validates protocol conformance."""
    # Compliant class should pass
    assert validate_class_structure(CompliantClass, TestProtocol) is CompliantClass

    # Non-compliant class should fail due to missing method
    with pytest.raises(ConformanceError) as excinfo:
        validate_class_structure(NonCompliantClass, TestProtocol)
    assert "does not implement method" in str(excinfo.value)

    # Class with wrong signature should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_class_structure(WrongSignatureClass, TestProtocol)
    assert "does not match type annotation" in str(excinfo.value)


def test_validate_class_hierarchy():
    """Test that validate_class_hierarchy validates inheritance."""
    # Derived class should pass
    assert validate_class_hierarchy(DerivedClass, AbstractBaseClass) is DerivedClass

    # Non-derived class should fail
    with pytest.raises(InheritanceError) as excinfo:
        validate_class_hierarchy(NonDerivedClass, AbstractBaseClass)
    assert "not subclass-of" in str(excinfo.value)


def test_validate_init():
    """Test that validate_init is a pass-through function."""
    assert validate_init(CompliantClass) is CompliantClass


# -------------------------------------------------------------------
# Tests for function validation
# -------------------------------------------------------------------


def expected_func(a: int, b: str) -> bool:
    return len(b) == a


def matching_func(a: int, b: str) -> bool:
    return a > len(b)


def missing_param_func(a: int) -> bool:
    return a > 0


def extra_param_func(a: int, b: str, c: float) -> bool:
    return len(b) == a and c > 0


def wrong_param_type_func(a: float, b: str) -> bool:
    return len(b) == int(a)


def wrong_return_type_func(a: int, b: str) -> str:
    return b * a


def test_validate_function():
    """Test that validate_function accepts functions and rejects non-functions."""
    # Functions should pass
    assert validate_function(expected_func) is expected_func

    # Non-functions should be rejected
    with pytest.raises(ValidationError):
        validate_function(42)
    with pytest.raises(ValidationError):
        validate_function("not a function")
    with pytest.raises(ValidationError):
        validate_function(CompliantClass)  # Class, not a function


def test_validate_function_signature():
    """Test that validate_function_signature validates function signatures."""
    # Matching function should pass
    assert validate_function_signature(matching_func, expected_func) is matching_func

    # Function with missing parameter should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_signature(missing_param_func, expected_func)
    assert "does not match argument count" in str(excinfo.value)

    # Function with extra parameter should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_signature(extra_param_func, expected_func)
    assert "does not match argument count" in str(excinfo.value)

    # Function with wrong parameter type should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_signature(wrong_param_type_func, expected_func)
    assert "does not match type annotation" in str(excinfo.value)

    # Function with wrong return type should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_signature(wrong_return_type_func, expected_func)
    assert "Return type" in str(excinfo.value)


def test_validate_function_parameters():
    """Test that validate_function_parameters validates function parameters against a type."""
    from typing import Callable

    # Create the expected callable type
    ExpectedType = Callable[[int, str], bool]

    # Matching function should pass
    assert validate_function_parameters(matching_func, ExpectedType) is matching_func

    # Function with missing parameter should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_parameters(missing_param_func, ExpectedType)
    assert "has 1 parameters, expected 2" in str(excinfo.value)

    # Function with extra parameter should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_parameters(extra_param_func, ExpectedType)
    assert "has 3 parameters, expected 2" in str(excinfo.value)

    # Function with wrong parameter type should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_parameters(wrong_param_type_func, ExpectedType)
    assert "has type float, expected int" in str(excinfo.value)

    # Function with wrong return type should fail
    with pytest.raises(ConformanceError) as excinfo:
        validate_function_parameters(wrong_return_type_func, ExpectedType)
    assert "has return type str, expected bool" in str(
        excinfo.value
    )


# -------------------------------------------------------------------
# Tests for instance validation
# -------------------------------------------------------------------


def test_validate_instance_hierarchy():
    """Test that validate_instance_hierarchy validates instance inheritance."""
    # Derived instance should pass
    derived_instance = DerivedClass()
    assert (
        validate_instance_hierarchy(derived_instance, AbstractBaseClass)
        is derived_instance
    )

    # Non-derived instance should fail
    non_derived_instance = NonDerivedClass()
    with pytest.raises(ValidationError) as excinfo:
        validate_instance_hierarchy(non_derived_instance, AbstractBaseClass)
    assert "not instance-of" in str(excinfo.value)

    # Non-instance should fail
    with pytest.raises(ValidationError):
        validate_instance_hierarchy(42, AbstractBaseClass)


def test_validate_instance_structure():
    """Test that validate_instance_structure validates instance structure."""
    # Compliant instance should pass
    compliant_instance = CompliantClass()
    assert (
        validate_instance_structure(compliant_instance, TestProtocol)
        is compliant_instance
    )

    # Non-compliant instance should fail due to missing method
    non_compliant_instance = NonCompliantClass()
    with pytest.raises(ConformanceError) as excinfo:
        validate_instance_structure(non_compliant_instance, TestProtocol)
    assert "does not implement attribute" in str(excinfo.value)

    # Instance with wrong signature should fail
    wrong_signature_instance = WrongSignatureClass()
    with pytest.raises(ConformanceError) as excinfo:
        validate_instance_structure(wrong_signature_instance, TestProtocol)
    assert "does not match type annotation" in str(excinfo.value)

    # Non-instance should fail
    with pytest.raises(ValidationError):
        validate_instance_structure(42, TestProtocol)


# -------------------------------------------------------------------
# Tests for validation error classes
# -------------------------------------------------------------------


def test_error_hierarchy():
    """Test that validation error classes have the correct inheritance hierarchy."""
    # ConformanceError should inherit from ValidationError and TypeError
    assert issubclass(ConformanceError, ValidationError)
    assert issubclass(ConformanceError, TypeError)

    # InheritanceError should inherit from ValidationError and TypeError
    assert issubclass(InheritanceError, ValidationError)
    assert issubclass(InheritanceError, TypeError)
