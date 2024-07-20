"""Validator module."""

from abc import ABC
import inspect
from typing import TypeVar, Callable, Any, Type
from typing_extensions import ParamSpec
from typeguard import check_type, TypeCheckError


Protocol = TypeVar("Protocol")


class CoercionError(ValueError):
    """Coercion error."""


class StructuringError(TypeError):
    """Static type-checking error."""


class NominatingError(TypeError):
    """Inheritance abc.ABC registry error."""


class ValidationError(CoercionError, StructuringError, NominatingError):
    """Validation error."""


def validate_class(subcls: Any) -> Type:
    """Validate a class."""
    print("validate_class", subcls)
    if not inspect.isclass(subcls):
        raise ValidationError(f"{subcls} is not a class")
    return subcls


def validate_class_structure(
    subcls: type, /, expected_type: type[Protocol], coerce_to_type: bool = False
) -> type:
    """Check the structure of a class."""
    print("validate_class_structure", subcls)
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(subcls, expected_type)
    except TypeCheckError as exc:
        error_msg = f"{subcls} is not of type {expected_type}"
        raise StructuringError(error_msg) from exc


def validate_class_hierarchy(subcls: type, /, abc_class: Type[ABC]) -> type:
    """Check the hierarchy of a class."""
    print("validate_class_hierarchy", subcls)
    if not issubclass(subcls, abc_class):
        raise NominatingError(f"{subcls} not subclass-of {abc_class}")
    return subcls


def validate_init(cls: Any) -> Any:
    """Validate a class."""
    return cls


R = TypeVar("R")
P = ParamSpec("P")


def validate_function(func: Any) -> Callable:
    """Validate a function."""
    if not inspect.isfunction(func):
        raise ValidationError(f"{func} is not a function")
    return func


def validate_function_parameters(
    func: Callable,
    /,
    expected_type: type[Callable[P, R]],  # pyright: ignore[reportInvalidTypeForm]
    coerce_to_type: bool = False,
) -> Callable[P, R]:
    """Check the structure of a class."""
    print("validate_function_parameters")
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(func, expected_type)
    except TypeCheckError as exc:
        error_msg = f"{func} is not of type {expected_type}"
        raise StructuringError(error_msg) from exc


def validate_instance(instance: Any, /, expected_type: type) -> Any:
    """Validate a class."""
    print("validate_instance", instance, "expected_type:=", expected_type.__name__)
    if not inspect.isclass(instance.__class__):
        raise ValidationError(f"{instance} is not a class instance")
    if not isinstance(instance, expected_type):
        raise ValidationError(f"{instance} not instance-of {expected_type}")
    return instance
