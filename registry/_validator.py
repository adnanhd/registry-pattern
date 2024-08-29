"""Validator module."""

import inspect
from abc import ABC
from typing import Any
from typing import Callable
from typing import Type
from typing import TypeVar

from typeguard import TypeCheckError
from typeguard import check_type
from typing_extensions import ParamSpec

Protocol = TypeVar("Protocol")


class ValidationError(Exception):
    """Validation error."""


class CoercionError(ValueError, ValidationError):
    """Coercion error."""


class ConformanceError(TypeError, ValidationError):
    """Static type-checking error."""


class InheritanceError(TypeError, ValidationError):
    """Inheritance abc.ABC registry error."""


def validate_class(subcls: Any) -> Type:
    """Validate a class."""
    print("validate_class", subcls)
    if not inspect.isclass(subcls):
        raise ValidationError(f"{subcls} is not a class")
    return subcls


def validate_class_structure(
    subcls: Type, /, expected_type: Type[Protocol], coerce_to_type: bool = False
) -> Type:
    """Check the structure of a class."""
    print("validate_class_structure", subcls)
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(subcls, expected_type)  # type: ignore TODO fix this part!
    except TypeCheckError as exc:
        error_msg = f"{subcls} is not of type {expected_type}: {exc}"
        raise ConformanceError(error_msg) from exc


def validate_class_hierarchy(subcls: Type, /, abc_class: Type[ABC]) -> Type:
    """Check the hierarchy of a class."""
    print("validate_class_hierarchy", subcls)
    if not issubclass(subcls, abc_class):
        raise InheritanceError(f"{subcls} not subclass-of {abc_class}")
    return subcls


def validate_init(cls: Any) -> Any:
    """Validate a class."""
    return cls


R = TypeVar("R")
P = ParamSpec("P")


def validate_function(func: Callable) -> Callable:
    """Validate a function."""
    if not (inspect.isfunction(func) or inspect.isbuiltin(func)):
        raise ValidationError(f"{func} is not a function")
    return func


def validate_function_parameters(
    func: Callable,
    /,
    expected_type: Type[Callable[P, R]],  # ff: igre[reportInvalidTypeForm]
    coerce_to_type: bool = False,
) -> Callable[P, R]:
    """Check the structure of a class."""
    print("validate_function_parameters")
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(func, expected_type)
    except TypeCheckError as exc:
        error_msg = f"{func} is not of type {expected_type}"
        raise ConformanceError(error_msg) from exc


Obj = TypeVar("Obj")


def validate_instance_hierarchy(instance: Obj, /, expected_type: Type) -> Obj:
    """Validate a class."""
    print(
        "validate_instance_hierarchy",
        instance,
        "expected_type:=",
        expected_type.__name__,
    )
    if not inspect.isclass(instance.__class__):
        raise ValidationError(f"{instance} is not a class instance")
    if not isinstance(instance, expected_type):
        raise ValidationError(f"{instance} not instance-of {expected_type}")
    return instance


def validate_instance_structure(
    obj: Obj, /, expected_type: Type[Obj], coerce_to_type: bool = False
) -> Obj:
    """Check the structure of a class."""
    print("validate_instance_structure", obj)
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(obj, expected_type)  # type: ignore TODO fix this part!
    except TypeCheckError as exc:
        error_msg = f"{obj} is not of type {expected_type}"
        raise ConformanceError(error_msg) from exc
