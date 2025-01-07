"""Validator module."""

import os
import inspect
import logging
from abc import ABC
from typing import Any
from typing import Callable
from typing import Type
from typing import TypeVar

from functools import wraps, partial, partialmethod
from typing_extensions import ParamSpec
# TODO: remove typeguard dependency
from typeguard import TypeCheckError
from typeguard import check_type

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("VALIDATOR_LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)-5s] [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",  # Custom date format
)

def log_debug(func: Callable) -> Callable:
    """Decorator to log function calls in debug mode."""
    if os.getenv("VALIDATOR_QUIET", "FALSE").upper() == "TRUE":
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        if logger.isEnabledFor(logging.DEBUG):
            arg_str = ", ".join(repr(a) for a in args)
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
            logger.debug(f"{func.__name__}({all_args})")
        return func(*args, **kwargs)

    return wrapper


def get_func_name(func: Callable) -> str:
    if isinstance(func, (partial, partialmethod)):
        return get_func_name(func.func)
    else:
        return func.__qualname__


class ValidationError(Exception):
    """Validation error."""


class CoercionError(ValueError, ValidationError):
    """Coercion error."""


class ConformanceError(TypeError, ValidationError):
    """Static type-checking error."""


class InheritanceError(TypeError, ValidationError):
    """Inheritance abc.ABC registry error."""


#############################################################################

Protocol = TypeVar("Protocol")


@log_debug
def validate_class(subcls: Any) -> Type:
    """Validate a class."""
    if not inspect.isclass(subcls):
        raise ValidationError(f"{subcls} is not a class")
    return subcls


@log_debug
def validate_class_structure(
    subcls: Type, /, expected_type: Type[Protocol], coerce_to_type: bool = False
) -> Type:
    """Check the structure of a class."""
    assert not coerce_to_type, "Coercion is not supported"

    for method_name in dir(expected_type):
        # Skip special methods and attributes
        if method_name.startswith("_"):
            continue

        # Ensure the class has the method
        if not hasattr(subcls, method_name):
            raise ConformanceError(
                f"Class {subcls.__name__} does not implement method '{method_name}'"
            )

        # Validate the method signature
        expected_type_method: Callable = getattr(expected_type, method_name)
        subcls_method: Callable = getattr(subcls, method_name)
        validate_function_signature(
            subcls_method,
            expected_type_method,
        )
    return subcls


@log_debug
def validate_class_hierarchy(subcls: Type, /, abc_class: Type[ABC]) -> Type:
    """Check the hierarchy of a class."""
    if not issubclass(subcls, abc_class):
        raise InheritanceError(f"{subcls} not subclass-of {abc_class}")
    return subcls


@log_debug
def validate_init(cls: Any) -> Any:
    """Validate a class."""
    return cls


#############################################################################

R = TypeVar("R")
P = ParamSpec("P")


@log_debug
def validate_function(func: Callable) -> Callable:
    """Validate a function."""
    if not (inspect.isfunction(func) or inspect.isbuiltin(func)):
        raise ValidationError(f"{func} is not a function")
    return func


@log_debug
def validate_function_signature(
    func: Callable[P, R], expected_func: Callable[P, R]
) -> Callable[P, R]:
    """
    Validate that a specific method in a class adheres to the signature
    defined in a protocol, including argument counts, types, and return type.
    """
    func_signature: inspect.Signature = inspect.signature(func)
    expected_signature: inspect.Signature = inspect.signature(expected_func)

    # Check if the argument counts match
    protocol_params = list(expected_signature.parameters.values())
    cls_params = list(func_signature.parameters.values())
    if len(protocol_params) != len(cls_params):
        raise ConformanceError(
            f"Function '{get_func_name(func)}' does not match "
            f"argument count of protocol.\n"
            f"Expected: {len(protocol_params)} arguments, Found: {len(cls_params)}"
        )

    # Validate argument types
    for protocol_param, cls_param in zip(protocol_params, cls_params):
        if protocol_param.annotation != cls_param.annotation:
            raise ConformanceError(
                f"Parameter '{protocol_param.name}' in function '{get_func_name(func)}' "
                f"does not match type annotation.\n"
                f"Expected: {protocol_param.annotation}, Found: {cls_param.annotation}"
            )

    # Validate return type
    if expected_signature.return_annotation != func_signature.return_annotation:
        raise ConformanceError(
            f"Return type of method '{get_func_name(func)}' does not match.\n"
            f"Expected: {expected_signature.return_annotation}, Found: {func_signature.return_annotation}"
        )

    return func


@log_debug
def validate_function_parameters(
    func: Callable,
    /,
    expected_type: Type[Callable[P, R]],  # ff: igre[reportInvalidTypeForm]
    coerce_to_type: bool = False,
) -> Callable[P, R]:
    """Check the structure of a class."""
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(func, expected_type)
    except TypeCheckError as exc:
        error_msg = f"{func} is not of type {expected_type}"
        raise ConformanceError(error_msg) from exc


#############################################################################


Obj = TypeVar("Obj")


@log_debug
def validate_instance_hierarchy(instance: Obj, /, expected_type: Type) -> Obj:
    """Validate a class."""
    if not inspect.isclass(instance.__class__):
        raise ValidationError(f"{instance} is not a class instance")
    if not isinstance(instance, expected_type):
        raise ValidationError(f"{instance} not instance-of {expected_type}")
    return instance


@log_debug
def validate_instance_structure(
    obj: Obj, /, expected_type: Type[Obj], coerce_to_type: bool = False
) -> Obj:
    """Check the structure of a class."""
    assert not coerce_to_type, "Coercion is not supported"
    try:
        return check_type(obj, expected_type)  # type: ignore TODO fix this part!
    except TypeCheckError as exc:
        error_msg = f"{obj} is not of type {expected_type}"
        raise ConformanceError(error_msg) from exc
