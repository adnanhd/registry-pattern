r"""Validator module.

This module provides functions and decorators for validating classes,
functions, and instances. It includes logging decorators for debugging,
validation of class structures against expected protocols, and checks
for function signature conformance.

Doxygen Dot Graph of Exception Hierarchy:
------------------------------------------
\dot
digraph ExceptionHierarchy {
    node [shape=rectangle];
    "Exception" -> "ValidationError";
    "ValidationError" -> "CoercionError";
    "ValidationError" -> "ConformanceError";
    "ValidationError" -> "InheritanceError";
}
\enddot
"""

import os
import inspect
import logging
from abc import ABC
from typing import Any, Callable, Type, TypeVar

from functools import wraps, partial, partialmethod
from typing_extensions import ParamSpec

# TODO: remove typeguard dependency
from typeguard import TypeCheckError, check_type

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

# Configure logging with environment-based settings.
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("VALIDATOR_LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)-5s] [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------


def log_debug(func: Callable) -> Callable:
    """
    Decorator to log function calls in debug mode.

    If the environment variable 'VALIDATOR_QUIET' is set to 'TRUE',
    logging is disabled for the decorated function.

    Parameters:
        func (Callable): The function to decorate.

    Returns:
        Callable: The decorated function.
    """
    if os.getenv("VALIDATOR_QUIET", "FALSE").upper() == "TRUE":
        return func

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Log function call with arguments if debug level is enabled.
        if logger.isEnabledFor(logging.DEBUG):
            arg_str = ", ".join(repr(a) for a in args)
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
            logger.debug(f"{func.__name__}({all_args})")
        return func(*args, **kwargs)

    return wrapper


def get_func_name(func: Callable) -> str:
    """
    Retrieve the qualified name of a function, resolving partial functions.

    Parameters:
        func (Callable): The function or partial function.

    Returns:
        str: The qualified name of the underlying function.
    """
    if isinstance(func, (partial, partialmethod)):
        return get_func_name(func.func)
    else:
        return func.__qualname__


# -----------------------------------------------------------------------------
# Exception Classes
# -----------------------------------------------------------------------------


class ValidationError(Exception):
    """Base exception for validation errors."""

    pass


class CoercionError(ValueError, ValidationError):
    """Exception raised when coercion of a value fails."""

    pass


class ConformanceError(TypeError, ValidationError):
    """Exception raised when a function or class does not conform to expected type signatures."""

    pass


class InheritanceError(TypeError, ValidationError):
    """Exception raised when a class does not inherit from an expected abstract base class."""

    pass


# -----------------------------------------------------------------------------
# Type Variables for Protocols and Functions
# -----------------------------------------------------------------------------

# 'Protocol' here is a type variable used to represent expected protocol types.
Protocol = TypeVar("Protocol")

# Type variables for function signature validation.
R = TypeVar("R")
P = ParamSpec("P")


# -----------------------------------------------------------------------------
# Class and Function Validation Functions
# -----------------------------------------------------------------------------


@log_debug
def validate_class(subcls: Any) -> Type:
    """
    Validate that the provided argument is a class.

    Parameters:
        subcls (Any): The object to validate.

    Returns:
        Type: The validated class.

    Raises:
        ValidationError: If the argument is not a class.
    """
    if not inspect.isclass(subcls):
        raise ValidationError(f"{subcls} is not a class")
    return subcls


@log_debug
def validate_class_structure(
    subcls: Type, /, expected_type: Type[Protocol], coerce_to_type: bool = False
) -> Type:
    """
    Validate that a class implements the structure of an expected type.

    This function checks that all public methods (non-underscore-prefixed)
    defined in the expected type are present in the class and that their
    signatures match.

    Parameters:
        subcls (Type): The class to validate.
        expected_type (Type[Protocol]): The expected protocol or interface.
        coerce_to_type (bool): Flag for coercion support (currently not supported).

    Returns:
        Type: The validated class.

    Raises:
        ConformanceError: If the class does not implement required methods or if
                          method signatures do not match.
    """
    # Coercion is not supported in this implementation.
    assert not coerce_to_type, "Coercion is not supported"

    # Iterate over attributes in the expected type.
    for method_name in dir(expected_type):
        # Skip private or special methods.
        if method_name.startswith("_"):
            continue

        # Check if the method exists in the class.
        if not hasattr(subcls, method_name):
            raise ConformanceError(
                f"Class {subcls.__name__} does not implement method '{method_name}'"
            )

        # Validate that the method signatures match.
        expected_type_method: Callable = getattr(expected_type, method_name)
        subcls_method: Callable = getattr(subcls, method_name)
        validate_function_signature(subcls_method, expected_type_method)

    return subcls


@log_debug
def validate_class_hierarchy(subcls: Type, /, abc_class: Type[ABC]) -> Type:
    """
    Validate that a class is a subclass of a given abstract base class.

    Parameters:
        subcls (Type): The class to validate.
        abc_class (Type[ABC]): The expected abstract base class.

    Returns:
        Type: The validated class.

    Raises:
        InheritanceError: If the class is not a subclass of the specified ABC.
    """
    if not issubclass(subcls, abc_class):
        raise InheritanceError(f"{subcls} not subclass-of {abc_class}")
    return subcls


@log_debug
def validate_init(cls: Any) -> Any:
    """
    Validate initialization of a class.

    This function currently acts as a pass-through and can be extended to
    perform more complex initialization validation if needed.

    Parameters:
        cls (Any): The class to validate.

    Returns:
        Any: The validated class.
    """
    return cls


@log_debug
def validate_function(func: Callable) -> Callable:
    """
    Validate that the provided argument is a function or a built-in function.

    Parameters:
        func (Callable): The function to validate.

    Returns:
        Callable: The validated function.

    Raises:
        ValidationError: If the argument is not a function.
    """
    if not (inspect.isfunction(func) or inspect.isbuiltin(func)):
        raise ValidationError(f"{func} is not a function")
    return func


@log_debug
def validate_function_signature(
    func: Callable[P, R], expected_func: Callable[P, R]
) -> Callable[P, R]:
    """
    Validate that a function's signature matches that of an expected function.

    This includes checking the number of parameters, parameter type annotations,
    and the return type.

    Parameters:
        func (Callable[P, R]): The function to validate.
        expected_func (Callable[P, R]): The function with the expected signature.

    Returns:
        Callable[P, R]: The validated function.

    Raises:
        ConformanceError: If the function's signature does not match the expected signature.
    """
    func_signature: inspect.Signature = inspect.signature(func)
    expected_signature: inspect.Signature = inspect.signature(expected_func)

    # Check that the number of parameters match.
    protocol_params = list(expected_signature.parameters.values())
    cls_params = list(func_signature.parameters.values())
    if len(protocol_params) != len(cls_params):
        raise ConformanceError(
            f"Function '{get_func_name(func)}' does not match "
            f"argument count of protocol.\n"
            f"Expected: {len(protocol_params)} arguments, Found: {len(cls_params)}"
        )

    # Validate each parameter's type annotation.
    for protocol_param, cls_param in zip(protocol_params, cls_params):
        if protocol_param.annotation != cls_param.annotation:
            raise ConformanceError(
                f"Parameter '{protocol_param.name}' in function '{get_func_name(func)}' "
                f"does not match type annotation.\n"
                f"Expected: {protocol_param.annotation}, Found: {cls_param.annotation}"
            )

    # Validate the return type annotation.
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
    expected_type: Type[Callable[P, R]],
    coerce_to_type: bool = False,
) -> Callable[P, R]:
    """
    Validate that a function's parameters match an expected callable type.

    This function compares the number of parameters and their type annotations
    against the expected callable's signature.

    Parameters:
        func (Callable): The function to validate.
        expected_type (Type[Callable[P, R]]): The expected function type.
        coerce_to_type (bool): Flag for coercion support (currently not supported).

    Returns:
        Callable[P, R]: The validated function.

    Raises:
        ConformanceError: If the function parameters or return type do not match.
    """
    # Coercion is not supported.
    assert not coerce_to_type, "Coercion is not supported"

    func_signature = inspect.signature(func)
    # Extract expected argument types and return type from the expected callable.
    expected_args = expected_type.__args__[:-1]
    expected_return = expected_type.__args__[-1]

    # Check that the number of parameters is correct.
    if len(func_signature.parameters) != len(expected_args):
        raise ConformanceError(
            f"Function {func.__name__} has {len(func_signature.parameters)} parameters, "
            f"expected {len(expected_args)}."
        )

    # Validate each parameter's type.
    for param, expected_arg_type in zip(
        func_signature.parameters.values(), expected_args
    ):
        if param.annotation != expected_arg_type:
            raise ConformanceError(
                f"Parameter {param.name} of function {func.__name__} has type {param.annotation}, "
                f"expected {expected_arg_type}."
            )

    # Validate the return type.
    if func_signature.return_annotation != expected_return:
        raise ConformanceError(
            f"Function {func.__name__} has return type {func_signature.return_annotation}, "
            f"expected {expected_return}."
        )

    return func


# -----------------------------------------------------------------------------
# Instance Validation Functions
# -----------------------------------------------------------------------------

Obj = TypeVar("Obj")


@log_debug
def validate_instance_hierarchy(instance: Obj, /, expected_type: Type) -> Obj:
    """
    Validate that an instance is of a given type.

    Parameters:
        instance (Obj): The instance to validate.
        expected_type (Type): The expected type or class.

    Returns:
        Obj: The validated instance.

    Raises:
        ValidationError: If the instance is not an instance of the expected type.
    """
    if not inspect.isclass(instance.__class__):
        raise ValidationError(f"{instance} is not a class instance")
    if not isinstance(instance, expected_type):
        raise ValidationError(f"{instance} not instance-of {expected_type}")
    return instance


@log_debug
def validate_instance_structure(
    obj: Obj, /, expected_type: Type[Obj], coerce_to_type: bool = False
) -> Obj:
    """
    Validate that an instance conforms to an expected type using runtime type checking.

    This function leverages 'check_type' from the typeguard package to
    enforce type conformity.

    Parameters:
        obj (Obj): The instance to validate.
        expected_type (Type[Obj]): The expected type of the instance.
        coerce_to_type (bool): Flag for coercion support (currently not supported).

    Returns:
        Obj: The validated instance.

    Raises:
        ConformanceError: If the instance does not conform to the expected type.
    """
    # Coercion is not supported.
    assert not coerce_to_type, "Coercion is not supported"
    try:
        # check_type performs runtime type checking.
        return check_type(obj, expected_type)  # type: ignore
    except TypeCheckError as exc:
        error_msg = f"{obj} is not of type {expected_type}"
        raise ConformanceError(error_msg) from exc
