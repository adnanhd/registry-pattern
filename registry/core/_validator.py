r"""Enhanced validator module.

This module provides functions and decorators for validating classes,
functions, and instances. It includes rich error context, validation caching,
and improved error messages with actionable suggestions.

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

from gc import callbacks
import inspect
import logging
import os
import time
import weakref
from abc import ABC
from functools import partial, partialmethod, wraps
from threading import RLock
from typing_compat import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    ParamSpec,
    Type,
    TypeAlias,
    TypeVar,
    get_args,
)

# -----------------------------------------------------------------------------
# Logging Configuration
# -----------------------------------------------------------------------------

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("VALIDATOR_LOG_LEVEL", "INFO").upper(),
    format="[%(asctime)s] [%(levelname)-5s] [%(name)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# -----------------------------------------------------------------------------
# Validation Cache
# -----------------------------------------------------------------------------


class ValidationCache:
    """Thread-safe cache for validation results to improve performance."""

    def __init__(self, max_size: int = 1000, ttl_seconds: float = 300.0):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: Dict[int, tuple] = {}  # hash -> (result, timestamp, suggestions)
        self._lock = RLock()

    def get(self, obj: Any, operation: str) -> Optional[tuple]:
        """Get cached validation result and suggestions."""
        cache_key = hash((id(obj), type(obj).__name__, operation))

        with self._lock:
            if cache_key in self._cache:
                result, timestamp, suggestions = self._cache[cache_key]
                if time.time() - timestamp < self.ttl:
                    return result, suggestions
                else:
                    del self._cache[cache_key]

        return None

    def set(
        self, obj: Any, operation: str, result: bool, suggestions: List[str]
    ) -> None:
        """Cache validation result with suggestions."""
        cache_key = hash((id(obj), type(obj).__name__, operation))

        with self._lock:
            # Simple LRU eviction
            if len(self._cache) >= self.max_size:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]

            self._cache[cache_key] = (result, time.time(), suggestions)


# Global validation cache
_validation_cache = ValidationCache()

# -----------------------------------------------------------------------------
# Enhanced Exception Classes
# -----------------------------------------------------------------------------


class ValidationError(Exception):
    """Base exception for validation errors with rich context."""

    def __init__(
        self,
        message: str,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.suggestions = suggestions or []
        self.context = context or {}

        # Build enhanced error message
        enhanced_message = self._build_enhanced_message()
        super().__init__(enhanced_message)

    def _build_enhanced_message(self) -> str:
        """Build enhanced error message with context and suggestions."""
        lines = [self.message]

        if self.context:
            if "expected_type" in self.context and "actual_type" in self.context:
                lines.append(f"  Expected: {self.context['expected_type']}")
                lines.append(f"  Actual: {self.context['actual_type']}")

            if "artifact_name" in self.context:
                lines.append(f"  Artifact: {self.context['artifact_name']}")

        if self.suggestions:
            lines.append("  Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"    • {suggestion}")

        return "\n".join(lines)


class CoercionError(ValidationError, ValueError):
    """Exception raised when coercion of a value fails."""

    pass


class ConformanceError(ValidationError, TypeError):
    """Exception raised when a function or class does not conform to expected type signatures."""

    pass


class InheritanceError(ValidationError, TypeError):
    """Exception raised when a class does not inherit from an expected abstract base class."""

    pass


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def get_func_name(func: Callable, qualname: bool = False) -> str:
    """
    Retrieve the (qualified) name of a function, resolving partials and wrappers.

    Parameters:
        func (Callable): The function, partial function, method, or wrapped callable.
        qualname (bool): If True, return the qualified name (e.g., including class/module).

    Returns:
        str: The resolved function name.

    Raises:
        TypeError: If the input is not a recognizable callable.
    """
    # Resolve functools.partial or partialmethod
    while isinstance(func, (partial, partialmethod)):
        func = func.func

    # Resolve wrapped functions (e.g., from decorators)
    while hasattr(func, "__wrapped__"):
        func = getattr(func, "__wrapped__")

    # Handle bound or unbound methods
    if hasattr(func, "__name__"):
        return getattr(func, "__qualname__" if qualname else "__name__")

    # Callable object (e.g., class with __call__)
    if hasattr(func, "__call__"):
        func = getattr(func, "__call__")
        return get_func_name(func, qualname)

    raise TypeError(f"Cannot resolve name from non-callable object: {func!r}")


def get_type_name(cls: Type, qualname: bool = False) -> str:
    """
    Retrieve the name or qualified name of a type.

    Parameters:
        cls (Type): The class or type object.
        qualname (bool): If True, return the qualified name.

    Returns:
        str: The type's name.

    Raises:
        TypeError: If the type cannot be resolved.
    """
    if qualname and hasattr(cls, "__qualname__"):
        return getattr(cls, "__qualname__")
    elif hasattr(cls, "__name__"):
        return getattr(cls, "__name__")
    else:
        return str(cls)


def get_mem_addr(obj: Any, with_prefix: bool = True) -> str:
    """
    Return the memory address of an object in hexadecimal format.

    Parameters:
        obj (Any): The object whose memory address is to be returned.
        with_prefix (bool): If True, return address with '0x' prefix (default: True).

    Returns:
        str: The memory address of the object as a hex string.
    """
    addr = id(obj)
    return f"{addr:#x}" if with_prefix else f"{addr:x}"


def _generate_suggestions(
    operation: str, expected_type: Optional[Type] = None, artifact: Any = None
) -> List[str]:
    """Generate helpful suggestions based on validation failure context."""
    suggestions = []

    if operation == "class_structure":
        suggestions.append("Ensure all required methods are implemented")
        suggestions.append("Check that method signatures match exactly")
        if expected_type:
            protocol_methods = [
                m
                for m in dir(expected_type)
                if not m.startswith("_") and callable(getattr(expected_type, m))
            ]
            if protocol_methods:
                suggestions.append(f"Required methods: {', '.join(protocol_methods)}")

    elif operation == "class_hierarchy":
        suggestions.append("Make sure your class inherits from the required base class")
        if expected_type:
            suggestions.append(
                f"Add 'class YourClass({get_type_name(expected_type)}):' to your class definition"
            )

    elif operation == "function_signature":
        suggestions.append("Check parameter types and return type annotations")
        suggestions.append(
            "Ensure parameter names and order match the expected signature"
        )

    elif operation == "function_parameters":
        suggestions.append(
            "Verify function signature matches the expected callable type"
        )
        suggestions.append("Check parameter count, types, and return type")

    elif operation == "instance_structure":
        suggestions.append("Ensure the instance implements all required methods")
        suggestions.append("Check that method signatures are correct")

    return suggestions


def log_debug(func: Callable) -> Callable:
    """
    Decorator to log function calls in debug mode with enhanced error context.
    """
    if os.getenv("REGISTRY_VALIDATOR_DEBUG", "FALSE").upper() != "TRUE":
        return func

    func_name = get_func_name(func)
    func_logger = logging.getLogger(func_name)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check cache first
        if len(args) > 0:
            cached_result = _validation_cache.get(args[0], func_name)
            if cached_result is not None:
                result, suggestions = cached_result
                if not result:
                    # Re-raise cached failure with suggestions
                    context = {
                        "operation": func_name,
                        "artifact_name": (
                            get_type_name(args[0])
                            if hasattr(args[0], "__name__")
                            else str(args[0])[:999]
                        ),
                    }
                    raise ValidationError(
                        f"Cached validation failure in {func_name}",
                        suggestions,
                        context,
                    )
                return args[0]  # Return validated object

        # Log function call with arguments if debug level is enabled
        if func_logger.isEnabledFor(logging.DEBUG):
            arg_str = ", ".join(repr(a) for a in args)
            kwarg_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
            all_args = ", ".join(filter(None, [arg_str, kwarg_str]))
            func_logger.debug(f"{func_name}({all_args})")

        try:
            result = func(*args, **kwargs)
            # Cache successful validation
            if len(args) > 0:
                _validation_cache.set(args[0], func_name, True, [])
            return result

        except (ValidationError, ConformanceError, InheritanceError) as e:
            # Cache failed validation with suggestions
            if len(args) > 0:
                suggestions = getattr(e, "suggestions", [])
                _validation_cache.set(args[0], func_name, False, suggestions)
            raise

        except Exception as e:
            # Convert generic exceptions to ValidationError with context
            suggestions = _generate_suggestions(
                func_name, kwargs.get("expected_type"), args[0] if args else None
            )
            context = {
                "operation": func_name,
                "artifact_name": (
                    get_type_name(args[0])
                    if args and hasattr(args[0], "__name__")
                    else str(args[0])[:999] if args else "unknown"
                ),
            }

            enhanced_error = ValidationError(str(e), suggestions, context)

            # Cache the failure
            if len(args) > 0:
                _validation_cache.set(args[0], func_name, False, suggestions)

            raise enhanced_error from e

    return wrapper


# -----------------------------------------------------------------------------
# Type Variables for Protocols and Functions
# -----------------------------------------------------------------------------

Protocol = TypeVar("Protocol")
R = TypeVar("R")
P = ParamSpec("P")

# -----------------------------------------------------------------------------
# Enhanced Validation Functions
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
        suggestions = [
            "Ensure you're passing a class, not an instance",
            "Check that the object is defined with 'class' keyword",
            f"Got {type(subcls).__name__}, expected a class",
        ]
        context = {
            "expected_type": "class",
            "actual_type": type(subcls).__name__,
            "artifact_name": str(subcls)[:999],
        }
        raise ValidationError(f"{subcls} is not a class", suggestions, context)
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
    assert not coerce_to_type, "Coercion is not supported"

    missing_methods = []
    signature_errors = []

    # Iterate over attributes in the expected type
    for method_name in dir(expected_type):
        if method_name.startswith("_"):
            continue

        # Check if the method exists in the class
        if not hasattr(subcls, method_name):
            missing_methods.append(method_name)
            continue

        # Validate that the method signatures match
        expected_method = getattr(expected_type, method_name)
        subcls_method = getattr(subcls, method_name)

        try:
            validate_function_signature(subcls_method, expected_method)
        except ConformanceError as e:
            signature_errors.append(f"{method_name}: {e.message}")

    # Build comprehensive error message
    if missing_methods or signature_errors:
        error_parts = []
        suggestions = []

        if missing_methods:
            error_parts.append(f"Missing methods: {', '.join(missing_methods)}")
            for method in missing_methods:
                expected_method = getattr(expected_type, method)
                try:
                    sig = inspect.signature(expected_method)
                    suggestions.append(f"Add method: def {method}{sig}: ...")
                except:
                    suggestions.append(f"Add method: def {method}(self, ...): ...")

        if signature_errors:
            error_parts.append("Signature mismatches:")
            error_parts.extend(f"  {error}" for error in signature_errors)
            suggestions.append("Check parameter types and return type annotations")

        context = {
            "expected_type": get_type_name(expected_type),
            "actual_type": get_type_name(subcls),
            "artifact_name": get_type_name(subcls),
        }

        message = (
            f"Class {get_type_name(subcls)} does not conform to protocol {get_type_name(expected_type)}:\n"
            + "\n".join(error_parts)
        )
        raise ConformanceError(message, suggestions, context)

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
        suggestions = [
            f"Make your class inherit from {get_type_name(abc_class)}",
            f"Change class definition to: class {get_type_name(subcls)}({get_type_name(abc_class)}):",
            "Check that you're importing the correct base class",
        ]
        context = {
            "expected_type": get_type_name(abc_class),
            "actual_type": get_type_name(subcls),
            "artifact_name": get_type_name(subcls),
        }
        raise InheritanceError(
            f"{get_type_name(subcls)} is not a subclass of {get_type_name(abc_class)}",
            suggestions,
            context,
        )
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
    if (inspect.isfunction(func) or inspect.isbuiltin(func)) and callable(func):
        return func
    suggestions = [
        "Ensure you're passing a function, not a variable",
        "Check that the object is defined with 'def' keyword or is callable",
        f"Got {type(func).__name__}, expected a callable",
    ]
    context = {
        "expected_type": "function",
        "actual_type": type(func).__name__,
        "artifact_name": (
            get_func_name(func) if hasattr(func, "__name__") else str(func)[:999]
        ),
    }
    raise ValidationError(f"{func} is not a function", suggestions, context)


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
    try:
        func_signature = inspect.signature(func)
        expected_signature = inspect.signature(expected_func)
    except (ValueError, TypeError) as e:
        suggestions = ["Check that both functions have valid signatures"]
        context = {
            "artifact_name": get_func_name(func),
            "operation": "signature_inspection",
        }
        raise ConformanceError(
            f"Cannot inspect function signature: {e}", suggestions, context
        )

    errors = []
    suggestions = []

    # Check that the number of parameters match
    protocol_params = list(expected_signature.parameters.values())
    cls_params = list(func_signature.parameters.values())

    if len(protocol_params) != len(cls_params):
        errors.append(
            f"Parameter count mismatch: expected {len(protocol_params)}, got {len(cls_params)}"
        )
        suggestions.append(
            f"Adjust function to have exactly {len(protocol_params)} parameters"
        )

    # Validate each parameter's type annotation
    for i, (protocol_param, cls_param) in enumerate(zip(protocol_params, cls_params)):
        if (
            protocol_param.annotation != cls_param.annotation
            and protocol_param.annotation != inspect.Parameter.empty
        ):
            if cls_param.annotation == inspect.Parameter.empty:
                errors.append(f"Parameter '{cls_param.name}' missing type annotation")
                suggestions.append(
                    f"Add type annotation: {cls_param.name}: {get_type_name(protocol_param.annotation)}"
                )
            else:
                errors.append(
                    f"Parameter '{cls_param.name}' type mismatch: expected {get_type_name(protocol_param.annotation)}, got {get_type_name(cls_param.annotation)}"
                )
                suggestions.append(
                    f"Change parameter type to: {cls_param.name}: {get_type_name(protocol_param.annotation)}"
                )

    # Validate the return type annotation
    if (
        expected_signature.return_annotation != func_signature.return_annotation
        and expected_signature.return_annotation != inspect.Signature.empty
    ):
        if func_signature.return_annotation == inspect.Signature.empty:
            errors.append("Missing return type annotation")
            suggestions.append(
                f"Add return annotation: -> {get_type_name(expected_signature.return_annotation)}"
            )
        else:
            errors.append(
                f"Return type mismatch: expected {get_type_name(expected_signature.return_annotation)}, got {get_type_name(func_signature.return_annotation)}"
            )
            suggestions.append(
                f"Change return type to: -> {get_type_name(expected_signature.return_annotation)}"
            )

    if errors:
        context = {
            "artifact_name": get_func_name(func),
            "expected_type": str(expected_signature),
            "actual_type": str(func_signature),
        }
        message = (
            f"Function '{get_func_name(func)}' signature validation failed:\n"
            + "\n".join(f"  • {error}" for error in errors)
        )
        raise ConformanceError(message, suggestions, context)

    return func


@log_debug
def validate_function_parameters(
    func: Callable[P, R],
    /,
    expected_type: TypeAlias,
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
    assert not coerce_to_type, "Coercion is not supported"

    try:
        func_signature = inspect.signature(func)
        expected_args, expected_return = get_args(expected_type)
    except Exception as e:
        suggestions = ["Check that the expected type is a valid Callable type"]
        context = {"artifact_name": get_func_name(func)}
        raise ValidationError(
            f"Cannot analyze function type: {e}", suggestions, context
        )

    errors = []
    suggestions = []

    # Check that the number of parameters is correct
    if len(func_signature.parameters) != len(expected_args):
        errors.append(
            f"Parameter count mismatch: expected {len(expected_args)}, got {len(func_signature.parameters)}"
        )
        suggestions.append(
            f"Function should have exactly {len(expected_args)} parameters"
        )

    # Validate each parameter's type
    for i, (param, expected_arg_type) in enumerate(
        zip(func_signature.parameters.values(), expected_args)
    ):
        if (
            param.annotation != expected_arg_type
            and param.annotation != inspect.Parameter.empty
        ):
            if param.annotation == inspect.Parameter.empty:
                errors.append(f"Parameter {param.name} missing type annotation")
                suggestions.append(
                    f"Add type annotation: {param.name}: {get_type_name(expected_arg_type)}"
                )
            else:
                errors.append(
                    f"Parameter {param.name} type mismatch: expected {get_type_name(expected_arg_type)}, got {get_type_name(param.annotation)}"
                )
                suggestions.append(
                    f"Change parameter type: {param.name}: {get_type_name(expected_arg_type)}"
                )

    # Validate the return type
    if (
        func_signature.return_annotation != expected_return
        and func_signature.return_annotation != inspect.Signature.empty
    ):
        if func_signature.return_annotation == inspect.Signature.empty:
            errors.append("Missing return type annotation")
            suggestions.append(
                f"Add return annotation: -> {get_type_name(expected_return)}"
            )
        else:
            errors.append(
                f"Return type mismatch: expected {get_type_name(expected_return)}, got {get_type_name(func_signature.return_annotation)}"
            )
            suggestions.append(
                f"Change return type: -> {get_type_name(expected_return)}"
            )

    if errors:
        context = {
            "artifact_name": get_func_name(func),
            "expected_type": str(expected_type),
            "actual_type": str(func_signature),
        }
        message = (
            f"Function '{get_func_name(func)}' parameter validation failed:\n"
            + "\n".join(f"  • {error}" for error in errors)
        )
        raise ConformanceError(message, suggestions, context)

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
        suggestions = ["Ensure the object is a proper class instance"]
        context = {
            "expected_type": "class instance",
            "actual_type": type(instance).__name__,
            "artifact_name": str(instance)[:999],
        }
        raise ValidationError(
            f"{instance} is not a class instance", suggestions, context
        )

    if not isinstance(instance, expected_type):
        suggestions = [
            f"Ensure the instance is of type {get_type_name(expected_type)}",
            f"Check that the class inherits from {get_type_name(expected_type)}",
            f"Got instance of {get_type_name(instance.__class__)}, expected {get_type_name(expected_type)}",
        ]
        context = {
            "expected_type": get_type_name(expected_type),
            "actual_type": get_type_name(instance.__class__),
            "artifact_name": str(instance)[:999],
        }
        raise InheritanceError(
            f"{instance} is not an instance of {get_type_name(expected_type)}",
            suggestions,
            context,
        )

    return instance


@log_debug
def validate_instance_structure(
    obj: Any, /, expected_type: Type, coerce_to_type: bool = False
) -> Any:
    """
    Validate that an instance conforms to the structure of an expected type.

    This function checks that all public (non-underscore-prefixed) attributes,
    and, in particular, callables (methods) defined in the expected type are
    present on the instance and that their signatures match.

    Parameters:
        obj (Any): The instance to validate.
        expected_type (Type): The expected protocol or interface.
        coerce_to_type (bool): Flag for coercion support (currently not supported).

    Returns:
        Any: The validated instance.

    Raises:
        ConformanceError: If the instance does not implement required methods or if
                          method signatures do not match.
        ValidationError: If the provided object is not a proper instance.
    """
    if coerce_to_type:
        raise ValueError("Coercion is not supported")

    # Ensure that obj is indeed an instance of a class
    if not hasattr(obj, "__class__") or not inspect.isclass(obj.__class__):
        suggestions = ["Ensure you're passing a class instance, not a primitive type"]
        context = {
            "expected_type": "class instance",
            "actual_type": type(obj).__name__,
            "artifact_name": str(obj)[:999],
        }
        raise ValidationError(f"{obj} is not a valid instance", suggestions, context)

    missing_attributes = []
    signature_errors = []

    # Iterate over attributes in the expected type
    for attr_name in dir(expected_type):
        if attr_name.startswith("_"):
            continue

        # Check that the attribute is present on the instance
        if not hasattr(obj, attr_name):
            missing_attributes.append(attr_name)
            continue

        expected_attr = getattr(expected_type, attr_name)
        obj_attr = getattr(obj, attr_name)

        # If both expected and object attributes are callable, validate their signatures
        if callable(expected_attr):
            if not callable(obj_attr):
                signature_errors.append(f"Attribute '{attr_name}' should be callable")
                continue

            try:
                # if obj_attr is a method, unwrap it to get the underlying function
                if inspect.ismethod(obj_attr):
                    obj_attr = obj_attr.__func__

                # if expected_attr is a method, unwrap it to get the underlying function
                if inspect.ismethod(expected_attr):
                    expected_attr = expected_attr.__func__

                # Validate that the function signatures match
                validate_function_signature(obj_attr, expected_attr)
            except ConformanceError as e:
                signature_errors.append(f"Method '{attr_name}': {e.message}")

    # Build comprehensive error message
    if missing_attributes or signature_errors:
        error_parts = []
        suggestions = []

        if missing_attributes:
            error_parts.append(f"Missing attributes: {', '.join(missing_attributes)}")
            suggestions.extend(
                [f"Add attribute/method: {attr}" for attr in missing_attributes]
            )

        if signature_errors:
            error_parts.append("Signature mismatches:")
            error_parts.extend(f"  {error}" for error in signature_errors)
            suggestions.append("Check method signatures match the expected protocol")

        context = {
            "expected_type": get_type_name(expected_type),
            "actual_type": get_type_name(obj.__class__),
            "artifact_name": str(obj)[:999],
        }

        message = (
            f"Instance of {get_type_name(obj.__class__)} does not conform to protocol {get_type_name(expected_type)}:\n"
            + "\n".join(error_parts)
        )
        raise ConformanceError(message, suggestions, context)

    return obj


# -----------------------------------------------------------------------------
# Cache Management Functions
# -----------------------------------------------------------------------------


def clear_validation_cache() -> int:
    """Clear the validation cache and return number of entries cleared."""
    with _validation_cache._lock:
        count = len(_validation_cache._cache)
        _validation_cache._cache.clear()
        return count


def get_cache_stats() -> Dict[str, Any]:
    """Get validation cache statistics."""
    with _validation_cache._lock:
        total_entries = len(_validation_cache._cache)
        expired_entries = 0
        current_time = time.time()

        for _, (_, timestamp, _) in _validation_cache._cache.items():
            if current_time - timestamp >= _validation_cache.ttl:
                expired_entries += 1

        return {
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "active_entries": total_entries - expired_entries,
            "max_size": _validation_cache.max_size,
            "ttl_seconds": _validation_cache.ttl,
        }


def configure_validation_cache(
    max_size: Optional[int] = None, ttl_seconds: Optional[float] = None
) -> None:
    """Configure validation cache settings."""
    global _validation_cache

    if max_size is not None:
        _validation_cache.max_size = max_size

    if ttl_seconds is not None:
        _validation_cache.ttl = ttl_seconds
