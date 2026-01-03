r"""Utility exceptions and helpers for registry mixins.

Exceptions:
    ValidationError: Base class carrying `suggestions` and `context` metadata.
    CoercionError: Raised when coercion of values fails.
    ConformanceError: Raised when callables/classes violate required signatures.
    InheritanceError: Raised when classes fail required inheritance.
    RegistryError: Raised when mapping errors occur.

Helpers:
    get_type_name: Return a human-readable type name.
    get_func_name: Return a human-readable function name.
    get_artifact_name: Return name from any artifact (class, function, or object).
    get_callable_signature: Extract signature from class or function.
    pydantic_to_dict: Convert Pydantic model to dict (v1/v2 compatible).
    build_error_context: Build standardized error context dict.
"""

import collections.abc
import inspect
import logging
import sys
import weakref
from functools import partial, reduce, wraps
from inspect import (
    Parameter,
    Signature,
    _empty,
    getmembers,
    isbuiltin,
    isclass,
    isfunction,
    ismethod,
    ismodule,
    signature,
)
from types import ModuleType
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

from typing_extensions import ParamSpec, get_args, runtime_checkable

# Python version compatibility
if sys.version_info >= (3, 9):
    from types import GenericAlias
    from typing import get_args, get_origin
elif sys.version_info >= (3, 8):
    from typing import get_args, get_origin

    try:
        from typing import _GenericAlias as GenericAlias
    except ImportError:
        GenericAlias = type(Callable[[int], int])
else:
    # Python 3.7
    def get_origin(tp):
        return getattr(tp, "__origin__", None)

    def get_args(tp):
        return getattr(tp, "__args__", ())

    try:
        from typing import _GenericAlias as GenericAlias
    except ImportError:
        GenericAlias = type(Callable[[int], int])

T = TypeVar("T")


logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Base exception for validation with structured context.

    Attributes:
        message: Human-readable error text.
        suggestions: List of short, imperative hints for remediation.
        context: Free-form key/value details safe to log and render.
    """

    def __init__(
        self,
        message: str,
        suggestions: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.suggestions = suggestions or []
        self.context = context or {}
        super().__init__(self._build_enhanced_message())

    def _build_enhanced_message(self) -> str:
        """Embed key context and suggestions into the exception string."""
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


class CoercionError(ValidationError):
    """Raised when a value cannot be coerced into the required representation."""


class ConformanceError(ValidationError):
    """Raised when a callable or class does not conform to required signatures."""


class InheritanceError(ValidationError):
    """Raised when a class does not inherit from a required base."""


class RegistryError(ValidationError):
    """Raised for key-related mapping errors with rich context attached."""


def get_type_name(cls: type, qualname: bool = False) -> str:
    """Return a readable name for a type.

    Args:
        cls: The class or type object.
        qualname: If True, return the qualified name when available.

    Returns:
        The type's `__qualname__`, `__name__`, or a string fallback.
    """
    if not isclass(cls):
        raise ValidationError(f"{cls} is not a class")
    if qualname and hasattr(cls, "__qualname__"):
        return getattr(cls, "__qualname__")
    elif hasattr(cls, "__name__"):
        return getattr(cls, "__name__")
    else:
        return str(cls)


def get_func_name(func: Callable[..., Any], qualname: bool = False) -> str:
    """Return a readable name for a function.

    Args:
        func: The function object.
        qualname: If True, return the qualified name when available.

    Returns:
        The function's `__qualname__`, `__name__`, or a string fallback.
    """
    if not callable(func):
        raise ValidationError(f"{func} is not callable")
    while hasattr(func, "__wrapped__"):
        func = getattr(func, "__wrapped__")
    return getattr(func, "__qualname__" if qualname else "__name__", str(func))


def get_artifact_name(artifact: Any, qualname: bool = False) -> str:
    """Return a readable name for any artifact (class, function, or object).

    Args:
        artifact: The artifact to name.
        qualname: If True, return the qualified name when available.

    Returns:
        The artifact's name or string representation.
    """
    if isclass(artifact):
        return get_type_name(artifact, qualname)
    if callable(artifact):
        return get_func_name(artifact, qualname)
    return getattr(artifact, "__name__", str(artifact))


def _def_checking(v: Any) -> Any:
    return v


def get_protocol(cls: type):
    """Get the protocol from the class."""
    assert isclass(cls), f"{cls} is not a class"
    assert Generic in cls.mro(), f"{cls} is not a generic class"

    type_arg = get_args(cls.__orig_bases__[0])[0]

    # If it's already a runtime_checkable Protocol, just return it
    if hasattr(type_arg, "_is_runtime_protocol") and type_arg._is_runtime_protocol:
        return type_arg

    # If it's a Protocol but not runtime_checkable, make it so
    if hasattr(type_arg, "_is_protocol") and type_arg._is_protocol:
        return runtime_checkable(type_arg)

    # Otherwise, it's just a regular type, so return it as is
    return type_arg


P = ParamSpec("P")
"""Type variable for the parameters."""
M = TypeVar("M")
"""Type variable for the middle value."""
R = TypeVar("R")
"""Type variable for the result value."""


def get_subclasses(cls: type) -> List[type]:
    """Get subclasses of a class."""
    if isclass(cls):
        return cls.__subclasses__()
    raise ValidationError(f"{cls} is not a class")


def get_module_name(module: ModuleType) -> str:
    """Get name of a module."""
    assert ismodule(module), f"{module} is not a module"
    return module.__name__


def get_object_name(obj: Any) -> str:
    """Get name of an object."""
    assert hasattr(obj, "__name__"), f"{obj} is not an object"
    return obj.__name__


def get_module_members(
    module: ModuleType, ignore_all_keyword: bool = False
) -> List[Any]:
    """Get members of a module."""
    assert ismodule(module), f"{module} is not a module"
    if ignore_all_keyword or not hasattr(module, "__all__"):
        _names, members = zip(*getmembers(module))
    else:
        _members = filter(lambda m: isinstance(m, str), module.__all__)
        _members = filter(lambda m: hasattr(module, m), _members)
        members = tuple(map(lambda m: getattr(module, m), _members))

    result = []
    for member in filter(
        lambda m: hasattr(m, "__name__") and not m.__name__.startswith("_"), members
    ):
        if isclass(member):
            result.append(member)
        elif (isfunction(member) or isbuiltin(member)) and callable(member):
            result.append(member)
        elif ismethod(member) and callable(member):
            logging.info(f"Method {member.__name__} found")
            # result.append(member)
        else:
            pass
    return result


def compose_two_funcs(
    f: Callable[P, M], g: Callable[[M], R], wrap: bool = True
) -> Callable[P, R]:
    """Compose two functions"""
    assert callable(f), "First function must be callable"
    assert callable(g), "Second function must be callable"

    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return g(f(*args, **kwargs))

    return wraps(f)(wrapper) if wrap else wrapper


def compose(*functions: Callable[..., Any], wrap: bool = True) -> Callable[..., Any]:
    """Compose functions"""
    composed_functions = reduce(
        partial(compose_two_funcs, wrap=False), reversed(functions)
    )
    return wraps(functions[0])(composed_functions) if wrap else composed_functions


def _validate_function_signature(
    func: Callable[..., Any], expected_callable_alias: GenericAlias
) -> None:
    """Validate that func's signature matches the expected Callable type annotation."""
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(
            "Validating method signature %s against %s",
            get_func_name(func),
            expected_callable_alias,
        )

    # First check if expected_callable_alias is a generic alias (has __origin__)
    if not hasattr(expected_callable_alias, "__origin__"):
        raise ConformanceError(
            f"expected_callable_alias must be a Callable type annotation, got {type(expected_callable_alias)}",
            ["Pass a Callable type like Callable[[int, str], bool]"],
            {"artifact_name": get_func_name(func), "operation": "signature_inspection"},
        )

    # Verify expected_callable_alias is a Callable type
    origin = get_origin(expected_callable_alias)
    # In Python 3.7+, origin could be typing.Callable or collections.abc.Callable
    is_callable = origin is collections.abc.Callable

    if not is_callable:
        raise ConformanceError(
            f"expected_callable_alias must be a Callable type annotation, got origin {origin}",
            ["Pass a Callable type like Callable[[int, str], bool]"],
            {"artifact_name": get_func_name(func), "operation": "signature_inspection"},
        )

    try:
        actual_sig = inspect.signature(func)
    except (ValueError, TypeError) as e:
        raise ConformanceError(
            f"Cannot inspect function signature: {e}",
            ["Ensure function is a valid callable"],
            {"artifact_name": get_func_name(func), "operation": "signature_inspection"},
        )

    # Extract parameter types and return type from Callable annotation
    # e.g., Callable[[int, str], bool] -> ([int, str], bool)
    type_args = get_args(expected_callable_alias)
    if len(type_args) != 2:
        raise ConformanceError(
            f"Invalid Callable type annotation: {expected_callable_alias}",
            ["Use format: Callable[[param_types...], return_type]"],
            {"artifact_name": get_func_name(func), "operation": "signature_inspection"},
        )

    expected_param_types, expected_return_type = type_args

    # Handle Callable[..., ReturnType] case (ellipsis means any params)
    if expected_param_types is Ellipsis:
        # Only validate return type
        if (
            expected_return_type != inspect.Signature.empty
            and actual_sig.return_annotation != expected_return_type
        ):
            if actual_sig.return_annotation == inspect.Signature.empty:
                raise ConformanceError(
                    "Missing return type annotation",
                    [f"Annotate return: -> {get_type_name(expected_return_type)}"],
                    {
                        "artifact_name": get_func_name(func),
                        "expected_type": str(expected_callable_alias),
                    },
                )
            else:
                raise ConformanceError(
                    f"Return type mismatch: expected {get_type_name(expected_return_type)}, "
                    + f"got {get_type_name(actual_sig.return_annotation)}",
                    ["Align return annotation"],
                    {
                        "artifact_name": get_func_name(func),
                        "expected_type": str(expected_callable_alias),
                    },
                )
        return

    errs: List[str] = []
    hints: List[str] = []

    actual_params = list(actual_sig.parameters.values())

    # Validate parameter count
    if len(actual_params) != len(expected_param_types):
        errs.append(
            f"Parameter count mismatch: expected {len(expected_param_types)}, got {len(actual_params)}"
        )
        hints.append(f"Use exactly {len(expected_param_types)} parameters")

    # Validate each parameter type
    for i, (actual_param, expected_type) in enumerate(
        zip(actual_params, expected_param_types)
    ):
        if actual_param.annotation == inspect.Parameter.empty:
            errs.append(f"Parameter '{actual_param.name}' missing type annotation")
            hints.append(
                f"Annotate: {actual_param.name}: {get_type_name(expected_type)}"
            )
        elif actual_param.annotation != expected_type:
            errs.append(
                f"Parameter '{actual_param.name}' type mismatch: expected {get_type_name(expected_type)}, "
                + f"got {get_type_name(actual_param.annotation)}"
            )
            hints.append("Align parameter type annotation")

    # Validate return type
    if expected_return_type != inspect.Signature.empty:
        if actual_sig.return_annotation == inspect.Signature.empty:
            errs.append("Missing return type annotation")
            hints.append(f"Annotate return: -> {get_type_name(expected_return_type)}")
        elif actual_sig.return_annotation != expected_return_type:
            errs.append(
                f"Return type mismatch: expected {get_type_name(expected_return_type)}, "
                + f"got {get_type_name(actual_sig.return_annotation)}"
            )
            hints.append("Align return annotation")

    if errs:
        raise ConformanceError(
            "Method signature validation failed:\n"
            + "\n".join(f"  • {e}" for e in errs),
            hints,
            {
                "artifact_name": get_func_name(func),
                "expected_type": str(expected_callable_alias),
                "actual_type": str(actual_sig),
            },
        )


# -----------------------------------------------------------------------------
# New Utility Functions for Code Simplification
# -----------------------------------------------------------------------------


def get_callable_signature(
    artifact: Union[Callable[..., Any], type],
) -> Tuple[str, Signature, List[Parameter]]:
    """Extract signature from a class or callable, removing 'self' for classes.

    Args:
        artifact: A class or callable to inspect.

    Returns:
        Tuple of (name, signature, parameters list without 'self').

    Raises:
        ValidationError: If artifact is not a class or callable.
    """
    if isclass(artifact):
        name = getattr(artifact, "__name__", str(artifact))
        try:
            sig = signature(artifact.__init__)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Cannot inspect __init__ of class {name}: {e}",
                ["Provide an explicit __init__ with type annotations"],
                {"artifact_name": name, "operation": "inspect_signature"},
            ) from e
        params = list(sig.parameters.values())
        if params and params[0].name == "self":
            params = params[1:]
        sig = sig.replace(parameters=params)
        return name, sig, params
    elif callable(artifact):
        name = get_func_name(artifact, qualname=False)
        try:
            sig = signature(artifact)
        except (TypeError, ValueError) as e:
            raise ValidationError(
                f"Cannot inspect callable {name}: {e}",
                ["Ensure it's a Python callable with inspectable signature"],
                {"artifact_name": name, "operation": "inspect_signature"},
            ) from e
        return name, sig, list(sig.parameters.values())
    else:
        raise ValidationError(
            f"{artifact!r} is neither a class nor a callable",
            ["Pass a function or class object"],
            {"actual_type": type(artifact).__name__},
        )


def pydantic_to_dict(model: Any) -> Dict[str, Any]:
    """Convert a Pydantic model to dict, compatible with v1 and v2.

    Args:
        model: A Pydantic BaseModel instance.

    Returns:
        Dictionary representation of the model.
    """
    if hasattr(model, "model_dump"):
        return dict(model.model_dump())
    return dict(model.dict())


def build_error_context(
    operation: str,
    registry_cls: Optional[type] = None,
    key: Optional[Any] = None,
    artifact: Optional[Any] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a standardized error context dictionary.

    Args:
        operation: The operation being performed.
        registry_cls: The registry class (optional).
        key: The key being operated on (optional).
        artifact: The artifact being operated on (optional).
        **extra: Additional context key-value pairs.

    Returns:
        Dictionary suitable for ValidationError context.
    """
    context: Dict[str, Any] = {"operation": operation}

    if registry_cls is not None:
        context["registry_name"] = getattr(registry_cls, "__name__", "Unknown")
        context["registry_type"] = get_type_name(type(registry_cls))
        if hasattr(registry_cls, "_get_mapping"):
            try:
                mapping = registry_cls._get_mapping()
                context["registry_size"] = len(mapping)
            except Exception:
                pass

    if key is not None:
        context["key"] = str(key)
        context["key_type"] = type(key).__name__

    if artifact is not None:
        context["artifact_name"] = get_artifact_name(artifact)
        context["artifact_type"] = type(artifact).__name__

    context.update(extra)
    return context


def is_hashable(value: Any) -> bool:
    """Check if a value is hashable.

    Args:
        value: The value to check.

    Returns:
        True if the value is hashable.
    """
    if isinstance(value, Hashable):
        return True
    try:
        hash(value)
        return True
    except Exception:
        return False


def cleanup_dead_weakrefs(mapping: Dict[Any, Any], key_is_weakref: bool = True) -> int:
    """Remove dead weakref entries from a mapping.

    Args:
        mapping: The mapping to clean up.
        key_is_weakref: If True, keys are weakrefs; if False, values are weakrefs.

    Returns:
        Number of dead entries removed.
    """
    dead = []
    for k, v in list(mapping.items()):
        try:
            target = k if key_is_weakref else v
            if isinstance(target, weakref.ref) and target() is None:
                dead.append(k)
        except Exception:
            dead.append(k)

    for k in dead:
        mapping.pop(k, None)

    if dead and logger.isEnabledFor(logging.DEBUG):
        logger.debug("Cleaned %d dead weakref entries", len(dead))

    return len(dead)
