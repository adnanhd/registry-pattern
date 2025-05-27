r"""Enhanced FunctionalRegistry with improved validation system.

This module defines the FunctionalRegistry class, a specialized mutable registry
for functions with comprehensive validation, rich error context, and performance
optimizations through validation caching.

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph FunctionalRegistry {
    "MutableRegistryValidatorMixin" -> "FunctionalRegistry";
}
\enddot
"""

import logging
import inspect
import sys
from abc import ABC
from warnings import warn

from typing_compat import (
    Any,
    Callable,
    Union,
    Optional,
    ClassVar,
    Hashable,
    ParamSpec,
    Tuple,
    Type,
    TypeVar,
    get_args,
)
from typing import overload

from ..mixin import MutableRegistryValidatorMixin
from ._dev_utils import get_module_members
from ._validator import (
    ValidationError,
    ConformanceError,
    get_func_name,
    validate_function,
    validate_function_parameters,
)

# Type variables for function return type and parameters.
R = TypeVar("R")
P = ParamSpec("P")
logger = logging.getLogger(__name__)


class FunctionalRegistry(MutableRegistryValidatorMixin[Hashable, Callable[P, R]], ABC):
    """
    Enhanced registry for registering functions with comprehensive validation.

    This registry performs runtime validations for functions with rich error context
    and helpful suggestions, ensuring that they conform to expected signatures and
    behaviors. It provides utility methods for registering and unregistering functions
    as well as for registering all functions found in a module.

    Attributes:
        _repository (dict): Dictionary for storing registered functions.
        _strict (bool): Whether to enforce signature conformance.
    """

    _repository: dict
    _strict: bool = False
    __orig_bases__: ClassVar[Tuple[Type, ...]]
    __slots__ = ()

    @classmethod
    def _get_mapping(cls):
        """Return the repository dictionary."""
        return cls._repository

    @classmethod
    def __class_getitem__(cls, params: Any) -> Any:
        """
        Enable subscripted type hints for the registry (e.g., FunctionalRegistry[...]).

        For Python versions prior to 3.10, adjust the provided parameters accordingly.

        Parameters:
            params (Any): The type parameters provided.

        Returns:
            Any: The adjusted type parameters via the superclass method.
        """
        if sys.version_info < (3, 10):
            args, kwargs = params  # Expecting a tuple of (args, kwargs)
            params = (Tuple[tuple(args)], kwargs)
        return super().__class_getitem__(params)  # type: ignore

    @classmethod
    def __init_subclass__(
        cls, strict: bool = False, coercion: bool = False, **kwargs: Any
    ) -> None:
        """
        Initialize a subclass of FunctionalRegistry with enhanced validation configuration.

        This method sets up the function repository and configures validation
        based on the provided flags.

        Parameters:
            strict (bool): If True, enforce strict signature validation with detailed feedback.
            coercion (bool): If True, attempt to enable coercion (currently not supported).
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        if coercion:
            warn("Coercion not yet supported! Thus, it has no effect :(")
        # Initialize the registry repository as an empty dictionary.
        cls._repository = dict()
        # Store validation flags
        cls._strict = strict

    @classmethod
    def _intern_artifact(cls, value: Any) -> Callable[P, R]:
        """
        Validate a function before registration with enhanced error reporting.

        The validation involves ensuring that the value is a function and that it
        conforms to the expected signature (if strict mode is enabled).

        Parameters:
            value (Any): The function to validate.

        Returns:
            Callable[P, R]: The validated function.

        Raises:
            ValidationError: If value is not a function, with suggestions for fixing.
            ConformanceError: If value does not match the expected signature, with detailed feedback.
        """
        # Check that 'value' is a valid function
        try:
            value = validate_function(value)
        except ValidationError as e:
            # Add registry-specific suggestions
            if hasattr(e, "suggestions"):
                e.suggestions.extend(
                    [
                        f"Registry {cls.__name__} only accepts callable functions",
                        "Make sure you're registering a function, not a variable or class",
                        "Use 'def function_name():' to define functions",
                    ]
                )
            if hasattr(e, "context"):
                e.context.update(
                    {
                        "registry_name": cls.__name__,
                        "registry_type": "FunctionalRegistry",
                    }
                )
            raise

        # Apply signature validation if strict mode is enabled
        if cls._strict:
            param, ret = get_args(cls.__orig_bases__[0])
            if sys.version_info < (3, 10):
                param = list(get_args(param))
            callable_type = Callable[param, ret]  # type: ignore
            try:
                value = validate_function_parameters(value, expected_type=callable_type)
            except ConformanceError as e:
                # Add registry-specific context
                if hasattr(e, "context"):
                    e.context.update(
                        {
                            "registry_name": cls.__name__,
                            "registry_mode": "strict",
                            "expected_signature": (
                                str(callable_type)
                                if "callable_type" in locals()
                                else "unknown"
                            ),
                        }
                    )
                if hasattr(e, "suggestions"):
                    e.suggestions.extend(
                        [
                            f"Use cls.validate_artifact(your_function) to check signature before registration",
                            f"Registry {cls.__name__} requires strict signature conformance",
                            "Check parameter types and return type annotations",
                        ]
                    )
                raise
            except Exception as e:
                # Convert unexpected errors to ValidationError with context
                suggestions = [
                    "Check that the function signature matches the expected type",
                    "Verify that type annotations are correct",
                    f"Expected signature type: {getattr(cls, '__orig_bases__', ['unknown'])[0] if hasattr(cls, '__orig_bases__') else 'unknown'}",
                ]
                context = {
                    "registry_name": cls.__name__,
                    "registry_mode": "strict",
                    "operation": "signature_validation",
                    "artifact_name": get_func_name(value),
                }
                enhanced_error = ValidationError(
                    f"Function signature validation failed: {e}", suggestions, context
                )
                raise enhanced_error from e

        return value

    @classmethod
    def _extern_artifact(cls, value: Callable[P, R]) -> Callable[P, R]:
        """
        Validate a function when retrieving from the registry.

        This method is a no-op by default, as validation is primarily
        done during registration. Override this method to add validation
        during retrieval if needed.

        Parameters:
            value (Callable[P, R]): The function to validate.

        Returns:
            Callable[P, R]: The validated function.
        """
        return value

    @classmethod
    def _identifier_of(cls, item: Callable[P, R]) -> Hashable:
        """
        Generate a unique identifier for a function.

        Parameters:
            item (Callable[P, R]): The function to generate an identifier for.

        Returns:
            Hashable: A unique identifier for the function.
        """
        return get_func_name(validate_function(item))

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Perform a runtime subclass check for functions with caching.

        This custom subclass check ensures that a function not only is callable
        but also passes the validation. Results are cached for performance.

        Parameters:
            value (Any): The function to check.

        Returns:
            bool: True if the function passes validation; False otherwise.
        """
        try:
            # Use _intern_artifact for validation during subclass check
            cls._intern_artifact(value)
            return True
        except (ValidationError, ConformanceError):
            return False

    @classmethod
    def register_module_functions(cls, module: Any, raise_error: bool = True) -> Any:
        """
        Register all callable functions found in a given module with enhanced error reporting.

        This function retrieves all members from the specified module, filters for
        callable objects, and registers each function in the registry with detailed
        error context and suggestions.

        Parameters:
            module (Any): The module from which to retrieve functions.
            raise_error (bool): If True, raise errors on validation failures;
                                otherwise, log them with context.

        Returns:
            Any: The module after processing its members.
        """
        # Retrieve all module members
        try:
            members = get_module_members(module)
        except Exception as e:
            enhanced_error = ValidationError(
                f"Failed to retrieve members from module: {e}",
                suggestions=[
                    "Ensure the module is properly imported",
                    "Check that the module exists and is accessible",
                    f"Module: {getattr(module, '__name__', str(module))}",
                ],
                context={
                    "operation": "register_module_functions",
                    "module_name": getattr(module, "__name__", str(module)),
                },
            )
            if raise_error:
                raise enhanced_error from e
            else:
                logger.debug(str(enhanced_error))
                return module

        registration_results = {
            "successful": [],
            "failed": [],
            "errors": [],
            "skipped": [],
        }

        # Try to register each member as a function
        for obj in members:
            obj_name = getattr(obj, "__name__", str(obj)[:999])

            try:
                # Pre-filter for functions to avoid unnecessary validation
                validate_function(obj)
                cls.register_artifact(obj)
                registration_results["successful"].append(obj_name)

            except ConformanceError as e:
                registration_results["failed"].append(obj_name)
                error_info = {
                    "name": obj_name,
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "type": "conformance_error",
                }
                registration_results["errors"].append(error_info)

                if raise_error:
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "module_name": getattr(module, "__name__", str(module)),
                                "operation": "register_module_functions",
                            }
                        )
                    raise
                else:
                    logger.debug(f"Conformance check failed for {obj_name}: {e}")

            except ValidationError as e:
                registration_results["failed"].append(obj_name)
                error_info = {
                    "name": obj_name,
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "type": "validation_error",
                }
                registration_results["errors"].append(error_info)

                if raise_error:
                    # Add module context to the error
                    if hasattr(e, "context"):
                        e.context.update(
                            {
                                "module_name": getattr(module, "__name__", str(module)),
                                "operation": "register_module_functions",
                            }
                        )
                    raise
                else:
                    logger.debug(f"Validation failed for {obj_name}: {e}")

            except Exception as e:
                # Handle unexpected errors
                enhanced_error = ValidationError(
                    f"Unexpected error registering {obj_name}: {e}",
                    suggestions=[
                        "Check that the object is a valid function",
                        "Verify the function signature if in strict mode",
                        "Ensure the module is properly structured",
                    ],
                    context={
                        "module_name": getattr(module, "__name__", str(module)),
                        "operation": "register_module_functions",
                        "artifact_name": obj_name,
                    },
                )

                registration_results["failed"].append(obj_name)
                registration_results["errors"].append(
                    {
                        "name": obj_name,
                        "error": str(enhanced_error),
                        "suggestions": enhanced_error.suggestions,
                        "type": "unexpected_error",
                    }
                )

                if raise_error:
                    raise enhanced_error from e
                else:
                    logger.debug(f"Unexpected error with {obj_name}: {enhanced_error}")

        # Log summary
        logger.info(
            f"Module function registration complete: {len(registration_results['successful'])} successful, {len(registration_results['failed'])} failed"
        )

        return module

    @classmethod
    def get_signature_info(cls) -> dict:
        """
        Get information about the expected function signature for this registry.

        Returns:
            Dictionary containing signature information and requirements.
        """
        info = {
            "registry_name": cls.__name__,
            "strict_mode": cls._strict,
            "expected_signature": None,
            "parameter_types": [],
            "return_type": None,
        }

        if hasattr(cls, "__orig_bases__") and cls.__orig_bases__:
            try:
                param, ret = get_args(cls.__orig_bases__[0])
                if sys.version_info < (3, 10):
                    param = list(get_args(param))

                info["expected_signature"] = f"Callable[{param}, {ret}]"
                info["parameter_types"] = (
                    param if isinstance(param, (list, tuple)) else [param]
                )
                info["return_type"] = ret
            except Exception as e:
                info["signature_error"] = str(e)

        return info

    @classmethod
    def diagnose_function_failure(cls, failed_function: Callable) -> dict:
        """
        Diagnose why a function failed to register and provide detailed suggestions.

        Parameters:
            failed_function: The function that failed to register.

        Returns:
            Dictionary containing diagnostic information and suggestions.
        """
        diagnosis = {
            "function_name": get_func_name(failed_function),
            "is_callable": callable(failed_function),
            "is_function": inspect.isfunction(failed_function)
            or inspect.isbuiltin(failed_function),
            "validation_errors": [],
            "suggestions": [],
            "signature_info": None,
            "registry_config": {
                "strict_mode": cls._strict,
                "registry_name": cls.__name__,
            },
        }

        # Test basic function validation
        try:
            validate_function(failed_function)
        except ValidationError as e:
            diagnosis["validation_errors"].append(
                {
                    "type": "function_validation",
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                }
            )
            diagnosis["suggestions"].extend(getattr(e, "suggestions", []))

        # Get function signature info
        try:
            sig = inspect.signature(failed_function)
            diagnosis["signature_info"] = {
                "parameters": list(sig.parameters.keys()),
                "parameter_annotations": {
                    name: param.annotation for name, param in sig.parameters.items()
                },
                "return_annotation": sig.return_annotation,
            }
        except Exception as e:
            diagnosis["signature_info"] = {"error": str(e)}

        # Test signature conformance if in strict mode
        if cls._strict:
            try:
                param, ret = get_args(cls.__orig_bases__[0])
                if sys.version_info < (3, 10):
                    param = list(get_args(param))
                callable_type = Callable[param, ret]  # type: ignore
                validate_function_parameters(
                    failed_function, expected_type=callable_type
                )
            except ConformanceError as e:
                diagnosis["validation_errors"].append(
                    {
                        "type": "signature_validation",
                        "error": str(e),
                        "suggestions": getattr(e, "suggestions", []),
                    }
                )
                diagnosis["suggestions"].extend(getattr(e, "suggestions", []))
            except Exception as e:
                diagnosis["validation_errors"].append(
                    {
                        "type": "signature_analysis_error",
                        "error": str(e),
                        "suggestions": [
                            "Check that the registry has a valid signature specification"
                        ],
                    }
                )

        # Add general suggestions based on registry configuration
        if cls._strict:
            diagnosis["suggestions"].append(
                "This registry requires strict signature conformance"
            )
            diagnosis["suggestions"].append(
                "Check parameter types and return type annotations"
            )

        diagnosis["suggestions"].append(
            f"Use {cls.__name__}.validate_artifact(your_function) to test before registration"
        )

        # Add signature-specific suggestions
        if diagnosis["signature_info"] and "error" not in diagnosis["signature_info"]:
            sig_info = diagnosis["signature_info"]
            if not sig_info["parameter_annotations"]:
                diagnosis["suggestions"].append(
                    "Add type annotations to function parameters"
                )
            if sig_info["return_annotation"] == inspect.Signature.empty:
                diagnosis["suggestions"].append(
                    "Add return type annotation to function"
                )

        return diagnosis

    @classmethod
    def validate_function_batch(cls, functions: dict) -> dict:
        """
        Validate multiple functions without registering them.

        Parameters:
            functions: Dictionary mapping names to functions.

        Returns:
            Dictionary containing validation results for each function.
        """
        results = {"valid": [], "invalid": [], "results": {}}

        for name, func in functions.items():
            try:
                cls._intern_artifact(func)
                results["valid"].append(name)
                results["results"][name] = {
                    "status": "valid",
                    "function_name": get_func_name(func),
                }
            except (ValidationError, ConformanceError) as e:
                results["invalid"].append(name)
                results["results"][name] = {
                    "status": "invalid",
                    "error": str(e),
                    "suggestions": getattr(e, "suggestions", []),
                    "function_name": (
                        get_func_name(func) if hasattr(func, "__name__") else "unknown"
                    ),
                }
            except Exception as e:
                results["invalid"].append(name)
                results["results"][name] = {
                    "status": "error",
                    "error": f"Unexpected error: {e}",
                    "suggestions": ["Check that the object is a valid function"],
                    "function_name": "unknown",
                }

        return results
