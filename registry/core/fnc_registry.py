r"""Baseclass for registering functions.

This module defines the FunctionalRegistry class, a specialized mutable registry
for functions. It validates functions at registration time and supports both
immediate and deferred registration (where applicable).

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph FunctionalRegistry {
    "MutableRegistryValidatorMixin" -> "FunctionalRegistry";
}
\enddot
"""

import logging
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
from ._dev_utils import get_module_members  # _dev_utils
from ._validator import ValidationError  # _validator
from ._validator import (
    get_func_name,
    ConformanceError,
    validate_function,
    validate_function_parameters,
)

# Type variables for function return type and parameters.
R = TypeVar("R")
P = ParamSpec("P")
logger = logging.getLogger(__name__)


# pylint: disable=abstract-method
class FunctionalRegistry(MutableRegistryValidatorMixin[Hashable, Callable[P, R]], ABC):
    """
    Class for registering functions.

    This registry performs runtime validations for functions, ensuring that they
    conform to expected signatures and behaviors. It provides utility methods for
    registering and unregistering functions as well as for registering all functions
    found in a module.

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
        Initialize a subclass of FunctionalRegistry.

        This method sets up the function repository and configures validation
        based on the provided flags.

        Parameters:
            strict (bool): If True, enforce strict signature validation.
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
        Validate a function before registration.

        The validation involves ensuring that the value is a function and that it
        conforms to the expected signature (if strict mode is enabled).

        Parameters:
            value (Any): The function to validate.

        Returns:
            Callable[P, R]: The validated function.

        Raises:
            ValidationError: If value is not a function.
            ConformanceError: If value does not match the expected signature.
        """
        # Check that 'value' is a valid function
        value = validate_function(value)

        # Apply signature validation if strict mode is enabled
        if cls._strict:
            param, ret = get_args(cls.__orig_bases__[0])
            if sys.version_info < (3, 10):
                param = list(get_args(param))
            callable_type = Callable[param, ret]  # type: ignore
            value = validate_function_parameters(value, expected_type=callable_type)

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
        Perform a runtime subclass check for functions.

        This custom subclass check ensures that a function not only is callable
        but also passes the validation.

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
        Register all callable functions found in a given module.

        This function retrieves all members from the specified module, filters for
        callable objects, and registers each function in the registry.

        Parameters:
            module (Any): The module from which to retrieve functions.
            raise_error (bool): If True, raise errors on validation failures;
                                otherwise, ignore them.

        Returns:
            Any: The module after processing its members.
        """
        # Retrieve all module members
        members = get_module_members(module)
        # Try to register each member as a function
        for obj in members:
            try:
                validate_function(obj)  # Pre-filter for functions
                cls.register_artifact(obj)
            except ValidationError as e:
                if raise_error:
                    raise ValidationError(e)
                else:
                    logger.debug(e)
        return module
