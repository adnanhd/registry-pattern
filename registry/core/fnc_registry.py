r"""Baseclass for registering functions.

This module defines the FunctionalRegistry class, a specialized mutable registry
for functions. It validates functions at registration time and supports both
immediate and deferred registration (where applicable).

Doxygen Dot Graph of Inheritance:
-----------------------------------
\dot
digraph FunctionalRegistry {
    "MutableRegistry" -> "FunctionalRegistry";
}
\enddot
"""

import sys
from functools import partial
from typing_compat import (
    Any,
    Callable,
    ClassVar,
    Hashable,
    Tuple,
    Type,
    TypeVar,
    Iterable,
    get_args,
    ParamSpec,
)
from warnings import warn

from ..utils import (
    # base
    MutableRegistry,
    # _dev_utils
    _def_checking,
    compose,
    get_module_members,
    # _validator
    ConformanceError,
    ValidationError,
    validate_function,
    validate_function_parameters,
)

# Type variables for function return type and parameters.
R = TypeVar("R")
P = ParamSpec("P")


# pylint: disable=abstract-method
class FunctionalRegistry(MutableRegistry[Hashable, Callable[P, R]]):
    """
    Metaclass for registering functions.

    This registry performs runtime validations for functions, ensuring that they
    conform to expected signatures and behaviors. It provides utility methods for
    registering and unregistering functions as well as for registering all functions
    found in a module.

    Attributes:
        runtime_conformance_checking (Callable[[Callable[P, R]], Callable[P, R]]):
            Function used to validate the conformance of registered functions.
    """

    runtime_conformance_checking: Callable[[Callable[P, R]], Callable[P, R]] = (
        _def_checking
    )
    __orig_bases__: ClassVar[Tuple[Type, ...]]
    __slots__ = ()

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

        This method sets up the function repository and configures runtime
        conformance checking based on the provided flags. The 'coercion' flag is
        currently not supported.

        Parameters:
            strict (bool): If True, enforce strict signature validation using
                           validate_function_parameters.
            coercion (bool): If True, attempt to enable coercion (currently not supported).
            **kwargs: Additional keyword arguments passed to the superclass initializer.
        """
        super().__init_subclass__(**kwargs)
        if coercion:
            warn("Coercion not yet supported! Thus, it has no effect :(")
        # Initialize the registry repository as an empty dictionary.
        cls._repository = dict()

        # Retrieve the function type parameters from the original bases.
        param, ret = get_args(cls.__orig_bases__[0])
        if sys.version_info < (3, 10):
            param = list(get_args(param))

        # Construct a Callable type for validation.
        callable_type: Type[Callable[P, R]] = Callable[param, ret]  # type: ignore

        # If strict mode is enabled, set the runtime conformance checking function.
        if strict:
            cls.runtime_conformance_checking = partial(
                validate_function_parameters, expected_type=callable_type
            )

    @classmethod
    def validate_artifact(cls, value: Callable[P, R]) -> Callable[P, R]:
        """
        Validate a function before registration.

        The validation involves ensuring that the value is a function and that it
        conforms to the expected signature (if strict mode is enabled).

        Parameters:
            value (Callable[P, R]): The function to validate.

        Returns:
            Callable[P, R]: The validated function.

        Raises:
            ValidationError: If the function fails validation.
        """
        # Check that 'value' is a valid function.
        validate_function(value)
        # Run the conformance checking (which may be a no-op if not in strict mode).
        return cls.runtime_conformance_checking(value)

    @classmethod
    def __subclasscheck__(cls, value: Any) -> bool:
        """
        Perform a runtime subclass check for functions.

        This custom subclass check ensures that a function not only is callable
        but also passes the conformance validation.

        Parameters:
            value (Any): The function to check.

        Returns:
            bool: True if the function passes validation; False otherwise.
        """
        try:
            validate_function(value)
        except ValidationError:
            print("no validation")
            return False

        try:
            cls.runtime_conformance_checking(value)
        except ConformanceError:
            print("no conformation")
            return False

        return True
        # Note: The following line is unreachable and has been removed:
        # cls._validate_artifact = compose(*validators)

    @classmethod
    def register_function(cls, func: Callable[P, R]) -> Callable[P, R]:
        """
        Register a function in the registry.

        The function is stored in the registry under its __name__ attribute.

        Parameters:
            func (Callable[P, R]): The function to register.

        Returns:
            Callable[P, R]: The registered function.
        """
        cls.register_artifact(func.__name__, func)
        return func

    @classmethod
    def unregister_function(cls, func: Callable[P, R]) -> Callable[P, R]:
        """
        Unregister a function from the registry.

        The function is removed from the registry using its __name__ attribute.

        Parameters:
            func (Callable[P, R]): The function to unregister.

        Returns:
            Callable[P, R]: The unregistered function.
        """
        cls.unregister_artifacts(func.__name__)
        return func

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
        # Retrieve all module members and filter for callables.
        members: Iterable[Callable] = filter(callable, get_module_members(module))
        for obj in members:
            cls.register_function(obj)
        return module
