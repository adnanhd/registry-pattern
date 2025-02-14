r"""Object configuration map registry.

This module defines the ObjectConfigMap class, a functional registry for
registering and tracking instances along with an associated configuration.
It uses a WeakKeyDictionary as its underlying repository so that the registry
does not prevent garbage collection of the keys. Runtime validation of keys
is performed via configurable conformance and inheritance checkers.

Doxygen Dot Graph for ObjectConfigMap:
----------------------------------------
\dot
digraph ObjectConfigMap {
    "BaseMutableRegistry" -> "ObjectConfigMap";
}
\enddot
"""

import sys
import weakref
from functools import partial, wraps
from typing_compat import (
    Any,
    Callable,
    Dict,
    Generic,
    Hashable,
    Type,
    TypeVar,
    get_args,
)

from weakref import WeakKeyDictionary

from ..utils import (
    # .base
    BaseMutableRegistry,
    # ._dev_utils
    _def_checking,
    get_protocol,
    # ._validator
    validate_instance_hierarchy,
    validate_instance_structure,
)

# -----------------------------------------------------------------------------
# Type Variables and Aliases
# -----------------------------------------------------------------------------

# K represents the type of the key (or object) being registered.
K = TypeVar("K", bound=Hashable)
# CfgT represents a configuration dictionary.
CfgT = Dict[str, Any]

# Use a subscriptable WeakKeyDictionary for Python 3.9+; otherwise, fall back.
if sys.version_info >= (3, 9):
    WeakKeyDictionaryT = weakref.WeakKeyDictionary[K, CfgT]
else:
    WeakKeyDictionaryT = weakref.WeakKeyDictionary


# -----------------------------------------------------------------------------
# ObjectConfigMap Class Definition
# -----------------------------------------------------------------------------


class ObjectConfigMap(BaseMutableRegistry[K, CfgT], Generic[K]):
    """
    Functional registry for registering instances with configuration.

    ObjectConfigMap maintains a mapping between objects (keys) and their associated
    configuration dictionaries. It leverages a WeakKeyDictionary to ensure that
    the registry does not prevent garbage collection of keys. Runtime validations
    for keys are performed via configurable conformance and inheritance checkers.

    Attributes:
        runtime_conformance_checking (Callable[..., Any]):
            A callable used for checking that a key conforms to an expected structure.
        runtime_inheritance_checking (Callable[..., Any]):
            A callable used for checking that a key meets inheritance requirements.
    """

    # Default checking functions; these may be overridden based on registry configuration.
    runtime_conformance_checking: Callable[..., Any] = _def_checking
    runtime_inheritance_checking: Callable[..., Any] = _def_checking
    __slots__ = ()

    @classmethod
    def __init_subclass__(cls, strict: bool = False, abstract: bool = False) -> None:
        """
        Initialize a subclass of ObjectConfigMap with appropriate runtime validators.

        This method initializes the repository using a WeakKeyDictionary and configures
        runtime validation functions for keys. If 'abstract' is True, then inheritance
        checking is enforced. If 'strict' is True, then conformance checking is enforced.

        Parameters:
            strict (bool): If True, enforce strict structure validation via
                           validate_instance_structure.
            abstract (bool): If True, enforce inheritance validation via
                             validate_instance_hierarchy.
        """
        super().__init_subclass__()
        cls._repository = WeakKeyDictionaryT()

        if abstract:
            cls.runtime_inheritance_checking = partial(
                validate_instance_hierarchy, expected_type=cls
            )

        if strict:
            cls.runtime_conformance_checking = partial(
                validate_instance_structure, expected_type=get_protocol(cls)
            )

    @classmethod
    def validate_artifact(cls, value: dict) -> CfgT:
        """
        Validate a configuration item.

        Ensures that the provided value is a dictionary.

        Parameters:
            value (dict): The configuration dictionary to validate.

        Returns:
            CfgT: The validated configuration dictionary.

        Raises:
            TypeError: If the provided value is not a dictionary.
        """
        if not isinstance(value, dict):
            raise TypeError(f"{value} is not a dict")
        return value

    @classmethod
    def validate_artifact_id(cls, key: Any) -> K:
        """
        Validate a key before registration.

        The key is checked for proper inheritance and structural conformance.

        Parameters:
            key (Any): The key to validate.

        Returns:
            K: The validated key.

        Raises:
            An exception as raised by the runtime checkers if the key does not conform.
        """
        cls.runtime_inheritance_checking(key)
        cls.runtime_conformance_checking(key)
        return key

    @classmethod
    def register_instance(cls, obj: K, cfg: Dict[str, Any]) -> K:
        """
        Register an instance along with its configuration.

        Parameters:
            obj (K): The instance to register.
            cfg (Dict[str, Any]): The configuration dictionary associated with the instance.

        Returns:
            K: The registered instance.
        """
        cls.register_artifact(obj, cfg)
        return obj

    @classmethod
    def unregister_instance(cls, obj: K) -> K:
        """
        Unregister an instance from the registry.

        Parameters:
            obj (K): The instance to unregister.

        Returns:
            K: The unregistered instance.
        """
        cls.unregister_artifacts(obj)
        return obj

    @classmethod
    def register_class_instances(cls, supercls: Type[K]) -> Type[K]:
        """
        Create a tracked version of a class so that all its instances are automatically registered.

        This method dynamically creates a new metaclass that overrides the __call__ method.
        Upon instantiation, each new object is registered in the configuration map with the
        instantiation keyword arguments as its configuration.

        Parameters:
            supercls (Type[K]): The class whose instances are to be tracked.

        Returns:
            Type[K]: A new class that wraps the original class and automatically registers instances.
        """
        meta: Type[Any] = type(supercls)

        class newmeta(meta):
            def __call__(self, **kwds: Any) -> K:
                # Create an instance of the original class.
                obj = super().__call__(**kwds)
                # Register the instance along with its configuration.
                return cls.register_instance(obj, kwds)

        # Copy metadata from the original metaclass.
        newmeta.__name__ = meta.__name__
        newmeta.__qualname__ = meta.__qualname__
        newmeta.__module__ = meta.__module__
        newmeta.__doc__ = meta.__doc__
        # Return the new tracked class.
        return newmeta(supercls.__name__, (supercls,), {})

    @classmethod
    def register_builder(cls, func: Callable[..., K]) -> Callable[..., K]:
        """
        Decorator to register a builder function.

        When the decorated function is called, its result (an instance) is automatically
        registered in the registry along with the keyword arguments used to build it.

        Parameters:
            func (Callable[..., K]): The builder function that creates an instance.

        Returns:
            Callable[..., K]: A wrapped version of the builder function that registers the instance.
        """

        @wraps(func)
        def wrapper(**kwds: Any) -> K:
            # Call the builder function to create an instance.
            obj = func(**kwds)
            # Register the instance along with its configuration.
            cls.register_instance(obj, kwds)
            return obj

        return wrapper
