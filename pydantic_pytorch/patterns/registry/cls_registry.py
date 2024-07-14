"""
Metaclass for registering classes.
"""

import abc
import inspect
from typing import (
    TypeVar,
    Generic,
    ClassVar,
    Dict,
    get_args,
    List,
    Any,
    Type,
    Protocol,
    runtime_checkable,
)
from ._reg_hooks import Composable as composable
from typeguard import check_type, TypeCheckError as TypeguardTypeCheckError
from .registry import RegistryMeta, ValidationError, RegistryError, TypeCheckError, _get_members


Artifact = TypeVar("Artifact")


@runtime_checkable
class AnyArtifact(Protocol):
    """A dummy protocol for type checking."""


class RegistryMetaConfig:
    """Configuration for the registry metaclass."""
    strict: bool = True
    structural: bool = True
    hierarchical: bool = True


class ClassKeyRegistry(Generic[Artifact], metaclass=RegistryMeta):
    """Metaclass for registering classes."""
    __artifact_protocol__: ClassVar[Type[Artifact]]
    __orig_bases__: ClassVar[tuple[type, ...]]
    _registrar: ClassVar[Dict[str, Artifact]]
    _structural_check: ClassVar[bool] = True
    _hierarchical_check: ClassVar[bool] = True

    @classmethod
    def __init_subclass__(cls) -> None:
        """Initialize the subclass."""
        print('__init_subclass__')

        @composable
        def validate_class(subcls: Type[Artifact]) -> Type[Artifact]:
            """Validate a class."""
            if not inspect.isclass(subcls):
                raise ValidationError(f"{subcls} is not a class")
            return subcls

        # Set __artifact_protocol__
        if cls._structural_check:
            __artifact_protocol__, = get_args(cls.__orig_bases__[0])

            # Check if the protocol is a protocol
            if not issubclass(__artifact_protocol__, Protocol):
                raise TypeError(f"{__artifact_protocol__} is not a protocol.")
            # Ensure if the protocol is a runtime protocol
            if not __artifact_protocol__._is_runtime_protocol:
                __artifact_protocol__ = runtime_checkable(
                    __artifact_protocol__)

            @validate_class.proloque_callback
            def structural_check(subcls: Type[Artifact]) -> Type[Artifact]:
                """Check the structure of a class."""
                try:
                    return check_type(subcls, __artifact_protocol__)
                except TypeguardTypeCheckError as exc:
                    raise ValidationError(
                        f"{subcls} is not of type {__artifact_protocol__}") from exc

        if cls._hierarchical_check:
            @validate_class.proloque_callback
            def hierarchical_check(subcls: Type[Artifact]) -> Type[Artifact]:
                """Check the hierarchy of a class."""
                if not issubclass(subcls, cls):
                    raise ValidationError(f"{subcls} is not of cls {cls}")
                return subcls

        cls.validate_class = validate_class

    @classmethod
    def validate_class_structure(cls, subcls: Type[Artifact]) -> Type[Artifact]:
        """Check the structure of a class."""
        try:
            return check_type(subcls, cls.__artifact_protocol__)
        except TypeguardTypeCheckError as exc:
            raise ValidationError(
                f"{subcls} is not of type {cls.__artifact_protocol__}") from exc


    @classmethod
    def validate_class_hierarchy(cls, subcls: Type[Artifact]) -> Type[Artifact]:
        """Check the hierarchy of a class."""
        if not issubclass(subcls, cls):
            raise ValidationError(f"{subcls} is not of cls {cls}")
        return subcls

    @classmethod
    def register_class(cls, subcls: Type[Artifact]) -> Type[Artifact]:
        """Register a subclass."""
        # Validate the class
        subcls = cls.validate_class(subcls)

        # Register the class
        return cls._add_registry(subcls.__name__, subcls)

    @classmethod
    def register_module_subclasses(cls, module):
        """Register all subclasses of a given module."""
        assert inspect.ismodule(module), f"{module} is not a module"
        for obj in _get_members(module).values():
            try:
                cls.register_class(obj)
            except (RegistryError, ValidationError, TypeCheckError, KeyError, AssertionError):
                pass
        return module

    @classmethod
    def register_subclasses(cls,
                            supercls: Artifact,
                            recursive: bool = False) -> Artifact:
        """Register all subclasses of a given superclass."""
        assert inspect.isclass(supercls), f"{supercls} is not a class"
        abc.ABCMeta.register(cls, supercls)
        for subcls in _get_subclasses(supercls):
            if recursive and _get_subclasses(subcls):
                cls.register_subclasses(subcls, recursive)
            cls.register_class(subcls)
        return supercls


def _get_subclasses(subcls) -> List[Type[Any]]:
    """Get members of a module."""
    assert inspect.isclass(subcls), f"{subcls} is not a class"
    return subcls.__subclasses__()
