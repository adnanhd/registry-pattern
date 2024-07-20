import sys
from typing import TypeVar, Protocol
from typing import Union, Optional, Type, List
from inspect import isclass

from .cls_registry import ClassRegistry
from .func_registry import FunctionalRegistry
from ._pydantic import BuilderValidator

if sys.version_info >= (3, 10):
    from typing import ParamSpec
    from types import EllipsisType
else:
    from typing_extensions import ParamSpec

    EllipsisType = type(Ellipsis)

if sys.version_info >= (3, 9):
    from typing import Annotated
else:
    from typing_extensions import Annotated


class AnyProtocol(Protocol):
    """Any protocol."""


T = TypeVar("T", bound=AnyProtocol)
P = ParamSpec("P")


def _dict_not_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def make_class_registry(
    name: str,
    *subcls: Type,
    protocol: Union[Type[T], Type] = type,
    coercion: bool = False,
    domain: Optional[str] = None,
    description: Optional[str] = None,
) -> ClassRegistry[T]:
    """Make a class registry."""
    assert domain is None, "Domains are not supported"
    assert not coercion, "Strict references are not supported"
    assert all(map(isclass, subcls)), "Only classes are supported"

    class Register(ClassRegistry[protocol]):
        """A class registry."""

        ignore_structural_subtyping = protocol is AnyProtocol
        ignore_abcnominal_subtyping = len(subcls) == 0

    Register.__name__ = f"{name}Registry"
    Register.__qualname__ = f"{name}Registry"
    Register.__module__ = protocol.__module__
    Register.__doc__ = description

    for _subcls in subcls:
        Register.register(_subcls)

    # TODO: return cacher later
    return Register()


def make_functional_registry(
    name: str,
    args: List[Type] = ...,
    ret: Type[T] = type,
    coercion: bool = False,
    domain: Optional[str] = None,
    description: Optional[str] = None,
) -> FunctionalRegistry[P, T]:
    """Make a functional registry."""
    assert domain == "", "Domains are not supported"
    assert not coercion, "Strict references are not supported"

    class Registry(FunctionalRegistry[args, ret]):  # type: ignore
        """A functional registry."""

    Registry.__name__ = f"{name}Registry"
    Registry.__qualname__ = f"{name}Registry"
    Registry.__module__ = ret.__module__
    Registry.__doc__ = description

    return Registry()


class PydanticBuilderValidatorMacro:
    """A macro for generating Pydantic builder validators."""

    @classmethod
    def __class_getitem__(cls, item):
        """Get the item."""
        source_type, registry = item
        return Annotated[source_type, BuilderValidator(registry=registry)]
