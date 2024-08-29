import inspect
import sys
from collections.abc import Callable
from inspect import isclass
from types import new_class
from typing import List
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
from typing import Union

from .fnc_registry import FunctionalRegistry
from .typ_registry import TypeRegistry

if sys.version_info >= (3, 10):
    from types import EllipsisType
    from typing import ParamSpec
else:
    from typing_extensions import ParamSpec

    EllipsisType = type(Ellipsis)

if sys.version_info >= (3, 9):
    pass
else:
    pass


class AnyProtocol(Protocol):
    """Any protocol."""


T = TypeVar("T", bound=AnyProtocol)
P = ParamSpec("P")


def _dict_not_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def type_registry_factory(
    name: str,
    *subcls: Type,
    protocol: Union[Type[T], Type] = type,
    coercion: bool = False,
    domain: Optional[str] = None,
    description: Optional[str] = None,
) -> Type[TypeRegistry[T]]:
    """Make a class registry."""
    assert domain is None, "Domains are not supported"
    assert not coercion, "Strict references are not supported"
    assert all(map(isclass, subcls)), "Only classes are supported"

    kwds = {"strict": protocol is not AnyProtocol, "abstract": len(subcls) != 0}

    class Register(TypeRegistry[protocol], **kwds):
        ...

    Register.__name__ = f"{name}Registry"
    Register.__qualname__ = f"{name}Registry"
    Register.__module__ = protocol.__module__
    Register.__doc__ = description

    for _subcls in subcls:
        Register.register(_subcls)

    # TODO: return cacher later
    return Register


def type_registry_decorator(
    strict: bool = False,
    abstract: bool = False,
    domain: Optional[str] = None,
) -> Callable[[Union[Type[T], Type]], Type[TypeRegistry[T]]]:
    def type_registry_wrapper(protocol: Union[Type[T], Type] = type):
        kwds = dict(strict=strict, abstract=abstract)
        return new_class(
            protocol.__name__ + "Registry", (TypeRegistry[protocol],), kwds
        )

    return type_registry_wrapper


def functional_registry_factory(
    name: str,
    args: List[type] = ...,
    ret: Type[T] = type,
    coercion: bool = False,
    domain: Optional[str] = None,
    description: Optional[str] = None,
) -> Type[FunctionalRegistry[P, T]]:
    """Make a functional registry."""
    assert domain is None, "Domains are not supported: "
    assert not coercion, "Strict references are not supported"

    kwds = {"strict": args != ... or ret is not type}

    class Registry(FunctionalRegistry[args, ret], **kwds):
        ...  # type: ignore

    Registry.__name__ = f"{name}Registry"
    Registry.__qualname__ = f"{name}Registry"
    Registry.__module__ = ret.__module__
    Registry.__doc__ = description

    return Registry


def functional_registry_decorator(
    strict: bool = False,
    coercion: bool = False,
    domain: Optional[str] = None,
):
    def functional_registry_wrapper(fn: Callable[P, T]):
        kwds = dict(strict=strict, coercion=coercion)
        param = [p.annotation for p in inspect.signature(fn).parameters.values()]
        ret = inspect.signature(fn).return_annotation
        return new_class(
            fn.__name__ + "Registry", (FunctionalRegistry[param, ret],), kwds
        )

    return functional_registry_wrapper
