from .registry import ClassRegistry, FunctionalRegistry
from typing import Type, TypeVar, _ProtocolMeta, ParamSpec
from types import new_class

Protocol = TypeVar("Protocol", bound=_ProtocolMeta)


def _dict_not_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def make_class_registry(
        name: str,
        cls: Type[Protocol],
        strict: bool = True,
        weak: bool = False,
        domain: str | None = None
) -> Type[ClassRegistry[Protocol]]:
    """Make a class registry."""
    if not weak:
        raise ValueError("Weak references are not supported")
    assert domain is None, "Domains are not supported"

    # kwds = _dict_not_none({'strict': strict, 'weak': weak, 'domain': domain})
    return new_class(f'{name}Registry', bases=(ClassRegistry[cls],), kwds=None)


def make_functional_registry(
        name: str,
        args: Type[ParamSpec],
        ret: Type,
        strict: bool = True,
        weak: bool = False,
        domain: str = ''
) -> Type[FunctionalRegistry[Type[ParamSpec], Type]]:
    """Make a functional registry."""
    if not weak:
        raise ValueError("Weak references are not supported")
    assert domain == '', "Domains are not supported"

    # kwds = _dict_not_none({'strict': strict, 'weak': weak, 'domain': domain})
    return new_class(f'{name}Registry',
                     bases=(FunctionalRegistry[args, ret],),
                     kwds=None)
