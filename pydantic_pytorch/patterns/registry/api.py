from typing import Type, TypeVar, _ProtocolMeta, ParamSpec, Annotated
from types import new_class
from .registry import ClassRegistry, FunctionalRegistry
from ._compat import BuilderValidator

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
    assert not weak, "Weak references are not supported"
    assert domain is None, "Domains are not supported"
    assert strict, "Strict references are not supported"

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
    assert not weak, "Weak references are not supported"
    assert domain == '', "Domains are not supported"
    assert strict, "Strict references are not supported"

    # kwds = _dict_not_none({'strict': strict, 'weak': weak, 'domain': domain})
    return new_class(f'{name}Registry', bases=(FunctionalRegistry[args, ret],), kwds=None)


class PydanticBuilderValidatorMacro:
    """A macro for generating Pydantic builder validators."""

    @classmethod
    def __class_getitem__(cls, item):
        """Get the item."""
        source_type, registry = item
        return Annotated[source_type, BuilderValidator(registry=registry)]
