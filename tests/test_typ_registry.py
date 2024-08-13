"""
Test class registry.
"""

from typing import Protocol, Type
import pytest

from registry import (
    type_registry_decorator,
    type_registry_factory,
    TypeRegistry,
    ConformanceError,
    RegistryError,
)


@type_registry_decorator()
class DecoratedProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int:
        ...


class FabricatedProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int:
        ...


@pytest.fixture
def Registry():
    return type_registry_factory("ExampleProtocol", protocol=FabricatedProtocol)


def test_typ_registry_with_subclass(Registry: Type[TypeRegistry[FabricatedProtocol]]):
    @Registry.register_class
    @Registry.register
    class Bar:
        def test_method(self, x: int, y: int) -> int:
            return 2

    assert Registry.get_registry_item("Bar") is Bar

    with pytest.raises(RegistryError) as exc_info:
        Registry.register_class(Bar)

    assert exc_info.value.args[0] == "ExampleProtocolRegistry: 'Bar' already registered"


def test_typ_registry_with_invalid_class(
    Registry: Type[TypeRegistry[FabricatedProtocol]],
):
    with pytest.raises(ConformanceError) as exc_info:

        @Registry.register_class
        @Registry.register
        class Baz:
            def test2_method(self, x: int, y: int) -> int:
                return 2

    msg = (
        "<class 'test_typ_registry.test_typ_registry_with_invalid_class.<locals>.Baz'>"
        " is not of type <class 'test_typ_registry.FabricatedProtocol'>: "
        "class test_typ_registry.test_typ_registry_with_invalid_class.<locals>.Baz is"
        " not compatible with the FabricatedProtocol protocol because it has no method"
        " named 'test_method'"
    )
    assert exc_info.value.args[0] == msg
