import pytest

from typing import Protocol
from registry import TypeRegistry, RegistryError, ConformanceError
from types import new_class


class CommandProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


@pytest.fixture
def CommandRegistry():
    return new_class("CommandRegistry", (TypeRegistry[CommandProtocol],), {})


def test_registry_valid_entry(CommandRegistry: TypeRegistry):

    @CommandRegistry.register_class
    class AdditionCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    assert CommandRegistry.has_registry_item(AdditionCommand)
    assert CommandRegistry.get_registry_item("AdditionCommand") == AdditionCommand


def test_registry_invalid_method_different_method_name(CommandRegistry: TypeRegistry):

    @CommandRegistry.register_class
    class SubtractionCommand:
        def test_method2(self, x: int, y: int) -> int:
            return x - y

    assert CommandRegistry.has_registry_item(SubtractionCommand)
    assert CommandRegistry.get_registry_item("SubtractionCommand") == SubtractionCommand


def test_unregister_class(CommandRegistry: TypeRegistry):
    """Test unregistering a class."""

    @CommandRegistry.register_class
    class TempClass:
        def execute(self, data: int) -> str:
            return f"Temporary {data}"

    CommandRegistry.unregister_class(TempClass)
    assert not CommandRegistry.has_registry_item(TempClass)
    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("TempClass")


def test_register_module_subclasses(CommandRegistry: TypeRegistry):
    """Test registering all subclasses from a module."""

    @CommandRegistry.register_class
    class ParentClass:
        def execute(self, data: int) -> str:
            return f"Parent {data}"

    class ChildClass(ParentClass):
        def execute(self, data: int) -> str:
            return f"Child {data}"

    CommandRegistry.register_subclasses(ParentClass)

    assert CommandRegistry.has_registry_item(ParentClass)
    assert CommandRegistry.has_registry_item(ChildClass)
