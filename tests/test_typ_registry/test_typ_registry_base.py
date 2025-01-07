import pytest

from typing import Protocol
from registry import TypeRegistry
from types import new_class


class CommandProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


@pytest.fixture
def CommandRegistry():
    return new_class("CommandRegistry", (TypeRegistry[CommandProtocol],), {})


def test_registry_strict_subclass_entry(CommandRegistry: TypeRegistry):

    @CommandRegistry.register_class
    class AdditionCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    assert "AdditionCommand" in CommandRegistry._repository
    assert CommandRegistry.has_registry_item(AdditionCommand)


def test_registry_nonstrict_subclass_entry(CommandRegistry: TypeRegistry):

    @CommandRegistry.register_class
    class SubtractionCommand:
        def test_method2(self, x: int, y: int) -> int:
            return x - y

    assert "SubtractionCommand" in CommandRegistry._repository
    assert CommandRegistry.has_registry_item(SubtractionCommand)
