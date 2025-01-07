import pytest

from typing import Protocol
from registry import TypeRegistry, ConformanceError
from types import new_class


class CommandProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


@pytest.fixture
def CommandRegistry():
    return new_class("CommandRegistry", (TypeRegistry[CommandProtocol],), {"strict": True})


def test_registry_valid_entry(CommandRegistry: TypeRegistry):

    @CommandRegistry.register_class
    class AdditionCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    assert "AdditionCommand" in CommandRegistry._repository
    assert CommandRegistry.has_registry_item(AdditionCommand)


def test_registry_invalid_entry_with_missing_method(CommandRegistry: TypeRegistry):

    with pytest.raises(ConformanceError):
        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method2(self, x: int, y: int) -> int:
                return x - y

    assert "SubtractionCommand" not in CommandRegistry._repository


def test_registry_invalid_entry_with_extra_argument(CommandRegistry: TypeRegistry):

    with pytest.raises(ConformanceError):
        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method(self, x: int, y: int, z: int) -> int:
                return x - y + z

    assert "SubtractionCommand" not in CommandRegistry._repository



def test_registry_invalid_entry_with_wrong_argument_type(CommandRegistry: TypeRegistry):

    with pytest.raises(ConformanceError):
        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method(self, x: float, y: float) -> float:
                return x - y

    assert "SubtractionCommand" not in CommandRegistry._repository
