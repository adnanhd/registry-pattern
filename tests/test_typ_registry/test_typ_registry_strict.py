import pytest

from typing import Protocol
from registry import TypeRegistry, ConformanceError
from types import new_class

from registry.base import RegistryError


class CommandProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


@pytest.fixture
def CommandRegistry():
    return new_class(
        "CommandRegistry", (TypeRegistry[CommandProtocol],), {"strict": True}
    )


def test_registry_valid_entry(CommandRegistry: TypeRegistry):

    @CommandRegistry.register_class
    class AdditionCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    assert CommandRegistry.has_registry_item(AdditionCommand)
    assert CommandRegistry.get_registry_item("AdditionCommand") == AdditionCommand


def test_registry_invalid_entry_with_missing_method(CommandRegistry: TypeRegistry):

    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method2(self, x: int, y: int) -> int:
                return x - y

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand")


def test_registry_invalid_entry_with_extra_argument(CommandRegistry: TypeRegistry):

    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method(self, x: int, y: int, z: int) -> int:
                return x - y + z

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand")


def test_registry_invalid_entry_with_wrong_argument_type(CommandRegistry: TypeRegistry):

    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand1:
            def test_method(self, x: float, y: int) -> int:
                return int(x - y)

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand1")

    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand2:
            def test_method(self, x: int, y: float) -> int:
                return int(x - y)

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand2")
    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand3:
            def test_method(self, x: int, y: int) -> float:
                return x - y

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand3")


def test_validate_item(CommandRegistry: TypeRegistry):
    """Test the validate_item method in strict mode."""

    class ValidClass:
        def test_method(self, x: int, y: int) -> int:
            return x - y

    # Validation should pass for a conforming class
    validated = CommandRegistry.validate_item(ValidClass)
    assert validated is ValidClass

    # Validation should fail for a non-conforming class
    class InvalidClass:
        def another_method(self, data: int) -> str:
            return f"Data {data}"

    with pytest.raises(ConformanceError):
        CommandRegistry.validate_item(InvalidClass)
