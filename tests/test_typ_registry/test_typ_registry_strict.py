import pytest
from typing import Protocol
from types import new_class

from registry import TypeRegistry, ConformanceError, RegistryError

# -------------------------------------------------------------------
# Define a protocol for command classes.
# -------------------------------------------------------------------


class CommandProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


# -------------------------------------------------------------------
# Fixture for creating a strict registry.
# -------------------------------------------------------------------


@pytest.fixture
def CommandRegistry():
    """
    Create a registry class for CommandProtocol using TypeRegistry in strict mode.
    Strict mode enforces that registered classes must correctly implement
    the protocol (i.e. have the 'test_method' with the correct signature).
    """
    return new_class(
        "CommandRegistry", (TypeRegistry[CommandProtocol],), {"strict": True}
    )


# -------------------------------------------------------------------
# Tests for strict registry functionality.
# -------------------------------------------------------------------


def test_registry_valid_entry(CommandRegistry: TypeRegistry):
    """
    Test that a valid class (implementing 'test_method') is accepted.
    """

    @CommandRegistry.register_class
    class AdditionCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    assert CommandRegistry.has_registry_item(AdditionCommand)
    assert CommandRegistry.get_registry_item("AdditionCommand") == AdditionCommand


def test_registry_invalid_entry_with_missing_method(CommandRegistry: TypeRegistry):
    """
    Test that a class missing the required method ('test_method') triggers
    a ConformanceError and is not registered.
    """
    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method2(self, x: int, y: int) -> int:
                return x - y

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand")


def test_registry_invalid_entry_with_extra_argument(CommandRegistry: TypeRegistry):
    """
    Test that a class with an extra argument in 'test_method' is rejected.
    """
    with pytest.raises(ConformanceError):

        @CommandRegistry.register_class
        class SubtractionCommand:
            def test_method(self, x: int, y: int, z: int) -> int:
                return x - y + z

    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("SubtractionCommand")


def test_registry_invalid_entry_with_wrong_argument_type(CommandRegistry: TypeRegistry):
    """
    Test that classes with incorrect argument types or return type for 'test_method'
    are rejected.
    """
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
    """
    Test the validate_item method in strict mode.
    Validation should pass for a conforming class and fail for a non-conforming one.
    """

    class ValidClass:
        def test_method(self, x: int, y: int) -> int:
            return x - y

    # Validation should pass for a conforming class.
    validated = CommandRegistry.validate_item(ValidClass)
    assert validated is ValidClass

    # Validation should fail for a non-conforming class.
    class InvalidClass:
        def another_method(self, data: int) -> str:
            return f"Data {data}"

    with pytest.raises(ConformanceError):
        CommandRegistry.validate_item(InvalidClass)


# -------------------------------------------------------------------
# Additional tests for strict registry functionality.
# -------------------------------------------------------------------


def test_duplicate_registration(CommandRegistry: TypeRegistry):
    """
    Test that attempting to register the same class twice raises a RegistryError.
    """

    @CommandRegistry.register_class
    class ValidCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    with pytest.raises(RegistryError):
        # Attempting to register the same class again should raise an error.
        CommandRegistry.register_class(ValidCommand)


def test_unregister_class(CommandRegistry: TypeRegistry):
    """
    Test that unregistering a class removes it from the registry.
    """

    @CommandRegistry.register_class
    class TempCommand:
        def test_method(self, x: int, y: int) -> int:
            return x * y

    # Ensure the class is registered.
    assert CommandRegistry.has_registry_item(TempCommand)
    # Unregister the class.
    CommandRegistry.unregister_class(TempCommand)
    # Verify that the class is no longer in the registry.
    with pytest.raises(RegistryError):
        CommandRegistry.get_registry_item("TempCommand")


def test_register_module_subclasses(CommandRegistry: TypeRegistry):
    """
    Test that all subclasses of a given class are registered correctly.
    """

    @CommandRegistry.register_class
    class ParentCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    class ChildCommand(ParentCommand):
        def test_method(self, x: int, y: int) -> int:
            return x - y

    CommandRegistry.register_subclasses(ParentCommand)

    assert CommandRegistry.has_registry_item(ParentCommand)
    assert CommandRegistry.has_registry_item(ChildCommand)
