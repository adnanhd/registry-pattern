import pytest
from typing import Protocol
from types import new_class

from registry import TypeRegistry, RegistryError, ConformanceError

# (Assuming that TypeRegistry, RegistryError, and ConformanceError are correctly imported from your project.)

# -------------------------------------------------------------------
# Define a protocol for command classes.
# -------------------------------------------------------------------


class CommandProtocol(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


# -------------------------------------------------------------------
# Fixtures for creating registries.
# -------------------------------------------------------------------


@pytest.fixture
def CommandRegistry():
    """
    Create a registry class for CommandProtocol using TypeRegistry.
    This registry does not enforce strict protocol conformance.
    """
    return new_class("CommandRegistry", (TypeRegistry[CommandProtocol],), {})


@pytest.fixture
def StrictCommandRegistry():
    """
    Create a registry class for CommandProtocol in strict mode.
    In strict mode, classes that do not implement the required protocol method
    (i.e. 'test_method') will be rejected.
    """

    # Define a new registry class using normal subclassing so that we can pass keyword arguments.
    class StrictCommandRegistry(TypeRegistry[CommandProtocol], strict=True):
        pass

    return StrictCommandRegistry


# -------------------------------------------------------------------
# Tests for basic registry functionality.
# -------------------------------------------------------------------


def test_registry_valid_entry(CommandRegistry: TypeRegistry):
    """
    Test that a valid class (one implementing 'test_method') is accepted.
    """

    @CommandRegistry.register_class
    class AdditionCommand:
        def test_method(self, x: int, y: int) -> int:
            return x + y

    assert CommandRegistry.artifact_exists(AdditionCommand)
    assert CommandRegistry.get_artifact("AdditionCommand") == AdditionCommand


def test_registry_invalid_method_different_method_name(CommandRegistry: TypeRegistry):
    """
    Test that a class with a method name different from the protocol's expected name
    is still registered when not in strict mode.

    (Note: In non-strict mode, validation is lax so the class is registered even if
    it does not implement 'test_method'.)
    """

    @CommandRegistry.register_class
    class SubtractionCommand:
        def test_method2(self, x: int, y: int) -> int:
            return x - y

    assert CommandRegistry.artifact_exists(SubtractionCommand)
    assert CommandRegistry.get_artifact("SubtractionCommand") == SubtractionCommand


def test_unregister_class(CommandRegistry: TypeRegistry):
    """Test unregistering a class from the registry."""

    @CommandRegistry.register_class
    class TempClass:
        def execute(self, data: int) -> str:
            return f"Temporary {data}"

    CommandRegistry.unregister_class(TempClass)
    assert not CommandRegistry.artifact_exists(TempClass)
    with pytest.raises(RegistryError):
        CommandRegistry.get_artifact("TempClass")


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

    assert CommandRegistry.artifact_exists(ParentClass)
    assert CommandRegistry.artifact_exists(ChildClass)


# -------------------------------------------------------------------
# Additional tests for strict mode validation.
# -------------------------------------------------------------------


def test_strict_registry_invalid(StrictCommandRegistry: TypeRegistry):
    """
    In strict mode, attempting to register a class that does not implement the protocol
    should result in a ConformanceError.
    """
    with pytest.raises(ConformanceError):

        @StrictCommandRegistry.register_class
        class InvalidCommand:
            # Note: This class intentionally uses a wrong method name.
            def wrong_method(self, x: int, y: int) -> int:
                return x * y


def test_strict_registry_valid(StrictCommandRegistry: TypeRegistry):
    """
    In strict mode, a class that correctly implements the protocol should be registered.
    """

    @StrictCommandRegistry.register_class
    class ValidCommand:
        def test_method(self, x: int, y: int) -> int:
            return x * y

    assert StrictCommandRegistry.artifact_exists(ValidCommand)
    assert StrictCommandRegistry.get_artifact("ValidCommand") == ValidCommand
