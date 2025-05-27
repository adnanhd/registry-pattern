import pytest
from typing import Protocol
from types import new_class

from registry import TypeRegistry, ConformanceError, RegistryError, InheritanceError
from registry.core._validator import ValidationCache, ValidationError

# -------------------------------------------------------------------
# Define a protocol for command classes.
# -------------------------------------------------------------------


class CommandProtocol(Protocol):
    def execute(self, x: int, y: int) -> int: ...


# -------------------------------------------------------------------
# Base test class for TypeRegistry
# -------------------------------------------------------------------


class BaseTypeRegistryTest:
    """Base class for TypeRegistry tests."""

    @pytest.fixture
    def registry_class(self):
        """Return a registry class for testing. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement this fixture")

    def test_register_valid_class(self, registry_class):
        """Test that a valid class can be registered."""

        @registry_class.register_artifact
        class AddCommand:
            def execute(self, x: int, y: int) -> int:
                return x + y

        assert registry_class.has_artifact(AddCommand)
        assert registry_class.get_artifact("AddCommand") is AddCommand

    def test_duplicate_registration(self, registry_class):
        """Test that registering the same class twice raises an error."""

        @registry_class.register_artifact
        class DuplicateCommand:
            def execute(self, x: int, y: int) -> int:
                return x + y

        with pytest.raises(ValidationError):
            registry_class.register_artifact(DuplicateCommand)

    # def test_unregister_artifact(self, registry_class):
    #     """Test that a class can be unregistered."""

    #     @registry_class.register_artifact
    #     class TempCommand:
    #         def execute(self, x: int, y: int) -> int:
    #             return x * y

    #     assert registry_class.has_artifact(TempCommand)
    #     registry_class.unregister_artifact(TempCommand)
    #     assert not registry_class.has_artifact(TempCommand)
    #     with pytest.raises(RegistryError):
    #         registry_class.get_artifact("TempCommand")

    def test_unregister_artifact_by_name(self, registry_class):
        """Test that a class can be unregistered by name."""

        @registry_class.register_artifact
        class TempCommand:
            def execute(self, x: int, y: int) -> int:
                return x * y

        assert registry_class.has_artifact(TempCommand)
        registry_class.unregister_artifact("TempCommand")
        assert not registry_class.has_artifact(TempCommand)
        with pytest.raises(RegistryError):
            registry_class.get_artifact("TempCommand")

    def test_register_subclasses(self, registry_class):
        """Test registering all subclasses of a class."""

        @registry_class.register_artifact
        class ParentCommand:
            def execute(self, x: int, y: int) -> int:
                return x + y

        class ChildCommand(ParentCommand):
            def execute(self, x: int, y: int) -> int:
                return x - y

        registry_class.register_subclasses(ParentCommand)

        assert registry_class.has_artifact(ParentCommand)
        assert registry_class.has_artifact(ChildCommand)


# -------------------------------------------------------------------
# Test class for non-strict TypeRegistry
# -------------------------------------------------------------------


class TestNonStrictTypeRegistry(BaseTypeRegistryTest):
    """Tests for TypeRegistry in non-strict mode."""

    @pytest.fixture
    def registry_class(self):
        """Create a non-strict registry for CommandProtocol."""
        return new_class("CommandRegistry", (TypeRegistry[CommandProtocol],))

    def test_register_non_conforming_class(self, registry_class):
        """Test that a non-conforming class can be registered in non-strict mode."""

        @registry_class.register_artifact
        class NonConformingCommand:
            def different_method(self, a: str, b: str) -> str:
                return a + b

        assert registry_class.has_artifact(NonConformingCommand)
        assert (
            registry_class.get_artifact("NonConformingCommand") is NonConformingCommand
        )

    def test_register_artifact_wrong_signature(self, registry_class):
        """Test that a class with wrong method signatures can still be registered in non-strict mode."""

        @registry_class.register_artifact
        class WrongSignatureCommand:
            def execute(self, x: str, y: str) -> str:
                return x + y

        assert registry_class.has_artifact(WrongSignatureCommand)
        assert (
            registry_class.get_artifact("WrongSignatureCommand")
            is WrongSignatureCommand
        )


# -------------------------------------------------------------------
# Test class for strict TypeRegistry
# -------------------------------------------------------------------


class TestStrictTypeRegistry:
    """Tests for TypeRegistry in strict mode."""

    @pytest.fixture
    def registry_class(self):
        """Create a strict registry for CommandProtocol."""
        return new_class(
            "StrictCommandRegistry", (TypeRegistry[CommandProtocol],), {"strict": True}
        )

    def test_register_valid_class(self, registry_class):
        """Test that a valid class can be registered in strict mode."""

        @registry_class.register_artifact
        class ValidCommand:
            def execute(self, x: int, y: int) -> int:
                return x * y

        assert registry_class.has_artifact(ValidCommand)
        assert registry_class.get_artifact("ValidCommand") is ValidCommand

    def test_register_non_conforming_class(self, registry_class):
        """Test that a non-conforming class cannot be registered in strict mode."""
        with pytest.raises(ConformanceError):

            @registry_class.register_artifact
            class NonConformingCommand:
                def different_method(self, a: str, b: str) -> str:
                    return a + b

        with pytest.raises(RegistryError):
            registry_class.get_artifact("NonConformingCommand")

    def test_register_artifact_wrong_signature(self, registry_class):
        """Test that a class with wrong method signatures cannot be registered in strict mode."""
        with pytest.raises(ConformanceError):

            @registry_class.register_artifact
            class WrongSignatureCommand:
                def execute(self, x: str, y: str) -> str:
                    return x + y

        with pytest.raises(RegistryError):
            registry_class.get_artifact("WrongSignatureCommand")

    def test_validate_artifact(self, registry_class):
        """Test that validate_artifact correctly validates a class in strict mode."""

        class ValidClass:
            def execute(self, x: int, y: int) -> int:
                return x - y

        # Validation should pass for a conforming class.
        validated = registry_class.validate_artifact(ValidClass)
        assert validated is ValidClass

        # Validation should fail for a non-conforming class.
        class InvalidClass:
            def another_method(self, data: int) -> str:
                return f"Data {data}"

        with pytest.raises(ConformanceError):
            registry_class.validate_artifact(InvalidClass)


# -------------------------------------------------------------------
# Test class for abstract TypeRegistry
# -------------------------------------------------------------------


class TestAbstractTypeRegistry:
    """Tests for TypeRegistry in abstract mode."""

    @pytest.fixture
    def base_class(self):
        """Create a base class for testing inheritance."""

        class BaseCommand:
            def execute(self, x: int, y: int) -> int:
                return x + y

        return BaseCommand

    @pytest.fixture
    def registry_class(self):
        """Create an abstract registry for CommandProtocol."""
        return new_class(
            "AbstractCommandRegistry",
            (TypeRegistry[CommandProtocol],),
            {"abstract": True},
        )

    def test_register_inheriting_class(self, registry_class):
        """Test that a class inheriting from the registry class can be registered in abstract mode."""

        class InheritingCommand(registry_class):
            def execute(self, x: int, y: int) -> int:
                return x * y

        registry_class.register_artifact(InheritingCommand)
        assert registry_class.has_artifact(InheritingCommand)
        assert registry_class.get_artifact("InheritingCommand") is InheritingCommand

    def test_register_non_inheriting_class(self, registry_class):
        """Test that a class not inheriting from the registry class cannot be registered in abstract mode."""

        class NonInheritingCommand:
            def execute(self, x: int, y: int) -> int:
                return x - y

        with pytest.raises(InheritanceError):
            registry_class.register_artifact(NonInheritingCommand)

        assert not registry_class.has_artifact(NonInheritingCommand)
        with pytest.raises(RegistryError):
            registry_class.get_artifact("NonInheritingCommand")


# -------------------------------------------------------------------
# Additional test cases for better coverage
# -------------------------------------------------------------------


class TestTypeRegistryAdvanced:
    """Advanced tests for TypeRegistry."""

    @pytest.fixture
    def registry_class(self):
        """Create a registry for CommandProtocol."""
        return new_class("CommandRegistry", (TypeRegistry[CommandProtocol],))

    def test_recursive_register_subclasses(self, registry_class):
        """Test recursive registration of subclasses."""

        @registry_class.register_artifact
        class GrandparentCommand:
            def execute(self, x: int, y: int) -> int:
                return x + y

        class ParentCommand(GrandparentCommand):
            def execute(self, x: int, y: int) -> int:
                return x - y

        class ChildCommand(ParentCommand):
            def execute(self, x: int, y: int) -> int:
                return x * y

        registry_class.register_subclasses(GrandparentCommand, recursive=True)

        assert registry_class.has_artifact(GrandparentCommand)
        assert registry_class.has_artifact(ParentCommand)
        assert registry_class.has_artifact(ChildCommand)

    def test_register_with_different_modes(self, registry_class):
        """Test registration with different modes."""

        @registry_class.register_artifact
        class BaseCommand:
            def execute(self, x: int, y: int) -> int:
                return x + y

        # Test immediate mode
        registry_class.register_subclasses(BaseCommand, mode="immediate")
        assert registry_class.has_artifact(BaseCommand)

        # Test deferred mode - create a class that will enable auto-registration of subclasses
        DeferredBaseCommand = registry_class.register_subclasses(
            BaseCommand, mode="deferred"
        )

        # Create a subclass which should be auto-registered
        class DeferredChildCommand(DeferredBaseCommand):
            def execute(self, x: int, y: int) -> int:
                return x - y

        # Should be registered automatically
        assert registry_class.has_artifact(DeferredChildCommand)

    # def test_validate_identifier(self, registry_class):
    #     """Test validating a string identifier."""
    #     # A string is a valid identifier
    #     key = "test_key"
    #     assert registry_class.validate_identifier(key) == key

    #     # A non-hashable object (like a list) should raise an error
    #     with pytest.raises(TypeError):
    #         registry_class.validate_identifier([1, 2, 3])
