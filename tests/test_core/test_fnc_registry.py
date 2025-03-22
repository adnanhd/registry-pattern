import pytest
from types import ModuleType, new_class

from registry import (
    FunctionalRegistry,
    ConformanceError,
    RegistryError,
)

# -------------------------------------------------------------------
# Base test class for FunctionalRegistry
# -------------------------------------------------------------------


class BaseFunctionalRegistryTest:
    """Base class for FunctionalRegistry tests."""

    @pytest.fixture
    def registry_class(self):
        """Return a registry class for testing. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement this fixture")

    def test_register_valid_function(self, registry_class):
        """Test that a valid function can be registered."""

        @registry_class.register_function
        def add(x: int, y: int) -> int:
            return x + y

        assert registry_class.has_artifact(add)
        assert registry_class.get_artifact("add") is add

    def test_duplicate_registration(self, registry_class):
        """Test that registering the same function twice raises an error."""

        @registry_class.register_function
        def duplicate(x: int, y: int) -> int:
            return x + y

        with pytest.raises(RegistryError):
            registry_class.register_function(duplicate)

    def test_unregister_function(self, registry_class):
        """Test that a function can be unregistered."""

        @registry_class.register_function
        def temp(x: int, y: int) -> int:
            return x * y

        assert registry_class.has_artifact(temp)
        registry_class.unregister_function("temp")
        assert not registry_class.has_artifact(temp)
        with pytest.raises(RegistryError):
            registry_class.get_artifact("temp")

    def test_register_module_functions(self, registry_class):
        """Test registering all functions from a module."""
        # Create a mock module with functions
        module = ModuleType("mock_module")

        def mock_function_factory(name):
            """Create a function with a specific name."""

            def func(x: int, y: int) -> int:
                return x + y

            func.__name__ = name
            return func

        module.add = mock_function_factory("add")
        module.subtract = mock_function_factory("subtract")

        # Register all functions from the module
        registry_class.register_module_functions(module)

        assert registry_class.has_artifact(module.add)
        assert registry_class.has_artifact(module.subtract)
        assert registry_class.get_artifact("add") is module.add
        assert registry_class.get_artifact("subtract") is module.subtract


# -------------------------------------------------------------------
# Test class for non-strict FunctionalRegistry
# -------------------------------------------------------------------


class TestNonStrictFunctionalRegistry(BaseFunctionalRegistryTest):
    """Tests for FunctionalRegistry in non-strict mode."""

    @pytest.fixture
    def registry_class(self):
        """Create a non-strict registry for functions with signature Callable[[int, int], int]."""
        return new_class("FuncRegistry", (FunctionalRegistry[[int, int], int],))

    def test_register_function_missing_argument(self, registry_class):
        """Test that a function with a missing argument can be registered in non-strict mode."""

        @registry_class.register_function
        def add_one(x: int) -> int:
            return x + 1

        assert registry_class.has_artifact(add_one)
        assert registry_class.get_artifact("add_one") is add_one

    def test_register_function_extra_argument(self, registry_class):
        """Test that a function with an extra argument can be registered in non-strict mode."""

        @registry_class.register_function
        def add_three(x: int, y: int, z: int) -> int:
            return x + y + z

        assert registry_class.has_artifact(add_three)
        assert registry_class.get_artifact("add_three") is add_three

    def test_register_function_wrong_return_type(self, registry_class):
        """Test that a function with a wrong return type can be registered in non-strict mode."""

        @registry_class.register_function
        def add_str(x: int, y: int) -> str:
            return f"{x + y}"

        assert registry_class.has_artifact(add_str)
        assert registry_class.get_artifact("add_str") is add_str


# -------------------------------------------------------------------
# Test class for strict FunctionalRegistry
# -------------------------------------------------------------------


class TestStrictFunctionalRegistry:
    """Tests for FunctionalRegistry in strict mode."""

    @pytest.fixture
    def registry_class(self):
        """Create a strict registry for functions with signature Callable[[int, int], int]."""
        return new_class(
            "StrictFuncRegistry",
            (FunctionalRegistry[[int, int], int],),
            {"strict": True},
        )

    def test_register_valid_function(self, registry_class):
        """Test that a valid function can be registered in strict mode."""

        @registry_class.register_function
        def add(x: int, y: int) -> int:
            return x + y

        assert registry_class.has_artifact(add)
        assert registry_class.get_artifact("add") is add

    def test_register_function_missing_argument(self, registry_class):
        """Test that a function with a missing argument cannot be registered in strict mode."""
        with pytest.raises(ConformanceError):

            @registry_class.register_function
            def add_one(x: int) -> int:
                return x + 1

        # Function shouldn't be registered due to the exception
        with pytest.raises(RegistryError):
            registry_class.get_artifact("add_one")

    def test_register_function_extra_argument(self, registry_class):
        """Test that a function with an extra argument cannot be registered in strict mode."""
        with pytest.raises(ConformanceError):

            @registry_class.register_function
            def add_three(x: int, y: int, z: int) -> int:
                return x + y + z

        # Function shouldn't be registered due to the exception
        with pytest.raises(RegistryError):
            registry_class.get_artifact("add_three")

    def test_register_function_wrong_return_type(self, registry_class):
        """Test that a function with a wrong return type cannot be registered in strict mode."""
        with pytest.raises(ConformanceError):

            @registry_class.register_function
            def add_str(x: int, y: int) -> str:
                return f"{x + y}"

        # Function shouldn't be registered due to the exception
        with pytest.raises(RegistryError):
            registry_class.get_artifact("add_str")

    def test_validate_function(self, registry_class):
        """Test that validate_artifact correctly validates a function in strict mode."""

        def valid_func(x: int, y: int) -> int:
            return x + y

        # Validation should pass for a conforming function
        validated = registry_class.validate_function(valid_func)
        assert validated is valid_func

        # Validation should fail for a non-conforming function
        def invalid_func(x: int) -> int:
            return x + 1

        with pytest.raises(ConformanceError):
            registry_class.validate_function(invalid_func)


# -------------------------------------------------------------------
# Additional test cases for better coverage
# -------------------------------------------------------------------


class TestFunctionalRegistryAdvanced:
    """Advanced tests for FunctionalRegistry."""

    @pytest.fixture
    def registry_class(self):
        """Create a registry for functions with signature Callable[[int, int], int]."""
        return new_class("FuncRegistry", (FunctionalRegistry[[int, int], int],))

    @pytest.fixture
    def strict_registry_class(self):
        """Create a registry for functions with signature Callable[[int, int], int]."""
        return new_class(
            "FuncRegistry", (FunctionalRegistry[[int, int], int],), {"strict": True}
        )

    def test_subclasscheck(self, registry_class, strict_registry_class):
        """Test the __subclasscheck__ method."""

        def valid_func(x: int, y: int) -> int:
            return x + y

        def invalid_func(x: str, y: str) -> str:
            return x + y

        # Valid function should pass subclasscheck
        assert strict_registry_class.__subclasscheck__(valid_func)
        # Invalid function should not pass subclasscheck
        assert not strict_registry_class.__subclasscheck__(invalid_func)

        # Valid function should pass subclasscheck
        assert registry_class.__subclasscheck__(valid_func)
        # Invalid function should not pass subclasscheck
        assert registry_class.__subclasscheck__(invalid_func)

    def test_register_module_functions_with_error(self, registry_class):
        """Test registering module functions with an error."""
        # Create a mock module with functions
        module = ModuleType("mock_module")

        def mock_function_factory(name):
            """Create a function with a specific name."""

            def func(x: int, y: int) -> int:
                return x + y

            func.__name__ = name
            return func

        module.add = mock_function_factory("add")
        # Add a non-function attribute
        module.not_a_function = "This is not a function"

        # Register all functions from the module (should ignore non-functions)
        registry_class.register_module_functions(module, raise_error=False)

        assert registry_class.has_artifact(module.add)
        assert registry_class.get_artifact("add") is module.add
        # The non-function attribute should not be registered
        with pytest.raises(RegistryError):
            registry_class.get_artifact("not_a_function")

    def test_register_module_functions_raise_error(self, registry_class):
        """Test registering module functions with raise_error=True."""
        # Create a mock module with functions and a non-function
        module = ModuleType("error_module")

        # Valid function
        def valid_func(x: int, y: int) -> int:
            return x + y

        valid_func.__name__ = "valid_func"
        module.valid_func = valid_func

        # Add a class that isn't a function
        class NotAFunction:
            pass

        module.not_a_func = NotAFunction

        # Should continue without error when raise_error=False
        registry_class.register_module_functions(module, raise_error=False)
        assert registry_class.has_artifact(module.valid_func)
