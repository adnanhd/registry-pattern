import pytest
from types import new_class, ModuleType
from typing import Callable, Tuple

from registry import FunctionalRegistry, ConformanceError, RegistryError

# -----------------------------------------------------------------------------
# Fixture for creating a strict FunctionalRegistry.
# -----------------------------------------------------------------------------


@pytest.fixture
def FuncRegistry():
    """
    Create a FunctionalRegistry for functions with signature Callable[[int, int], int]
    in strict mode. In strict mode, functions must exactly match the expected signature.
    """
    return new_class(
        "FuncRegistry", (FunctionalRegistry[[int, int], int],), {"strict": True}
    )


# -----------------------------------------------------------------------------
# Tests for function registration with strict mode.
# -----------------------------------------------------------------------------


def test_register_valid_function(FuncRegistry: FunctionalRegistry):
    """
    Test that a function with a valid signature is registered successfully.
    """

    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    # Check that the function is in the registry.
    assert FuncRegistry.has_registry_item(add)
    # Also, verify that get_registry_item returns the same function.
    assert FuncRegistry.get_registry_item("add") == add


def test_register_invalid_function_missing_argument(FuncRegistry: FunctionalRegistry):
    """
    Test that a function missing a required argument raises a ConformanceError.
    The function should not be added to the registry.
    """
    with pytest.raises(ConformanceError):

        @FuncRegistry.register_function
        def add(x: int) -> int:
            return x + 1

    # Verify that the function "add" is not present in the repository.
    assert "add" not in FuncRegistry._repository


def test_register_invalid_function_extra_argument(FuncRegistry: FunctionalRegistry):
    """
    Test that a function with an extra argument raises a ConformanceError.
    The function should not be added to the registry.
    """
    with pytest.raises(ConformanceError):

        @FuncRegistry.register_function
        def add(x: int, y: int, z: int) -> int:
            return x + y + z

    # Verify that the function "add" is not present in the repository.
    assert "add" not in FuncRegistry._repository


def test_register_invalid_function_wrong_return_type(FuncRegistry: FunctionalRegistry):
    """
    Test that a function with a return type that does not match the expected type
    raises a ConformanceError and is not registered.
    """
    with pytest.raises(ConformanceError):

        @FuncRegistry.register_function
        def add(x: int, y: int) -> str:
            return f"{x + y}"

    # Verify that the function "add" is not present in the repository.
    assert "add" not in FuncRegistry._repository


def test_register_module_functions(FuncRegistry: FunctionalRegistry):
    """
    Test that registering module functions triggers a ConformanceError if any function
    in the module does not meet the strict signature requirements.
    """
    module = ModuleType("custom_module")

    def func_mocker(name: str) -> Callable:
        """
        Helper to assign a specific __name__ to a function.
        """

        def wrapper(fn: Callable) -> Callable:
            fn.__name__ = name
            return fn

        return wrapper

    # Create a function with an incorrect signature (extra parameter 'z').
    module.add = func_mocker("add")(lambda x, y, z: x + y)

    # Expect registration to fail due to ConformanceError.
    with pytest.raises(ConformanceError):
        FuncRegistry.register_module_functions(module)

    # Verify that the function is not registered.
    assert not FuncRegistry.has_registry_item(module.add)

    with pytest.raises(RegistryError):
        FuncRegistry.get_registry_item("add")
