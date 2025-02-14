import pytest
from types import new_class, ModuleType
from typing import Callable, Tuple

from registry import FunctionalRegistry, ConformanceError, RegistryError

# -------------------------------------------------------------------
# Fixture for creating a FunctionalRegistry.
# -------------------------------------------------------------------


@pytest.fixture
def FuncRegistry():
    """
    Create a FunctionalRegistry for functions with signature Callable[[int, int], int].
    (Note: By default, the registry uses a no-op runtime conformance checker.)
    """
    return new_class("FuncRegistry", (FunctionalRegistry[[int, int], int],))


# -------------------------------------------------------------------
# Tests for function registration.
# -------------------------------------------------------------------


def test_register_valid_function(FuncRegistry: FunctionalRegistry):
    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    assert FuncRegistry.artifact_exists(add)
    assert FuncRegistry.get_artifact("add") == add


def test_register_invalid_function_missing_argument(FuncRegistry: FunctionalRegistry):
    """
    Test that a function with a missing argument is still registered in non-strict mode.
    """

    @FuncRegistry.register_function
    def add(x: int) -> int:
        return x + 1

    assert FuncRegistry.artifact_exists(add)
    assert FuncRegistry.get_artifact("add") == add


def test_register_invalid_function_extra_argument(FuncRegistry: FunctionalRegistry):
    """
    Test that a function with an extra argument is still registered in non-strict mode.
    """

    @FuncRegistry.register_function
    def add(x: int, y: int, z: int) -> int:
        return x + y + z

    assert FuncRegistry.artifact_exists(add)
    assert FuncRegistry.get_artifact("add") == add


def test_register_invalid_function_wrong_return_type(FuncRegistry: FunctionalRegistry):
    """
    Test that a function with an unexpected return type is still registered.
    """

    @FuncRegistry.register_function
    def add(x: int, y: int) -> str:
        return f"{x + y}"

    assert FuncRegistry.artifact_exists(add)
    assert FuncRegistry.get_artifact("add") == add


def test_register_module_functions(FuncRegistry: FunctionalRegistry):
    """
    Test that functions in a module are automatically registered.
    """
    module = ModuleType("custom_module")

    def func_mocker(name: str) -> Callable:
        def wrapper(fn: Callable) -> Callable:
            fn.__name__ = name
            return fn

        return wrapper

    module.add = func_mocker("add")(lambda x, y: x + y)
    module.sub = func_mocker("sub")(lambda x, y: x - y)

    FuncRegistry.register_module_functions(module)

    assert FuncRegistry.artifact_exists(module.add)
    assert FuncRegistry.get_artifact("add") == module.add

    assert FuncRegistry.artifact_exists(module.sub)
    assert FuncRegistry.get_artifact("sub") == module.sub


def test_unregister_function(FuncRegistry: FunctionalRegistry):
    """
    Test that a registered function can be unregistered.
    """

    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    assert FuncRegistry.artifact_exists(add)
    assert FuncRegistry.get_artifact("add") == add

    FuncRegistry.unregister_function(add)
    assert not FuncRegistry.artifact_exists(add)
    with pytest.raises(RegistryError):
        FuncRegistry.get_artifact("add")


# -------------------------------------------------------------------
# Additional tests for extended functionality.
# -------------------------------------------------------------------


def test_duplicate_registration(FuncRegistry: FunctionalRegistry):
    """
    Test that attempting to register the same function twice raises a RegistryError.
    """

    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    with pytest.raises(RegistryError):
        FuncRegistry.register_function(add)


def test_contains_operator(FuncRegistry: FunctionalRegistry):
    """
    Test that the registry's membership operator (__contains__) works correctly.
    """

    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    instance = FuncRegistry()
    assert "add" in instance
    # Test that a non-registered function name is not reported as present.
    assert "nonexistent" not in instance


def test_instance_setitem_and_delitem(FuncRegistry: FunctionalRegistry):
    """
    Test that the instance-level __setitem__ and __delitem__ methods work as expected.
    """
    instance = FuncRegistry()
    # Use __setitem__ to register a function.
    instance["multiply"] = lambda x, y: x * y
    assert instance["multiply"](3, 4) == 12

    # Use __delitem__ to unregister the function.
    del instance["multiply"]
    with pytest.raises(RegistryError):
        _ = instance["multiply"]
