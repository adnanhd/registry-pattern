import pytest

from types import new_class, ModuleType

from typing import Callable, Tuple
from registry import FunctionalRegistry, ConformanceError, RegistryError


@pytest.fixture
def FuncRegistry():
    return new_class("FuncRegistry", (FunctionalRegistry[[int, int], int],))


def test_register_valid_function(FuncRegistry: FunctionalRegistry):
    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    assert FuncRegistry.has_registry_item(add)
    assert FuncRegistry.get_registry_item("add") == add


def test_register_invalid_function_missing_argument(FuncRegistry: FunctionalRegistry):
    @FuncRegistry.register_function
    def add(x: int) -> int:
        return x + 1

    assert FuncRegistry.has_registry_item(add)
    assert FuncRegistry.get_registry_item("add") == add


def test_register_invalid_function_extra_argument(FuncRegistry: FunctionalRegistry):

    @FuncRegistry.register_function
    def add(x: int, y: int, z: int) -> int:
        return x + y + z

    assert FuncRegistry.has_registry_item(add)
    assert FuncRegistry.get_registry_item("add") == add


def test_register_invalid_function_wrong_return_type(FuncRegistry: FunctionalRegistry):

    @FuncRegistry.register_function
    def add(x: int, y: int) -> str:
        return f"{x + y}"

    assert FuncRegistry.has_registry_item(add)
    assert FuncRegistry.get_registry_item("add") == add


def test_register_module_functions(FuncRegistry: FunctionalRegistry):
    module = ModuleType("custom_module")

    def func_mocker(name):
        def wrapper(fn):
            fn.__name__ = name
            return fn

        return wrapper

    module.add = func_mocker("add")(lambda x, y: x + y)
    module.sub = func_mocker("sub")(lambda x, y: x - y)

    FuncRegistry.register_module_functions(module)

    assert FuncRegistry.has_registry_item(module.add)
    assert FuncRegistry.get_registry_item("add") == module.add

    assert FuncRegistry.has_registry_item(module.sub)
    assert FuncRegistry.get_registry_item("sub") == module.sub


def test_unregister_function(FuncRegistry: FunctionalRegistry):
    @FuncRegistry.register_function
    def add(x: int, y: int) -> int:
        return x + y

    assert FuncRegistry.has_registry_item(add)
    assert FuncRegistry.get_registry_item("add") == add
    FuncRegistry.unregister_function(add)
    assert not FuncRegistry.has_registry_item(add)
    with pytest.raises(RegistryError):
        FuncRegistry.get_registry_item("add")
