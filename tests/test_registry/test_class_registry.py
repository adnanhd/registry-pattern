"""
Test class registry.
"""

from typing import Protocol
import pytest

from pydantic_pytorch.patterns.registry import ClassRegistry, RegistryError


class Foo(Protocol):
    def foo(self, x: int, y: int) -> int:
        ...


class FooClassRegistry(ClassRegistry[Foo]):
    """Class registry for Foo."""


def test_class_registry_with_non_subclass():
    with pytest.raises(RegistryError) as exc_info:
        @FooClassRegistry.register_class
        class Bar:
            def foo(self, x: int, y: int) -> int:
                return 2
    err_msg = exc_info.value.args[0]
    assert err_msg == "<class 'test_class_registry.test_class_registry_with_non_subclass.<locals>.Bar'> is not of cls <class 'test_class_registry.FooClassRegistry'>"



def test_class_registry_with_subclass():
    @FooClassRegistry.register_class
    @FooClassRegistry.register
    class Bar:
        def foo(self, x: int, y: int) -> int:
            return 2

    assert FooClassRegistry.get_registered('Bar') is Bar


    with pytest.raises(RegistryError) as exc_info:
        FooClassRegistry.register_class(Bar)

    assert exc_info.value.args[0] == "FooClassRegistry: 'Bar' is already registered"

def test_class_registry_with_invalid_class():
    with pytest.raises(RegistryError) as exc_info:
        @FooClassRegistry.register_class
        @FooClassRegistry.register
        class Baz:
            ...

    assert exc_info.value.args[0] == "<class 'test_class_registry.test_class_registry_with_invalid_class.<locals>.Baz'> is not of type <class 'test_class_registry.Foo'>"
