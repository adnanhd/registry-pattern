"""
Test class registry.
"""

from typing import Protocol
import pytest

from registry import make_class_registry, ClassRegistry, StructuringError, RegistryError


class Foo(Protocol):
    def test_method(self, x: int, y: int) -> int: ...


@pytest.fixture
def Registry():
    return make_class_registry("FooClass", protocol=Foo)


def test_class_registry_with_subclass(Registry: ClassRegistry):
    @Registry.register_class
    @Registry.__class__.register
    class Bar:
        def test_method(self, x: int, y: int) -> int:
            return 2

    assert Registry.get_registry_item("Bar") is Bar

    with pytest.raises(RegistryError) as exc_info:
        Registry.register_class(Bar)

    assert exc_info.value.args[0] == "FooClassRegistry: 'Bar' already registered"


def test_class_registry_with_invalid_class(Registry: ClassRegistry):
    with pytest.raises(StructuringError) as exc_info:

        @Registry.register_class
        @Registry.__class__.register
        class Baz:
            def test2_method(self, x: int, y: int) -> int:
                return 2

    assert (
        exc_info.value.args[0]
        == "<class 'test_class_registry.test_class_registry_with_invalid_class.<locals>.Baz'>"
        " is not of type <class 'test_class_registry.Foo'>"
    )
