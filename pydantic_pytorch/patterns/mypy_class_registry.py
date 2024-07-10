import registry
import typing
from pydantic import BaseModel
from registry import ClassRegistry


class Foo(typing.Protocol):
    def foo(self, x: int, y: int) -> int: ...


class FooClassRegistry(ClassRegistry[Foo]):
    """Class registry for Foo."""


@FooClassRegistry.register_class
@FooClassRegistry.register
class Bar:
    def foo(self, x: int, y: int) -> int:
        return 2


@FooClassRegistry.register_class
@FooClassRegistry.register
class Bar2:
    ...
