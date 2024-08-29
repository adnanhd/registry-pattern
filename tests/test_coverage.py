from typing import Protocol
from typing import Type
from typing import runtime_checkable

import pytest

from registry import ConformanceError
from registry import RegistryError
from registry import TypeRegistry
from registry.api import functional_registry_decorator
from registry.api import type_registry_decorator


# Mock classes and functions for testing
@runtime_checkable
class MockProtocol(Protocol):
    def method(self, x: int) -> int:
        ...


from typing import Iterable
from typing import Protocol
from typing import SupportsIndex
from typing import TypeVar

T = TypeVar("T")


@runtime_checkable
class SequenceProtocol(Protocol):
    __slots__ = ()

    def __getitem__(self, index: SupportsIndex):
        ...

    def __len__(self) -> int:
        ...

    def __contains__(self, item) -> bool:
        ...

    def __iter__(self) -> Iterable:
        ...

    def index(self, value, start: int = 0, stop: int = 9223372036854775807) -> int:
        ...

    def count(self, value) -> int:
        ...


class ValidClass(MockProtocol):
    def method(self, x: int) -> int:
        return x


class InvalidClass:
    def method2(self, y: int) -> int:
        return y


def valid_function(x: int) -> int:
    return x


def invalid_function(y: str) -> str:
    return y


def invalid_function2(y: str, z: str) -> str:
    return y


@pytest.fixture
def type_registry():
    return type_registry_decorator(strict=True, abstract=False)(MockProtocol)


@pytest.fixture
def sequence_registry():
    return type_registry_decorator(strict=True, abstract=False)(SequenceProtocol)


@pytest.fixture
def functional_registry():
    return functional_registry_decorator(strict=True)(valid_function)


def test_register_valid_class(type_registry):
    type_registry.register_class(ValidClass)
    assert type_registry.get_registry_item("ValidClass") is ValidClass


def test_register_invalid_class(type_registry):
    with pytest.raises(ConformanceError):
        type_registry.register_class(InvalidClass)


def test_register_valid_function(functional_registry):
    functional_registry.register_function(valid_function)
    assert functional_registry.get_registry_item("valid_function") is valid_function


def test_register_invalid_function(functional_registry):
    # TODO: keep in mind, invalid function means num of args is different
    functional_registry.register_function(invalid_function)
    with pytest.raises(ConformanceError):
        functional_registry.register_function(invalid_function2)


def test_unregister_class(type_registry):
    type_registry.register_class(ValidClass)
    type_registry.unregister_class(ValidClass)
    with pytest.raises(RegistryError):
        type_registry.get_registry_item("ValidClass")


def test_unregister_function(functional_registry):
    functional_registry.register_function(valid_function)
    functional_registry.unregister_function(valid_function)
    with pytest.raises(RegistryError):
        functional_registry.get_registry_item("valid_function")


def test_register_module_subclasses(
    sequence_registry: Type[TypeRegistry[SequenceProtocol]],
):
    # Assuming get_module_members is mocked or tested with actual modules
    import collections.abc

    sequence_registry.register_module_subclasses(collections.abc, raise_error=False)

    # Assert conditions based on the mocked module's contents
    assert sequence_registry.get_registry_item("Sequence") is collections.abc.Sequence
    assert (
        sequence_registry.get_registry_item("MutableSequence")
        is collections.abc.MutableSequence
    )
    assert (
        sequence_registry.get_registry_item("ByteString") is collections.abc.ByteString
    )


def test_register_module_functions(functional_registry):
    import math

    functional_registry.register_module_functions(math, raise_error=False)
    # Assert conditions based on the mocked module's contents
    assert functional_registry.len() == 45

    assert functional_registry.get_registry_item("acos") == math.acos
    assert functional_registry.get_registry_item("acosh") == math.acosh
    assert functional_registry.get_registry_item("asin") == math.asin
    assert functional_registry.get_registry_item("asinh") == math.asinh
    assert functional_registry.get_registry_item("atan") == math.atan
    assert functional_registry.get_registry_item("atanh") == math.atanh
    assert functional_registry.get_registry_item("cbrt") == math.cbrt
    assert functional_registry.get_registry_item("ceil") == math.ceil
    assert functional_registry.get_registry_item("cos") == math.cos
    assert functional_registry.get_registry_item("cosh") == math.cosh
    assert functional_registry.get_registry_item("degrees") == math.degrees
    assert functional_registry.get_registry_item("erf") == math.erf
    assert functional_registry.get_registry_item("erfc") == math.erfc
    assert functional_registry.get_registry_item("exp") == math.exp
    assert functional_registry.get_registry_item("exp2") == math.exp2
    assert functional_registry.get_registry_item("expm1") == math.expm1
    assert functional_registry.get_registry_item("fabs") == math.fabs
    assert functional_registry.get_registry_item("factorial") == math.factorial
    assert functional_registry.get_registry_item("floor") == math.floor
    assert functional_registry.get_registry_item("frexp") == math.frexp
    assert functional_registry.get_registry_item("fsum") == math.fsum
    assert functional_registry.get_registry_item("gamma") == math.gamma
    assert functional_registry.get_registry_item("gcd") == math.gcd
    assert functional_registry.get_registry_item("hypot") == math.hypot
    assert functional_registry.get_registry_item("isfinite") == math.isfinite
    assert functional_registry.get_registry_item("isinf") == math.isinf
    assert functional_registry.get_registry_item("isnan") == math.isnan
    assert functional_registry.get_registry_item("isqrt") == math.isqrt
    assert functional_registry.get_registry_item("lcm") == math.lcm
    assert functional_registry.get_registry_item("lgamma") == math.lgamma
    assert functional_registry.get_registry_item("log") == math.log
    assert functional_registry.get_registry_item("log10") == math.log10
    assert functional_registry.get_registry_item("log1p") == math.log1p
    assert functional_registry.get_registry_item("log2") == math.log2
    assert functional_registry.get_registry_item("modf") == math.modf
    assert functional_registry.get_registry_item("perm") == math.perm
    assert functional_registry.get_registry_item("prod") == math.prod
    assert functional_registry.get_registry_item("radians") == math.radians
    assert functional_registry.get_registry_item("sin") == math.sin
    assert functional_registry.get_registry_item("sinh") == math.sinh
    assert functional_registry.get_registry_item("sqrt") == math.sqrt
    assert functional_registry.get_registry_item("tan") == math.tan
    assert functional_registry.get_registry_item("tanh") == math.tanh
    assert functional_registry.get_registry_item("trunc") == math.trunc
    assert functional_registry.get_registry_item("ulp") == math.ulp
