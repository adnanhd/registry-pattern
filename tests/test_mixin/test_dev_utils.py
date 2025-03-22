import pytest
from types import ModuleType
from typing import Generic, TypeVar, Protocol

from registry.core._dev_utils import (
    get_protocol,
    get_subclasses,
    get_module_members,
    compose_two_funcs,
    compose,
)


# -------------------------------------------------------------------
# Test classes and protocols
# -------------------------------------------------------------------


T = TypeVar("T")


class TestProtocol(Protocol):
    __test__ = False

    def method(self, x: int) -> str: ...


class GenericClass(Generic[T]):
    pass


class SpecializedClass(GenericClass[TestProtocol]):
    pass


class BaseClass:
    pass


class ChildClass(BaseClass):
    pass


class GrandchildClass(ChildClass):
    pass


# -------------------------------------------------------------------
# Tests for protocol extraction
# -------------------------------------------------------------------


def test_get_protocol():
    """Test that get_protocol extracts the protocol from a generic class."""
    protocol = get_protocol(SpecializedClass)

    # Check that the protocol is the TestProtocol
    assert protocol is TestProtocol, f"Expected TestProtocol, got {protocol}"

    # Verify it's a runtime_checkable protocol
    assert hasattr(
        protocol, "_is_runtime_protocol"
    ), f"Does not have _is_runtime_protocol attribute, got {protocol}"

    # Non-generic class should raise an assertion error
    with pytest.raises(AssertionError):
        get_protocol(BaseClass)

    # Non-class should raise an assertion error
    with pytest.raises(AssertionError):
        get_protocol(42)  # type: ignore


# -------------------------------------------------------------------
# Tests for subclass detection
# -------------------------------------------------------------------


def test_get_subclasses():
    """Test that get_subclasses returns the correct subclasses."""
    # BaseClass has ChildClass as a direct subclass
    subclasses = get_subclasses(BaseClass)
    assert ChildClass in subclasses
    assert GrandchildClass not in subclasses  # Only direct subclasses

    # ChildClass has GrandchildClass as a direct subclass
    subclasses = get_subclasses(ChildClass)
    assert GrandchildClass in subclasses

    # GrandchildClass has no subclasses
    subclasses = get_subclasses(GrandchildClass)
    assert len(subclasses) == 0

    # Non-class should raise a validation error
    with pytest.raises(Exception):
        get_subclasses(42)  # type: ignore


# -------------------------------------------------------------------
# Tests for module member extraction
# -------------------------------------------------------------------


def test_get_module_members():
    """Test that get_module_members extracts members from a module."""
    # Create a test module
    module = ModuleType("test_module")

    # Add various members
    module.class1 = BaseClass
    module.class2 = ChildClass
    module.func1 = lambda x: x
    module.func2 = lambda x, y: x + y
    module.var1 = 42
    module.var2 = "string"
    module._private = "private"

    # Define __all__ to limit exported members
    module.__all__ = ["class1", "func1", "var1"]

    # Get members with __all__ respected
    members = get_module_members(module, ignore_all_keyword=False)

    # Only class1 and func1 should be extracted (var1 is not a class or function)
    assert BaseClass in members
    assert ChildClass not in members
    assert len(members) == 2

    # Get all members ignoring __all__
    members = get_module_members(module, ignore_all_keyword=True)

    # class1, class2, func1, and func2 should be extracted
    assert BaseClass in members
    assert ChildClass in members
    assert len(members) == 4  # Both classes and both functions

    # Non-module should raise an assertion error
    with pytest.raises(AssertionError):
        get_module_members(42)


# -------------------------------------------------------------------
# Tests for function composition
# -------------------------------------------------------------------


def test_compose_two_funcs():
    """Test that compose_two_funcs correctly composes two functions."""

    def f(x):
        return x + 1

    def g(x):
        return x * 2

    # f then g: (x + 1) * 2
    fg = compose_two_funcs(f, g)
    assert fg(3) == 8

    # g then f: (x * 2) + 1
    gf = compose_two_funcs(g, f)
    assert gf(3) == 7

    # Test wrapping (preserving metadata)
    f.__name__ = "add_one"
    fg_wrapped = compose_two_funcs(f, g, wrap=True)
    assert fg_wrapped.__name__ == "add_one"

    # Test no wrapping
    fg_unwrapped = compose_two_funcs(f, g, wrap=False)
    assert not hasattr(fg_unwrapped, "__name__") or fg_unwrapped.__name__ != "add_one"

    # Non-callable should raise an assertion error
    with pytest.raises(AssertionError):
        compose_two_funcs(42, g)  # type: ignore
    with pytest.raises(AssertionError):
        compose_two_funcs(f, 42)  # type: ignore


def test_compose():
    """Test that compose correctly composes multiple functions."""

    def f(x):
        return x + 1

    def g(x):
        return x * 2

    def h(x):
        return x - 3

    # f then g then h: ((x + 1) * 2) - 3
    fgh = compose(f, g, h)
    assert fgh(3) == 1

    # h then g then f: (((x - 3) * 2) + 1)
    hgf = compose(h, g, f)
    assert hgf(3) == 5

    # Test with a single function
    f_composed = compose(f)
    assert f_composed(3) == 4

    # Test wrapping (preserving metadata)
    f.__name__ = "add_one"
    fgh_wrapped = compose(f, g, h, wrap=True)
    assert fgh_wrapped.__name__ == "add_one"

    # Test no wrapping
    fgh_unwrapped = compose(f, g, h, wrap=False)
    assert not hasattr(fgh_unwrapped, "__name__") or fgh_unwrapped.__name__ != "add_one"
