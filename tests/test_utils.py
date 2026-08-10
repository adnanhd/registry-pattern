"""Coverage for registry.utils naming, composition, and signature helpers.

Exercises the public helpers documented in the module (get_type_name,
get_func_name, get_artifact_name, get_callable_signature, pydantic_to_dict,
build_error_context) plus the composition, module-introspection, hashability,
weakref-cleanup, and Callable-signature-validation utilities.
"""

import functools
import gc
import types
import weakref
from typing import Any, Callable, Generic, List, Protocol, TypeVar, runtime_checkable

import pytest
from pydantic import BaseModel

from registry import TypeRegistry
from registry.utils import (
    ConformanceError,
    ValidationError,
    _validate_function_signature,
    build_error_context,
    cleanup_dead_weakrefs,
    compose,
    compose_two_funcs,
    get_artifact_name,
    get_callable_signature,
    get_func_name,
    get_module_members,
    get_module_name,
    get_object_name,
    get_protocol,
    get_subclasses,
    get_type_name,
    is_hashable,
    pydantic_to_dict,
)

_T = TypeVar("_T")


class CovBaseCls:
    pass


class CovSubCls(CovBaseCls):
    pass


class CovInit:
    def __init__(self, x: int, y: str = "d") -> None:
        self.x = x
        self.y = y


class CovParams(BaseModel):
    a: int
    b: str = "z"


class CovFakeV1:
    """Stands in for a pydantic v1 model: exposes ``dict`` but not ``model_dump``."""

    def dict(self) -> dict:
        return {"a": 1, "b": 2}


@runtime_checkable
class CovRunProto(Protocol):
    def do(self) -> None: ...


class CovPlainProto(Protocol):
    def do(self) -> None: ...


class CovGenericBase(Generic[_T]):
    pass


class CovHoldsRunProto(CovGenericBase[CovRunProto]):
    pass


class CovHoldsPlainProto(CovGenericBase[CovPlainProto]):
    pass


class CovHoldsInt(CovGenericBase[int]):
    pass


# -- get_type_name -----------------------------------------------------------


def test_get_type_name_simple() -> None:
    assert get_type_name(CovInit) == "CovInit"


def test_get_type_name_qualname() -> None:
    assert get_type_name(CovInit, qualname=True) == CovInit.__qualname__


def test_get_type_name_non_class_raises() -> None:
    with pytest.raises(ValidationError):
        get_type_name(5)


# -- get_func_name -----------------------------------------------------------


def test_get_func_name_unwraps_wrapped() -> None:
    def inner() -> None:
        return None

    @functools.wraps(inner)
    def wrapper() -> None:
        return None

    assert get_func_name(wrapper) == "inner"


def test_get_func_name_non_callable_raises() -> None:
    with pytest.raises(ValidationError):
        get_func_name(5)


# -- get_artifact_name -------------------------------------------------------


def test_get_artifact_name_class() -> None:
    assert get_artifact_name(CovInit) == "CovInit"


def test_get_artifact_name_function() -> None:
    def cov_fn() -> None:
        return None

    assert get_artifact_name(cov_fn) == "cov_fn"


def test_get_artifact_name_plain_value() -> None:
    assert get_artifact_name(5) == "5"


# -- get_protocol ------------------------------------------------------------


def test_get_protocol_runtime_checkable() -> None:
    assert get_protocol(CovHoldsRunProto) is CovRunProto


def test_get_protocol_makes_runtime_checkable() -> None:
    proto = get_protocol(CovHoldsPlainProto)
    assert getattr(proto, "_is_runtime_protocol", False)


def test_get_protocol_plain_type() -> None:
    assert get_protocol(CovHoldsInt) is int


# -- get_subclasses / module helpers -----------------------------------------


def test_get_subclasses() -> None:
    assert CovSubCls in get_subclasses(CovBaseCls)


def test_get_subclasses_non_class_raises() -> None:
    with pytest.raises(ValidationError):
        get_subclasses(5)


def _make_module() -> types.ModuleType:
    mod = types.ModuleType("cov_utils_mod")

    def pub_fn() -> int:
        return 1

    class PubCls:
        pass

    def _priv_fn() -> int:
        return 0

    mod.pub_fn = pub_fn  # type: ignore[attr-defined]
    mod.PubCls = PubCls  # type: ignore[attr-defined]
    mod._priv_fn = _priv_fn  # type: ignore[attr-defined]
    mod.CONST = 5  # type: ignore[attr-defined]
    return mod


def test_get_module_name() -> None:
    mod = _make_module()
    assert get_module_name(mod) == "cov_utils_mod"


def test_get_module_name_non_module_raises() -> None:
    with pytest.raises(AssertionError):
        get_module_name(5)


def test_get_object_name() -> None:
    assert get_object_name(CovInit) == "CovInit"


def test_get_object_name_no_name_raises() -> None:
    with pytest.raises(AssertionError):
        get_object_name(5)


def test_get_module_members_without_all() -> None:
    mod = _make_module()
    members = get_module_members(mod)
    names = {m.__name__ for m in members}
    assert "pub_fn" in names
    assert "PubCls" in names
    # Private and non-named members are excluded.
    assert "_priv_fn" not in names


def test_get_module_members_respects_all() -> None:
    mod = _make_module()
    mod.__all__ = ["pub_fn"]  # type: ignore[attr-defined]
    names = {m.__name__ for m in get_module_members(mod)}
    assert names == {"pub_fn"}


def test_get_module_members_ignore_all_keyword() -> None:
    mod = _make_module()
    mod.__all__ = ["pub_fn"]  # type: ignore[attr-defined]
    names = {m.__name__ for m in get_module_members(mod, ignore_all_keyword=True)}
    assert "PubCls" in names


def test_get_module_members_non_module_raises() -> None:
    with pytest.raises(AssertionError):
        get_module_members(5)


# -- compose -----------------------------------------------------------------


def _inc(x: int) -> int:
    return x + 1


def _double(x: int) -> int:
    return x * 2


def test_compose_two_funcs_wrapped() -> None:
    composed = compose_two_funcs(_inc, _double)
    assert composed(3) == 8  # double(inc(3)) == double(4)


def test_compose_two_funcs_unwrapped() -> None:
    composed = compose_two_funcs(_inc, _double, wrap=False)
    assert composed(3) == 8


def test_compose_two_funcs_non_callable_raises() -> None:
    with pytest.raises(AssertionError):
        compose_two_funcs(5, _double)


def test_compose_many() -> None:
    composed = compose(_double, _inc)
    # compose applies right-to-left: double(inc(3)) == 8
    assert composed(3) == 8


# -- _validate_function_signature --------------------------------------------


def test_validate_signature_match() -> None:
    def cov_good(a: int, b: str) -> bool:
        return True

    _validate_function_signature(cov_good, Callable[[int, str], bool])


def test_validate_signature_not_generic_alias() -> None:
    def cov_fn(a: int) -> bool:
        return True

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, int)


def test_validate_signature_origin_not_callable() -> None:
    def cov_fn(a: int) -> bool:
        return True

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, List[int])


def test_validate_signature_ellipsis_return_match() -> None:
    def cov_fn() -> int:
        return 1

    _validate_function_signature(cov_fn, Callable[..., int])


def test_validate_signature_ellipsis_missing_return() -> None:
    def cov_fn():
        return 1

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[..., int])


def test_validate_signature_ellipsis_return_mismatch() -> None:
    def cov_fn() -> str:
        return "x"

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[..., int])


def test_validate_signature_param_count_mismatch() -> None:
    def cov_fn(a: int) -> bool:
        return True

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[[int, str], bool])


def test_validate_signature_param_missing_annotation() -> None:
    def cov_fn(a) -> bool:
        return True

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[[int], bool])


def test_validate_signature_param_type_mismatch() -> None:
    def cov_fn(a: str) -> bool:
        return True

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[[int], bool])


def test_validate_signature_return_missing() -> None:
    def cov_fn(a: int):
        return True

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[[int], bool])


def test_validate_signature_return_mismatch() -> None:
    def cov_fn(a: int) -> str:
        return "x"

    with pytest.raises(ConformanceError):
        _validate_function_signature(cov_fn, Callable[[int], bool])


# -- get_callable_signature --------------------------------------------------


def test_get_callable_signature_class_strips_self() -> None:
    name, sig, params = get_callable_signature(CovInit)
    assert name == "CovInit"
    param_names = [p.name for p in params]
    assert "self" not in param_names
    assert param_names == ["x", "y"]


def test_get_callable_signature_function() -> None:
    def cov_fn(a: int, b: str = "d") -> str:
        return b

    name, sig, params = get_callable_signature(cov_fn)
    assert name == "cov_fn"
    assert [p.name for p in params] == ["a", "b"]


def test_get_callable_signature_non_callable_raises() -> None:
    with pytest.raises(ValidationError):
        get_callable_signature(5)


# -- pydantic_to_dict --------------------------------------------------------


def test_pydantic_to_dict_v2_model() -> None:
    model = CovParams(a=1, b="hi")
    assert pydantic_to_dict(model) == {"a": 1, "b": "hi"}


def test_pydantic_to_dict_v1_style() -> None:
    assert pydantic_to_dict(CovFakeV1()) == {"a": 1, "b": 2}


# -- build_error_context -----------------------------------------------------


class CovErrCtxRegistry(TypeRegistry[Any], repo="cov_utils_errctx"):
    pass


def test_build_error_context_minimal() -> None:
    ctx = build_error_context("do_thing")
    assert ctx["operation"] == "do_thing"


def test_build_error_context_full() -> None:
    ctx = build_error_context(
        "register",
        registry_cls=CovErrCtxRegistry,
        key="widget",
        artifact=CovInit,
        extra_flag=True,
    )
    assert ctx["operation"] == "register"
    assert ctx["registry_name"] == "CovErrCtxRegistry"
    assert "registry_size" in ctx
    assert ctx["key"] == "widget"
    assert ctx["key_type"] == "str"
    assert ctx["artifact_name"] == "CovInit"
    assert ctx["extra_flag"] is True


# -- is_hashable -------------------------------------------------------------


def test_is_hashable_true() -> None:
    assert is_hashable("x") is True
    assert is_hashable((1, 2)) is True


def test_is_hashable_false() -> None:
    assert is_hashable([1, 2]) is False


# -- cleanup_dead_weakrefs ---------------------------------------------------


class CovWeakTarget:
    pass


def test_cleanup_dead_weakrefs_key() -> None:
    obj = CovWeakTarget()
    ref = weakref.ref(obj)
    mapping = {ref: "payload"}
    del obj
    gc.collect()
    removed = cleanup_dead_weakrefs(mapping)
    assert removed == 1
    assert mapping == {}


def test_cleanup_dead_weakrefs_value() -> None:
    obj = CovWeakTarget()
    ref = weakref.ref(obj)
    mapping = {"k": ref}
    del obj
    gc.collect()
    removed = cleanup_dead_weakrefs(mapping, key_is_weakref=False)
    assert removed == 1
    assert mapping == {}


def test_cleanup_dead_weakrefs_keeps_live() -> None:
    obj = CovWeakTarget()
    ref = weakref.ref(obj)
    mapping = {ref: "payload"}
    removed = cleanup_dead_weakrefs(mapping)
    assert removed == 0
    assert mapping == {ref: "payload"}
    # Keep obj referenced until after the assertion.
    assert obj is not None
