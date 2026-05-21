"""Tests for the recursive factory pipeline (registry.factory.build)."""

from __future__ import annotations

from typing import Annotated, Any

import pytest

from registry import BuildCfg, FunctionalRegistry, TypeRegistry, build, resolve


class _SampleModelRegistry(TypeRegistry[Any]):
    pass


class _SampleFnRegistry(FunctionalRegistry):
    pass


@pytest.fixture(autouse=True)
def _clean_registries():
    _SampleModelRegistry.clear_artifacts()
    _SampleFnRegistry.clear_artifacts()
    yield
    _SampleModelRegistry.clear_artifacts()
    _SampleFnRegistry.clear_artifacts()


class _Adder:
    def __init__(self, base: int = 0, mult: int = 1):
        self.base = base
        self.mult = mult

    def __call__(self, x: int) -> int:
        return self.base + self.mult * x


def test_build_constructs_class() -> None:
    _SampleModelRegistry.register_artifact(_Adder)
    obj = build({"type": "_Adder", "data": {"base": 5, "mult": 3}})
    assert isinstance(obj, _Adder)
    assert obj.base == 5 and obj.mult == 3
    assert obj(10) == 35


def test_build_invokes_function() -> None:
    @_SampleFnRegistry.register_artifact
    def sum_two(a: int, b: int = 0) -> int:
        return a + b

    out = build({"type": "sum_two", "data": {"a": 7, "b": 8}})
    assert out == 15


def test_build_recurses_nested_envelope() -> None:
    class _Wrapper:
        def __init__(self, inner: object) -> None:
            self.inner = inner

    _SampleModelRegistry.register_artifact(_Adder)
    _SampleModelRegistry.register_artifact(_Wrapper)

    obj = build(
        {
            "type": "_Wrapper",
            "data": {"inner": {"type": "_Adder", "data": {"base": 2}}},
        }
    )
    assert isinstance(obj, _Wrapper)
    assert isinstance(obj.inner, _Adder)
    assert obj.inner.base == 2


def test_build_resolves_dollar_ref_from_ctx() -> None:
    @_SampleFnRegistry.register_artifact
    def takes_thing(thing: Any) -> Any:
        return thing

    sentinel = object()
    out = build({"type": "takes_thing", "data": {"thing": "$external"}}, ctx={"external": sentinel})
    assert out is sentinel


def test_build_resolves_dollar_ref_from_sibling() -> None:
    @_SampleFnRegistry.register_artifact
    def with_siblings(left: Any, right: int) -> int:
        return left.base + right

    _SampleModelRegistry.register_artifact(_Adder)
    # First sibling is built (becomes _Adder), second references its .base.
    out = build(
        {
            "type": "with_siblings",
            "data": {
                "left": {"type": "_Adder", "data": {"base": 4}},
                "right": "$left.base",
            },
        }
    )
    assert out == 8


def test_build_call_method_ref() -> None:
    @_SampleFnRegistry.register_artifact
    def takes_value(adder: Any, value: int) -> int:
        return value

    _SampleModelRegistry.register_artifact(_Adder)
    out = build(
        {
            "type": "takes_value",
            "data": {
                "adder": {"type": "_Adder", "data": {"base": 10}},
                "value": "$adder.base",
            },
        }
    )
    assert out == 10


def test_resolve_raises_on_missing() -> None:
    with pytest.raises(KeyError, match="not registered"):
        resolve("NotARealType")


def test_resolve_disambiguates_with_repo() -> None:
    class _AltModelRegistry(TypeRegistry[Any]):
        pass

    _SampleModelRegistry.register_artifact(_Adder)
    _AltModelRegistry.register_artifact(_Adder)
    # Without repo: ambiguous
    with pytest.raises(KeyError, match="ambiguous"):
        resolve("_Adder")
    # With repo: disambiguated
    reg, art = resolve("_Adder", repo="_AltModelRegistry")
    assert reg is _AltModelRegistry
    _AltModelRegistry.clear_artifacts()


def test_build_envelope_meta_propagates() -> None:
    _SampleModelRegistry.register_artifact(_Adder)
    cfg = BuildCfg(type="_Adder", data={"base": 1}, meta={"label": "hello"})
    obj = build(cfg)
    assert obj.__meta__ == {"label": "hello"}
    assert cfg.meta == {"label": "hello"}


def test_build_writes_meta_back_to_input_dict() -> None:
    """When cfg is passed as a dict, the dict's meta key should be updated in place."""

    class _MetaWriter:
        @classmethod
        def post_init(cls, instance, meta):
            meta["sentinel"] = "written"

    class _MetaWriterRegistry(TypeRegistry[Any]):
        @classmethod
        def post_init(cls, instance, meta):
            meta["from_post_init"] = True

    _MetaWriterRegistry.register_artifact(_Adder)
    cfg = {"type": "_Adder", "data": {"base": 1}, "meta": {}}
    build(cfg)
    assert cfg["meta"] == {"from_post_init": True}
    _MetaWriterRegistry.clear_artifacts()


def test_build_uses_noop_validator_when_requested() -> None:
    _SampleModelRegistry.register_artifact(_Adder)
    # Pass an unknown extra key; pydantic validator strips it (not in schema),
    # noop validator passes everything through unchanged.
    cfg = {"type": "_Adder", "data": {"base": 9, "unused_kwarg": "x"}}
    # noop will then try _Adder(**{"base": 9, "unused_kwarg": "x"}) -> TypeError
    with pytest.raises(TypeError):
        build(cfg, validator="noop")
    # pydantic strips unknowns (extra="ignore" default for create_model fields)
    obj = build({"type": "_Adder", "data": {"base": 9}})
    assert obj.base == 9
