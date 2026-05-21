"""Tests for schema derivation (registry.schema)."""

from typing import Annotated, Any

from pydantic import BaseModel

from registry import Buildable
from registry.schema import (
    derive_config_schema,
    derive_meta_schema,
    process_compute,
    process_validate,
    resolve_data_schema,
    resolve_meta_schema,
)


class _Thing:
    def __init__(self, x: int = 0):
        self.x = x


class _ValidateMarker:
    def validate(self, value, kwargs, ctx):
        _ValidateMarker.calls.append((value, dict(kwargs), dict(ctx)))

    calls: list = []


class _ComputeMarker:
    def __init__(self, name: str):
        self.name = name

    def compute(self, value: Any) -> str:
        return f"hash:{value}"


def _fn_config_passthrough(thing: _Thing, count: int = 1, label: str = "hi") -> None: ...
def _fn_only_thing(thing: _Thing) -> None: ...
def _fn_with_compute_marker(thing: Annotated[_Thing, _ComputeMarker("thing_score")]) -> None: ...
def _fn_with_validate_marker(x: Annotated[int, _ValidateMarker()]) -> None: ...
def _fn_with_compute_marker_x(x: Annotated[int, _ComputeMarker("x_hash")]) -> None: ...


def _fn_no_markers(x: int, y: str) -> None: ...


def _fn_unrelated(unrelated: int) -> None: ...


def test_derive_config_schema_rewrites_arbitrary_to_buildable() -> None:
    schema = derive_config_schema(_fn_config_passthrough)
    fields = schema.model_fields

    assert fields["count"].annotation is int
    assert fields["label"].annotation is str
    assert fields["thing"].annotation is Buildable[_Thing]


def test_derive_config_schema_no_arbitrary_types_allowed() -> None:
    schema = derive_config_schema(_fn_only_thing)
    cfg = schema.model_config
    assert not cfg.get("arbitrary_types_allowed", False)


def test_derive_meta_schema_picks_up_compute_markers() -> None:
    schema = derive_meta_schema(_fn_with_compute_marker)
    assert schema is not None
    assert "thing_score" in schema.model_fields
    assert schema.model_fields["thing_score"].annotation is str


def test_derive_meta_schema_none_when_no_compute_markers() -> None:
    assert derive_meta_schema(_fn_only_thing) is None


def test_explicit_data_schema_overrides_derived() -> None:
    class Explicit(BaseModel):
        name: str

    class _Reg:
        data_schema = Explicit

    assert resolve_data_schema(_Reg, _fn_unrelated) is Explicit


def test_process_validate_runs_markers() -> None:
    _ValidateMarker.calls = []
    process_validate(_fn_with_validate_marker, {"x": 7}, {"ctxkey": "ctxval"})
    assert _ValidateMarker.calls == [(7, {"x": 7}, {"ctxkey": "ctxval"})]


def test_process_compute_writes_to_meta() -> None:
    meta: dict[str, Any] = {}
    process_compute(_fn_with_compute_marker_x, {"x": 42}, meta)
    assert meta == {"x_hash": "hash:42"}


def test_process_validate_skips_kwargs_without_markers() -> None:
    process_validate(_fn_no_markers, {"x": 1, "y": "z"}, {})


def test_resolve_meta_schema_explicit_wins() -> None:
    class Explicit(BaseModel):
        m: int

    class _Reg:
        meta_schema = Explicit

    assert resolve_meta_schema(_Reg, _fn_unrelated) is Explicit
