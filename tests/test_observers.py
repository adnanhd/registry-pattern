"""Tests for the factory observer pattern."""

from typing import Any

import pytest

from registry import FactoryObserver, FunctionalRegistry, TypeRegistry, attach, build, detach
from registry.observers import emit, observers


class _ObsTestModelReg(TypeRegistry[object]):
    pass


class _ObsTestFnReg(FunctionalRegistry):
    pass


class _ObsAdder:
    def __init__(self, base: int = 0):
        self.base = base


@_ObsTestFnReg.register_artifact
def _ok_fn(x: int) -> int:
    return x * 2


@_ObsTestFnReg.register_artifact
def _bad_fn(x: int) -> int:
    raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _isolate_obs_registries():
    _ObsTestModelReg.clear_artifacts()
    _ObsTestModelReg.register_artifact(_ObsAdder)
    yield
    _ObsTestModelReg.clear_artifacts()


class _Recorder(FactoryObserver):
    name = "test_recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_build_start(self, *, cfg, ctx):
        self.events.append(("start", {"type": cfg.type}))

    def on_validated(self, *, target, kwargs, ctx):
        self.events.append(("validated", {"target": target.__name__, "kwargs": dict(kwargs)}))

    def on_built(self, *, target, result, meta, ctx):
        self.events.append(("built", {"target": target.__name__, "result_type": type(result).__name__}))

    def on_error(self, *, cfg, exc, ctx):
        self.events.append(("error", {"type": cfg.type, "exc": type(exc).__name__}))


@pytest.fixture(autouse=True)
def _clean_observers():
    # Snapshot + restore so tests don't leak observers.
    before = dict(observers())
    yield
    for name in list(observers()):
        detach(name)
    for name, obs in before.items():
        attach(obs)


def test_observer_fires_on_success() -> None:
    rec = attach(_Recorder())
    build({"type": "_ok_fn", "data": {"x": 5}})
    phases = [e[0] for e in rec.events]
    assert phases == ["start", "validated", "built"]
    assert rec.events[1][1]["target"] == "_ok_fn"


def test_observer_fires_on_error() -> None:
    rec = attach(_Recorder())
    with pytest.raises(RuntimeError, match="boom"):
        build({"type": "_bad_fn", "data": {"x": 5}})
    phases = [e[0] for e in rec.events]
    # start + validated (succeeded) + error (target raised)
    assert "start" in phases
    assert "error" in phases
    assert phases[-1] == "error"


def test_multiple_observers_attach_replace_by_name() -> None:
    a = _Recorder()
    a.name = "same"
    b = _Recorder()
    b.name = "same"
    attach(a)
    attach(b)
    build({"type": "_ok_fn", "data": {"x": 1}})
    # Only `b` should have fired (replaced `a`).
    assert a.events == []
    assert b.events != []


def test_detach_removes_observer() -> None:
    rec = attach(_Recorder())
    detach("test_recorder")
    build({"type": "_ok_fn", "data": {"x": 1}})
    assert rec.events == []


def test_observer_exceptions_do_not_break_build() -> None:
    class _Broken(FactoryObserver):
        name = "broken"

        def on_validated(self, **kw):
            raise RuntimeError("observer is broken")

    attach(_Broken())
    # Build must not raise — emit() swallows observer errors.
    out = build({"type": "_ok_fn", "data": {"x": 3}})
    assert out == 6


def test_emit_no_op_when_method_missing() -> None:
    class _OnlyStart(FactoryObserver):
        name = "only_start"
        starts: int = 0

        def on_build_start(self, **kw):
            type(self).starts += 1

    attach(_OnlyStart())
    build({"type": "_ok_fn", "data": {"x": 1}})
    # No exception even though only on_build_start is overridden.
    assert _OnlyStart.starts == 1


def test_nested_build_emits_per_envelope() -> None:
    rec = attach(_Recorder())

    @_ObsTestFnReg.register_artifact
    def _wrapper(inner: Any) -> Any:
        return inner

    build({
        "type": "_wrapper",
        "data": {"inner": {"type": "_ObsAdder", "data": {"base": 7}}},
    })
    # Two builds happened: _ObsAdder (inner) and _wrapper (outer). Each fires start+validated+built.
    phases = [e[0] for e in rec.events]
    assert phases.count("start") == 2
    assert phases.count("built") == 2
