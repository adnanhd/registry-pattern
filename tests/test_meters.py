"""Tests for FactoryMeter and the built-in meters."""

from typing import Any

import pytest

from registry import FunctionalRegistry, TypeRegistry, build
from registry.meters import (
    CPUMeter,
    FactoryMeter,
    HeapMeter,
    IOMeter,
    LifetimeMeter,
    MemoryMeter,
    NetworkMeter,
    RecursionMeter,
    attach_meter,
    detach_meter,
    emit_meter,
    meters,
)


class _MeterTestModelReg(TypeRegistry[object]):
    pass


class _MeterTestFnReg(FunctionalRegistry):
    pass


class _MeterAdder:
    def __init__(self, base: int = 0):
        self.base = base


@_MeterTestFnReg.register_artifact
def _meter_ok(x: int) -> int:
    return x * 2


@_MeterTestFnReg.register_artifact
def _meter_bad(x: int) -> int:
    raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _isolate_meter_registries():
    _MeterTestModelReg.clear_artifacts()
    _MeterTestModelReg.register_artifact(_MeterAdder)
    for name in list(meters()):
        detach_meter(name)
    yield
    for name in list(meters()):
        detach_meter(name)
    _MeterTestModelReg.clear_artifacts()


class _RecorderMeter(FactoryMeter):
    name = "recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_build_start(self, *, cfg, ctx, meta):
        self.events.append(("start", {"type": cfg.type}))

    def on_validated(self, *, target, kwargs, ctx, meta):
        self.events.append(("validated", {"target": target.__name__}))

    def on_built(self, *, target, result, meta, ctx):
        self.events.append(("built", {"target": target.__name__}))
        meta["recorder_wrote"] = True

    def on_error(self, *, cfg, exc, ctx, meta):
        self.events.append(("error", {"type": cfg.type, "exc": type(exc).__name__}))


def test_meter_fires_on_success() -> None:
    rec = attach_meter(_RecorderMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    assert [e[0] for e in rec.events] == ["start", "validated", "built"]
    assert cfg["meta"]["recorder_wrote"] is True


def test_meter_fires_on_error() -> None:
    rec = attach_meter(_RecorderMeter())
    with pytest.raises(RuntimeError, match="boom"):
        build({"type": "_meter_bad", "data": {"x": 1}})
    phases = [e[0] for e in rec.events]
    assert "start" in phases
    assert phases[-1] == "error"


def test_lifetime_meter_writes_seconds() -> None:
    attach_meter(LifetimeMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    assert "lifetime_seconds" in cfg["meta"]
    assert cfg["meta"]["lifetime_seconds"] >= 0.0


def test_cpu_meter_writes_user_sys() -> None:
    attach_meter(CPUMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    assert "cpu_user_seconds" in cfg["meta"]
    assert "cpu_system_seconds" in cfg["meta"]


def test_memory_meter_writes_rss() -> None:
    attach_meter(MemoryMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    assert cfg["meta"]["rss_max_kb"] > 0


def test_io_meter_writes_bytes() -> None:
    attach_meter(IOMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    for k in ("io_read_bytes", "io_write_bytes", "io_rchar_bytes", "io_wchar_bytes"):
        assert k in cfg["meta"]


def test_network_meter_writes_rx_tx() -> None:
    attach_meter(NetworkMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    assert "net_rx_bytes" in cfg["meta"]
    assert "net_tx_bytes" in cfg["meta"]


def test_heap_meter_writes_tracemalloc() -> None:
    attach_meter(HeapMeter())
    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 1}, "meta": {}}
    build(cfg)
    for k in ("heap_current_bytes", "heap_delta_bytes", "heap_peak_bytes"):
        assert k in cfg["meta"]


def test_recursion_meter_tracks_depth() -> None:
    attach_meter(RecursionMeter())

    @_MeterTestFnReg.register_artifact
    def _outer(inner: Any) -> Any:
        return inner

    cfg: dict[str, Any] = {
        "type": "_outer",
        "data": {"inner": {"type": "_MeterAdder", "data": {"base": 1}, "meta": {}}},
        "meta": {},
    }
    build(cfg)
    assert cfg["meta"]["build_depth"] == 1
    assert cfg["meta"]["build_max_depth"] == 2
    assert cfg["data"]["inner"]["meta"]["build_depth"] == 2


def test_all_meters_compose() -> None:
    attach_meter(LifetimeMeter())
    attach_meter(CPUMeter())
    attach_meter(MemoryMeter())
    attach_meter(IOMeter())
    attach_meter(NetworkMeter())
    attach_meter(HeapMeter())

    cfg: dict[str, Any] = {"type": "_meter_ok", "data": {"x": 2}, "meta": {}}
    out = build(cfg)
    assert out == 4
    assert {"lifetime_seconds", "cpu_user_seconds", "rss_max_kb",
            "io_read_bytes", "net_rx_bytes", "heap_current_bytes"} <= set(cfg["meta"])


def test_meter_exceptions_do_not_break_build() -> None:
    class _Broken(FactoryMeter):
        name = "broken"

        def on_validated(self, **kw):
            raise RuntimeError("meter broke")

    attach_meter(_Broken())
    assert build({"type": "_meter_ok", "data": {"x": 3}}) == 6


def test_attach_replaces_by_name() -> None:
    a = _RecorderMeter()
    a.name = "same"
    b = _RecorderMeter()
    b.name = "same"
    attach_meter(a)
    attach_meter(b)
    build({"type": "_meter_ok", "data": {"x": 1}})
    assert a.events == []
    assert b.events != []
