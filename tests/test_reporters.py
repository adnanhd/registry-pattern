"""Tests for FactoryReporter and the built-in reporters."""

from typing import Any

import pytest

from registry import FunctionalRegistry, build
from registry.reporters import (
    FactoryReporter,
    HTTPDashboardReporter,
    attach_reporter,
    detach_reporter,
    reporters,
)


class _RepTestFnReg(FunctionalRegistry):
    pass


@_RepTestFnReg.register_artifact
def _rep_ok(x: int) -> int:
    return x * 3


@_RepTestFnReg.register_artifact
def _rep_bad(x: int) -> int:
    raise RuntimeError("boom")


@pytest.fixture(autouse=True)
def _isolate_reporters():
    for name in list(reporters()):
        detach_reporter(name)
    yield
    for name in list(reporters()):
        detach_reporter(name)


class _RecorderReporter(FactoryReporter):
    name = "recorder"

    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def on_build_start(self, *, cfg, ctx, meta):
        self.events.append(("start", {"type": cfg.type}))

    def on_validated(self, *, target, kwargs, ctx, meta):
        self.events.append(("validated", {"target": target.__name__}))

    def on_built(self, *, target, result, meta, ctx):
        self.events.append(("built", {"target": target.__name__, "meta_keys": list(meta)}))

    def on_error(self, *, cfg, exc, ctx, meta):
        self.events.append(("error", {"type": cfg.type}))


def test_reporter_fires_on_success() -> None:
    rec = attach_reporter(_RecorderReporter())
    build({"type": "_rep_ok", "data": {"x": 1}})
    assert [e[0] for e in rec.events] == ["start", "validated", "built"]


def test_reporter_fires_on_error() -> None:
    rec = attach_reporter(_RecorderReporter())
    with pytest.raises(RuntimeError, match="boom"):
        build({"type": "_rep_bad", "data": {"x": 1}})
    assert rec.events[-1][0] == "error"


def test_reporter_sees_meter_output_in_meta() -> None:
    """Pipeline contract: meters run before reporters at every stage."""
    from registry.meters import LifetimeMeter, attach_meter, detach_meter

    attach_meter(LifetimeMeter())
    rec = attach_reporter(_RecorderReporter())
    try:
        build({"type": "_rep_ok", "data": {"x": 1}})
    finally:
        detach_meter("lifetime")
    # The reporter's on_built saw the meta AFTER LifetimeMeter wrote lifetime_seconds.
    built_event = next(e for e in rec.events if e[0] == "built")
    assert "lifetime_seconds" in built_event[1]["meta_keys"]


def test_attach_replaces_by_name() -> None:
    a = _RecorderReporter()
    a.name = "same"
    b = _RecorderReporter()
    b.name = "same"
    attach_reporter(a)
    attach_reporter(b)
    build({"type": "_rep_ok", "data": {"x": 1}})
    assert a.events == []
    assert b.events != []


def test_reporter_exceptions_do_not_break_build() -> None:
    class _Broken(FactoryReporter):
        name = "broken"

        def on_validated(self, **kw):
            raise RuntimeError("reporter broke")

    attach_reporter(_Broken())
    assert build({"type": "_rep_ok", "data": {"x": 4}}) == 12


def test_http_dashboard_serves_events() -> None:
    import json
    import urllib.request

    dash = attach_reporter(HTTPDashboardReporter(port=8767))
    build({"type": "_rep_ok", "data": {"x": 5}})
    with urllib.request.urlopen(dash.url) as r:
        events = json.loads(r.read())
    assert any(e.get("phase") == "start" for e in events)
    assert any(e.get("phase") == "built" for e in events)


def test_journal_reporter_imports() -> None:
    """Smoke: JournalReporter constructs without raising even if journald absent."""
    from registry.reporters import JournalReporter

    j = JournalReporter(ident="test-reporter")
    assert j.name == "journal"


def test_opentelemetry_reporter_import_optional() -> None:
    """OpenTelemetryReporter requires the [otel] extra; gracefully reports missing."""
    from registry.reporters import OpenTelemetryReporter

    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError):
            OpenTelemetryReporter()
    else:
        # If installed, should construct fine
        r = OpenTelemetryReporter()
        assert r.name == "opentelemetry"
