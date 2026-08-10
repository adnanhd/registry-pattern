"""Concurrency tests for registry mutation, resolve(), and meters.

These target three races found by audit. Each core test widens the race
window deterministically (an injected delay or an explicit event handshake
at the exact interleaving point) instead of relying on scheduler luck, so
it fails reliably against the pre-fix code and passes reliably against the
fix -- a bare thread-count-and-hope harness rarely preempts inside a check-
then-act on a few dict/list operations often enough to be a trustworthy
regression test.

  - Duplicate-key registration is a check-then-act (assert-absent, then
    write) that used to be non-atomic: concurrent `register_artifact` calls
    for the same key could both pass the absence check and both write,
    silently losing one registration instead of raising for the loser.

  - `resolve()` memoizes into `RESOLVE_CACHE` with a miss -> scan -> write
    sequence that used to have no guard against a concurrent unregister
    invalidating the cache mid-scan; a stale write could land after the
    invalidation and resurrect an already-unregistered artifact.

  - `_StackedMeter` (`LifetimeMeter`/`CPUMeter`/`MemoryMeter`) is a shared
    singleton instance (attached once via `attach_meter`) reused across
    every concurrent `build()` call. Its baseline stack used to be a single
    shared list, so two threads racing non-nested `build()` calls could pop
    each other's baseline and silently write a corrupted delta into `meta`.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, List

import pytest

from registry import (
    FunctionalRegistry,
    RegistryError,
    TypeRegistry,
    build,
    resolve,
)
from registry.extra.meters import LifetimeMeter, _StackedMeter
from registry.meters import attach_meter, detach_meter
from registry.resolve_cache import invalidate_resolve_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    invalidate_resolve_cache()
    yield
    invalidate_resolve_cache()


def _run_concurrently(fns: List[Any], timeout: float = 10.0) -> None:
    """Start every callable in `fns` on its own thread; join them all."""
    threads = [threading.Thread(target=fn) for fn in fns]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout)
        assert not t.is_alive(), "thread did not finish in time"


class TestConcurrentDuplicateRegistration:
    """Concurrent registration of the same key must yield exactly one winner.

    Both tests widen the check-then-act gap in `_assert_absence` with a
    monkeypatched delay: on the fix, the whole check-then-act happens under
    `_mapping_lock()`, so the delay just makes the loser wait longer for the
    lock and still lose deterministically; on pre-fix code, the delay gives
    a concurrent caller ample time to also pass the (now stale) absence
    check, so the lost-update always fires.
    """

    def test_exactly_one_winner_rest_raise_registry_error(self):
        class R(TypeRegistry[Any], repo="conc.test.duplicate"):
            pass

        class Foo:
            def __init__(self, x: int = 0):
                self.x = x

        import registry.mixin.mutator as mutator_mod

        original = mutator_mod.RegistryMutatorMixin._assert_absence.__func__

        def delayed_assert_absence(cls, key):
            result = original(cls, key)  # raises RegistryError if already present
            time.sleep(0.2)
            return result

        mutator_mod.RegistryMutatorMixin._assert_absence = classmethod(
            delayed_assert_absence
        )
        try:
            barrier = threading.Barrier(2)
            results: List[Any] = [None, None]

            def worker(i: int) -> None:
                barrier.wait()
                try:
                    R.register_artifact(Foo)
                    results[i] = "ok"
                except RegistryError:
                    results[i] = "conflict"

            _run_concurrently([lambda i=i: worker(i) for i in range(2)])
        finally:
            mutator_mod.RegistryMutatorMixin._assert_absence = classmethod(original)

        assert results.count("ok") == 1, f"expected exactly one winner, got {results}"
        assert results.count("conflict") == 1
        assert R.get_artifact("Foo") is Foo

    def test_functional_registry_same_race(self):
        class FR(FunctionalRegistry, repo="conc.test.duplicate.fn"):
            pass

        def step(x: int = 0) -> int:
            return x

        import registry.mixin.mutator as mutator_mod

        original = mutator_mod.RegistryMutatorMixin._assert_absence.__func__

        def delayed_assert_absence(cls, key):
            result = original(cls, key)
            time.sleep(0.2)
            return result

        mutator_mod.RegistryMutatorMixin._assert_absence = classmethod(
            delayed_assert_absence
        )
        try:
            barrier = threading.Barrier(2)
            results: List[Any] = [None, None]

            def worker(i: int) -> None:
                barrier.wait()
                try:
                    FR.register_artifact(step)
                    results[i] = "ok"
                except RegistryError:
                    results[i] = "conflict"

            _run_concurrently([lambda i=i: worker(i) for i in range(2)])
        finally:
            mutator_mod.RegistryMutatorMixin._assert_absence = classmethod(original)

        assert results.count("ok") == 1
        assert results.count("conflict") == 1


class TestConcurrentResolveRegisterUnregister:
    """resolve() must never hand back an artifact that is no longer registered."""

    def test_no_stale_write_survives_invalidation(self):
        """Directly targets the miss -> scan -> write race in resolve().

        `R.get_artifact` is monkeypatched so a `resolve()` call, once it has
        already captured the artifact reference for its match, pauses right
        there. While paused, a concurrent `unregister_identifier` removes
        the artifact and invalidates the cache. The paused `resolve()` then
        resumes and tries to write its (now-stale) result -- that write
        must not land, and a fresh `resolve()` afterwards must raise
        `KeyError` rather than returning the ghost artifact.
        """
        from registry import resolve_cache

        class R(TypeRegistry[Any], repo="conc.test.resolve.stale"):
            pass

        class Slow:
            def __init__(self, x: int = 0):
                self.x = x

        R.register_artifact(Slow)

        reached = threading.Event()
        proceed = threading.Event()
        delay_applied = threading.Event()
        original_get_artifact = R.get_artifact.__func__

        def delayed_get_artifact(cls, key):
            result = original_get_artifact(cls, key)
            # Only pause the FIRST caller (the resolver's scan). `unregister_
            # identifier` also calls `get_artifact` internally (to drop the
            # schema cache entry) -- letting that reentrant call through
            # immediately keeps the two delay points from serializing on
            # each other's `proceed` wait.
            if not delay_applied.is_set():
                delay_applied.set()
                reached.set()
                proceed.wait(timeout=5.0)
            return result

        R.get_artifact = classmethod(delayed_get_artifact)
        try:
            result_holder: List[Any] = []

            def slow_resolver() -> None:
                result_holder.append(resolve("Slow", repo="conc.test.resolve.stale"))

            t = threading.Thread(target=slow_resolver)
            t.start()
            assert reached.wait(timeout=5.0), "resolve() never reached get_artifact"

            R.unregister_identifier("Slow")
            proceed.set()
            t.join(5.0)
            assert not t.is_alive()
        finally:
            del R.get_artifact  # restore the inherited classmethod

        assert len(result_holder) == 1, "resolve() call did not complete"

        cached = resolve_cache.RESOLVE_CACHE.get(("Slow", "conc.test.resolve.stale"))
        assert (
            cached is None
        ), "stale write resurrected an unregistered artifact in RESOLVE_CACHE"

        with pytest.raises(KeyError):
            resolve("Slow", repo="conc.test.resolve.stale")

    def test_resolve_never_returns_unregistered_artifact_under_churn(self):
        """Best-effort stress companion to the deterministic test above."""

        class R(TypeRegistry[Any], repo="conc.test.resolve"):
            pass

        class Widget:
            def __init__(self, x: int = 0):
                self.x = x

        stop = threading.Event()
        violations: List[str] = []
        lock = threading.Lock()

        def churn() -> None:
            while not stop.is_set():
                try:
                    R.register_artifact(Widget)
                except RegistryError:
                    pass
                try:
                    R.unregister_identifier("Widget")
                except RegistryError:
                    pass

        def resolver() -> None:
            while not stop.is_set():
                try:
                    _, artifact = resolve("Widget", repo="conc.test.resolve")
                except KeyError:
                    continue
                if artifact is not Widget:
                    with lock:
                        violations.append(
                            f"resolve returned wrong artifact: {artifact!r}"
                        )

        threads = [threading.Thread(target=churn) for _ in range(4)] + [
            threading.Thread(target=resolver) for _ in range(4)
        ]
        for t in threads:
            t.start()
        time.sleep(1.0)
        stop.set()
        for t in threads:
            t.join(5.0)
            assert not t.is_alive()

        assert not violations, violations


class TestConcurrentBuildWithStackedMeter:
    """A shared `_StackedMeter` must not cross-contaminate concurrent builds."""

    def test_stack_baseline_not_shared_across_threads(self):
        """White-box: forces the exact non-nested push/push/pop/pop interleave
        that corrupts a single shared LIFO stack, and checks each thread's
        `on_built` used its own baseline, not the other thread's.
        """

        class _TaggedMeter(_StackedMeter):
            name = "conc-test-tagged"

            def _sample(self):
                return (threading.get_ident(),)

            def _write(self, meta, before, after):
                meta["before_thread"] = before[0]
                meta["after_thread"] = after[0]

        meter = _TaggedMeter()
        meta_a: Dict[str, Any] = {}
        meta_b: Dict[str, Any] = {}

        a_pushed = threading.Event()
        b_pushed = threading.Event()
        a_popped = threading.Event()

        def thread_a() -> None:
            meter.on_build_start(cfg=None, ctx={}, meta=meta_a)
            a_pushed.set()
            assert b_pushed.wait(timeout=5.0)
            # A pops FIRST even though B pushed after A -- overlapping,
            # non-nested builds, the case a naive shared stack gets wrong.
            meter.on_built(target=None, result=None, meta=meta_a, ctx={})
            a_popped.set()

        def thread_b() -> None:
            assert a_pushed.wait(timeout=5.0)
            meter.on_build_start(cfg=None, ctx={}, meta=meta_b)
            b_pushed.set()
            assert a_popped.wait(timeout=5.0)
            meter.on_built(target=None, result=None, meta=meta_b, ctx={})

        ta = threading.Thread(target=thread_a)
        tb = threading.Thread(target=thread_b)
        ta.start()
        tb.start()
        ta.join(5.0)
        tb.join(5.0)
        assert not ta.is_alive() and not tb.is_alive()

        assert (
            meta_a["before_thread"] == meta_a["after_thread"]
        ), f"thread A's on_built used a foreign (cross-thread) baseline: {meta_a}"
        assert (
            meta_b["before_thread"] == meta_b["after_thread"]
        ), f"thread B's on_built used a foreign (cross-thread) baseline: {meta_b}"

    def test_lifetime_meter_sane_under_concurrent_build(self):
        """Integration-style companion through the real `build()` pipeline."""

        class R(TypeRegistry[Any], repo="conc.test.meter"):
            pass

        class Slowish:
            def __init__(self, delay_ms: int = 0):
                time.sleep(delay_ms / 1000.0)

        R.register_artifact(Slowish)

        meter = LifetimeMeter()
        attach_meter(meter)
        try:
            n_threads = 12
            metas: List[Any] = [None] * n_threads
            errors: List[Any] = [None] * n_threads

            def worker(i: int, delay_ms: int) -> None:
                try:
                    obj = build(
                        {
                            "type": "Slowish",
                            "repo": "conc.test.meter",
                            "data": {"delay_ms": delay_ms},
                        }
                    )
                    metas[i] = obj.__meta__
                except Exception as e:  # pragma: no cover - diagnostic aid
                    errors[i] = e

            delays = [((i % 4) * 15) for i in range(n_threads)]
            _run_concurrently(
                [lambda i=i, d=delays[i]: worker(i, d) for i in range(n_threads)]
            )

            assert not any(errors), errors
            for i, meta in enumerate(metas):
                assert meta is not None, f"build {i} produced no meta"
                lifetime = meta.get("lifetime_seconds")
                assert lifetime is not None, f"build {i} meta missing lifetime_seconds"
                assert lifetime >= 0, f"build {i} got negative lifetime: {lifetime}"
                assert lifetime < 2.0, f"build {i} got implausible lifetime: {lifetime}"
        finally:
            detach_meter(meter.name)
