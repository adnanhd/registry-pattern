"""Lifecycle tests for the per-target schema cache.

Cache contract:
  - Populated on register_artifact.
  - Hit on every build() / Buildable[T] validation.
  - Dropped on unregister_artifact / unregister_identifier.
  - Auto-evicted when the target type is garbage-collected (weak keys).
  - Falls back to on-demand derivation for never-registered targets.
"""

from __future__ import annotations

import gc
import weakref
from typing import Any

from pydantic import BaseModel

from registry import FunctionalRegistry, TypeRegistry, build
from registry.schema import (
    ArtifactSchema,
    cache_schema,
    drop_schema,
    ensure_schema,
    get_schema,
)

# ---------------------------------------------------------------------------
# Population at registration time
# ---------------------------------------------------------------------------


class TestPopulatedOnRegister:
    def test_type_registry_caches_schema_on_register(self):
        class R(TypeRegistry[Any], repo="cache.test.tr1"):
            pass

        class C:
            def __init__(self, x: int, y: str = "hi"):
                self.x = x
                self.y = y

        assert get_schema(C) is None
        R.register_artifact(C)
        cached = get_schema(C)
        assert cached is not None
        assert isinstance(cached, ArtifactSchema)
        assert cached.config is not None
        # hints should include both annotated params
        assert "x" in cached.hints
        assert "y" in cached.hints
        # Cleanup so other tests do not see this class.
        R.unregister_artifact(C)

    def test_func_registry_caches_schema_on_register(self):
        class FR(FunctionalRegistry[[Any], Any], repo="cache.test.fr1"):
            pass

        def fn(a: int, b: str = "hi") -> str:
            return f"{b}-{a}"

        assert get_schema(fn) is None
        FR.register_artifact(fn)
        cached = get_schema(fn)
        assert cached is not None
        assert "a" in cached.hints
        FR.unregister_artifact(fn)

    def test_explicit_params_model_takes_precedence(self):
        class R(TypeRegistry[Any], repo="cache.test.tr2"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        class Explicit(BaseModel):
            x: int
            extra_field: str = "from-explicit"

        R.register_artifact(C, params_model=Explicit)
        cached = get_schema(C)
        assert cached is not None
        assert cached.config is Explicit
        R.unregister_artifact(C)


# ---------------------------------------------------------------------------
# Hit on build()
# ---------------------------------------------------------------------------


class TestHitOnBuild:
    def test_build_uses_cached_schema(self):
        class R(TypeRegistry[Any], repo="cache.test.build1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        cached_before = get_schema(C)
        assert cached_before is not None

        # Build many times: schema cache identity must not change.
        for _ in range(10):
            build({"type": "C", "repo": "cache.test.build1", "data": {"x": 1}})
        cached_after = get_schema(C)
        assert cached_after is cached_before
        R.unregister_artifact(C)


# ---------------------------------------------------------------------------
# Eviction on unregister
# ---------------------------------------------------------------------------


class TestEvictionOnUnregister:
    def test_unregister_artifact_drops_entry(self):
        class R(TypeRegistry[Any], repo="cache.test.un1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        assert get_schema(C) is not None
        R.unregister_artifact(C)
        assert get_schema(C) is None

    def test_unregister_identifier_drops_entry(self):
        class R(TypeRegistry[Any], repo="cache.test.un2"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        assert get_schema(C) is not None
        R.unregister_identifier("C")
        assert get_schema(C) is None

    def test_unregister_then_reregister_rebuilds(self):
        class R(TypeRegistry[Any], repo="cache.test.un3"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        first = get_schema(C)
        R.unregister_artifact(C)
        assert get_schema(C) is None
        R.register_artifact(C)
        second = get_schema(C)
        # New ArtifactSchema instance after re-register (caches are rebuilt).
        assert second is not None
        assert second is not first


# ---------------------------------------------------------------------------
# Weak-key GC behavior
# ---------------------------------------------------------------------------


class TestWeakKeySemantics:
    def test_entry_vanishes_when_type_is_gc(self):
        # Build a class purely inside this scope so the registry's reference
        # is the only strong one; then unregister and let GC collect.
        class R(TypeRegistry[Any], repo="cache.test.weak1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        wref = weakref.ref(C)
        # Unregister to release the registry's strong reference.
        R.unregister_artifact(C)
        # Drop the local strong reference.
        del C
        gc.collect()
        assert wref() is None, "type was not garbage-collected"


# ---------------------------------------------------------------------------
# Fallback for never-registered targets
# ---------------------------------------------------------------------------


class TestFallbackForUnregistered:
    def test_ensure_schema_derives_on_demand(self):
        class Standalone:
            def __init__(self, x: int):
                self.x = x

        assert get_schema(Standalone) is None
        schema = ensure_schema(Standalone)
        assert schema is not None
        assert get_schema(Standalone) is schema  # now cached
        drop_schema(Standalone)
        assert get_schema(Standalone) is None

    def test_cache_schema_with_override_then_lookup(self):
        class C:
            def __init__(self, x: int):
                self.x = x

        class Explicit(BaseModel):
            x: int

        cache_schema(C, config_override=Explicit)
        assert get_schema(C).config is Explicit
        drop_schema(C)


# ---------------------------------------------------------------------------
# Identity / no-rebuild on hot path
# ---------------------------------------------------------------------------


class TestIdentity:
    def test_repeated_ensure_returns_same_instance(self):
        class C:
            def __init__(self, x: int):
                self.x = x

        s1 = ensure_schema(C)
        s2 = ensure_schema(C)
        assert s1 is s2
        drop_schema(C)
