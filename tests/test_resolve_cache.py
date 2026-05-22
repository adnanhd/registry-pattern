"""Lifecycle tests for the resolve() memoization cache.

Cache contract:
  - Populated on the first successful resolve(type_name, repo).
  - Hit on every subsequent call with the same (type_name, repo) key.
  - Cleared whenever an artifact is registered or unregistered (any
    registry, anywhere) since either event can change the resolution.
  - KeyError outcomes are NOT cached (a later registration could change
    the answer).
"""

from __future__ import annotations

from typing import Any

import pytest

from registry import FunctionalRegistry, TypeRegistry, resolve
from registry.resolve_cache import RESOLVE_CACHE, invalidate_resolve_cache


@pytest.fixture(autouse=True)
def _clean_cache():
    """Each test starts with an empty resolve cache."""
    invalidate_resolve_cache()
    yield
    invalidate_resolve_cache()


class TestPopulationAndHits:
    def test_first_resolve_populates_cache(self):
        class R(TypeRegistry[Any], repo="rc.test.pop1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        assert ("C", "rc.test.pop1") not in RESOLVE_CACHE
        reg, art = resolve("C", "rc.test.pop1")
        assert reg is R and art is C
        assert RESOLVE_CACHE[("C", "rc.test.pop1")] == (R, C)
        R.unregister_artifact(C)

    def test_subsequent_resolves_hit_cache(self):
        class R(TypeRegistry[Any], repo="rc.test.hit1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        resolve("C", "rc.test.hit1")
        snapshot = RESOLVE_CACHE.copy()
        # Many repeats -- nothing new should appear.
        for _ in range(50):
            resolve("C", "rc.test.hit1")
        assert RESOLVE_CACHE == snapshot
        R.unregister_artifact(C)

    def test_repo_none_and_specific_repo_are_distinct_keys(self):
        class R(TypeRegistry[Any], repo="rc.test.keys1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        resolve("C", None)
        resolve("C", "rc.test.keys1")
        assert ("C", None) in RESOLVE_CACHE
        assert ("C", "rc.test.keys1") in RESOLVE_CACHE
        R.unregister_artifact(C)


class TestInvalidationOnRegister:
    def test_register_clears_existing_entries(self):
        class R(TypeRegistry[Any], repo="rc.test.inv1"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        resolve("C", "rc.test.inv1")
        assert RESOLVE_CACHE  # populated

        class D:
            def __init__(self, y: int):
                self.y = y

        R.register_artifact(D)
        # Newly registered artifact wipes the cache so previous
        # ambiguous-or-now-distinct resolutions are re-derived.
        assert not RESOLVE_CACHE
        R.unregister_artifact(C)
        R.unregister_artifact(D)


class TestInvalidationOnUnregister:
    def test_unregister_clears_existing_entries(self):
        class R(TypeRegistry[Any], repo="rc.test.inv2"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        resolve("C", "rc.test.inv2")
        assert RESOLVE_CACHE

        R.unregister_artifact(C)
        assert not RESOLVE_CACHE


class TestCorrectness:
    def test_repo_prefix_match_cached(self):
        # `repo="rc.test"` should match a registry at `repo="rc.test.prefix"`.
        class R(TypeRegistry[Any], repo="rc.test.prefix"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        reg1, art1 = resolve("C", "rc.test")
        reg2, art2 = resolve("C", "rc.test")
        assert reg1 is reg2 and art1 is art2
        assert RESOLVE_CACHE[("C", "rc.test")] == (R, C)
        R.unregister_artifact(C)

    def test_new_registration_then_resolve_returns_fresh(self):
        class R(TypeRegistry[Any], repo="rc.test.fresh"):
            pass

        class C:
            def __init__(self, x: int):
                self.x = x

        R.register_artifact(C)
        first = resolve("C", "rc.test.fresh")

        # Unregister and re-register with a fresh class.
        R.unregister_artifact(C)

        class C2:  # noqa: N801
            def __init__(self, x: int):
                self.x = x

        C2.__name__ = "C"
        R.register_artifact(C2)

        second = resolve("C", "rc.test.fresh")
        assert second != first  # different artifact returned, no stale hit
        assert second[1] is C2
        R.unregister_artifact(C2)

    def test_func_registry_invalidates_too(self):
        class FR(FunctionalRegistry[[Any], Any], repo="rc.test.fr1"):
            pass

        def f1(a: int) -> int:
            return a

        FR.register_artifact(f1)
        resolve("f1", "rc.test.fr1")
        assert RESOLVE_CACHE

        def f2(b: int) -> int:
            return b

        FR.register_artifact(f2)
        assert not RESOLVE_CACHE
        FR.unregister_artifact(f1)
        FR.unregister_artifact(f2)


class TestNegativeResultsNotCached:
    def test_keyerror_does_not_pollute_cache(self):
        with pytest.raises(KeyError):
            resolve("DoesNotExist", "rc.test.never")
        # KeyError outcomes must not be cached -- a later registration could
        # legitimately make the same call succeed.
        assert ("DoesNotExist", "rc.test.never") not in RESOLVE_CACHE
