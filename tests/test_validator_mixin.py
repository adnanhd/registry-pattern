"""Coverage for registry.mixin.validator: ValidationCache and validator mixins.

Two surfaces are exercised:
  - ``ValidationCache`` and the module-level cache helpers (hit / miss / expiry /
    eviction / stats / configure / clear).
  - The mutable validator surface via a ``TypeRegistry`` (register / unregister /
    clear / batch validate / safe batch register / registry-state report), plus
    the hashability guard on the read paths.
"""

from typing import Any, Dict

import pytest

from registry import TypeRegistry
from registry.mixin.validator import (
    MutableValidatorMixin,
    ValidationCache,
    clear_validation_cache,
    configure_validation_cache,
    get_cache_stats,
)
from registry.utils import RegistryError, ValidationError


# -- ValidationCache ---------------------------------------------------------


class _CacheKey:
    pass


def test_cache_set_then_get_hit() -> None:
    cache = ValidationCache(max_size=10, ttl_seconds=300.0)
    obj = _CacheKey()
    cache.set(obj, "op", True, ["hint-a"])
    result = cache.get(obj, "op")
    assert result == (True, ["hint-a"])
    # Returned suggestions are a copy, not the stored tuple.
    result[1].append("mutated")
    assert cache.get(obj, "op") == (True, ["hint-a"])


def test_cache_get_miss_returns_none() -> None:
    cache = ValidationCache()
    assert cache.get(_CacheKey(), "op") is None


def test_cache_entry_expires() -> None:
    cache = ValidationCache(max_size=10, ttl_seconds=0.0)
    obj = _CacheKey()
    cache.set(obj, "op", True, [])
    # ttl=0 means any elapsed time counts as expired on the next read.
    assert cache.get(obj, "op") is None


def test_cache_evicts_when_full() -> None:
    cache = ValidationCache(max_size=1, ttl_seconds=300.0)
    first = _CacheKey()
    second = _CacheKey()
    cache.set(first, "op", True, [])
    cache.set(second, "op", True, [])
    assert len(cache._cache) == 1
    assert cache.get(first, "op") is None
    assert cache.get(second, "op") == (True, [])


# -- module-level cache helpers ----------------------------------------------


def test_get_cache_stats_shape() -> None:
    stats = get_cache_stats()
    for key in (
        "total_entries",
        "expired_entries",
        "active_entries",
        "max_size",
        "ttl_seconds",
    ):
        assert key in stats


def test_configure_and_clear_validation_cache() -> None:
    configure_validation_cache(max_size=321, ttl_seconds=42.0)
    stats = get_cache_stats()
    assert stats["max_size"] == 321
    assert stats["ttl_seconds"] == 42.0
    removed = clear_validation_cache()
    assert isinstance(removed, int)
    # Restore the process-wide defaults for other tests.
    configure_validation_cache(max_size=1000, ttl_seconds=300.0)


# -- mutable validator surface via TypeRegistry ------------------------------


class CovValWidget:
    def __init__(self, n: int = 0) -> None:
        self.n = n


class CovValGadget:
    def __init__(self) -> None:
        pass


class CovValidatorReg(TypeRegistry[Any], repo="cov_validator_mixin"):
    pass


@pytest.fixture(autouse=True)
def _isolate_registry():
    CovValidatorReg.clear_artifacts()
    yield
    CovValidatorReg.clear_artifacts()


def test_register_inferred_key_and_get() -> None:
    CovValidatorReg.register_artifact(CovValWidget)
    assert CovValidatorReg.has_identifier("CovValWidget")
    assert CovValidatorReg.get_artifact("CovValWidget") is CovValWidget


def test_has_artifact_and_validate_artifact() -> None:
    CovValidatorReg.register_artifact(CovValWidget)
    assert CovValidatorReg.has_artifact(CovValWidget) is True
    assert CovValidatorReg.has_artifact(CovValGadget) is False
    assert CovValidatorReg.validate_artifact(CovValWidget) is CovValWidget


def test_iter_identifiers() -> None:
    CovValidatorReg.register_artifact(CovValWidget)
    CovValidatorReg.register_artifact(CovValGadget)
    assert set(CovValidatorReg.iter_identifiers()) == {"CovValWidget", "CovValGadget"}


def test_has_identifier_unhashable_returns_false() -> None:
    assert CovValidatorReg.has_identifier([1, 2]) is False


def test_unregister_identifier() -> None:
    CovValidatorReg.register_artifact(CovValWidget)
    CovValidatorReg.unregister_identifier("CovValWidget")
    assert not CovValidatorReg.has_identifier("CovValWidget")


def test_unregister_missing_identifier_raises() -> None:
    with pytest.raises(RegistryError):
        CovValidatorReg.unregister_identifier("NopeMissing")


def test_unregister_artifact() -> None:
    CovValidatorReg.register_artifact(CovValWidget)
    CovValidatorReg.unregister_artifact(CovValWidget)
    assert not CovValidatorReg.has_identifier("CovValWidget")


def test_unregister_artifact_missing_raises() -> None:
    with pytest.raises(RegistryError):
        CovValidatorReg.unregister_artifact(CovValGadget)


def test_validate_registry_state() -> None:
    CovValidatorReg.register_artifact(CovValWidget)
    CovValidatorReg.register_artifact(CovValGadget)
    report = CovValidatorReg.validate_registry_state()
    assert report["total_artifacts"] == 2
    assert report["validation_errors"] == []
    assert "cache_stats" in report


def test_batch_validate_reports_per_key() -> None:
    results = CovValidatorReg.batch_validate({"CovValWidget": CovValWidget})
    assert results["CovValWidget"] is True


def test_batch_validate_bad_artifact_yields_error() -> None:
    # A non-class artifact fails class validation, so its value is an error.
    results = CovValidatorReg.batch_validate({"bad": [1, 2]})
    assert isinstance(results["bad"], ValidationError)


def test_registry_cache_management_classmethods() -> None:
    assert isinstance(CovValidatorReg.clear_validation_cache(), int)
    CovValidatorReg.configure_validation(cache_size=555, cache_ttl=60.0)
    stats = CovValidatorReg.get_validation_stats()
    assert stats["max_size"] == 555
    assert stats["ttl_seconds"] == 60.0
    # Restore defaults so other tests observe the standard configuration.
    CovValidatorReg.configure_validation(cache_size=1000, cache_ttl=300.0)


# -- base MutableValidatorMixin (explicit-key + batch registration) ----------
#
# ``TypeRegistry`` overrides ``register_artifact`` to a single-argument (key
# inferred from the class) form, so the explicit ``register_artifact(key,
# item)`` path and ``safe_register_batch`` -- which passes ``(key, item)`` --
# are exercised through a minimal registry built directly on the base mixin.
# It is stringly-typed (its own value is its key) and never enters the global
# registry tables, so it cannot collide with other suites.


class CovStrReg(MutableValidatorMixin[str, str]):
    _store: Dict[str, str] = {}

    @classmethod
    def _get_mapping(cls) -> Dict[str, str]:
        return cls._store

    @classmethod
    def _identifier_of(cls, item: str) -> str:
        return item


@pytest.fixture(autouse=True)
def _isolate_str_reg():
    CovStrReg._store.clear()
    yield
    CovStrReg._store.clear()


def test_base_register_explicit_key() -> None:
    CovStrReg.register_artifact("alpha", "value-a")
    assert CovStrReg.get_artifact("alpha") == "value-a"


def test_base_register_inferred_key() -> None:
    CovStrReg.register_artifact("beta")
    assert CovStrReg.get_artifact("beta") == "beta"


def test_base_batch_validate_success() -> None:
    results = CovStrReg.batch_validate({"k1": "k1", "k2": "k2"})
    assert results == {"k1": True, "k2": True}


def test_safe_register_batch_mixed() -> None:
    CovStrReg.register_artifact("dup")
    outcome = CovStrReg.safe_register_batch({"fresh": "fresh", "dup": "dup"})
    assert outcome["total"] == 2
    assert "fresh" in outcome["successful"]
    assert "dup" in outcome["failed"]
    assert outcome["errors"]


def test_safe_register_batch_raises_when_not_skipping() -> None:
    CovStrReg.register_artifact("dup")
    with pytest.raises(RegistryError):
        CovStrReg.safe_register_batch({"dup": "dup"}, skip_invalid=False)
