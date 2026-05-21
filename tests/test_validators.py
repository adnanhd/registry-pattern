"""Tests for the ValidatorRegistry."""

from __future__ import annotations

import pytest

from registry.validators import ValidatorRegistry


class _Sample:
    def __init__(self, name: str = "anon", count: int = 0):
        self.name = name
        self.count = count


def test_pydantic_validator_round_trips_dict() -> None:
    fn = ValidatorRegistry.get_artifact("pydantic")
    out = fn(_Sample, {"name": "x", "count": 7})
    assert out == {"name": "x", "count": 7}


def test_pydantic_validator_rejects_wrong_type() -> None:
    fn = ValidatorRegistry.get_artifact("pydantic")
    # Pydantic should reject a non-int for an int field (strict mode in our derived schema).
    import pytest
    with pytest.raises(Exception):
        fn(_Sample, {"name": "x", "count": []})


def test_pydantic_validator_uses_defaults() -> None:
    fn = ValidatorRegistry.get_artifact("pydantic")
    out = fn(_Sample, {})
    assert out == {"name": "anon", "count": 0}


def test_noop_validator_passthrough() -> None:
    fn = ValidatorRegistry.get_artifact("noop")
    out = fn(_Sample, {"anything": "goes", "count": "still_a_string"})
    assert out == {"anything": "goes", "count": "still_a_string"}


def test_unknown_validator_raises_at_lookup() -> None:
    with pytest.raises(Exception):
        ValidatorRegistry.get_artifact("does-not-exist")
