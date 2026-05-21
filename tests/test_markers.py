"""Tests for the Annotated[T, ...] marker library.

These tests require torch (the markers themselves work on torch tensors and
modules). If torch isn't installed, the entire module is skipped.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
nn = pytest.importorskip("torch.nn")

from registry.markers import (
    BoundTo,
    Checksum,
    Device,
    EffectiveLr,
    InputShapeMatches,
    NumParams,
    SameDeviceAs,
)


class _Tiny(nn.Module):
    def __init__(self, in_features: int = 4, hidden: int = 3):
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.net = nn.Linear(in_features, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Validation markers
# ---------------------------------------------------------------------------


def test_same_device_as_passes_when_devices_match() -> None:
    model = _Tiny()  # cpu
    kwargs = {"model": model, "device": "cpu"}
    SameDeviceAs("device").validate(model, kwargs, {})


def test_same_device_as_raises_on_mismatch() -> None:
    model = _Tiny()  # cpu
    kwargs = {"model": model, "device": "cuda"}
    with pytest.raises(ValueError, match="device mismatch"):
        SameDeviceAs("device").validate(model, kwargs, {})


def test_bound_to_passes_when_optimizer_holds_model_params() -> None:
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    BoundTo("model").validate(opt, {"model": model, "optimizer": opt}, {})


def test_bound_to_raises_when_optimizer_unrelated_to_model() -> None:
    model = _Tiny()
    other = _Tiny()
    opt = torch.optim.SGD(other.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="optimizer not bound"):
        BoundTo("model").validate(opt, {"model": model, "optimizer": opt}, {})


def test_input_shape_matches_passes() -> None:
    model = _Tiny(in_features=4)
    batch = (torch.zeros(2, 4), torch.zeros(2, dtype=torch.long))
    InputShapeMatches("model").validate(batch, {"model": model, "batch": batch}, {})


def test_input_shape_matches_raises_on_wrong_dim() -> None:
    model = _Tiny(in_features=4)
    batch = (torch.zeros(2, 8), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError, match="feature dim"):
        InputShapeMatches("model").validate(batch, {"model": model, "batch": batch}, {})


# ---------------------------------------------------------------------------
# Compute markers
# ---------------------------------------------------------------------------


def test_checksum_returns_sha256_prefix() -> None:
    model = _Tiny()
    out = Checksum("h").compute(model)
    assert out.startswith("sha256:")
    assert Checksum("h").compute(model) == out  # determinism


def test_num_params_counts_parameters() -> None:
    model = _Tiny(in_features=4, hidden=3)
    # weight: 4*3 + bias: 3 = 15
    assert NumParams("n").compute(model) == 15


def test_device_returns_device_string() -> None:
    model = _Tiny()
    out = Device("d").compute(model)
    assert out in ("cpu", "cuda:0") or out.startswith("cuda")


def test_effective_lr_returns_first_group_lr() -> None:
    model = _Tiny()
    opt = torch.optim.SGD(model.parameters(), lr=0.123)
    assert EffectiveLr("lr").compute(opt) == pytest.approx(0.123)
