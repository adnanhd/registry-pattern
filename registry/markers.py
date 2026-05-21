"""Declarative ``Annotated[T, ...]`` markers for the recursive factory.

Two protocols:

- ``validate(value, kwargs, ctx) -> None`` — runtime cross-arg validation.
  Fires after kwargs are assembled, before the target is called/instantiated.
- ``compute(value) -> Any`` — populates the ``meta`` dict with a value derived
  from the built instance. Fires after the target is called.

Both protocols are duck-typed: a marker can implement either or both.

Example::

    def train_one_step(
        model: Annotated[nn.Module, SameDeviceAs("device"), Checksum("model_hash")],
        optimizer: Annotated[Optimizer, BoundTo("model")],
        device: str,
    ) -> dict[str, float]: ...

torch is imported lazily — markers stay importable without it; only the
torch-specific helpers raise when used without torch installed.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

__all__ = [
    "SameDeviceAs",
    "BoundTo",
    "InputShapeMatches",
    "Checksum",
    "NumParams",
    "Device",
    "EffectiveLr",
]


def _device_of(value: Any) -> Any:
    """Best-effort device extraction for tensors / modules / optimizers / sequences."""
    import torch
    from torch import nn

    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, nn.Module):
        params = list(value.parameters())
        if params:
            return params[0].device
        return torch.device("cpu")
    if isinstance(value, torch.optim.Optimizer):
        for group in value.param_groups:
            if group["params"]:
                return group["params"][0].device
        return torch.device("cpu")
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0].device
    return value


def _hash_state_dict(sd: dict[str, Any]) -> str:
    import torch

    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        t = sd[k]
        if isinstance(t, torch.Tensor):
            h.update(t.detach().cpu().numpy().tobytes())
        else:
            h.update(repr(t).encode())
    return f"sha256:{h.hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# Validation markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SameDeviceAs:
    """Require ``value`` to live on the device referenced by ``kwargs[ref]`` (or ctx)."""

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        target = kwargs.get(self.ref, ctx.get(self.ref))
        if target is None:
            raise ValueError(f"SameDeviceAs: '{self.ref}' not found in kwargs or ctx")
        actual = str(_device_of(value))
        expected = str(target if isinstance(target, str) else _device_of(target))
        if actual != expected:
            raise ValueError(f"device mismatch: got {actual!r}, expected {expected!r}")


@dataclass(frozen=True)
class BoundTo:
    """Optimizer's params must come from ``kwargs[ref]``'s parameters."""

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        from torch import nn

        ref_obj = kwargs.get(self.ref, ctx.get(self.ref))
        if not isinstance(ref_obj, nn.Module):
            raise ValueError(f"BoundTo: '{self.ref}' is not a Module")
        opt_ids = {id(p) for g in value.param_groups for p in g["params"]}
        ref_ids = {id(p) for p in ref_obj.parameters()}
        if not (opt_ids & ref_ids):
            raise ValueError(f"optimizer not bound to '{self.ref}'")


@dataclass(frozen=True)
class InputShapeMatches:
    """Batch's feature dim must match ``kwargs[ref]``'s first Linear in_features."""

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        import torch
        from torch import nn

        x = value[0] if isinstance(value, (tuple, list)) else value
        if not isinstance(x, torch.Tensor):
            return
        model = kwargs.get(self.ref, ctx.get(self.ref))
        if model is None:
            raise ValueError(f"InputShapeMatches: '{self.ref}' not found")
        first_linear = next((m for m in model.modules() if isinstance(m, nn.Linear)), None)
        if first_linear is None:
            return
        actual = x.flatten(1).shape[1]
        if actual != first_linear.in_features:
            raise ValueError(
                f"batch feature dim {actual} != {self.ref}.in_features {first_linear.in_features}"
            )


# ---------------------------------------------------------------------------
# Compute markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checksum:
    """Write a sha256-prefix of the module's state_dict to ``meta[name]``."""

    name: str

    def compute(self, value: Any) -> str:
        return _hash_state_dict(value.state_dict())


@dataclass(frozen=True)
class NumParams:
    """Write ``sum(p.numel() for p in module.parameters())`` to ``meta[name]``."""

    name: str

    def compute(self, value: Any) -> int:
        return sum(p.numel() for p in value.parameters())


@dataclass(frozen=True)
class Device:
    """Write the device of the value to ``meta[name]``."""

    name: str

    def compute(self, value: Any) -> str:
        return str(_device_of(value))


@dataclass(frozen=True)
class EffectiveLr:
    """Write the optimizer's first-group learning rate to ``meta[name]``."""

    name: str

    def compute(self, value: Any) -> float:
        return float(value.param_groups[0]["lr"])
