"""Torch-flavoured extensions for the factory pipeline.

Concrete markers, a torch.profiler-backed meter, and a TensorBoard reporter --
all packaged here so consumers don't reinvent them. Importing this module
requires ``torch`` (and ``torch.utils.tensorboard`` for the reporter).

Install with::

    pip install 'registry-pattern[torch]'

Layout
------

Markers (subclass :class:`registry.markers.ValidateMarker` or
:class:`~registry.markers.ComputeMarker`):

- :class:`SameDeviceAs`     -- validate
- :class:`BoundTo`          -- validate
- :class:`InputShapeMatches`-- validate
- :class:`Checksum`         -- compute
- :class:`NumParams`        -- compute
- :class:`Device`           -- compute
- :class:`EffectiveLr`      -- compute

Pipeline extensions:

- :class:`TorchProfilerMeter`  -- wraps each build in ``torch.profiler.profile``
- :class:`TensorBoardReporter` -- writes numeric meta fields as TB scalars
"""

from __future__ import annotations

import hashlib
import threading
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from ..markers import ComputeMarker, ValidateMarker
from ..meters import FactoryMeter
from ..reporters import FactoryReporter

__all__ = [
    # helpers
    "device_of",
    "hash_state_dict",
    # validate markers (cross-arg / runtime invariants -- raise on mismatch)
    "SameDeviceAs",
    "BoundTo",
    "InputShapeMatches",
    "VerifyChecksum",
    "MatchesInputShape",
    "MatchesOutputShape",
    "MatchesTargetShape",
    # compute markers (emit provenance / serialize into meta)
    "Checksum",
    "NumParams",
    "Device",
    "EffectiveLr",
    "InputShape",
    "OutputShape",
    # meters
    "TorchProfilerMeter",
    # reporters
    "TensorBoardReporter",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def device_of(value: Any) -> Any:
    """Best-effort device extraction for tensors / modules / optimizers / sequences."""
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, nn.Module):
        params = list(value.parameters())
        return params[0].device if params else torch.device("cpu")
    if isinstance(value, torch.optim.Optimizer):
        for group in value.param_groups:
            if group["params"]:
                return group["params"][0].device
        return torch.device("cpu")
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0].device
    return value


def hash_state_dict(sd: dict[str, torch.Tensor]) -> str:
    """Stable sha256 prefix of a state_dict for provenance / cache keys."""
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
# Validate markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SameDeviceAs(ValidateMarker):
    """Require ``value`` to live on the device referenced by ``kwargs[ref]`` (or ctx)."""

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        target = kwargs.get(self.ref, ctx.get(self.ref))
        if target is None:
            raise ValueError(f"SameDeviceAs: {self.ref!r} not in kwargs or ctx")
        actual = str(device_of(value))
        expected = str(target if isinstance(target, str) else device_of(target))
        if actual != expected:
            raise ValueError(f"device mismatch: got {actual!r}, expected {expected!r}")


@dataclass(frozen=True)
class BoundTo(ValidateMarker):
    """Optimizer's params must come from ``kwargs[ref]``'s parameters."""

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        ref_obj = kwargs.get(self.ref, ctx.get(self.ref))
        if not isinstance(ref_obj, nn.Module):
            raise ValueError(f"BoundTo: {self.ref!r} is not an nn.Module")
        opt_ids = {id(p) for g in value.param_groups for p in g["params"]}
        ref_ids = {id(p) for p in ref_obj.parameters()}
        if not (opt_ids & ref_ids):
            raise ValueError(f"optimizer not bound to {self.ref!r}")


@dataclass(frozen=True)
class VerifyChecksum(ValidateMarker):
    """Validate that ``value``'s state_dict hash equals an expected sha256 prefix.

    Conceptually paired with -- but distinct from -- the :class:`Checksum`
    compute marker:

    - ``VerifyChecksum("sha256:abc...")`` raises on mismatch (gatekeeper).
    - ``Checksum("model_hash")`` writes the hash into meta (provenance emitter).

    Use this when you have a pinned-weights contract (e.g. an IMAGENET1K_V2
    artifact ships with a known sha256) and you want a hard guarantee that
    the loaded weights match.
    """

    expected: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        actual = hash_state_dict(value.state_dict())
        if actual != self.expected:
            raise ValueError(f"checksum mismatch: got {actual} != expected {self.expected}")


def _shape_or_none(value: Any) -> tuple[int, ...] | None:
    """``value.shape[1:]`` for tensors / (tensor, ...) tuples; else None."""
    x = value[0] if isinstance(value, (tuple, list)) and value else value
    if isinstance(x, torch.Tensor):
        return tuple(x.shape[1:])
    return None


def _meta_shape(obj: Any, key: str) -> tuple[int, ...] | None:
    """Look ``key`` up in ``obj.__meta__`` (if any). Returns a tuple or None."""
    meta = getattr(obj, "__meta__", None) or {}
    value = meta.get(key)
    return tuple(value) if value is not None else None


@dataclass(frozen=True)
class MatchesInputShape(ValidateMarker):
    """Batch shape (sans batch dim) must equal ``kwargs[ref].__meta__['input_shape']``.

    Convention: shape-aware models declare ``input_shape`` in their meta (via
    a registry's ``serialize_meta`` or ``post_init`` hook). This marker
    enforces that whatever is annotated as a batch matches that contract.
    """

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        ref_obj = kwargs.get(self.ref, ctx.get(self.ref))
        if ref_obj is None:
            return
        expected = _meta_shape(ref_obj, "input_shape")
        actual = _shape_or_none(value)
        if expected is None or actual is None:
            return
        if expected != actual:
            raise ValueError(
                f"input shape mismatch: {actual} != {self.ref}.input_shape {expected}"
            )


@dataclass(frozen=True)
class MatchesOutputShape(ValidateMarker):
    """``value``'s declared input_shape meta must equal ``kwargs[ref].__meta__['output_shape']``.

    Useful for criteria / heads / decoders that consume a model's output.
    """

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        ref_obj = kwargs.get(self.ref, ctx.get(self.ref))
        if ref_obj is None:
            return
        ref_output = _meta_shape(ref_obj, "output_shape")
        self_input = _meta_shape(value, "input_shape")
        if ref_output is None or self_input is None:
            return
        if ref_output != self_input:
            raise ValueError(
                f"output/input shape mismatch: {self.ref}.output_shape {ref_output} != "
                f"this.input_shape {self_input}"
            )


@dataclass(frozen=True)
class MatchesTargetShape(ValidateMarker):
    """The target tensor in a (input, target) batch matches ``kwargs[ref].__meta__['target_shape']``.

    The marker reads ``value[1]`` (the target) and compares to the criterion's
    declared target_shape, typically populated by serialize_meta.
    """

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        ref_obj = kwargs.get(self.ref, ctx.get(self.ref))
        if ref_obj is None:
            return
        expected = _meta_shape(ref_obj, "target_shape")
        if expected is None:
            return
        target = value[1] if isinstance(value, (tuple, list)) and len(value) > 1 else None
        if not isinstance(target, torch.Tensor):
            return
        actual = tuple(target.shape[1:])
        if expected != actual:
            raise ValueError(
                f"target shape mismatch: {actual} != {self.ref}.target_shape {expected}"
            )


@dataclass(frozen=True)
class InputShape(ComputeMarker):
    """Write the value's input shape (skipping batch dim) to meta[name]."""

    name: str

    def compute(self, value: Any) -> tuple[int, ...] | None:
        return _shape_or_none(value)


@dataclass(frozen=True)
class OutputShape(ComputeMarker):
    """Write the value's output shape (skipping batch dim) to meta[name].

    Only meaningful for callable modules whose forward returns a Tensor of
    a deterministic shape; otherwise the caller writes ``output_shape`` via
    a custom ``serialize_meta`` / ``post_init`` hook.
    """

    name: str

    def compute(self, value: Any) -> tuple[int, ...] | None:
        return _shape_or_none(value)


@dataclass(frozen=True)
class InputShapeMatches(ValidateMarker):
    """Batch feature dim must match ``kwargs[ref]``'s first Linear.in_features."""

    ref: str

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        x = value[0] if isinstance(value, (tuple, list)) else value
        if not isinstance(x, torch.Tensor):
            return
        model = kwargs.get(self.ref, ctx.get(self.ref))
        if model is None:
            raise ValueError(f"InputShapeMatches: {self.ref!r} not found")
        first_linear = next((m for m in model.modules() if isinstance(m, nn.Linear)), None)
        if first_linear is None:
            return
        actual = x.flatten(1).shape[1]
        if actual != first_linear.in_features:
            raise ValueError(
                f"batch feature dim {actual} != {self.ref}.in_features "
                f"{first_linear.in_features}"
            )


# ---------------------------------------------------------------------------
# Compute markers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Checksum(ComputeMarker):
    """Write a sha256 prefix of the module's state_dict to meta[name]."""

    name: str

    def compute(self, value: nn.Module) -> str:
        return hash_state_dict(value.state_dict())


@dataclass(frozen=True)
class NumParams(ComputeMarker):
    """Write ``sum(p.numel() for p in module.parameters())`` to meta[name]."""

    name: str

    def compute(self, value: nn.Module) -> int:
        return sum(p.numel() for p in value.parameters())


@dataclass(frozen=True)
class Device(ComputeMarker):
    """Write the device of the value (as a string) to meta[name]."""

    name: str

    def compute(self, value: Any) -> str:
        return str(device_of(value))


@dataclass(frozen=True)
class EffectiveLr(ComputeMarker):
    """Write the optimizer's first-group learning rate to meta[name]."""

    name: str

    def compute(self, value: torch.optim.Optimizer) -> float:
        return float(value.param_groups[0]["lr"])


# ---------------------------------------------------------------------------
# Meters
# ---------------------------------------------------------------------------


class TorchProfilerMeter(FactoryMeter):
    """Wraps each build() call in ``torch.profiler.profile`` and writes a
    summary (self-CPU / self-CUDA totals + top ops) to the envelope meta.

    Nested builds get per-level profiles via a private stack.
    """

    name: str = "torch_profiler"

    def __init__(self, top_n: int = 5, record_shapes: bool = False) -> None:
        self._top_n: int = top_n
        self._record_shapes: bool = record_shapes
        self._stack: list[Any] = []

    def on_build_start(self, *, cfg: Any, ctx: dict[str, Any], meta: dict[str, Any]) -> None:
        activities = [ProfilerActivity.CPU]
        if torch.cuda.is_available():
            activities.append(ProfilerActivity.CUDA)
        prof = profile(activities=activities, record_shapes=self._record_shapes)
        prof.__enter__()
        self._stack.append(prof)

    def on_built(self, *, target: Any, result: Any, meta: dict[str, Any], ctx: dict[str, Any]) -> None:
        if not self._stack:
            return
        prof = self._stack.pop()
        prof.__exit__(None, None, None)

        events = prof.key_averages()
        meta["torch_profile_self_cpu_time_total_us"] = sum(e.self_cpu_time_total for e in events)
        meta["torch_profile_self_cuda_time_total_us"] = sum(
            getattr(e, "self_cuda_time_total", 0) for e in events
        )
        top = sorted(events, key=lambda e: e.self_cpu_time_total, reverse=True)[: self._top_n]
        meta["torch_profile_top_ops"] = [
            {"op": e.key, "self_cpu_us": e.self_cpu_time_total} for e in top
        ]

    def on_error(self, *, cfg: Any, exc: BaseException, ctx: dict[str, Any], meta: dict[str, Any]) -> None:
        if not self._stack:
            return
        prof = self._stack.pop()
        try:
            prof.__exit__(type(exc), exc, exc.__traceback__)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Reporters
# ---------------------------------------------------------------------------


class TensorBoardReporter(FactoryReporter):
    """Writes every numeric meta field as a TensorBoard scalar.

    Step number auto-increments per build. A single ``SummaryWriter`` is
    shared across the process; pass ``log_dir`` to control where events go.
    """

    name: str = "tensorboard"

    def __init__(self, log_dir: str = "runs") -> None:
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=log_dir)
        self._step: int = 0
        self._lock = threading.Lock()

    def on_built(self, *, target: Any, result: Any, meta: dict[str, Any], ctx: dict[str, Any]) -> None:
        target_name = getattr(target, "__name__", str(target))
        with self._lock:
            for k, v in meta.items():
                if isinstance(v, (int, float, bool)):
                    self._writer.add_scalar(f"{target_name}/{k}", float(v), self._step)
            self._step += 1
