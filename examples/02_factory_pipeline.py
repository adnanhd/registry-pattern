#!/usr/bin/env python
"""End-to-end training using the new recursive factory pipeline.

What this example demonstrates:

- 4 registries (model / optimizer / loader / criterion) + 1 functional registry (step).
- `build(cfg, ctx=...)` recursively constructs the whole object graph from a
  YAML-shaped dict, with sibling cross-references via ``$``-prefixed strings.
- Per-registry ``post_init`` hook (NaN sanity check on the model).
- ``Annotated[T, ...]`` markers on the training function:
  - ``SameDeviceAs``: cross-arg device check
  - ``BoundTo``: optimizer must hold the model's parameters
  - ``Checksum`` / ``NumParams`` / ``EffectiveLr``: write provenance into meta
- Implicit ``meta_schema`` derived from the compute markers' return annotations.

Run::

    PYTHONPATH=. python examples/06_factory_pipeline.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Annotated, Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from registry import FunctionalRegistry, TypeRegistry, build
from registry.markers import ComputeMarker, ValidateMarker   # protocol contracts


# =============================================================================
# Torch-specific Annotated markers (downstream extension; not shipped by the lib)
# =============================================================================
# `registry.schema.process_validate` / `process_compute` duck-type on `.validate`
# and `.compute`, so any object with these methods works. Subclassing the
# Protocol (`ValidateMarker` / `ComputeMarker`) is OPTIONAL -- it just enables
# static-type-checker support. Both forms are equivalent at runtime.


def _device_of(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, nn.Module):
        params = list(value.parameters())
        return params[0].device if params else torch.device("cpu")
    if isinstance(value, torch.optim.Optimizer):
        for g in value.param_groups:
            if g["params"]:
                return g["params"][0].device
        return torch.device("cpu")
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0].device
    return value


def _hash_state_dict(sd: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for k in sorted(sd):
        h.update(k.encode())
        t = sd[k]
        h.update(t.detach().cpu().numpy().tobytes() if isinstance(t, torch.Tensor) else repr(t).encode())
    return f"sha256:{h.hexdigest()[:16]}"


@dataclass(frozen=True)
class SameDeviceAs(ValidateMarker):
    """Validate that `value` lives on the same device as `kwargs[ref]`."""
    ref: str
    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        target = kwargs.get(self.ref, ctx.get(self.ref))
        actual = str(_device_of(value))
        expected = str(target if isinstance(target, str) else _device_of(target))
        if actual != expected:
            raise ValueError(f"device mismatch: got {actual!r}, expected {expected!r}")


@dataclass(frozen=True)
class BoundTo(ValidateMarker):
    """Validate that an optimizer holds the parameters of `kwargs[ref]`."""
    ref: str
    def validate(self, value: torch.optim.Optimizer, kwargs: dict, ctx: dict) -> None:
        model: nn.Module = kwargs[self.ref]
        opt_ids = {id(p) for g in value.param_groups for p in g["params"]}
        model_ids = {id(p) for p in model.parameters()}
        if not (opt_ids & model_ids):
            raise ValueError(f"optimizer not bound to '{self.ref}'")


@dataclass(frozen=True)
class Checksum(ComputeMarker):
    """Compute marker: writes a sha256 prefix of the state_dict to meta[name]."""
    name: str
    def compute(self, value: nn.Module) -> str:
        return _hash_state_dict(value.state_dict())


@dataclass(frozen=True)
class NumParams(ComputeMarker):
    """Compute marker: writes parameter count to meta[name]."""
    name: str
    def compute(self, value: nn.Module) -> int:
        return sum(p.numel() for p in value.parameters())


@dataclass(frozen=True)
class EffectiveLr(ComputeMarker):
    """Compute marker: writes the first param-group's lr to meta[name]."""
    name: str
    def compute(self, value: torch.optim.Optimizer) -> float:
        return float(value.param_groups[0]["lr"])


# =============================================================================
# Registries
# =============================================================================


class ModelRegistry(TypeRegistry[nn.Module]):
    @classmethod
    def post_init(cls, instance: nn.Module, meta: dict[str, Any]) -> None:
        for name, p in instance.named_parameters():
            if torch.isnan(p).any():
                raise ValueError(f"NaN in {name}")


class OptimizerRegistry(TypeRegistry[torch.optim.Optimizer]): ...


class LoaderRegistry(TypeRegistry[DataLoader]): ...


class CriterionRegistry(TypeRegistry[nn.Module]): ...


class StepRegistry(FunctionalRegistry): ...


# stdlib + torch classes -- register by reference
OptimizerRegistry.register_artifact(torch.optim.Adam)
LoaderRegistry.register_artifact(DataLoader)
CriterionRegistry.register_artifact(nn.CrossEntropyLoss)


# =============================================================================
# Model
# =============================================================================


@ModelRegistry.register_artifact
class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 784,
        hidden: int = 128,
        out_features: int = 10,
    ) -> None:
        super().__init__()
        # Mirror constructor args as attributes so meta encoders can recover them.
        self.in_features: int = in_features
        self.hidden: int = hidden
        self.out_features: int = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.flatten(1))


# =============================================================================
# Training function -- Annotated drives both validation and meta provenance
# =============================================================================


@StepRegistry.register_artifact
def train_one_epoch(
    model: Annotated[
        nn.Module,
        SameDeviceAs("device"),
        Checksum("model_checksum_before"),
        NumParams("model_params"),
    ],
    loader: DataLoader,
    optimizer: Annotated[
        torch.optim.Optimizer,
        BoundTo("model"),
        EffectiveLr("effective_lr"),
    ],
    criterion: nn.Module,
    device: str,
) -> dict[str, float]:
    """One epoch of training. JSON-native return for easy wire-shipping."""
    model.train()
    total: float = 0.0
    n: int = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss: torch.Tensor = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item()
        n += 1
    return {"avg_loss": total / max(n, 1), "num_batches": float(n)}


# =============================================================================
# Driver
# =============================================================================


def main() -> dict[str, float]:
    # Runtime-only deps that don't travel through JSON go into ctx:
    dataset = MNIST(".", train=True, download=True, transform=ToTensor())

    cfg: dict[str, Any] = {
        "type": "train_one_epoch",
        "data": {
            # leaf envelopes -- built recursively. Each can use $-refs to siblings.
            "model": {"type": "MLP", "data": {"hidden": 256}},
            "loader": {
                "type": "DataLoader",
                "data": {
                    "dataset": "$dataset",     # <- injected from ctx
                    "batch_size": 64,
                    "shuffle": True,
                    "num_workers": 2,
                },
            },
            "optimizer": {
                "type": "Adam",
                "data": {
                    "params": "$model.parameters()",   # <- sibling-ref: model is already built
                    "lr": 1e-3,
                },
            },
            "criterion": {"type": "CrossEntropyLoss", "data": {}},
            # cuda.is_available() is unreliable on +cpu wheels; default to cpu here.
            "device": "cpu",
        },
        "meta": {},
    }

    # Recursive build -> calls train_one_epoch at the root and returns its dict.
    result: dict[str, float] = build(cfg, ctx={"dataset": dataset})

    # Provenance the markers + post_init left on the envelope's meta:
    print("result:", result)
    print("meta:  ", cfg["meta"])
    return result


if __name__ == "__main__":
    main()
