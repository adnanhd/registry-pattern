# Registry Pattern

[![Build Status](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml/badge.svg)](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/adnanhd/registry-pattern/branch/main/graph/badge.svg)](https://codecov.io/gh/adnanhd/registry-pattern)

A Python library for name-based class/function registries plus a recursive
factory that builds object graphs from JSON-friendly envelopes. Pydantic-
validated, observable, hierarchical, and stdlib-only at the core.

## What it gives you

- **Registries**: `TypeRegistry[T]` and `FunctionalRegistry` -- declare a
  registry with a dotted `repo` path; classes / functions register themselves
  with `@MyReg.register_artifact`.
- **Factory**: a single recursive `registry.build(cfg, ...)` that consumes a
  `BuildCfg`-shaped envelope (or a class / a string), validates the kwargs
  against the target's Pydantic schema, recurses on nested envelopes,
  resolves `$ref` strings against sibling and ctx scope, invokes the target,
  and runs registry hooks. Symmetric `registry.serialize(instance, ...)`
  for the outbound side.
- **Tree-shaped sub-registrars**: sub-classes inherit `post_init` / `pre_call`
  hooks via Python inheritance; `resolve(name, repo="prefix")` does prefix-
  or-exact matching across the registry tree.
- **Annotated marker contract**: the library ships duck-typed Protocols
  (`ValidateMarker`, `ComputeMarker`) in `registry.markers`. Concrete
  torch-flavoured markers (`SameDeviceAs`, `BoundTo`, `Checksum`, ...) live
  in `registry.experimental.torch_compat` together with `TorchProfilerMeter`
  and `TensorBoardReporter`. Validate markers fire pre-invocation, compute
  markers populate the envelope's meta dict.
- **Observability split**:
  - *Meters* (`LifetimeMeter`, `CPUMeter`, `MemoryMeter`, `IOMeter`,
    `NetworkMeter`, `HeapMeter`, `RecursionMeter`) write measurements into
    the envelope's meta.
  - *Reporters* (`JournalReporter`, `HTTPDashboardReporter`,
    `OpenTelemetryReporter`) ship events to external sinks.
  - Pipeline contract: meters fire BEFORE reporters at every stage.
- **stdlib logging**: `logging.basicConfig(level=INFO)` is enough to see
  every registry creation, artifact registration, and build event.

## Installation

```bash
pip install registry-pattern
```

Core depends only on `pydantic` and `typing-extensions`. Optional extras:

```bash
pip install 'registry-pattern[yaml]'   # ConfigFileEngine.yaml loader
pip install 'registry-pattern[otel]'   # OpenTelemetryReporter
pip install 'registry-pattern[torch]'  # registry.experimental.torch_compat
pip install 'registry-pattern[all]'    # everything above + docs / dev
```

## Quick start

### Register a class, build it from a dict

```python
import torch.nn as nn
from registry import TypeRegistry, build


class ModelRegistry(TypeRegistry[nn.Module]):
    pass


@ModelRegistry.register_artifact
class MLP(nn.Module):
    def __init__(self, in_features: int = 784, hidden: int = 128,
                 out_features: int = 10) -> None:
        super().__init__()
        self.in_features = in_features
        self.hidden = hidden
        self.out_features = out_features
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, out_features),
        )


# Build from an envelope
model = build({"type": "MLP", "data": {"hidden": 256}})

# Or directly from the class + a medium
model = build(MLP, {"hidden": 256}, validator="python")
model = build(MLP, "hidden: 256\n", validator="yaml")
```

### Round-trip through `serialize`

```python
from registry import serialize

data = serialize(model, serializator="python")   # -> {"in_features": 784, ...}
yaml = serialize(model, serializator="yaml")     # -> yaml string
json = serialize(model, serializator="json")     # -> json string
```

### Tree-shaped sub-registries

```python
import torchvision.models as tvm
from torch import nn
from registry import TypeRegistry, build
from registry.experimental.torch_compat import hash_state_dict


class Models(TypeRegistry[nn.Module], repo="my.models"):
    @classmethod
    def post_init(cls, instance, meta):
        meta["family"] = "models"


class CNNModels(Models, repo="my.models.cnn"):
    @classmethod
    def post_init(cls, instance, meta):
        super().post_init(instance, meta)            # parent first
        if not any(isinstance(m, nn.Conv2d) for m in instance.modules()):
            raise ValueError("not a CNN")


class Pretrained(Models, repo="my.models.pretrained"):
    @classmethod
    def post_init(cls, instance, meta):
        super().post_init(instance, meta)
        # post_init runs AFTER target.__init__, which loaded the weights
        # via torchvision's `weights=` API. The checksum reflects the
        # actually-loaded state_dict, not a freshly-initialized one.
        meta["checksum"] = hash_state_dict(instance.state_dict())


# Same class in multiple sub-registries -- different disciplines.
@CNNModels.register_artifact
@Pretrained.register_artifact
class ResNet50(nn.Module):
    """Wraps torchvision.resnet50. Weight loading happens inside __init__,
    before post_init runs -- so Pretrained.post_init can verify or record
    the loaded weights' checksum.
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = tvm.resnet50(weights=weights)

    def forward(self, x):
        return self.net(x)


build("ResNet50", {"pretrained": True}, repo="my.models.cnn")          # CNN check runs
build("ResNet50", {"pretrained": True}, repo="my.models.pretrained")   # checksum runs
build("ResNet50", {"pretrained": True}, repo="my.models")              # ambiguous: pick a sub
```

### Annotated markers: declarative cross-arg checks and meta provenance

```python
from typing import Annotated
from registry import FunctionalRegistry, build
# SameDeviceAs / BoundTo / Checksum / EffectiveLr are downstream markers --
# copy them from examples/02_factory_pipeline.py or define your own. The
# library ships only the Protocol contract (ValidateMarker / ComputeMarker)
# in registry.markers; concrete markers live in YOUR project.
from .markers import SameDeviceAs, BoundTo, Checksum, EffectiveLr


class StepRegistry(FunctionalRegistry, repo="my.steps"):
    pass


@StepRegistry.register_artifact
def train_one_step(
    model: Annotated[nn.Module, SameDeviceAs("device"), Checksum("model_checksum")],
    batch: Annotated[tuple, SameDeviceAs("device")],
    optimizer: Annotated[
        torch.optim.Optimizer, BoundTo("model"), EffectiveLr("effective_lr"),
    ],
    criterion: nn.Module,
    device: str,
) -> float:
    x, y = batch
    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()
    return loss.item()
```

Each marker is checked / computed before / after the function call; the
envelope's meta ends up with `model_checksum`, `effective_lr`, etc.

### Observability: localhost dashboard + journald

```python
import logging
from registry import attach_meter, attach_reporter, build
from registry import LifetimeMeter, CPUMeter, MemoryMeter
from registry import JournalReporter, HTTPDashboardReporter

logging.basicConfig(level=logging.INFO)             # stdlib log shows everything

attach_meter(LifetimeMeter())
attach_meter(CPUMeter())
attach_meter(MemoryMeter())

attach_reporter(JournalReporter(ident="my-trainer"))      # journalctl -t my-trainer
dash = attach_reporter(HTTPDashboardReporter(port=8765))  # curl localhost:8765

build(...)   # meters fire -> meta populated -> reporters ship the populated meta
```

For OpenTelemetry, install `registry-pattern[otel]` and
`attach_reporter(OpenTelemetryReporter())`. Each build becomes a span with the
envelope's meta as attributes; lifetime goes into a `registry.build.duration`
Histogram.

## End-user one-liners

A class with `from_X` / `to_X` methods becomes one-line wrappers around the
factory primitives:

```python
@MyReg.register_artifact
class MyConfig:
    def __init__(self, lr: float = 1e-3, epochs: int = 10): ...

    @classmethod
    def from_yaml(cls, text):  return build(cls, text, validator="yaml")
    @classmethod
    def from_args(cls, args):  return build(cls, args, validator="argparse")
    @classmethod
    def from_dict(cls, data):  return build(cls, data, validator="python")

    def to_yaml(self):         return serialize(self, serializator="yaml")
    def to_dict(self):         return serialize(self, serializator="python")
```

No bespoke `add_args` / `from_args` / `to_config` triplets per class.

## API at a glance

```python
from registry import (
    # Registries
    TypeRegistry, FunctionalRegistry, BuildCfg, Buildable,

    # Factory + serialize
    build, resolve, validate, serialize,

    # Meters (measure -> meta)
    FactoryMeter, attach_meter, detach_meter, meters,
    LifetimeMeter, CPUMeter, MemoryMeter, IOMeter, NetworkMeter,
    HeapMeter, RecursionMeter,

    # Reporters (ship event externally)
    FactoryReporter, attach_reporter, detach_reporter, reporters,
    JournalReporter, HTTPDashboardReporter, OpenTelemetryReporter,

    # Container (mostly legacy)
    ContainerMixin,

    # Exceptions
    ValidationError, RegistryError, CoercionError,
    ConformanceError, InheritanceError,
)
```

Custom markers, validator mediums, and serializer mediums register from the
relevant submodule (`registry.markers`, `registry.validators`,
`registry.factory.SerializerRegistry`).

## Examples

The `examples/` directory walks through the major patterns:

- `01_registry_basics.py`         -- bare registries (`@register_artifact` + `get_artifact`)
- `02_factory_pipeline.py`        -- the recursive `build()` pipeline end-to-end with PyTorch + MNIST
- `03_custom_reporters.py`        -- WandB / TensorBoard reporter extensions
- `04_one_liner_methods.py`       -- one-liner `from_X` / `to_X` methods via `build` / `serialize`
- `05_registry_tree_stress.py`    -- 4-level deep tree, diamond MRO, meta_schema escalation, cross-axis $ref

## CLI

```bash
python -m registry --version              # version
python -m registry info                   # full env diagnostics
python -m registry build cfg.yaml [--dry-run] [-o out.json] [-v]
python -m registry run   cfg.yaml [--entry main] [-v]
```

## Development

```bash
pip install -e '.[dev]'
pytest -vv --cov
pyright registry/
black --check .
ruff check registry/ tests/
```

## License

MIT. See `LICENSE`.
