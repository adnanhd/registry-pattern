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

```python
import torch.nn as nn
from registry import TypeRegistry, build


class ModelRegistry(TypeRegistry[nn.Module]):
    pass


@ModelRegistry.register_artifact
class MLP(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.hidden = hidden


model = build({"type": "MLP", "data": {"hidden": 256}})
assert isinstance(model, MLP) and model.hidden == 256
```

That's it for the smallest path. Everything below is opt-in.

### Other ways to build

```python
build(MLP, {"hidden": 256}, validator="python")     # class + kwargs dict
build(MLP, "hidden: 256\n", validator="yaml")       # class + raw YAML
build("MLP", {"hidden": 256})                       # string name + kwargs
```

### Round-trip through `serialize`

```python
from registry import serialize

env = serialize(model, serializator="python")
# -> {"type": "MLP", "data": {"in_features": 784, ...}, "meta": {"checksum": ...}}
yaml = serialize(model, serializator="yaml")     # YAML of the envelope
json = serialize(model, serializator="json")     # JSON of the envelope
```

### Tree-shaped sub-registries

Two separate hooks across the tree:

- **`post_init(cls, instance, meta)`** -- *validation* during build. Raises on mismatch. Runs after `target.__init__`.
- **`serialize_meta(cls, instance, meta)`** -- *emission* during `serialize()`. Writes provenance into the envelope's `meta`. Cooperative `super()` cascades up the chain.

```python
import torchvision.models as tvm
from torch import nn
from registry import TypeRegistry, build, serialize
from registry.experimental.torch_compat import hash_state_dict


class Models(TypeRegistry[nn.Module], repo="my.models"):
    """Every registered model emits its checksum on serialize."""

    @classmethod
    def serialize_meta(cls, instance, meta):
        meta["family"] = "models"
        meta["checksum"] = hash_state_dict(instance.state_dict())


class CNNModels(Models, repo="my.models.cnn"):
    @classmethod
    def post_init(cls, instance, meta):
        # build-time validation -- must be a CNN
        if not any(isinstance(m, nn.Conv2d) for m in instance.modules()):
            raise ValueError("not a CNN")

    @classmethod
    def serialize_meta(cls, instance, meta):
        super().serialize_meta(instance, meta)  # checksum + family
        meta["axis"] = "cnn"


class Pretrained(Models, repo="my.models.pretrained"):
    """Validates that loaded weights match a pinned expected hash.

    Emission is inherited from Models -- no need to redeclare checksum here.
    """

    @classmethod
    def post_init(cls, instance, meta):
        expected = getattr(instance, "_expected_hash", None)
        if expected is None:
            raise ValueError("Pretrained: instance must declare _expected_hash")
        actual = hash_state_dict(instance.state_dict())
        if actual != expected:
            raise ValueError(f"weight hash mismatch: {actual} != {expected}")


@CNNModels.register_artifact
@Pretrained.register_artifact
class ResNet50(nn.Module):
    """Weight loading happens inside __init__. Pretrained.post_init
    VALIDATES the loaded hash; Models.serialize_meta EMITS it on serialize."""

    _expected_hash = "sha256:..."

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.net = tvm.resnet50(weights=weights)

    def forward(self, x):
        return self.net(x)


build("ResNet50", {"pretrained": True}, repo="my.models.cnn")          # CNN check
build("ResNet50", {"pretrained": True}, repo="my.models.pretrained")   # hash check
serialize(model, repo="my.models.cnn")
# -> {"type": "ResNet50", "data": {"pretrained": True},
#     "meta": {"family": "models", "checksum": "sha256:...", "axis": "cnn"}}
build("ResNet50", {"pretrained": True}, repo="my.models")              # ambiguous: pick a sub
```

#### Cross-arg shape contracts

The marker mechanism extends to shape checks across multiple kwargs of a
single function. `torch_compat` ships `MatchesInputShape` /
`MatchesOutputShape` / `MatchesTargetShape`, which read each peer's
`__meta__["input_shape" | "output_shape" | "target_shape"]` (populated by a
registry's `serialize_meta` hook). A training step then validates that the
batch's shape matches the model's declared input, the criterion's input
shape matches the model's output, and the batch's target shape matches the
criterion's target_shape -- without explicit boilerplate inside the
function.

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
