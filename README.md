# Registry Pattern

[![Build Status](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml/badge.svg)](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml)
[![codecov](https://codecov.io/gh/adnanhd/registry-pattern/branch/main/graph/badge.svg)](https://codecov.io/gh/adnanhd/registry-pattern)

A Python library for name-based class/function registries plus a recursive
factory that builds object graphs from JSON-friendly envelopes. Pydantic-
validated, observable, and hierarchical -- and domain-agnostic: the core knows
nothing about any particular framework.

## What it gives you

- **Registries**: `TypeRegistry[T]` and `FunctionalRegistry` -- declare one with
  a dotted `repo` path; classes/functions register with `@MyReg.register_artifact`.
- **Factory**: a single recursive `build(cfg, ...)` that consumes a `BuildCfg`
  envelope (`{type, data, meta}`) -- or a class, or a string name -- validates
  the kwargs against the target's Pydantic schema, recurses on nested envelopes,
  resolves `$ref` strings against sibling/ctx scope, invokes the target, and runs
  registry hooks. Symmetric `serialize(instance, ...)` for the outbound side.
- **Tree-shaped registries**: sub-registries inherit `post_init` / `serialize_meta`
  hooks through Python inheritance; `resolve(name, repo="prefix")` does prefix-or-
  exact matching across the tree.
- **Annotated markers**: the library ships only the Protocol *contract*
  (`ValidateMarker`, `ComputeMarker`) in `registry.markers`. Validate markers fire
  before the target runs; compute markers write into the envelope's `meta`.
  Concrete markers are your code.
- **Observability**: *meters* measure pipeline stages into `meta`; *reporters*
  ship events to external sinks (journald, an HTTP dashboard, OpenTelemetry).
  Meters fire before reporters at every stage.
- **Pydantic integration**: `ArtifactOf[Registry]` -- an argument/field that
  accepts a live artifact *or* a config it builds.
- **stdlib logging**: `logging.basicConfig(level=INFO)` shows every registry
  creation, registration, and build event.

The core ships only the observability *buses* (base class + attach / detach /
emit). Concrete meters and reporters are opt-in batteries in `registry.extra`.
Framework-flavoured extras (torch state-dict hashing, tensor-shape markers, a
torch profiler meter) live in `examples/torch_compat.py`
(`pip install 'registry-pattern[torch]'`) -- never in the core.

## Installation

```bash
pip install registry-pattern
```

Core depends on `pydantic` and `typing-extensions`. Optional extras:

```bash
pip install 'registry-pattern[yaml]'   # ConfigFileEngine YAML loader
pip install 'registry-pattern[otel]'   # OpenTelemetryReporter
pip install 'registry-pattern[torch]'  # deps for examples/torch_compat.py
pip install 'registry-pattern[all]'    # everything + docs / dev
```

## Quick start

```python
from registry import TypeRegistry, build


class Shape:
    def area(self) -> float: ...


class ShapeRegistry(TypeRegistry[Shape]):
    pass


@ShapeRegistry.register_artifact
class Circle(Shape):
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def area(self) -> float:
        return 3.14159 * self.radius ** 2


circle = build({"type": "Circle", "data": {"radius": 2.0}})
assert isinstance(circle, Circle) and round(circle.area(), 2) == 12.57
```

That is the smallest path. Everything below is opt-in.

### Other ways to build

```python
build(Circle, {"radius": 2.0}, validator="python")   # class + kwargs dict
build(Circle, "radius: 2.0\n", validator="yaml")      # class + raw YAML
build("Circle", {"radius": 2.0})                      # string name + kwargs
```

### Round-trip through `serialize`

```python
from registry import serialize

env = serialize(circle, serializator="python")
# -> {"type": "Circle", "data": {"radius": 2.0}, "meta": {...}}
yaml = serialize(circle, serializator="yaml")   # YAML of the envelope
json = serialize(circle, serializator="json")   # JSON of the envelope
```

## Tree-shaped sub-registries

Two hooks cascade through the registry tree:

- **`post_init(cls, instance, meta)`** -- *validation* after `__init__`, raises on mismatch.
- **`serialize_meta(cls, instance, meta)`** -- *emission* during `serialize()`, writes
  provenance into `meta`. Cooperative `super()` cascades up the chain.

```python
from registry import TypeRegistry, build, serialize


class Shapes(TypeRegistry[Shape], repo="shapes"):
    """Every registered shape emits its area on serialize."""

    @classmethod
    def serialize_meta(cls, instance, meta):
        meta["family"] = "shapes"
        meta["area"] = instance.area()


class Rounded(Shapes, repo="shapes.rounded"):
    @classmethod
    def post_init(cls, instance, meta):            # build-time validation
        if instance.area() <= 0:
            raise ValueError("degenerate shape")

    @classmethod
    def serialize_meta(cls, instance, meta):
        super().serialize_meta(instance, meta)     # area + family (inherited)
        meta["axis"] = "rounded"


@Rounded.register_artifact
class Ring(Shape):
    def __init__(self, outer: float, inner: float) -> None:
        self.outer, self.inner = outer, inner

    def area(self) -> float:
        return 3.14159 * (self.outer ** 2 - self.inner ** 2)


build("Ring", {"outer": 3.0, "inner": 1.0}, repo="shapes.rounded")
serialize(ring, repo="shapes.rounded")
# -> {"type": "Ring", "data": {"outer": 3.0, "inner": 1.0},
#     "meta": {"family": "shapes", "area": 25.13, "axis": "rounded"}}
build("Ring", {...}, repo="shapes")   # ambiguous prefix -> resolve picks a sub
```

## Annotated markers: cross-arg checks and meta provenance

The library ships only the `ValidateMarker` / `ComputeMarker` Protocol contract
in `registry.markers` (the factory dispatches on method presence, so a marker may
implement either). Concrete markers are your code -- here, two tiny ones:

```python
from typing import Annotated, Any, Dict
from registry import FunctionalRegistry, build


class FitsInside:                     # ValidateMarker: reads the sibling at `ref`
    def __init__(self, ref: str) -> None:
        self.ref = ref

    def validate(self, value: Any, kwargs: Dict[str, Any], ctx: Dict[str, Any]) -> None:
        peer = kwargs.get(self.ref, ctx.get(self.ref))
        if peer is not None and value.area() > peer.area():
            raise ValueError(f"{value!r} does not fit inside {self.ref!r}")


class AreaOf:                          # ComputeMarker: writes into the envelope meta
    def __init__(self, name: str) -> None:
        self.name = name

    def compute(self, value: Any) -> Any:
        return value.area()


class LayoutRegistry(FunctionalRegistry, repo="layouts"):
    pass


@LayoutRegistry.register_artifact
def place(
    outer: Annotated[Shape, AreaOf("outer_area")],
    inner: Annotated[Shape, FitsInside("outer"), AreaOf("inner_area")],
) -> tuple:
    return outer, inner
```

`FitsInside` fires before `place` runs (raising aborts the build); `AreaOf`
fires after, leaving `outer_area` / `inner_area` in the envelope's `meta`.

### `ArtifactOf[Registry]`: accept an instance or a config, serialize back

`ArtifactOf[SomeRegistry]` (in `registry.integrations.pydantic`) is the
pydantic-native form of `build()`: a field or `@validate_call` argument typed
`ArtifactOf[SomeRegistry]` accepts **either** a live artifact of that registry's
element type **or** a config (`{type, data, ...}`) it builds -- and serializes
mode-aware (`model_dump()` -> the artifact, `model_dump(mode="json")` -> its config).

```python
from typing import Annotated
from pydantic import validate_call
from registry.integrations.pydantic import ArtifactOf

@validate_call
def frame(
    outer: ArtifactOf[ShapeRegistry],
    inner: Annotated[ArtifactOf[ShapeRegistry], FitsInside("outer")],
) -> tuple: ...

frame(a_circle, a_smaller_circle)                     # live artifacts
frame({"type": "Circle", "data": {"radius": 3.0}}, ...)  # a config -> built
```

Several registries are accepted as a union, either spelling:
`ArtifactOf[CircleRegistry, SquareRegistry]` or
`Union[ArtifactOf[CircleRegistry], ArtifactOf[SquareRegistry]]`. The base type
builds; downstream `Annotated` markers stay plain pydantic after-validators, and
cross-argument ones read the already-validated sibling from `info.data`.

## Observability: localhost dashboard + journald

```python
import logging
from registry import attach_meter, attach_reporter, build
from registry.extra.meters import LifetimeMeter, CPUMeter, MemoryMeter
from registry.extra.reporters import JournalReporter, HTTPDashboardReporter

logging.basicConfig(level=logging.INFO)              # stdlib log shows everything

attach_meter(LifetimeMeter())
attach_meter(CPUMeter())
attach_meter(MemoryMeter())

attach_reporter(JournalReporter(ident="my-app"))         # journalctl -t my-app
dash = attach_reporter(HTTPDashboardReporter(port=8765)) # curl localhost:8765

build(...)   # meters fire -> meta populated -> reporters ship the populated meta
```

For OpenTelemetry, install `registry-pattern[otel]` and
`attach_reporter(OpenTelemetryReporter())` (also in `registry.extra.reporters`).
Each build becomes a span with the envelope's meta as attributes; lifetime goes
into a `registry.build.duration` histogram.

## End-user one-liners

A class with `from_X` / `to_X` methods becomes one-line wrappers around the
factory primitives -- no bespoke `add_args` / `from_args` / `to_config` triplets:

```python
@MyReg.register_artifact
class Settings:
    def __init__(self, width: int = 640, height: int = 480): ...

    @classmethod
    def from_yaml(cls, text):  return build(cls, text, validator="yaml")
    @classmethod
    def from_args(cls, args):  return build(cls, args, validator="argparse")
    @classmethod
    def from_dict(cls, data):  return build(cls, data, validator="python")

    def to_yaml(self):         return serialize(self, serializator="yaml")
    def to_dict(self):         return serialize(self, serializator="python")
```

## API at a glance

```python
from registry import (
    # Registries + factory
    TypeRegistry, FunctionalRegistry, BuildCfg, Buildable,
    build, resolve, validate, serialize,

    # Meter bus (measure -> meta)
    FactoryMeter, attach_meter, detach_meter, meters,

    # Reporter bus (ship events externally)
    FactoryReporter, attach_reporter, detach_reporter, reporters,

    # Exceptions
    ValidationError, RegistryError, CoercionError,
    ConformanceError, InheritanceError,
)

# Opt-in batteries (concrete meters / reporters)
from registry.extra.meters import (
    LifetimeMeter, CPUMeter, MemoryMeter, IOMeter, NetworkMeter,
    HeapMeter, RecursionMeter,
)
from registry.extra.reporters import (
    JournalReporter, HTTPDashboardReporter, OpenTelemetryReporter,
)
from registry.integrations.pydantic import ArtifactOf
```

Custom markers, validator mediums, and serializer mediums register from the
relevant submodule (`registry.markers`, `registry.validators`,
`registry.factory.SerializerRegistry`).

## Examples

The `examples/` directory walks through the major patterns:

- `01_registry_basics.py`      -- bare registries (`@register_artifact` + `get_artifact`)
- `02_factory_pipeline.py`     -- the recursive `build()` pipeline end-to-end
- `03_custom_reporters.py`     -- custom reporter extensions
- `04_one_liner_methods.py`    -- `from_X` / `to_X` methods via `build` / `serialize`
- `05_registry_tree_stress.py` -- deep tree, diamond MRO, meta-schema escalation, cross-axis `$ref`

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
