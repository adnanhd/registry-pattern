# Migration sketch: cofinn (and similar consumers)

This is a concrete walkthrough for moving an ML codebase like `cofinn` off the
`add_args` / `from_args` / `to_config` triplet onto `registry-pattern`'s
recursive factory. Same pattern applies to `lift`, `magfree`, `torchestrator`
consumers, etc.

The migration is *incremental*. The `from_args` API stays working until you
delete it; no big-bang rewrite required.

---

## Before / after, one class at a time

### Before

```python
# cofinn/network/cnnfoil.py  (representative ~50 LOC pattern)
@dataclass
class CNNFoil(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.dropout = dropout
        # ...

    @classmethod
    def add_args(cls, parser):
        parser.add_argument("--in-channels", type=int, default=3)
        parser.add_argument("--hidden", type=int, default=64)
        parser.add_argument("--dropout", type=float, default=0.1)

    @classmethod
    def from_args(cls, args):
        return cls(in_channels=args.in_channels,
                   hidden=args.hidden,
                   dropout=args.dropout)

    def to_config(self) -> Namespace:
        return Namespace(in_channels=self.in_channels,
                         hidden=self.hidden,
                         dropout=self.dropout)
```

### After

```python
# cofinn/registries.py  -- one-time setup
from registry import TypeRegistry, FunctionalRegistry

class Networks(TypeRegistry[nn.Module], repo="cofinn.networks"): ...
class Losses(TypeRegistry[nn.Module],    repo="cofinn.losses"):   ...
class Steps(FunctionalRegistry,          repo="cofinn.steps"):    ...
```

```python
# cofinn/network/cnnfoil.py  -- per-class change is ~3 lines added
from registry import build, serialize
from cofinn.registries import Networks


@Networks.register_artifact
class CNNFoil(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden = hidden
        self.dropout = dropout
        # ...

    @classmethod
    def from_args(cls, args): return build(cls, args, validator="argparse")

    @classmethod
    def from_yaml(cls, text): return build(cls, text, validator="yaml")

    def to_config(self):      return Namespace(**serialize(self, serializator="python"))
```

`add_args` disappears entirely -- the CLI side switches to
`jsonargparse.add_class_arguments(parser, CNNFoil)` (one call, derives flags
from the `__init__` signature).

---

## Step-by-step plan

### Step 1: declare your registries (one file, ~10 lines)

```python
# cofinn/registries.py
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from registry import TypeRegistry, FunctionalRegistry

class Networks (TypeRegistry[nn.Module],   repo="cofinn.networks"):  ...
class Losses   (TypeRegistry[nn.Module],   repo="cofinn.losses"):    ...
class Metrics  (TypeRegistry[nn.Module],   repo="cofinn.metrics"):   ...
class Optimizers(TypeRegistry[Optimizer],  repo="cofinn.optimizers"): ...
class Loaders  (TypeRegistry[DataLoader],  repo="cofinn.loaders"):   ...
class Steps    (FunctionalRegistry,        repo="cofinn.steps"):     ...

# Register third-party classes once.
Optimizers.register_artifact(__import__("torch").optim.Adam)
Optimizers.register_artifact(__import__("torch").optim.SGD)
Loaders.register_artifact(DataLoader)
Losses.register_artifact(nn.CrossEntropyLoss)
Losses.register_artifact(nn.MSELoss)
```

### Step 2: register existing classes (one decorator per class)

```python
@Networks.register_artifact
class CNNFoil(nn.Module): ...

@Networks.register_artifact
class UNetPP(nn.Module): ...

@Losses.register_artifact
class HLLCLoss(nn.Module): ...
```

This step alone makes `build("CNNFoil", data, repo="cofinn.networks")` work.
The `add_args` / `from_args` / `to_config` methods keep functioning unchanged.

### Step 3: replace the bodies of `from_args` / `to_config`

```python
@classmethod
def from_args(cls, args):
    return build(cls, args, validator="argparse")

def to_config(self):
    return Namespace(**serialize(self, serializator="python"))
```

Delete `add_args` and add `--config train.yaml` to your CLI via
`jsonargparse.ArgumentParser().add_class_arguments(cls, prefix="...")`.

### Step 4: add registry-level invariants (optional, very high ROI)

Move per-architecture safety checks into the registry's `post_init`:

```python
class Networks(TypeRegistry[nn.Module], repo="cofinn.networks"):
    @classmethod
    def post_init(cls, instance, meta):
        for name, p in instance.named_parameters():
            if torch.isnan(p).any():
                raise ValueError(f"NaN in {name}")
```

You now get NaN checks for every registered network -- automatic, single
place.

### Step 5: add Annotated markers to your training step (declarative
cross-arg validation + provenance)

```python
from typing import Annotated
from registry.markers import SameDeviceAs, BoundTo, Checksum, EffectiveLr

@Steps.register_artifact
def train_one_epoch(
    model:     Annotated[nn.Module, SameDeviceAs("device"), Checksum("model_hash")],
    loader:    DataLoader,
    optimizer: Annotated[Optimizer, BoundTo("model"), EffectiveLr("lr")],
    criterion: nn.Module,
    device:    str,
) -> dict[str, float]:
    ...
```

After every call: meta has `model_hash` + `lr`. Wrong device or unbound
optimizer raises before the function runs.

### Step 6: wire observability (~5 lines)

```python
import logging
from registry import attach_meter, attach_reporter
from registry import LifetimeMeter, CPUMeter, MemoryMeter, JournalReporter

logging.basicConfig(level=logging.INFO)               # stdlib INFO logs everything
attach_meter(LifetimeMeter())
attach_meter(CPUMeter())
attach_meter(MemoryMeter())
attach_reporter(JournalReporter(ident="cofinn"))      # journalctl -t cofinn
```

`build(...)` now logs to stdout and journald automatically, with timing,
CPU, memory in every envelope's meta.

### Step 7: switch the run script

```python
# Before
parser = argparse.ArgumentParser()
CNNFoil.add_args(parser)
TrainerConfig.add_args(parser)
args = parser.parse_args()
model = CNNFoil.from_args(args)
trainer = TrainerConfig.from_args(args)
trainer.train(model)

# After (yaml-driven)
import yaml
cfg = yaml.safe_load(open("train.yaml"))
build(cfg, ctx={"dataset": MNIST(".", train=True, download=True, transform=ToTensor())})
```

with `train.yaml`:

```yaml
type: train_one_epoch
data:
  model:     {type: CNNFoil,    repo: cofinn.networks, data: {hidden: 256}}
  loader:    {type: DataLoader, repo: cofinn.loaders,  data: {dataset: $dataset, batch_size: 64}}
  optimizer: {type: Adam,       repo: cofinn.optimizers, data: {params: $model.parameters(), lr: 1e-3}}
  criterion: {type: HLLCLoss,   repo: cofinn.losses}
  device:    cuda
```

---

## What you can delete after the migration

- All `add_args` methods (jsonargparse derives flags from `__init__`).
- All `from_args` / `to_config` bodies if you don't keep them as one-liners.
- Any bespoke `BaseConfig` / `Namespace`-builder dataclasses.

For a class with three fields, the migration drops 30+ lines to ~3.

---

## Where this DOESN'T help

- Per-batch hooks inside the model's forward (`nn.Module.register_forward_hook`).
- Custom serialization of `torch.Tensor` parameters -- factory's `serialize()`
  reads constructor-mirrored attributes, not learned weights. Use `state_dict()`
  + `torch.save` for that.
- Distributed orchestration -- factory builds one process's object graph.
  Cross-process dispatch is callpyback's job.

---

## Rough effort estimate (cofinn-sized project)

| Step | Effort |
|---|---|
| Declare registries (Step 1) | 30 min |
| Add `@register_artifact` decorators (Step 2) | 1-2 hr for ~30 classes |
| Replace `from_args` / `to_config` bodies (Step 3) | 1-2 hr |
| Per-registry `post_init` invariants (Step 4) | 1-3 hr depending on rules |
| Annotated markers on training step (Step 5) | 30 min |
| Observability wiring (Step 6) | 15 min |
| Switch run script + write `train.yaml` (Step 7) | 1 hr |
| **Total** | **~half a day** |

Mostly mechanical. The big wins are Steps 4-5 (registry-level invariants
catch class of bugs the old triplet couldn't).
