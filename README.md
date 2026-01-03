# Registry Pattern

[![Build Status](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml/badge.svg)](https://github.com/adnanhd/registry-pattern/actions/workflows/build.yml)
[![Coverage Status](https://coveralls.io/repos/github/adnanhd/registry-pattern/badge.svg)](https://coveralls.io/github/adnanhd/registry-pattern)

A Python library implementing the **Registry Pattern**, **Factory Pattern**, and **Dependency Injection Container** with Pydantic integration for configuration-driven object construction.

## Features

- **Registry Pattern**: Central storage for classes and functions by name
- **Factory Pattern**: Configuration-driven instantiation with Pydantic validation
- **DI Container**: Recursive object graph construction with context injection
- **Type Coercion**: Automatic string-to-type conversion via Pydantic
- **Extras Handling**: Unknown config fields captured in `meta._unused_data`
- **Multi-Repository**: Namespace-based artifact organization
- **Deeply Nested Configs**: Support for arbitrarily nested `BuildCfg` envelopes

## Installation

```bash
pip install registry-pattern
```

Or install from source:

```bash
git clone https://github.com/adnanhd/registry-pattern.git
cd registry-pattern
pip install -e .
```

## Quick Start

### Registry Pattern

```python
from registry import TypeRegistry, FunctionalRegistry

# Create a registry for model classes
class ModelRegistry(TypeRegistry[object]):
    pass

# Register with decorator
@ModelRegistry.register_artifact
class MyModel:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size

# Retrieve and instantiate
model_cls = ModelRegistry.get_artifact("MyModel")
model = model_cls(hidden_size=256)
```

### Factory Pattern with Pydantic

```python
from pydantic import BaseModel, Field
from registry import TypeRegistry, BuildCfg, ContainerMixin

class ModelRegistry(TypeRegistry[object]):
    pass

class ModelParams(BaseModel):
    hidden_size: int = Field(ge=1, le=4096)
    dropout: float = Field(ge=0, le=1, default=0.1)

@ModelRegistry.register_artifact(params_model=ModelParams)
class ValidatedModel:
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.dropout = dropout

# Configure and build
ContainerMixin.configure_repos({"models": ModelRegistry, "default": ModelRegistry})

cfg = BuildCfg(
    type="ValidatedModel",
    repo="models",
    data={"hidden_size": "512", "dropout": "0.2"},  # Strings are coerced
    meta={"experiment": "test"}
)

model = ContainerMixin.build_cfg(cfg)
# model.hidden_size == 512 (int, not str)
```

### DI Container with Nested Configs

```python
from registry import TypeRegistry, BuildCfg, ContainerMixin

class OptimizerRegistry(TypeRegistry[object]):
    pass

class TrainerRegistry(TypeRegistry[object]):
    pass

@OptimizerRegistry.register_artifact
class Adam:
    def __init__(self, lr: float = 0.001):
        self.lr = lr

@TrainerRegistry.register_artifact
class Trainer:
    def __init__(self, model: object, optimizer: object, ctx: dict = None):
        self.model = model
        self.optimizer = optimizer
        self.ctx = ctx or {}

ContainerMixin.configure_repos({
    "optimizers": OptimizerRegistry,
    "trainers": TrainerRegistry,
    "default": TrainerRegistry,
})

# Nested configuration - optimizer is built recursively
cfg = BuildCfg(
    type="Trainer",
    repo="trainers",
    data={
        "model": some_model,
        "optimizer": BuildCfg(
            type="Adam",
            repo="optimizers",
            data={"lr": 0.0001}
        )
    }
)

trainer = ContainerMixin.build_cfg(cfg)
# trainer.optimizer is a fully constructed Adam instance
```

### Context Injection

```python
# Build named objects that can be referenced later
ContainerMixin.build_named("encoder", encoder_cfg)
ContainerMixin.build_named("decoder", decoder_cfg)

# Objects with `ctx` parameter receive the shared context
@TrainerRegistry.register_artifact
class MultiModelTrainer:
    def __init__(self, main_model: object, ctx: dict = None):
        self.main_model = main_model
        self.encoder = ctx.get("encoder")  # Access named objects
        self.decoder = ctx.get("decoder")
```

## BuildCfg Envelope Schema

```python
BuildCfg(
    type="ClassName",           # Required: artifact name in registry
    repo="namespace",           # Optional: registry namespace (default: "default")
    data={"param": "value"},    # Optional: constructor arguments
    meta={"tag": "info"}        # Optional: metadata attached to built object
)
```

- **type**: Name of the registered class/function
- **repo**: Registry namespace to look up the artifact
- **data**: Arguments passed to the constructor (validated via `params_model`)
- **meta**: Metadata attached to the built object as `__meta__` attribute

Unknown fields in `data` are moved to `meta._unused_data`.

## Buildable Type Guard

The `Buildable[T]` type annotation enables Pydantic models to accept either:
- An already-constructed instance of type `T`
- A `BuildCfg` (or dict) that gets built into an instance of `T`

```python
from pydantic import BaseModel
from registry import Buildable, TypeRegistry, ContainerMixin

class ModelRegistry(TypeRegistry[object]):
    pass

@ModelRegistry.register_artifact
class MyModel:
    def __init__(self, size: int):
        self.size = size

ContainerMixin.configure_repos({"models": ModelRegistry, "default": ModelRegistry})

class TrainerConfig(BaseModel):
    model: Buildable[MyModel]  # Accepts MyModel instance OR BuildCfg

# Works with direct instance
config1 = TrainerConfig(model=MyModel(size=10))

# Works with BuildCfg dict - automatically built
config2 = TrainerConfig(model={
    "type": "MyModel",
    "data": {"size": 20}
})
assert isinstance(config2.model, MyModel)
assert config2.model.size == 20
```

## Examples

See the `examples/` directory for complete examples:

- `01_registry_basics.py` - Registry pattern fundamentals
- `02_factory_pattern.py` - Factory pattern with Pydantic validation
- `03_di_container.py` - DI container with nested object graphs
- `04_pytorch_mnist.py` - PyTorch ResNet18 training on MNIST
- `05_full_experiment_config.py` - Complete ML experiment from config

Run an example:

```bash
PYTHONPATH=. python examples/01_registry_basics.py
```

## API Reference

### TypeRegistry

```python
class MyRegistry(TypeRegistry[BaseClass]):
    pass

# Registration
@MyRegistry.register_artifact
@MyRegistry.register_artifact(params_model=MyParams)
class MyClass: ...

# Retrieval
cls = MyRegistry.get_artifact("MyClass")
exists = MyRegistry.has_identifier("MyClass")
names = list(MyRegistry.iter_identifiers())

# Management
MyRegistry.unregister_identifier("MyClass")
MyRegistry.clear_artifacts()
```

### FunctionalRegistry

```python
class FuncRegistry(FunctionalRegistry):
    pass

@FuncRegistry.register_artifact
def my_function(x: int) -> int:
    return x * 2

fn = FuncRegistry.get_artifact("my_function")
```

### ContainerMixin

```python
# Configure repositories
ContainerMixin.configure_repos({
    "models": ModelRegistry,
    "optimizers": OptimizerRegistry,
    "default": ModelRegistry,
})

# Build from config
obj = ContainerMixin.build_cfg(cfg)

# Build and store in context
obj = ContainerMixin.build_named("key", cfg)

# Access/clear context
ContainerMixin._ctx["shared_data"] = value
ContainerMixin.clear_context()
```

## Testing

```bash
# Run all tests
PYTHONPATH=. pytest tests/ -v

# Run specific test file
PYTHONPATH=. pytest tests/test_nested_envelopes.py -v
```

## Continuous Integration

This project uses GitHub Actions for CI:

- **Tests**: `pytest` with coverage
- **Type Checking**: `pyright`
- **Code Formatting**: `black`
- **Linting**: `flake8`

### Running Checks Locally

```bash
# Install dev dependencies
pip install pytest black flake8 pyright

# Run tests
pytest -vv --cov

# Type checking
pyright

# Code formatting
black --check .

# Linting
flake8 .
```

## License

MIT License - see LICENSE file for details.
