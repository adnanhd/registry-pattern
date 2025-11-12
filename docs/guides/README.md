# Registry: Type-Safe, Extensible Artifact Management

A hierarchical registry system for managing functions, classes, and type schemes with optional strict signature validation, local/remote storage, and recursive factorization.

## Overview

The `registry` package provides a flexible framework for organizing, validating, and instantiating artifacts (functions, classes, schemes) across your codebase. It's designed for research workflows where you need:

- **Consistent validation** across different artifact types
- **Flexible storage** (in-process or network-synchronized)
- **Recursive instantiation** from configuration
- **Rich error context** with actionable suggestions
- **Scheme extraction** for callable/class introspection

## Core Concepts

The package provides three complementary registries:

| Registry | Stores | Key Use Case |
|----------|--------|--------------|
| `FunctionalRegistry` | Functions | Algorithm implementations, loss functions |
| `TypeRegistry` | Classes | Model architectures, data handlers |
| `SchemeRegistry` | Callable→Pydantic schemes | Configuration-driven instantiation |

Each registry offers:
- **Type-safe artifact lookup** with validation
- **Optional strict mode** for signature enforcement
- **Batch operations** with error capture
- **Local or remote storage** via pluggable backends
- **Recursive factorization** for nested dependencies

## Installation

```bash
# Basic installation
pip install registry

# With remote storage support
pip install registry[remote]

# Development
git clone <repo>
cd registry
pip install -e .
```

## Quick Start

### 1. Function Registry

```python
from registry import FunctionalRegistry

class MyFunctions(FunctionalRegistry):
    """Registry of loss functions."""
    pass

@MyFunctions.register_artifact
def mse(y_true, y_pred):
    """Mean squared error."""
    return ((y_true - y_pred) ** 2).mean()

@MyFunctions.register_artifact
def mae(y_true, y_pred):
    """Mean absolute error."""
    return abs(y_true - y_pred).mean()

# Retrieve and use
loss_fn = MyFunctions.get_artifact("mse")
result = loss_fn(y_true, y_pred)

# Iterate all registered functions
for name in MyFunctions.iter_identifiers():
    print(f"Available loss: {name}")
```

### 2. Type Registry

```python
from registry import TypeRegistry
from abc import ABC, abstractmethod

class ModelInterface(ABC):
    @abstractmethod
    def forward(self, x): pass
    
    @abstractmethod
    def backward(self, grad): pass

class Models(TypeRegistry[ModelInterface], abstract=True):
    """Registry of model implementations."""
    pass

class ConvNet(ModelInterface):
    def __init__(self, in_channels: int, out_channels: int):
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x): ...
    def backward(self, grad): ...

Models.register_artifact(ConvNet)
model = Models.get_artifact("ConvNet")(in_channels=3, out_channels=64)
```

### 3. Scheme Registry

```python
from registry import SchemeRegistry

class Configurations(SchemeRegistry):
    """Registry of config schemes."""
    pass

def create_optimizer(learning_rate: float, momentum: float = 0.9):
    """Create optimizer with given parameters."""
    return {"lr": learning_rate, "momentum": momentum}

Configurations.register_artifact(create_optimizer)

# Build validated config
config = Configurations.build_config("create_optimizer", learning_rate=0.01)
# -> create_optimizerConfig(learning_rate=0.01, momentum=0.9)

# Execute with validation
optimizer = Configurations.execute("create_optimizer", config)
# -> {"lr": 0.01, "momentum": 0.9}
```

## Storage Backends

By default, registries use thread-safe local storage. For distributed research workflows, use remote storage:

```python
class DistributedRegistry(FunctionalRegistry, logic_namespace="my.research.functions"):
    """Synced across network via central registry server."""
    pass

# Start server (separate process)
# $ python -m registry.__main__ --host 0.0.0.0 --port 8001

# Client connects automatically and syncs operations
DistributedRegistry.register_artifact(my_function)
```

## Validation & Error Handling

Validation is integrated throughout:

```python
from registry import TypeRegistry, ValidationError, InheritanceError

class StrictModels(TypeRegistry, strict=True):
    """Registry with signature enforcement."""
    pass

# This will raise ConformanceError if doesn't match protocol
try:
    StrictModels.register_artifact(IncorrectClass)
except (ValidationError, InheritanceError) as e:
    print(e.message)
    print(e.suggestions)
    print(e.context)
```

See [VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md) for comprehensive error semantics.

## Architecture

The package uses layered mixins for clean separation:

```
RegistryAccessorMixin (read operations)
    ↓
RegistryMutatorMixin (write operations)
    ↓
ImmutableValidatorMixin (read-side validation)
    ↓
MutableValidatorMixin (write-side validation)
    ↓
RegistryFactorizorMixin (recursive instantiation)
    ↓
Concrete Registries (FunctionalRegistry, TypeRegistry, SchemeRegistry)
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design patterns.

## Documentation

- **[GETTING_STARTED.md](GETTING_STARTED.md)** — Step-by-step tutorials
- **[API_REFERENCE.md](API_REFERENCE.md)** — Complete method reference
- **[ARCHITECTURE.md](ARCHITECTURE.md)** — Design patterns and internals
- **[STORAGE_BACKENDS.md](STORAGE_BACKENDS.md)** — Local vs remote storage
- **[VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md)** — Error model
- **[FACTORIZATION_PATTERN.md](FACTORIZATION_PATTERN.md)** — Instantiation system
- **[EXAMPLES.md](EXAMPLES.md)** — Real-world research examples

## Key Features

✓ **Type-safe access** with rich error context  
✓ **Optional strict validation** for signature enforcement  
✓ **Batch operations** with per-item error capture  
✓ **Local or distributed** storage backends  
✓ **Recursive factorization** from configs/files/network  
✓ **Thread-safe** operations with RLock semantics  
✓ **Pydantic integration** for validated instantiation  
✓ **Extensible engine system** for config loaders (JSON, YAML, TOML)  

## License

MIT

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.
