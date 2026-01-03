# Registry Pattern - DI Container / IoC Framework

A Python library implementing the Registry, Factory, and Dependency Injection patterns with Pydantic validation.

## Features

- **Registry Pattern**: Central storage for classes and functions by name
- **Factory Pattern**: Create objects from configuration with validation
- **DI Container**: Recursive object graph construction with context injection
- **Pydantic Integration**: Type coercion and validation via auto-extracted or explicit schemas
- **Multi-Repository**: Organize artifacts into namespaced registries

## Installation

```bash
pip install registry-pattern
```

## Quick Start

### Registry Pattern

Register and retrieve classes/functions by name:

```python
from registry import TypeRegistry

class ModelRegistry(TypeRegistry[object]):
    pass

@ModelRegistry.register_artifact
class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

# Retrieve by name
model_cls = ModelRegistry.get_artifact("ResNet18")
model = model_cls(num_classes=100)
```

### Factory Pattern

Build objects from configuration:

```python
from registry import TypeRegistry, BuildCfg, ContainerMixin

class ModelRegistry(TypeRegistry[object]):
    pass

@ModelRegistry.register_artifact
class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes

ContainerMixin.configure_repos({"models": ModelRegistry})

cfg = BuildCfg(
    type="ResNet18",
    repo="models",
    data={"num_classes": 100}
)
model = ContainerMixin.build_cfg(cfg)
```

### DI Container

Build complex object graphs with nested dependencies:

```python
from registry import TypeRegistry, FunctionalRegistry, BuildCfg, ContainerMixin

class ModelRegistry(TypeRegistry[object]):
    pass

class OptimizerRegistry(TypeRegistry[object]):
    pass

class UtilsRegistry(FunctionalRegistry):
    pass

@ModelRegistry.register_artifact
class ResNet18:
    def __init__(self, num_classes: int = 10):
        self.num_classes = num_classes
    def parameters(self):
        return ["param1", "param2"]

@UtilsRegistry.register_artifact
def model_parameters(model_key: str, ctx: dict):
    """Get parameters from a model in context."""
    return ctx[model_key].parameters()

@OptimizerRegistry.register_artifact
class SGD:
    def __init__(self, params, lr: float = 0.01):
        self.params = params
        self.lr = lr

ContainerMixin.configure_repos({
    "models": ModelRegistry,
    "optimizers": OptimizerRegistry,
    "utils": UtilsRegistry,
})

# Build model first, store in context
model = ContainerMixin.build_named("model", {
    "type": "ResNet18",
    "repo": "models",
    "data": {"num_classes": 10}
})

# Build optimizer with reference to model's parameters
optimizer = ContainerMixin.build_cfg({
    "type": "SGD",
    "repo": "optimizers",
    "data": {
        "lr": 0.01,
        "params": {
            "type": "model_parameters",
            "repo": "utils",
            "data": {"model_key": "model"}
        }
    }
})

print(optimizer.params)  # ["param1", "param2"]
```

## Configuration Schema

The `BuildCfg` model defines the configuration envelope:

```python
BuildCfg(
    type="ClassName",           # Required: artifact identifier
    repo="registry_name",       # Optional: which registry (default: "default")
    data={"key": "value"},      # Optional: kwargs for the builder
    meta={"tag": "v1"}          # Optional: metadata attached to built object
)
```

### Extras Handling

- Unknown top-level keys in BuildCfg → `meta._extra_cfg_keys`
- Unknown data keys (not in builder signature) → `meta._unused_data`

```python
cfg = BuildCfg(
    type="MyClass",
    data={"known_arg": 1, "unknown_arg": 2},
    unknown_top_level="value"
)
obj = ContainerMixin.build_cfg(cfg)
print(obj.__meta__)
# {
#     "_extra_cfg_keys": {"unknown_top_level": "value"},
#     "_unused_data": {"unknown_arg": 2}
# }
```

## Pydantic Integration

### Auto-Extraction

Parameter schemas are automatically extracted from function/class signatures:

```python
@ModelRegistry.register_artifact
class MyModel:
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        ...

# Schema auto-extracted: MyModelParams(hidden_size: int, dropout: float = 0.1)
```

### Explicit Schema

Provide explicit Pydantic models for more control:

```python
from pydantic import BaseModel, Field

class MyModelParams(BaseModel):
    hidden_size: int = Field(ge=1)
    dropout: float = Field(ge=0, le=1, default=0.1)

@ModelRegistry.register_artifact(params_model=MyModelParams)
class MyModel:
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        ...
```

### Type Coercion

Pydantic handles type coercion automatically:

```python
cfg = BuildCfg(
    type="MyModel",
    data={"hidden_size": "256", "dropout": "0.5"}  # strings
)
model = ContainerMixin.build_cfg(cfg)
# hidden_size is int 256, dropout is float 0.5
```

## Context Injection

Builders that accept a `ctx` parameter receive the shared context:

```python
@UtilsRegistry.register_artifact
def ref(key: str, ctx: dict):
    """Reference a previously built object."""
    return ctx[key]

# Build and store
model = ContainerMixin.build_named("my_model", model_cfg)

# Reference later
ref_cfg = BuildCfg(type="ref", repo="utils", data={"key": "my_model"})
same_model = ContainerMixin.build_cfg(ref_cfg)
assert same_model is model
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
MyRegistry.get_artifact("MyClass")
MyRegistry.has_identifier("MyClass")
MyRegistry.iter_identifiers()

# Removal
MyRegistry.unregister_identifier("MyClass")
```

### FunctionalRegistry

```python
class MyFuncRegistry(FunctionalRegistry):
    pass

@MyFuncRegistry.register_artifact
def my_function(x: int) -> str: ...
```

### ContainerMixin

```python
# Configuration
ContainerMixin.configure_repos({"name": Registry, ...})
ContainerMixin.clear_context()
ContainerMixin.get_context()

# Building
ContainerMixin.build_cfg(cfg)           # Build from BuildCfg
ContainerMixin.build_named("key", cfg)  # Build and store in context
ContainerMixin.build_value(value)       # Recursively resolve nested configs
```

### BuildCfg

```python
from registry import BuildCfg, is_build_cfg, normalize_cfg

# Create
cfg = BuildCfg(type="Name", repo="repo", data={}, meta={})

# Helpers
is_build_cfg({"type": "Name"})  # True
normalize_cfg({"type": "Name"})  # BuildCfg instance
```

## Examples

See the `examples/` directory for complete examples:

- `examples/01_registry_basics.py` - Registry pattern fundamentals
- `examples/02_factory_pattern.py` - Factory pattern with validation
- `examples/03_di_container.py` - DI container with object graphs
- `examples/04_pytorch_mnist.py` - PyTorch ResNet18 training on MNIST

## License

MIT
