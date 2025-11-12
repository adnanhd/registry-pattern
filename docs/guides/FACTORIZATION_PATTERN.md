# Factorization Pattern

## Overview

Factorization is the process of instantiating artifacts (functions/classes) from configurations with automatic recursive dependency resolution and Pydantic validation.

## Core Concepts

### What is Factorization?

Factorization transforms a configuration (dict/Pydantic model/file) into a fully instantiated object:

```
Configuration Input
       ↓
Pydantic Validation
       ↓
Recursive Dependency Resolution
       ↓
Instantiation
       ↓
Object Output
```

### Why Use Factorization?

- **Configuration-driven:** Define objects in JSON/YAML/dicts
- **Recursive:** Nested configs auto-instantiate
- **Validated:** Pydantic ensures type safety
- **Distributed:** Load configs from files or network
- **Traceable:** Full error context for debugging

## Basic Usage

### Registering Callables

Any callable (function or class) can be registered:

```python
from registry import SchemeRegistry

class MySchemes(SchemeRegistry):
    pass

def create_optimizer(learning_rate: float, momentum: float = 0.9):
    """Create an optimizer config."""
    return {
        "lr": learning_rate,
        "momentum": momentum
    }

class Model:
    """A trainable model."""
    
    def __init__(self, hidden_dim: int, num_layers: int = 3):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

MySchemes.register_artifact(create_optimizer)
MySchemes.register_artifact(Model)
```

### Extracting Schemes

Schemes are automatically extracted from signatures:

```python
# For create_optimizer:
# create_optimizerConfig(learning_rate: float, momentum: float = 0.9)

# For Model:
# ModelConfig(hidden_dim: int, num_layers: int = 3)

model_config = MySchemes.get_model("Model")
print(model_config.__fields__)
# → {'hidden_dim': FieldInfo(...), 'num_layers': FieldInfo(...)}
```

### Building and Executing

```python
# Build config (with validation)
config = MySchemes.build_config(
    "create_optimizer",
    learning_rate=0.001,
    momentum=0.95
)

# Execute (instantiate)
result = MySchemes.execute("create_optimizer", config)
# → {"lr": 0.001, "momentum": 0.95}

# Or directly with dict (auto-validates)
result = MySchemes.execute("create_optimizer", {
    "learning_rate": 0.001,
    "momentum": 0.95
})
```

## Recursive Factorization

### Problem: Nested Dependencies

Often objects depend on other registered types:

```python
def create_trainer(
    model: "Model",           # Depends on registered Model
    optimizer: "Optimizer",   # Depends on registered Optimizer
    loss_fn: "LossFunction"   # Depends on registered LossFunction
):
    return {"model": model, "optimizer": optimizer, "loss": loss_fn}
```

### Solution: Recursive Resolution

Factorization automatically resolves and instantiates nested types:

```python
# Config with nested types
config = {
    "model": {
        "_type": "Model",
        "hidden_dim": 256,
        "num_layers": 4
    },
    "optimizer": {
        "_type": "create_optimizer",
        "learning_rate": 0.001
    },
    "loss_fn": {
        "_type": "mse_loss"
    }
}

# Automatic recursive resolution
trainer = MySchemes.factorize_artifact("create_trainer", **config)

# Results in:
# trainer = {
#     "model": Model(hidden_dim=256, num_layers=4),
#     "optimizer": {"lr": 0.001, "momentum": 0.9},
#     "loss": <function mse_loss>
# }
```

### How Recursion Works

For each parameter with a registered type annotation:

1. **Check type annotation:** Is this parameter's type registered?
2. **Extract type identifier:** Get `_type` from dict (or use type name)
3. **Recursively factorize:** Call `factorize_artifact` on that type
4. **Substitute:** Use instantiated object as parameter

```
create_trainer(
    model: Model,
    optimizer: Optimizer,
    loss_fn: LossFunction
)

Input config:
{
    "model": {"_type": "Model", "hidden_dim": 256},
    "optimizer": {"_type": "create_optimizer", "learning_rate": 0.001},
    "loss_fn": {"_type": "mse"}
}

Processing:
- model: Type[Model] → config has _type → recursively instantiate → Model(256, ...)
- optimizer: Type[Optimizer] → config has _type → recursively instantiate → Optimizer(0.001)
- loss_fn: Callable → config has _type → recursively instantiate → mse_loss()

Result:
create_trainer(
    model=Model(256, ...),
    optimizer=Optimizer(0.001),
    loss_fn=mse_loss
)
```

## Sources of Configurations

### 1. Direct Dict

```python
result = MySchemes.factorize_artifact(
    "create_trainer",
    model={"_type": "Model", "hidden_dim": 256},
    optimizer={"_type": "create_optimizer", "learning_rate": 0.001}
)
```

### 2. Pydantic Model

```python
from pydantic import BaseModel

class TrainerConfig(BaseModel):
    model: dict  # {"_type": "Model", ...}
    optimizer: dict  # {"_type": "create_optimizer", ...}

config = TrainerConfig(
    model={"_type": "Model", "hidden_dim": 256},
    optimizer={"_type": "create_optimizer", "learning_rate": 0.001}
)

result = MySchemes.execute("create_trainer", config)
```

### 3. JSON File

**config.json:**
```json
{
  "model": {"_type": "Model", "hidden_dim": 256, "num_layers": 4},
  "optimizer": {"_type": "create_optimizer", "learning_rate": 0.001},
  "loss_fn": {"_type": "mse"}
}
```

**Python:**
```python
from pathlib import Path

result = MySchemes.factorize_from_file(
    "create_trainer",
    filepath=Path("config.json")
)
```

### 4. YAML File

**config.yaml:**
```yaml
model:
  _type: Model
  hidden_dim: 256
  num_layers: 4
optimizer:
  _type: create_optimizer
  learning_rate: 0.001
loss_fn:
  _type: mse
```

**Python:**
```python
result = MySchemes.factorize_from_file(
    "create_trainer",
    filepath=Path("config.yaml")
)
```

### 5. Network (RPC/HTTP)

```python
result = MySchemes.factorize_from_socket(
    "create_trainer",
    socket_config={
        "url": "http://config-server.example.com/api/config",
        "timeout": 10
    },
    protocol="http"
)
```

## Validation During Factorization

### Type Checking

Pydantic validates types:

```python
def create_trainer(
    learning_rate: float,  # Must be float
    epochs: int = 100      # Must be int
):
    pass

# Valid
config = {"learning_rate": 0.001, "epochs": 50}
result = MySchemes.execute("create_trainer", config)  # ✓

# Invalid
config = {"learning_rate": "not a float", "epochs": 100}
result = MySchemes.execute("create_trainer", config)  # ✗ ValidationError
```

### Required vs Optional

```python
def create_model(
    hidden_dim: int,              # Required (no default)
    num_layers: int = 3,          # Optional (has default)
    activation: str = "relu"      # Optional
):
    pass

# Valid (required provided)
config = {"hidden_dim": 256}
result = MySchemes.execute("create_model", config)  # ✓

# Invalid (required missing)
config = {"num_layers": 4}
result = MySchemes.execute("create_model", config)  # ✗ ValidationError
```

### Nested Validation

Recursive factorization validates at each level:

```python
config = {
    "model": {
        "_type": "Model",
        "hidden_dim": "not an int"  # Invalid type
    }
}

try:
    result = MySchemes.factorize_artifact("create_trainer", **config)
except ValidationError as e:
    print(e.message)
    # "Validation failed for Model: value is not a valid integer"
    print(e.context)
    # {"type": "Model", "errors": "..."}
```

## Advanced Patterns

### Pattern 1: Deeply Nested Configs

```python
experiment_config = {
    "trainer": {
        "_type": "create_trainer",
        "model": {
            "_type": "create_resnet",
            "depth": 50,
            "optimizer": {
                "_type": "create_adam",
                "learning_rate": 0.001
            }
        }
    }
}

# All levels automatically resolved
result = MySchemes.factorize_artifact("setup_experiment", **experiment_config)
```

### Pattern 2: Conditional Instantiation

```python
def create_model_from_config(model_type: str, **kwargs):
    """Select model type at runtime."""
    # This function name doesn't matter; recursive resolution happens
    # at parameter annotation level
    ModelClass = Models.get_artifact(model_type)
    return ModelClass(**kwargs)

# Config specifies type
config = {
    "model_type": "ResNet50",
    "num_classes": 1000
}

model = MySchemes.factorize_artifact("create_model_from_config", **config)
```

### Pattern 3: Factory with Preprocessing

```python
def create_optimizer(
    algo: str,                           # Algorithm name
    learning_rate: float,
    config_override: Optional[dict] = None
):
    """Create optimizer with optional config override."""
    base_config = {
        "adam": {"beta1": 0.9, "beta2": 0.999},
        "sgd": {"momentum": 0.9}
    }
    config = base_config[algo].copy()
    if config_override:
        config.update(config_override)
    return {"algorithm": algo, **config}

config = {
    "algo": "adam",
    "learning_rate": 0.001,
    "config_override": {"beta1": 0.95}
}

optimizer = MySchemes.execute("create_optimizer", config)
# → {"algorithm": "adam", "beta1": 0.95, "beta2": 0.999, "learning_rate": 0.001}
```

### Pattern 4: Composition

```python
class Pipeline:
    def __init__(
        self,
        preprocessor: "Preprocessor",
        model: "Model",
        postprocessor: "Postprocessor"
    ):
        self.preprocessor = preprocessor
        self.model = model
        self.postprocessor = postprocessor

config = {
    "preprocessor": {"_type": "StandardScaler"},
    "model": {"_type": "Model", "hidden_dim": 256},
    "postprocessor": {"_type": "SoftmaxNormalizer"}
}

pipeline = MySchemes.execute("Pipeline", config)
# → Pipeline with all components instantiated
```

## Error Handling in Factorization

### Common Errors

**Type Not Registered:**
```python
config = {"model": {"_type": "UnknownModel"}}

try:
    MySchemes.factorize_artifact("create_trainer", **config)
except ValidationError as e:
    print(e.message)
    # "Scheme not found: UnknownModel"
    print(e.suggestions)
    # ["Register the artifact first", "Check the identifier spelling"]
```

**Missing Required Parameter:**
```python
def create_model(hidden_dim: int):
    pass

try:
    MySchemes.execute("create_model", {})
except ValidationError as e:
    print(e.message)
    # "Validation failed for create_model: field required"
```

**Type Mismatch:**
```python
def create_model(hidden_dim: int):
    pass

try:
    MySchemes.execute("create_model", {"hidden_dim": "not an int"})
except ValidationError as e:
    print(e.message)
    # "Validation failed: value is not a valid integer"
```

### Debugging

```python
from registry import ValidationError

try:
    result = MySchemes.factorize_from_file("my_type", "config.json")
except ValidationError as e:
    print(f"Error: {e.message}")
    print(f"Suggestions:")
    for s in e.suggestions:
        print(f"  - {s}")
    print(f"Context:")
    import json
    print(json.dumps(e.context, indent=2))
```

## Performance Considerations

### Schema Caching

Schemes are extracted once and cached:

```python
# First call extracts and caches scheme
config1 = MySchemes.build_config("MyClass", param=value)

# Subsequent calls reuse cached scheme
config2 = MySchemes.build_config("MyClass", param=value)
# Much faster
```

### Validation Caching

Validation results are cached per artifact:

```python
# First validation
result1 = MySchemes.execute("MyClass", config)

# Second identical validation hits cache
result2 = MySchemes.execute("MyClass", config)
```

### Batching

For multiple instantiations, batch when possible:

```python
# ❌ Inefficient
for config in configs:
    result = MySchemes.factorize_artifact("MyClass", **config)

# ✅ Better (uses connection pooling for remote registries)
results = [
    MySchemes.factorize_artifact("MyClass", **config)
    for config in configs
]
```

## Serialization

Factorization handles complex types through Pydantic:

### JSON-Serializable

```python
def create_config(
    values: list,
    params: dict,
    name: str
):
    pass

config = {
    "values": [1, 2, 3],
    "params": {"alpha": 0.5},
    "name": "experiment_1"
}

result = MySchemes.execute("create_config", config)  # ✓ Works
```

### Complex Objects

```python
class CustomModel:
    def __init__(self, data: np.ndarray):
        self.data = data

# numpy arrays aren't JSON-serializable
# Use pickle instead or convert to list

config = {
    "data": [[1, 2], [3, 4]]  # As list
}

model = MySchemes.execute("CustomModel", {"data": np.array(config["data"])})
```

## Integration with Config Files

### Experiment Config Structure

```yaml
experiment:
  name: "ResNet50_ImageNet"
  seed: 42
  
  model:
    _type: "ResNet"
    depth: 50
    pretrained: false
  
  data:
    _type: "ImageNetDataLoader"
    batch_size: 256
    num_workers: 8
    augmentation:
      _type: "AugmentationPipeline"
      transforms: ["RandomCrop", "RandomFlip", "ColorJitter"]
  
  training:
    optimizer:
      _type: "create_adam"
      learning_rate: 0.1
      weight_decay: 0.0001
    
    scheduler:
      _type: "CosineAnnealing"
      T_max: 100
    
    epochs: 100
    early_stopping:
      patience: 10
```

**Loading:**
```python
import yaml
from pathlib import Path

config_path = Path("experiment.yaml")
config = yaml.safe_load(config_path.read_text())

# Recursively instantiate
experiment = MySchemes.factorize_artifact(
    "setup_experiment",
    **config["experiment"]
)
```

## Factory Registry Pattern

Combine multiple registries for specialized factories:

```python
class Losses(FunctionalRegistry):
    """Just functions."""
    pass

class Models(TypeRegistry):
    """Just classes."""
    pass

class Configs(SchemeRegistry):
    """Callable schemes."""
    pass

# Use together
@Losses.register_artifact
def my_loss(y_true, y_pred):
    pass

@Models.register_artifact
class MyModel:
    pass

@Configs.register_artifact
def create_trainer(loss_type: str, model_type: str):
    loss = Losses.get_artifact(loss_type)
    model = Models.get_artifact(model_type)
    return {"loss": loss(), "model": model()}

# Factorization coordinates all three
trainer = Configs.execute("create_trainer", {
    "loss_type": "my_loss",
    "model_type": "MyModel"
})
```

## Next Steps

- See [EXAMPLES.md](EXAMPLES.md) for research-specific factorization patterns
- See [API_REFERENCE.md](API_REFERENCE.md) for `factorize_*` method details
- See [VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md) for error semantics
