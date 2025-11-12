# Getting Started Guide

## Table of Contents

1. [Installation & Setup](#installation--setup)
2. [FunctionalRegistry Tutorial](#functionalregistry-tutorial)
3. [TypeRegistry Tutorial](#typeregistry-tutorial)
4. [SchemeRegistry Tutorial](#schemeregistry-tutorial)
5. [Working with Remote Storage](#working-with-remote-storage)
6. [Batch Operations](#batch-operations)

## Installation & Setup

### Basic Installation

```bash
pip install registry
```

### Verify Installation

```python
>>> from registry import TypeRegistry, FunctionalRegistry, SchemeRegistry
>>> print("Registry imported successfully!")
```

## FunctionalRegistry Tutorial

### Use Case

Use `FunctionalRegistry` to organize function implementations:
- Loss functions
- Optimization algorithms
- Data augmentation pipelines
- Metrics/evaluation functions

### Creating Your First Registry

```python
from registry import FunctionalRegistry

class LossFunctions(FunctionalRegistry):
    """Registry for loss function implementations."""
    pass
```

### Registering Functions

#### Method 1: Decorator

```python
@LossFunctions.register_artifact
def mse(y_true, y_pred):
    """Mean Squared Error loss."""
    import numpy as np
    return np.mean((y_true - y_pred) ** 2)

@LossFunctions.register_artifact
def mae(y_true, y_pred):
    """Mean Absolute Error loss."""
    import numpy as np
    return np.mean(np.abs(y_true - y_pred))

@LossFunctions.register_artifact
def huber(y_true, y_pred, delta=1.0):
    """Huber loss with adjustable delta."""
    import numpy as np
    diff = np.abs(y_true - y_pred)
    return np.where(
        diff <= delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    ).mean()
```

#### Method 2: Explicit Registration

```python
def cross_entropy(logits, labels):
    """Cross-entropy loss."""
    import numpy as np
    softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    return -np.mean(np.log(softmax[np.arange(len(labels)), labels]))

LossFunctions.register_artifact(cross_entropy)
```

### Using Registered Functions

```python
import numpy as np

y_true = np.array([1, 2, 3])
y_pred = np.array([1.1, 2.2, 2.9])

# Get and use
loss_fn = LossFunctions.get_artifact("mse")
loss_value = loss_fn(y_true, y_pred)
print(f"MSE Loss: {loss_value}")

# List all available
print("Available loss functions:")
for name in LossFunctions.iter_identifiers():
    print(f"  - {name}")

# Check if exists
if LossFunctions.has_identifier("mae"):
    print("MAE is registered")
```

### Organizing into Subregistries

For large projects, organize functions into logical groups:

```python
class OptimizationLosses(FunctionalRegistry):
    """Losses for optimization problems."""
    pass

class RegressionLosses(FunctionalRegistry):
    """Losses for regression tasks."""
    pass

class ClassificationLosses(FunctionalRegistry):
    """Losses for classification tasks."""
    pass

@RegressionLosses.register_artifact
def rmse(y_true, y_pred):
    import numpy as np
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

@ClassificationLosses.register_artifact
def focal_loss(logits, labels, gamma=2.0):
    """Focal loss for addressing class imbalance."""
    import numpy as np
    # Implementation...
    pass
```

## TypeRegistry Tutorial

### Use Case

Use `TypeRegistry` to organize class implementations:
- Model architectures
- Data loaders
- Optimizers
- Callbacks

### Creating a Type Registry

```python
from abc import ABC, abstractmethod
from registry import TypeRegistry

class ModelInterface(ABC):
    """Base interface for all models."""
    
    @abstractmethod
    def forward(self, x):
        """Forward pass."""
        pass
    
    @abstractmethod
    def backward(self, loss):
        """Backward pass."""
        pass

class Models(TypeRegistry[ModelInterface], abstract=True):
    """Registry for model implementations.
    
    abstract=True requires all registered classes to inherit from ModelInterface.
    """
    pass
```

### Registering Classes

```python
class MLPModel(ModelInterface):
    """Multi-layer perceptron."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.layers = self._build_layers()
    
    def _build_layers(self):
        # Implementation...
        pass
    
    def forward(self, x):
        # Implementation...
        pass
    
    def backward(self, loss):
        # Implementation...
        pass

class ConvNetModel(ModelInterface):
    """Convolutional neural network."""
    
    def __init__(self, in_channels: int, num_classes: int):
        self.in_channels = in_channels
        self.num_classes = num_classes
    
    def forward(self, x):
        # Implementation...
        pass
    
    def backward(self, loss):
        # Implementation...
        pass

# Register both
Models.register_artifact(MLPModel)
Models.register_artifact(ConvNetModel)
```

### Using Type Registry

```python
# Get and instantiate
ModelClass = Models.get_artifact("MLPModel")
model = ModelClass(input_dim=28*28, hidden_dims=[256, 128], output_dim=10)

# List available models
print("Available models:")
for name in Models.iter_identifiers():
    print(f"  - {name}")

# Dynamic instantiation
model_name = "ConvNetModel"
model = Models.get_artifact(model_name)(in_channels=3, num_classes=1000)
```

### Strict Mode for Signature Enforcement

```python
class StrictModels(TypeRegistry[ModelInterface], abstract=True, strict=True):
    """Enforce exact method signatures."""
    pass
```

In strict mode, method signatures must exactly match the base class. This prevents subtle bugs from signature mismatches.

## SchemeRegistry Tutorial

### Use Case

Use `SchemeRegistry` when you need configuration-driven instantiation:
- Hyperparameter management
- Experiment configuration
- Nested config hierarchies

### Creating a Scheme Registry

```python
from registry import SchemeRegistry
from typing import Optional

class Configurations(SchemeRegistry):
    """Registry of configuration schemes."""
    pass
```

### Registering Configuration Schemes

Anything callable can be registered:

```python
def create_adam_optimizer(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8
):
    """Adam optimizer configuration."""
    return {
        "name": "adam",
        "lr": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
    }

def create_sgd_optimizer(
    learning_rate: float,
    momentum: float = 0.0
):
    """SGD optimizer configuration."""
    return {
        "name": "sgd",
        "lr": learning_rate,
        "momentum": momentum,
    }

Configurations.register_artifact(create_adam_optimizer)
Configurations.register_artifact(create_sgd_optimizer)
```

### Building Validated Configs

```python
# Build config with Pydantic validation
config = Configurations.build_config(
    "create_adam_optimizer",
    learning_rate=0.001,
    beta1=0.9
)

print(config)
# -> create_adam_optimizerConfig(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Access as dict
config_dict = config.model_dump()
# -> {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8}
```

### Executing with Validation

```python
# Execute function with validated arguments
optimizer_config = Configurations.execute(
    "create_adam_optimizer",
    {"learning_rate": 0.001, "beta1": 0.95}
)

print(optimizer_config)
# -> {'name': 'adam', 'lr': 0.001, 'beta1': 0.95, ...}
```

### Loading from Configuration Files

JSON config file (`config.json`):
```json
{
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999
}
```

Load and instantiate:
```python
from pathlib import Path

# Auto-detects JSON format from extension
optimizer = Configurations.factorize_from_file(
    "create_adam_optimizer",
    filepath=Path("config.json")
)
```

Supports JSON, YAML, TOML:
```python
# Load from YAML
config = Configurations.factorize_from_file(
    "create_adam_optimizer",
    filepath="config.yaml",
    engine="yaml"  # Auto-detected from .yaml extension
)
```

### Registering Classes in Scheme Registry

```python
class Trainer:
    """Training loop."""
    
    def __init__(self, model_name: str, learning_rate: float, epochs: int = 100):
        self.model_name = model_name
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def train(self):
        # Implementation...
        pass

Configurations.register_artifact(Trainer)

# Build config
trainer_config = Configurations.build_config(
    "Trainer",
    model_name="ResNet50",
    learning_rate=0.001
)

# Instantiate
trainer = Configurations.execute("Trainer", trainer_config)
print(f"Training {trainer.model_name} for {trainer.epochs} epochs")
```

## Working with Remote Storage

### Setting Up Remote Storage

1. **Start the registry server** (separate process):

```bash
python -m registry.__main__ --host 0.0.0.0 --port 8001
```

Or with environment variables:
```bash
REGISTRY_SERVER_HOST=0.0.0.0 \
REGISTRY_SERVER_PORT=8001 \
python -m registry.__main__
```

2. **Create registries with remote storage**:

```python
from registry import FunctionalRegistry

class DistributedFunctions(FunctionalRegistry, logic_namespace="research.functions"):
    """Synced across network."""
    pass

class DistributedModels(FunctionalRegistry, logic_namespace="research.models"):
    """Different namespace for models."""
    pass
```

### Using Remote Registries

```python
# Register locally (syncs to server)
@DistributedFunctions.register_artifact
def my_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# Another process can access the same artifact
loss_fn = DistributedFunctions.get_artifact("my_loss")
```

### Checking Server Health

```python
import requests

response = requests.get("http://localhost:8001/health")
print(response.json())
# -> {'status': 'healthy', 'service': 'registry-server'}

# Get statistics
response = requests.get("http://localhost:8001/stats")
stats = response.json()
print(f"Total namespaces: {stats['namespaces']}")
print(f"Total entries: {stats['total_entries']}")
```

## Batch Operations

### Safe Batch Registration

```python
from registry import FunctionalRegistry

class Losses(FunctionalRegistry):
    pass

losses_dict = {
    "mse": lambda y_true, y_pred: ((y_true - y_pred) ** 2).mean(),
    "mae": lambda y_true, y_pred: abs(y_true - y_pred).mean(),
    "huber": lambda y_true, y_pred: ...,
}

# Register with error capture
result = Losses.safe_register_batch(losses_dict, skip_invalid=True)

print(f"Successful: {len(result['successful'])}")
print(f"Failed: {len(result['failed'])}")
if result['errors']:
    for error in result['errors']:
        print(f"  {error['key']}: {error['error']}")
```

### Batch Validation

```python
# Validate without registering
results = Losses.batch_validate(losses_dict)

for key, result in results.items():
    if result is True:
        print(f"{key}: Valid")
    else:
        print(f"{key}: Invalid - {result.message}")
```

### Validate Registry State

```python
# Check all artifacts
report = Losses.validate_registry_state()

print(f"Total artifacts: {report['total_artifacts']}")
if report['validation_errors']:
    print("Validation errors found:")
    for error in report['validation_errors']:
        print(f"  {error['key']}: {error['error']}")
```

## Common Patterns

### Pattern 1: Plugin Architecture

```python
class PluginInterface(ABC):
    @abstractmethod
    def process(self, data): pass

class Plugins(TypeRegistry[PluginInterface], abstract=True):
    pass

def load_plugins_from_module(module):
    """Auto-discover and register plugins."""
    Plugins.register_module_subclasses(module, raise_error=False)
```

### Pattern 2: Algorithm Selection

```python
class Algorithms(FunctionalRegistry):
    pass

def run_pipeline(algorithm_name: str, **kwargs):
    """Run named algorithm with kwargs."""
    algo = Algorithms.get_artifact(algorithm_name)
    return algo(**kwargs)

# In config file
algorithm = "my_algorithm"  # Selected at runtime
result = run_pipeline(algorithm, param1=value1, param2=value2)
```

### Pattern 3: Hierarchical Configuration

```python
class TrainingConfig:
    def __init__(self, optimizer: str, loss: str, model: str):
        self.optimizer_config = Optimizers.get_model(optimizer)
        self.loss_config = Losses.get_model(loss)
        self.model_config = Models.get_model(model)

# From nested JSON
config = TrainingConfig(
    optimizer="adam",
    loss="cross_entropy",
    model="resnet50"
)
```

## Next Steps

- See [API_REFERENCE.md](API_REFERENCE.md) for complete method documentation
- See [VALIDATION_ERROR_HANDLING.md](VALIDATION_ERROR_HANDLING.md) for error handling
- See [EXAMPLES.md](EXAMPLES.md) for research-specific examples
