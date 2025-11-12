# Examples: Research Applications

Practical examples for using registries in research workflows—from deep learning to signal processing to optimization.

## Example 1: Loss Function Registry for Neural Networks

```python
from registry import FunctionalRegistry
import numpy as np

class Losses(FunctionalRegistry, strict=False):
    """Registry for loss functions."""
    pass

@Losses.register_artifact
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Squared Error."""
    return np.mean((y_true - y_pred) ** 2)

@Losses.register_artifact
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return np.mean(np.abs(y_true - y_pred))

@Losses.register_artifact
def huber(y_true: np.ndarray, y_pred: np.ndarray, delta: float = 1.0) -> float:
    """Huber loss for robustness."""
    diff = np.abs(y_true - y_pred)
    return np.where(
        diff <= delta,
        0.5 * diff ** 2,
        delta * (diff - 0.5 * delta)
    ).mean()

@Losses.register_artifact
def cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Cross-entropy for classification."""
    softmax = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ce = -np.log(softmax[np.arange(len(labels)), labels])
    return ce.mean()

# Usage
y_true = np.random.randn(100)
y_pred = y_true + np.random.randn(100) * 0.1

loss_names = ["mse", "mae", "huber"]
for loss_name in loss_names:
    loss_fn = Losses.get_artifact(loss_name)
    loss_value = loss_fn(y_true, y_pred)
    print(f"{loss_name}: {loss_value:.4f}")
```

## Example 2: Model Architecture Registry

```python
from registry import TypeRegistry
from abc import ABC, abstractmethod

class ModelBase(ABC):
    """Base interface for all models."""
    
    @abstractmethod
    def forward(self, x): pass
    
    @abstractmethod
    def get_parameters(self): pass

class NeuralNetworks(TypeRegistry[ModelBase], abstract=True):
    """Registry for neural network architectures."""
    pass

class SimpleMLPModel(ModelBase):
    """Multi-layer perceptron."""
    
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self._initialize_layers()
    
    def _initialize_layers(self):
        """Initialize layer weights."""
        self.layers = []
        dims = [self.input_dim] + self.hidden_dims + [self.output_dim]
        for i in range(len(dims) - 1):
            self.layers.append({
                "W": np.random.randn(dims[i], dims[i+1]) * 0.01,
                "b": np.zeros((1, dims[i+1]))
            })
    
    def forward(self, x):
        """Forward pass."""
        activation = x
        for layer in self.layers[:-1]:
            activation = np.maximum(0, activation @ layer["W"] + layer["b"])  # ReLU
        return activation @ self.layers[-1]["W"] + self.layers[-1]["b"]
    
    def get_parameters(self):
        """Get layer parameters."""
        return self.layers

class ResNetBlock(ModelBase):
    """Residual network block."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.use_shortcut = stride > 1 or in_channels != out_channels
    
    def forward(self, x):
        """Forward with residual connection."""
        identity = x
        if self.use_shortcut:
            identity = self._downsample(x)
        
        out = x  # Simplified: just pass through
        return out + identity
    
    def _downsample(self, x):
        return x[:, ::self.stride, ::self.stride, :]  # Simplified
    
    def get_parameters(self):
        return {}

# Register architectures
NeuralNetworks.register_artifact(SimpleMLPModel)
NeuralNetworks.register_artifact(ResNetBlock)

# Usage
architecture = input("Choose architecture [SimpleMLPModel/ResNetBlock]: ")
ModelClass = NeuralNetworks.get_artifact(architecture)

if architecture == "SimpleMLPModel":
    model = ModelClass(input_dim=784, hidden_dims=[256, 128], output_dim=10)
else:
    model = ModelClass(in_channels=64, out_channels=128)

print(f"Model: {architecture}")
print(f"Parameters: {len(model.get_parameters())}")
```

## Example 3: Optimization Algorithm Configuration

```python
from registry import SchemeRegistry

class OptimizationConfig(SchemeRegistry):
    """Registry for optimizer configurations."""
    pass

def adam_optimizer(
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0
):
    """Adam optimizer configuration."""
    return {
        "algorithm": "adam",
        "learning_rate": learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "epsilon": epsilon,
        "weight_decay": weight_decay,
        "m": None,  # First moment
        "v": None   # Second moment
    }

def sgd_optimizer(
    learning_rate: float,
    momentum: float = 0.9,
    nesterov: bool = False
):
    """SGD optimizer configuration."""
    return {
        "algorithm": "sgd",
        "learning_rate": learning_rate,
        "momentum": momentum,
        "nesterov": nesterov,
        "velocity": None
    }

OptimizationConfig.register_artifact(adam_optimizer)
OptimizationConfig.register_artifact(sgd_optimizer)

# Build from config
adam_config = OptimizationConfig.build_config(
    "adam_optimizer",
    learning_rate=0.001,
    beta1=0.9,
    weight_decay=1e-4
)

sgd_config = OptimizationConfig.build_config(
    "sgd_optimizer",
    learning_rate=0.01,
    momentum=0.95
)

print("Adam config:", adam_config.model_dump())
print("SGD config:", sgd_config.model_dump())
```

## Example 4: Kalman Filter Variants Registry

```python
from registry import TypeRegistry
from abc import ABC, abstractmethod

class KalmanFilterBase(ABC):
    """Base interface for Kalman filters."""
    
    @abstractmethod
    def predict(self, x): pass
    
    @abstractmethod
    def update(self, z): pass

class KalmanFilters(TypeRegistry[KalmanFilterBase], abstract=True):
    """Registry for Kalman filter implementations."""
    pass

class StandardKalmanFilter(KalmanFilterBase):
    """Standard discrete Kalman filter."""
    
    def __init__(self, F: np.ndarray, H: np.ndarray, Q: np.ndarray, R: np.ndarray):
        self.F = F  # State transition matrix
        self.H = H  # Observation matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.x = None  # State
        self.P = None  # State covariance
    
    def predict(self, x):
        """Predict step."""
        self.x = self.F @ x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x
    
    def update(self, z):
        """Update step."""
        K = self.P @ self.H.T @ np.linalg.inv(self.H @ self.P @ self.H.T + self.R)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return self.x

class ExtendedKalmanFilter(KalmanFilterBase):
    """Extended Kalman filter for nonlinear systems."""
    
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R):
        self.f = f  # Nonlinear state transition
        self.h = h  # Nonlinear observation model
        self.F_jac = F_jacobian  # Jacobian of f
        self.H_jac = H_jacobian  # Jacobian of h
        self.Q = Q
        self.R = R
        self.x = None
        self.P = None
    
    def predict(self, x):
        """Predict step for nonlinear system."""
        self.x = self.f(x)
        F = self.F_jac(x)
        self.P = F @ self.P @ F.T + self.Q
        return self.x
    
    def update(self, z):
        """Update step for nonlinear observation."""
        H = self.H_jac(self.x)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)
        self.x = self.x + K @ (z - self.h(self.x))
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P
        return self.x

KalmanFilters.register_artifact(StandardKalmanFilter)
KalmanFilters.register_artifact(ExtendedKalmanFilter)

# Usage
F = np.eye(2)  # Identity transition
H = np.eye(2)  # Direct observation
Q = np.eye(2) * 0.01
R = np.eye(2) * 0.1

kf_class = KalmanFilters.get_artifact("StandardKalmanFilter")
kf = kf_class(F=F, H=H, Q=Q, R=R)
```

## Example 5: Experiment Configuration from YAML

**experiments/baseline.yaml:**
```yaml
experiment_name: "baseline_kalman_filter"
seed: 42

model:
  _type: "StandardKalmanFilter"
  F: [[1, 0], [0, 1]]
  H: [[1, 0], [0, 1]]
  Q: [[0.01, 0], [0, 0.01]]
  R: [[0.1, 0], [0, 0.1]]

optimizer:
  _type: "adam_optimizer"
  learning_rate: 0.001
  beta1: 0.9
  beta2: 0.999

loss:
  _type: "mse"

training:
  epochs: 100
  batch_size: 32
  early_stopping: true
  patience: 10
```

**Python:**
```python
from pathlib import Path
import yaml

class ExperimentFactory(SchemeRegistry):
    """Manages experiment configuration."""
    pass

def setup_experiment(
    experiment_name: str,
    model: dict,
    optimizer: dict,
    loss: dict,
    training: dict,
    seed: int = 42
):
    """Set up complete experiment."""
    np.random.seed(seed)
    
    return {
        "name": experiment_name,
        "config": {
            "model": model,
            "optimizer": optimizer,
            "loss": loss,
            "training": training
        }
    }

ExperimentFactory.register_artifact(setup_experiment)

# Load from YAML
config_path = Path("experiments/baseline.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Recursive factorization
experiment = ExperimentFactory.factorize_artifact(
    "setup_experiment",
    **config
)

print(f"Experiment: {experiment['name']}")
print(f"Model: {experiment['config']['model']}")
```

## Example 6: Multi-Modal Data Processing Pipeline

```python
from registry import TypeRegistry
from abc import ABC, abstractmethod

class DataProcessor(ABC):
    @abstractmethod
    def process(self, data): pass

class DataProcessors(TypeRegistry[DataProcessor], abstract=True):
    """Registry for data processors."""
    pass

class SensorDataProcessor(DataProcessor):
    """Process IMU/UWB sensor data."""
    
    def __init__(self, filter_type: str = "kalman", fs: float = 100.0):
        self.filter_type = filter_type
        self.fs = fs  # Sampling frequency
    
    def process(self, raw_data):
        """Apply filtering and normalization."""
        # High-pass filter to remove drift
        filtered = self._high_pass_filter(raw_data)
        # Normalize
        return (filtered - filtered.mean()) / filtered.std()
    
    def _high_pass_filter(self, data):
        """Simple high-pass filter."""
        alpha = 0.05
        filtered = np.zeros_like(data)
        filtered[0] = data[0]
        for i in range(1, len(data)):
            filtered[i] = alpha * (filtered[i-1] + data[i] - data[i-1])
        return filtered

class SignalProcessor(DataProcessor):
    """Process frequency-domain signals."""
    
    def __init__(self, window: str = "hann", nperseg: int = 256):
        self.window = window
        self.nperseg = nperseg
    
    def process(self, signal):
        """Compute spectrogram."""
        from scipy import signal as sig
        frequencies, times, Sxx = sig.spectrogram(
            signal,
            window=self.window,
            nperseg=self.nperseg
        )
        return {"frequencies": frequencies, "times": times, "power": Sxx}

DataProcessors.register_artifact(SensorDataProcessor)
DataProcessors.register_artifact(SignalProcessor)

# Select processor from config
processor_config = {"processor_type": "SensorDataProcessor", "fs": 100.0}
ProcessorClass = DataProcessors.get_artifact(processor_config["processor_type"])
processor = ProcessorClass(fs=processor_config["fs"])

# Process data
raw_sensor_data = np.random.randn(1000)
processed = processor.process(raw_sensor_data)
```

## Example 7: Batch Experiment Grid Search

```python
from registry import SchemeRegistry

class TrainingConfigs(SchemeRegistry):
    pass

def create_training_config(
    model_arch: str,
    optimizer: str,
    learning_rate: float,
    batch_size: int,
    epochs: int = 100
):
    return {
        "model": model_arch,
        "optimizer": optimizer,
        "lr": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs
    }

TrainingConfigs.register_artifact(create_training_config)

# Grid search
hyperparams = {
    "model_arch": ["SimpleMLPModel", "ResNetBlock"],
    "optimizer": ["adam_optimizer", "sgd_optimizer"],
    "learning_rate": [0.001, 0.01],
    "batch_size": [32, 64]
}

experiments = []
for model in hyperparams["model_arch"]:
    for opt in hyperparams["optimizer"]:
        for lr in hyperparams["learning_rate"]:
            for bs in hyperparams["batch_size"]:
                config = {
                    "model_arch": model,
                    "optimizer": opt,
                    "learning_rate": lr,
                    "batch_size": bs
                }
                experiments.append(config)

print(f"Total experiments: {len(experiments)}")

# Register in batch
results = TrainingConfigs.safe_register_batch(
    {f"exp_{i}": exp for i, exp in enumerate(experiments)},
    skip_invalid=True
)

print(f"Registered: {len(results['successful'])}")
print(f"Failed: {len(results['failed'])}")
```

## Example 8: Remote Registry for Distributed Training

**Server setup (on master node):**
```bash
python -m registry.__main__ --host 0.0.0.0 --port 8001
```

**Client setup (on worker nodes):**
```python
from registry import FunctionalRegistry

class DistributedMetrics(FunctionalRegistry, logic_namespace="training.metrics"):
    """Metrics synced across all workers."""
    pass

@DistributedMetrics.register_artifact
def accuracy(predictions, labels):
    """Classification accuracy."""
    return (predictions == labels).mean()

@DistributedMetrics.register_artifact
def f1_score(predictions, labels):
    """F1 score for binary classification."""
    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return 2 * (precision * recall) / (precision + recall + 1e-10)

# All workers can access the same metrics
def evaluate(predictions, labels):
    accuracy_fn = DistributedMetrics.get_artifact("accuracy")
    f1_fn = DistributedMetrics.get_artifact("f1_score")
    
    return {
        "accuracy": accuracy_fn(predictions, labels),
        "f1": f1_fn(predictions, labels)
    }
```

## Example 9: Validation Pipeline

```python
from registry import TypeRegistry
from abc import ABC, abstractmethod

class ValidationStep(ABC):
    @abstractmethod
    def validate(self, data): pass

class ValidationSteps(TypeRegistry[ValidationStep], abstract=True):
    pass

class DataIntegrityCheck(ValidationStep):
    """Check for NaN, Inf, shape mismatches."""
    
    def __init__(self, expected_shape: tuple):
        self.expected_shape = expected_shape
    
    def validate(self, data):
        if data.shape != self.expected_shape:
            raise ValueError(f"Shape mismatch: {data.shape} vs {self.expected_shape}")
        if np.isnan(data).any():
            raise ValueError("Data contains NaN")
        if np.isinf(data).any():
            raise ValueError("Data contains Inf")
        return True

class StatisticalValidation(ValidationStep):
    """Check statistical properties."""
    
    def __init__(self, mean_range: tuple, std_range: tuple):
        self.mean_range = mean_range
        self.std_range = std_range
    
    def validate(self, data):
        mean = data.mean()
        std = data.std()
        
        if not (self.mean_range[0] <= mean <= self.mean_range[1]):
            raise ValueError(f"Mean {mean} outside range {self.mean_range}")
        if not (self.std_range[0] <= std <= self.std_range[1]):
            raise ValueError(f"Std {std} outside range {self.std_range}")
        return True

ValidationSteps.register_artifact(DataIntegrityCheck)
ValidationSteps.register_artifact(StatisticalValidation)

# Use in pipeline
def validate_dataset(data, validation_configs):
    for step_name, params in validation_configs.items():
        StepClass = ValidationSteps.get_artifact(step_name)
        validator = StepClass(**params)
        assert validator.validate(data), f"Validation failed: {step_name}"
    return True

# Config
validation_config = {
    "DataIntegrityCheck": {"expected_shape": (1000, 10)},
    "StatisticalValidation": {"mean_range": (-1, 1), "std_range": (0.5, 2.0)}
}

test_data = np.random.randn(1000, 10) * 0.8
validate_dataset(test_data, validation_config)
```

## Example 10: Experimental Tracking and Logging

```python
from registry import SchemeRegistry
import json
from datetime import datetime

class ExperimentLogger(SchemeRegistry):
    pass

def log_experiment(
    experiment_name: str,
    model_config: dict,
    training_config: dict,
    metrics: dict,
    timestamp: str = None
):
    """Log experiment metadata and results."""
    if timestamp is None:
        timestamp = datetime.now().isoformat()
    
    log_entry = {
        "timestamp": timestamp,
        "name": experiment_name,
        "model_config": model_config,
        "training_config": training_config,
        "metrics": metrics
    }
    
    return log_entry

ExperimentLogger.register_artifact(log_experiment)

# Create multiple experiments
experiments_results = [
    {
        "experiment_name": "exp_001",
        "model_config": {"type": "MLP", "hidden_dims": [256, 128]},
        "training_config": {"lr": 0.001, "epochs": 100},
        "metrics": {"accuracy": 0.92, "loss": 0.08}
    },
    {
        "experiment_name": "exp_002",
        "model_config": {"type": "CNN", "depth": 5},
        "training_config": {"lr": 0.01, "epochs": 50},
        "metrics": {"accuracy": 0.95, "loss": 0.05}
    }
]

# Log all
logs = []
for exp_result in experiments_results:
    log = ExperimentLogger.execute("log_experiment", exp_result)
    logs.append(log)

# Save
with open("experiment_log.json", "w") as f:
    json.dump(logs, f, indent=2)

print(f"Logged {len(logs)} experiments")
```

## Integration Pattern: Full Research Workflow

```python
from pathlib import Path
import yaml

# 1. Define registries
class Models(TypeRegistry):
    pass

class LossFunctions(FunctionalRegistry):
    pass

class Experiments(SchemeRegistry):
    pass

# 2. Register implementations
@Models.register_artifact
class ConvNet: ...

@LossFunctions.register_artifact
def mse(y_true, y_pred): ...

@Experiments.register_artifact
def run_training(model_type, loss_type, learning_rate, epochs):
    Model = Models.get_artifact(model_type)
    loss = LossFunctions.get_artifact(loss_type)
    
    model = Model()
    for epoch in range(epochs):
        # Training loop...
        pass
    
    return {"model": model, "final_loss": ...}

# 3. Run from config
config = yaml.safe_load(Path("experiment.yaml").read_text())
results = Experiments.factorize_artifact("run_training", **config)
```

See the [API_REFERENCE.md](API_REFERENCE.md) for complete method documentation.
