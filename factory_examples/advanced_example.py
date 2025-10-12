"""Advanced example: Multi-level recursive factorization.

Demonstrates a realistic scenario with deep nesting similar to ML pipelines:
- Optimizer -> contains learning rate schedule
- Model -> contains layers with configurations  
- Trainer -> combines model, optimizer, dataset
"""

from registry import TypeRegistry, FunctionalRegistry
from typing import List, Optional


# ============================================================================
# Setup Registries
# ============================================================================

class ComponentRepo(TypeRegistry):
    """Repository for ML components."""
    pass


class PipelineRepo(FunctionalRegistry):
    """Repository for training pipelines."""
    pass


# ============================================================================
# Level 1: Basic Components
# ============================================================================

@ComponentRepo.register_artifact
class LRSchedule:
    """Learning rate schedule."""
    def __init__(self, initial_lr: float, decay_rate: float = 0.95):
        self.initial_lr = initial_lr
        self.decay_rate = decay_rate
    
    def __repr__(self):
        return f"LRSchedule(initial_lr={self.initial_lr}, decay_rate={self.decay_rate})"


@ComponentRepo.register_artifact
class LayerConfig:
    """Configuration for a neural network layer."""
    def __init__(self, units: int, activation: str = "relu", dropout: float = 0.0):
        self.units = units
        self.activation = activation
        self.dropout = dropout
    
    def __repr__(self):
        return f"LayerConfig(units={self.units}, activation={self.activation!r}, dropout={self.dropout})"


# ============================================================================
# Level 2: Composed Components
# ============================================================================

@ComponentRepo.register_artifact
class Optimizer:
    """Optimizer with learning rate schedule."""
    def __init__(self, name: str, lr_schedule: LRSchedule, momentum: float = 0.9):
        self.name = name
        self.lr_schedule = lr_schedule
        self.momentum = momentum
    
    def __repr__(self):
        return f"Optimizer(name={self.name!r}, lr_schedule={self.lr_schedule}, momentum={self.momentum})"


@ComponentRepo.register_artifact
class Model:
    """Model with multiple layers."""
    def __init__(self, name: str, layers: List[LayerConfig], input_dim: int):
        self.name = name
        self.layers = layers
        self.input_dim = input_dim
    
    def __repr__(self):
        return f"Model(name={self.name!r}, layers={self.layers}, input_dim={self.input_dim})"


@ComponentRepo.register_artifact
class Dataset:
    """Dataset configuration."""
    def __init__(self, path: str, batch_size: int = 32, shuffle: bool = True):
        self.path = path
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def __repr__(self):
        return f"Dataset(path={self.path!r}, batch_size={self.batch_size}, shuffle={self.shuffle})"


# ============================================================================
# Level 3: Training Pipeline
# ============================================================================

@ComponentRepo.register_artifact  
class Trainer:
    """Training pipeline combining model, optimizer, and dataset."""
    def __init__(
        self,
        model: Model,
        optimizer: Optimizer,
        dataset: Dataset,
        epochs: int = 10,
        validate: bool = True
    ):
        self.model = model
        self.optimizer = optimizer
        self.dataset = dataset
        self.epochs = epochs
        self.validate = validate
    
    def __repr__(self):
        return (
            f"Trainer(\n"
            f"  model={self.model},\n"
            f"  optimizer={self.optimizer},\n"
            f"  dataset={self.dataset},\n"
            f"  epochs={self.epochs},\n"
            f"  validate={self.validate}\n"
            f")"
        )


@PipelineRepo.register_artifact
def train(trainer: Trainer, checkpoint_dir: Optional[str] = None) -> dict:
    """Execute training pipeline."""
    print(f"Training {trainer.model.name} for {trainer.epochs} epochs...")
    print(f"Dataset: {trainer.dataset.path}")
    print(f"Optimizer: {trainer.optimizer.name}")
    
    # Simulate training
    return {
        "status": "completed",
        "final_loss": 0.123,
        "epochs_trained": trainer.epochs
    }


# ============================================================================
# Example 1: Deep Recursive Factorization
# ============================================================================

def example_deep_nesting():
    print("=" * 70)
    print("Example 1: Deep Recursive Factorization (4 levels)")
    print("=" * 70)
    
    # Create a deeply nested config
    config = {
        "model": {
            "name": "ResNet50",
            "input_dim": 224,
            "layers": [
                {"units": 64, "activation": "relu", "dropout": 0.1},
                {"units": 128, "activation": "relu", "dropout": 0.2},
                {"units": 256, "activation": "relu", "dropout": 0.3},
            ]
        },
        "optimizer": {
            "name": "Adam",
            "momentum": 0.95,
            "lr_schedule": {
                "initial_lr": 0.001,
                "decay_rate": 0.96
            }
        },
        "dataset": {
            "path": "/data/imagenet",
            "batch_size": 64,
            "shuffle": True
        },
        "epochs": 100,
        "validate": True
    }
    
    # Single call factorizes ALL nested components!
    trainer = ComponentRepo.factorize_artifact("Trainer", **config)
    
    print("\nFactorized trainer:")
    print(trainer)
    print("\nNOTE: All nested objects (Model, Optimizer, LRSchedule, LayerConfig, Dataset)")
    print("      were automatically factorized from the dict!")
    print()


# ============================================================================
# Example 2: Factorize and Execute Pipeline
# ============================================================================

def example_pipeline_execution():
    print("=" * 70)
    print("Example 2: Factorize and Execute Training Pipeline")
    print("=" * 70)
    
    # Config for the entire pipeline
    pipeline_config = {
        "trainer": {
            "model": {
                "name": "MobileNet",
                "input_dim": 128,
                "layers": [
                    {"units": 32, "activation": "relu"},
                    {"units": 64, "activation": "relu"},
                ]
            },
            "optimizer": {
                "name": "SGD",
                "momentum": 0.9,
                "lr_schedule": {"initial_lr": 0.01, "decay_rate": 0.98}
            },
            "dataset": {
                "path": "/data/cifar10",
                "batch_size": 128
            },
            "epochs": 50
        },
        "checkpoint_dir": "/checkpoints/experiment_001"
    }
    
    # Factorize and execute!
    result = PipelineRepo.factorize_artifact("train", **pipeline_config)
    
    print("\nTraining result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
    print()


# ============================================================================
# Example 3: From Config File
# ============================================================================

def example_from_config_file():
    print("=" * 70)
    print("Example 3: Load Complete Pipeline from Config File")
    print("=" * 70)
    
    import json
    from pathlib import Path
    
    # Create a config file
    config = {
        "model": {
            "name": "EfficientNet",
            "input_dim": 256,
            "layers": [
                {"units": 128, "activation": "swish"},
                {"units": 256, "activation": "swish"},
            ]
        },
        "optimizer": {
            "name": "AdamW",
            "momentum": 0.99,
            "lr_schedule": {"initial_lr": 0.0001, "decay_rate": 0.99}
        },
        "dataset": {
            "path": "/data/custom_dataset",
            "batch_size": 32
        },
        "epochs": 200,
        "validate": True
    }
    
    config_path = Path("training_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"Created config file: {config_path}")
    print("\nConfig contents:")
    print(json.dumps(config, indent=2))
    
    # Factorize from file - everything is recursive!
    trainer = ComponentRepo.factorize_from_file("Trainer", config_path)
    
    print("\nFactorized trainer from file:")
    print(trainer)
    
    # Clean up
    config_path.unlink()
    print()


# ============================================================================
# Example 4: Partial Configs with Defaults
# ============================================================================

def example_partial_configs():
    print("=" * 70)
    print("Example 4: Partial Configs Use Default Values")
    print("=" * 70)
    
    # Minimal config - most values will use defaults
    minimal_config = {
        "model": {
            "name": "SimpleNet",
            "input_dim": 100,
            "layers": [{"units": 50}]  # Uses default activation and dropout
        },
        "optimizer": {
            "name": "Adam",
            "lr_schedule": {"initial_lr": 0.001}  # Uses default decay_rate
        },
        "dataset": {
            "path": "/data/simple"  # Uses default batch_size and shuffle
        }
    }
    
    trainer = ComponentRepo.factorize_artifact("Trainer", **minimal_config)
    
    print("\nFactorized with minimal config (lots of defaults):")
    print(trainer)
    print("\nNOTE: Missing values were filled with defaults from __init__ signatures")
    print()


# ============================================================================
# Example 5: Inspect Auto-generated Schemes
# ============================================================================

def example_scheme_inspection():
    print("=" * 70)
    print("Example 5: Inspect Auto-generated Pydantic Schemes")
    print("=" * 70)
    
    # Get the scheme for Trainer
    trainer_scheme = ComponentRepo._get_scheme("Trainer")
    
    print(f"Scheme for 'Trainer': {trainer_scheme.__name__}")
    print("\nFields and their types:")
    for field_name, field_info in trainer_scheme.model_fields.items():
        print(f"  {field_name}: {field_info.annotation}")
    
    print("\nJSON Schema:")
    import json
    schema = trainer_scheme.model_json_schema()
    print(json.dumps(schema, indent=2)[:500] + "...")  # Truncate for readability
    print()


# ============================================================================
# Run All Examples
# ============================================================================

if __name__ == "__main__":
    example_deep_nesting()
    example_pipeline_execution()
    example_from_config_file()
    example_partial_configs()
    example_scheme_inspection()
    
    print("=" * 70)
    print("All examples completed successfully!")
    print("=" * 70)
    print("\nKey takeaway: Just define your classes with type hints,")
    print("register them, and factorize from nested dicts/files.")
    print("The registry handles all the recursive instantiation!")
