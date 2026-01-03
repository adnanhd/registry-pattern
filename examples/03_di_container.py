#!/usr/bin/env python
"""Example 03: Dependency Injection Container with Nested Object Graphs.

The DI Container Pattern extends the Factory Pattern by adding:

- Recursive building of nested configurations
- Context injection for cross-references between objects
- Named object storage in shared context
- Complex object graph construction

This example demonstrates:
1. Nested BuildCfg resolution (objects containing objects)
2. Context injection via the `ctx` parameter
3. Named object references with build_named()
4. Complex multi-layer object graphs
5. Clear and reset of context between builds
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field

from registry import BuildCfg, ContainerMixin, TypeRegistry

# =============================================================================
# Part 1: Define Registries and Models
# =============================================================================

print("=" * 60)
print("Part 1: Setting Up Registries")
print("=" * 60)


class OptimizerRegistry(TypeRegistry[object]):
    """Registry for optimizers."""

    pass


class SchedulerRegistry(TypeRegistry[object]):
    """Registry for learning rate schedulers."""

    pass


class ModelRegistry(TypeRegistry[object]):
    """Registry for neural network models."""

    pass


class TrainerRegistry(TypeRegistry[object]):
    """Registry for training loops."""

    pass


# Configure all repos
ContainerMixin.configure_repos(
    {
        "optimizers": OptimizerRegistry,
        "schedulers": SchedulerRegistry,
        "models": ModelRegistry,
        "trainers": TrainerRegistry,
        "default": ModelRegistry,
    }
)


# =============================================================================
# Part 2: Register Components
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Registering Components")
print("=" * 60)


# --- Optimizers ---
class OptimizerParams(BaseModel):
    lr: float = Field(ge=0, default=0.001)
    weight_decay: float = Field(ge=0, default=0.0)


@OptimizerRegistry.register_artifact(params_model=OptimizerParams)
class SGD:
    def __init__(self, lr: float = 0.001, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def __repr__(self):
        return f"SGD(lr={self.lr}, weight_decay={self.weight_decay})"


class AdamParams(BaseModel):
    lr: float = Field(ge=0, default=0.001)
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@OptimizerRegistry.register_artifact(params_model=AdamParams)
class Adam:
    def __init__(
        self, lr: float = 0.001, betas: tuple = (0.9, 0.999), eps: float = 1e-8
    ):
        self.lr = lr
        self.betas = betas
        self.eps = eps

    def __repr__(self):
        return f"Adam(lr={self.lr}, betas={self.betas})"


# --- Schedulers ---
@SchedulerRegistry.register_artifact
class StepLR:
    def __init__(self, step_size: int = 10, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma

    def __repr__(self):
        return f"StepLR(step_size={self.step_size}, gamma={self.gamma})"


@SchedulerRegistry.register_artifact
class CosineAnnealingLR:
    def __init__(self, T_max: int = 100, eta_min: float = 0.0):
        self.T_max = T_max
        self.eta_min = eta_min

    def __repr__(self):
        return f"CosineAnnealingLR(T_max={self.T_max})"


# --- Models ---
@ModelRegistry.register_artifact
class MLP:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def __repr__(self):
        return f"MLP({self.input_dim} -> {self.hidden_dim} -> {self.output_dim})"


@ModelRegistry.register_artifact
class CNN:
    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 32):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels

    def __repr__(self):
        return f"CNN(in={self.in_channels}, out={self.num_classes}, base={self.base_channels})"


print("Registered:")
print(f"  Optimizers: {list(OptimizerRegistry.iter_identifiers())}")
print(f"  Schedulers: {list(SchedulerRegistry.iter_identifiers())}")
print(f"  Models: {list(ModelRegistry.iter_identifiers())}")


# =============================================================================
# Part 3: Nested Configuration Building
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Nested Configuration Building")
print("=" * 60)


@TrainerRegistry.register_artifact
class SimpleTrainer:
    """Trainer that receives nested objects."""

    def __init__(self, model: Any, optimizer: Any, scheduler: Optional[Any] = None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def __repr__(self):
        parts = [
            f"SimpleTrainer(",
            f"  model={self.model},",
            f"  optimizer={self.optimizer},",
            f"  scheduler={self.scheduler}",
            f")",
        ]
        return "\n".join(parts)


# Build with nested configurations
ContainerMixin.clear_context()

trainer_cfg = BuildCfg(
    type="SimpleTrainer",
    repo="trainers",
    data={
        # Nested BuildCfg for model
        "model": BuildCfg(
            type="MLP",
            repo="models",
            data={"input_dim": 784, "hidden_dim": 256, "output_dim": 10},
        ),
        # Nested BuildCfg for optimizer
        "optimizer": BuildCfg(
            type="Adam", repo="optimizers", data={"lr": 0.001, "betas": (0.9, 0.999)}
        ),
        # Nested BuildCfg for scheduler
        "scheduler": BuildCfg(
            type="StepLR", repo="schedulers", data={"step_size": 5, "gamma": 0.5}
        ),
    },
    meta={"experiment": "nested_test"},
)

trainer = ContainerMixin.build_cfg(trainer_cfg)
print(f"\nBuilt trainer with nested objects:")
print(trainer)
print(f"\nTrainer meta: {getattr(trainer, '__meta__', {})}")


# =============================================================================
# Part 4: Context Injection
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Context Injection")
print("=" * 60)


@ModelRegistry.register_artifact
class ContextAwareModel:
    """Model that receives the shared context."""

    def __init__(self, hidden_dim: int, ctx: dict = None):
        self.hidden_dim = hidden_dim
        self.ctx = ctx or {}

        # Access shared objects from context
        if "shared_config" in self.ctx:
            print(f"  -> Received shared_config from ctx: {self.ctx['shared_config']}")
        if "base_model" in self.ctx:
            print(f"  -> Received base_model from ctx: {self.ctx['base_model']}")

    def __repr__(self):
        ctx_keys = list(self.ctx.keys()) if self.ctx else []
        return f"ContextAwareModel(hidden={self.hidden_dim}, ctx_keys={ctx_keys})"


# Set up shared context
ContainerMixin.clear_context()
ContainerMixin._ctx["shared_config"] = {"dropout": 0.1, "batch_size": 32}

# First, build a base model and store it in context
base_model_cfg = BuildCfg(
    type="MLP",
    repo="models",
    data={"input_dim": 100, "hidden_dim": 50, "output_dim": 10},
)
ContainerMixin.build_named("base_model", base_model_cfg)
print(f"Built and stored 'base_model' in context")

# Now build a context-aware model that can access the shared context
aware_cfg = BuildCfg(type="ContextAwareModel", repo="models", data={"hidden_dim": 128})
aware_model = ContainerMixin.build_cfg(aware_cfg)
print(f"\nBuilt: {aware_model}")


# =============================================================================
# Part 5: Named Object References
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Named Object References")
print("=" * 60)


@TrainerRegistry.register_artifact
class MultiModelTrainer:
    """Trainer that uses context to reference named objects."""

    def __init__(self, main_model: Any, ctx: dict = None):
        self.main_model = main_model
        self.ctx = ctx or {}

        # Access named objects from context
        self.encoder = self.ctx.get("encoder")
        self.decoder = self.ctx.get("decoder")

    def __repr__(self):
        return (
            f"MultiModelTrainer(\n"
            f"  main={self.main_model},\n"
            f"  encoder={self.encoder},\n"
            f"  decoder={self.decoder}\n"
            f")"
        )


# Clear and set up fresh context
ContainerMixin.clear_context()

# Build and name encoder
ContainerMixin.build_named(
    "encoder",
    BuildCfg(
        type="MLP",
        repo="models",
        data={"input_dim": 784, "hidden_dim": 256, "output_dim": 64},
    ),
)

# Build and name decoder
ContainerMixin.build_named(
    "decoder",
    BuildCfg(
        type="MLP",
        repo="models",
        data={"input_dim": 64, "hidden_dim": 256, "output_dim": 784},
    ),
)

# Build trainer that references named objects via ctx
trainer_cfg = BuildCfg(
    type="MultiModelTrainer",
    repo="trainers",
    data={
        "main_model": BuildCfg(
            type="CNN", repo="models", data={"in_channels": 1, "num_classes": 10}
        )
    },
)

multi_trainer = ContainerMixin.build_cfg(trainer_cfg)
print(f"\nBuilt multi-model trainer:")
print(multi_trainer)


# =============================================================================
# Part 6: Complex Object Graphs
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Complex Object Graphs")
print("=" * 60)


@TrainerRegistry.register_artifact
class ExperimentRunner:
    """Complex runner with deeply nested configuration."""

    def __init__(self, name: str, trainer: Any, epochs: int = 10, ctx: dict = None):
        self.name = name
        self.trainer = trainer
        self.epochs = epochs
        self.ctx = ctx or {}

    def describe(self):
        """Describe the full object graph."""
        lines = [
            f"Experiment: {self.name}",
            f"  Epochs: {self.epochs}",
            f"  Trainer: {type(self.trainer).__name__}",
            f"    Model: {self.trainer.model}",
            f"    Optimizer: {self.trainer.optimizer}",
            f"    Scheduler: {self.trainer.scheduler}",
            f"  Context keys: {list(self.ctx.keys())}",
        ]
        return "\n".join(lines)


# Build a complex nested experiment
ContainerMixin.clear_context()
ContainerMixin._ctx["run_id"] = "exp_001"
ContainerMixin._ctx["device"] = "cuda:0"

experiment_cfg = BuildCfg(
    type="ExperimentRunner",
    repo="trainers",
    data={
        "name": "MNIST Classification",
        "epochs": 50,
        "trainer": BuildCfg(
            type="SimpleTrainer",
            repo="trainers",
            data={
                "model": BuildCfg(
                    type="CNN",
                    repo="models",
                    data={"in_channels": 1, "num_classes": 10, "base_channels": 64},
                ),
                "optimizer": BuildCfg(
                    type="Adam",
                    repo="optimizers",
                    data={"lr": 0.0003, "weight_decay": 0.0001},
                ),
                "scheduler": BuildCfg(
                    type="CosineAnnealingLR",
                    repo="schedulers",
                    data={"T_max": 50, "eta_min": 1e-6},
                ),
            },
        ),
    },
    meta={"author": "researcher", "tags": ["mnist", "cnn", "baseline"]},
)

experiment = ContainerMixin.build_cfg(experiment_cfg)
print(f"\nBuilt complex experiment:")
print(experiment.describe())
print(f"\nExperiment meta: {getattr(experiment, '__meta__', {})}")


# =============================================================================
# Part 7: Dictionary-Based Configuration
# =============================================================================

print("\n" + "=" * 60)
print("Part 7: Dictionary-Based Configuration")
print("=" * 60)

# Configurations can also be plain dicts that get converted to BuildCfg
ContainerMixin.clear_context()

# Using dict syntax (will be converted internally)
config_dict = {
    "type": "SimpleTrainer",
    "repo": "trainers",
    "data": {
        "model": {
            "type": "MLP",
            "repo": "models",
            "data": {"input_dim": 100, "hidden_dim": 64, "output_dim": 10},
        },
        "optimizer": {
            "type": "SGD",
            "repo": "optimizers",
            "data": {"lr": 0.01, "weight_decay": 0.0001},
        },
    },
    "meta": {"source": "config_file"},
}

# Convert dict to BuildCfg
cfg = BuildCfg.model_validate(config_dict)
trainer = ContainerMixin.build_cfg(cfg)
print(f"\nBuilt from dict config:")
print(trainer)


print("\n" + "=" * 60)
print("Done! DI Container demonstration complete.")
print("=" * 60)
