#!/usr/bin/env python
"""Example 05: Full Experiment Configuration with Deep Nesting.

This example demonstrates a complete ML experiment configuration using the
registry pattern with deeply nested BuildCfg envelopes. It shows:

1. Multiple registries for different component types (models, transforms, etc.)
2. Deeply nested configurations (5+ levels deep)
3. Type coercion (strings to floats/ints)
4. Context injection for cross-references (e.g., optimizer referencing model params)
5. Extras handling (unknown fields -> meta)
6. Full training loop from configuration

This is an adaptation of reg_ex_pydantic.py to use the current registry package.

Requirements:
    pip install torch torchvision
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from registry import BuildCfg, ContainerMixin, FunctionalRegistry, TypeRegistry

# =============================================================================
# Registry Definitions
# =============================================================================


class ModelRegistry(TypeRegistry[object]):
    """Registry for neural network models."""

    pass


class TransformRegistry(FunctionalRegistry):
    """Registry for data transforms."""

    pass


class DatasetRegistry(FunctionalRegistry):
    """Registry for datasets."""

    pass


class DataLoaderRegistry(FunctionalRegistry):
    """Registry for data loaders."""

    pass


class OptimizerRegistry(FunctionalRegistry):
    """Registry for optimizers."""

    pass


class SchedulerRegistry(FunctionalRegistry):
    """Registry for learning rate schedulers."""

    pass


class LossRegistry(FunctionalRegistry):
    """Registry for loss functions."""

    pass


class UtilsRegistry(FunctionalRegistry):
    """Registry for utility functions."""

    pass


# Configure all repos
ContainerMixin.configure_repos(
    {
        "models": ModelRegistry,
        "transforms": TransformRegistry,
        "datasets": DatasetRegistry,
        "dataloaders": DataLoaderRegistry,
        "optimizers": OptimizerRegistry,
        "schedulers": SchedulerRegistry,
        "losses": LossRegistry,
        "utils": UtilsRegistry,
        "default": UtilsRegistry,
    }
)


# =============================================================================
# Parameter Schemas
# =============================================================================


class ParamsBase(BaseModel):
    """Base class for parameter schemas that allows extras."""

    model_config = ConfigDict(extra="allow")


class ResNet18Params(ParamsBase):
    num_classes: int = 10


class ComposeParams(ParamsBase):
    ops: list[Any]


class ToTensorParams(ParamsBase):
    pass


class NormalizeParams(ParamsBase):
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


class MNISTDatasetParams(ParamsBase):
    root: str
    train: bool
    download: bool
    transform: Optional[Any] = None


class DataLoaderParams(ParamsBase):
    dataset: Any
    batch_size: int
    shuffle: bool
    num_workers: int = 2
    pin_memory: bool = True
    drop_last: bool = False


class ModelParametersParams(ParamsBase):
    model_key: str


class RefParams(ParamsBase):
    key: str


class SGDParams(ParamsBase):
    params: Any
    lr: float
    momentum: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


class StepLRParams(ParamsBase):
    optimizer: Any
    step_size: int
    gamma: float = 0.1


class CrossEntropyParams(ParamsBase):
    pass


# =============================================================================
# Component Registration
# =============================================================================


# --- Models ---
@ModelRegistry.register_artifact(params_model=ResNet18Params)
class ResNet18MNIST(nn.Module):
    """ResNet18 adapted for MNIST."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.model = models.resnet18(weights=None, num_classes=num_classes)
        # Modify first conv for 1-channel input
        self.model.conv1 = nn.Conv2d(
            1, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.model.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# --- Transforms ---
@TransformRegistry.register_artifact(params_model=ComposeParams)
def compose(*, ops: list[Any]) -> transforms.Compose:
    """Compose multiple transforms."""
    return transforms.Compose(ops)


@TransformRegistry.register_artifact(params_model=ToTensorParams)
def to_tensor() -> transforms.ToTensor:
    """Convert to tensor."""
    return transforms.ToTensor()


@TransformRegistry.register_artifact(params_model=NormalizeParams)
def normalize(
    *, mean: Tuple[float, ...], std: Tuple[float, ...]
) -> transforms.Normalize:
    """Normalize with mean and std."""
    return transforms.Normalize(mean=mean, std=std)


# --- Datasets ---
@DatasetRegistry.register_artifact(params_model=MNISTDatasetParams)
def mnist(
    *, root: str, train: bool, download: bool, transform: Optional[Any] = None
) -> datasets.MNIST:
    """Load MNIST dataset."""
    return datasets.MNIST(
        root=root, train=train, download=download, transform=transform
    )


# --- DataLoaders ---
@DataLoaderRegistry.register_artifact(params_model=DataLoaderParams)
def torch_dataloader(
    *,
    dataset: Any,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader[Any]:
    """Create a PyTorch DataLoader."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )


# --- Utilities (context-aware) ---
@UtilsRegistry.register_artifact(params_model=ModelParametersParams)
def model_parameters(*, model_key: str, ctx: Dict[str, Any]) -> Any:
    """Get model parameters from context."""
    if model_key not in ctx:
        raise ValueError(
            f"model_key '{model_key}' not in context. Available: {list(ctx.keys())}"
        )
    model = ctx[model_key]
    if not isinstance(model, nn.Module):
        raise ValueError(f"ctx['{model_key}'] is not an nn.Module")
    return model.parameters()


@UtilsRegistry.register_artifact(params_model=RefParams)
def ref(*, key: str, ctx: Dict[str, Any]) -> Any:
    """Reference an object from context."""
    if key not in ctx:
        raise ValueError(
            f"ref key '{key}' not in context. Available: {list(ctx.keys())}"
        )
    return ctx[key]


# --- Optimizers ---
@OptimizerRegistry.register_artifact(params_model=SGDParams)
def sgd(
    *,
    params: Any,
    lr: float,
    momentum: float = 0.0,
    weight_decay: float = 0.0,
    nesterov: bool = False,
) -> torch.optim.SGD:
    """Create SGD optimizer."""
    return torch.optim.SGD(
        params=params,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )


# --- Schedulers ---
@SchedulerRegistry.register_artifact(params_model=StepLRParams)
def steplr(
    *, optimizer: Any, step_size: int, gamma: float = 0.1
) -> torch.optim.lr_scheduler.StepLR:
    """Create StepLR scheduler."""
    return torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer, step_size=step_size, gamma=gamma
    )


# --- Losses ---
@LossRegistry.register_artifact(params_model=CrossEntropyParams)
def cross_entropy() -> nn.CrossEntropyLoss:
    """Create CrossEntropyLoss."""
    return nn.CrossEntropyLoss()


# =============================================================================
# Training Settings
# =============================================================================


class TrainSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = 2
    device: str = Field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    log_every: int = 200


class ExperimentConfig(BaseModel):
    """Full experiment configuration."""

    model_config = ConfigDict(extra="forbid")

    model: BuildCfg
    train_loader: BuildCfg
    test_loader: BuildCfg
    criterion: BuildCfg
    optimizer: BuildCfg
    scheduler: Optional[BuildCfg] = None
    train: TrainSettings = Field(default_factory=TrainSettings)


# =============================================================================
# Training Functions
# =============================================================================


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_every: int,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0

    for it, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        seen += bs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits.detach(), y) * bs

        if log_every > 0 and it % log_every == 0:
            print(
                f"epoch={epoch} iter={it} "
                f"loss={total_loss / seen:.4f} acc={total_acc / seen:.4f} "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

    return total_loss / max(seen, 1), total_acc / max(seen, 1)


@torch.no_grad()
def validate(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    seen = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        seen += bs
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs

    return total_loss / max(seen, 1), total_acc / max(seen, 1)


# =============================================================================
# Example Configuration (with string -> number coercion)
# =============================================================================

RAW_CONFIG: Dict[str, Any] = {
    "model": {
        "repo": "models",
        "type": "ResNet18MNIST",
        "data": {
            "num_classes": "10",  # String will be coerced to int
            "unused_flag_goes_to_meta": True,  # Extra field -> meta
        },
        "meta": {"tag": "resnet18_mnist_baseline"},
    },
    "train_loader": {
        "repo": "dataloaders",
        "type": "torch_dataloader",
        "data": {
            "dataset": {
                "repo": "datasets",
                "type": "mnist",
                "data": {
                    "root": "./data",
                    "train": True,
                    "download": True,
                    "transform": {
                        "repo": "transforms",
                        "type": "compose",
                        "data": {
                            "ops": [
                                {"repo": "transforms", "type": "to_tensor", "data": {}},
                                {
                                    "repo": "transforms",
                                    "type": "normalize",
                                    "data": {
                                        "mean": ["0.1307"],  # String -> float coercion
                                        "std": ["0.3081"],
                                    },
                                },
                            ]
                        },
                    },
                },
            },
            "batch_size": "128",  # String -> int coercion
            "shuffle": True,
            "num_workers": "2",
            "pin_memory": True,
        },
    },
    "test_loader": {
        "repo": "dataloaders",
        "type": "torch_dataloader",
        "data": {
            "dataset": {
                "repo": "datasets",
                "type": "mnist",
                "data": {
                    "root": "./data",
                    "train": False,
                    "download": True,
                    "transform": {
                        "repo": "transforms",
                        "type": "compose",
                        "data": {
                            "ops": [
                                {"repo": "transforms", "type": "to_tensor", "data": {}},
                                {
                                    "repo": "transforms",
                                    "type": "normalize",
                                    "data": {"mean": ["0.1307"], "std": ["0.3081"]},
                                },
                            ]
                        },
                    },
                },
            },
            "batch_size": "256",
            "shuffle": False,
            "num_workers": "2",
            "pin_memory": True,
        },
    },
    "criterion": {"repo": "losses", "type": "cross_entropy", "data": {}},
    "optimizer": {
        "repo": "optimizers",
        "type": "sgd",
        "data": {
            "params": {
                "repo": "utils",
                "type": "model_parameters",
                "data": {"model_key": "model"},  # References model from context
            },
            "lr": "0.1",  # String -> float coercion
            "momentum": "0.9",
            "weight_decay": "5e-4",
            "nesterov": True,
            "this_key_is_not_in_schema": 123,  # Extra field -> meta
        },
    },
    "scheduler": {
        "repo": "schedulers",
        "type": "steplr",
        "data": {
            "optimizer": {
                "repo": "utils",
                "type": "ref",
                "data": {"key": "optimizer"},  # References optimizer from context
            },
            "step_size": "5",
            "gamma": "0.1",
        },
    },
    "train": {
        "epochs": "2",  # String -> int coercion
        "log_every": "200",
    },
}


def main() -> None:
    print("=" * 70)
    print("Full Experiment Configuration with Deep Nesting")
    print("=" * 70)

    # Parse and validate configuration
    print("\nParsing configuration...")
    cfg = ExperimentConfig.model_validate(RAW_CONFIG)

    # Clear context for fresh run
    ContainerMixin.clear_context()

    # Build components in order (respecting dependencies)
    print("\nBuilding components from nested configuration...")

    # Build and store model in context (so optimizer can reference it)
    model: nn.Module = ContainerMixin.build_named("model", cfg.model)
    print(f"  Model: {type(model).__name__}")
    print(f"    Meta: {getattr(model, '__meta__', {})}")

    # Build data loaders
    train_loader: DataLoader[Any] = ContainerMixin.build_named(
        "train_loader", cfg.train_loader
    )
    test_loader: DataLoader[Any] = ContainerMixin.build_named(
        "test_loader", cfg.test_loader
    )
    print(f"  Train loader: {len(train_loader)} batches")
    print(f"  Test loader: {len(test_loader)} batches")

    # Build criterion
    criterion: nn.Module = ContainerMixin.build_named("criterion", cfg.criterion)
    print(f"  Criterion: {type(criterion).__name__}")

    # Build optimizer (references model.parameters() via context)
    optimizer: torch.optim.Optimizer = ContainerMixin.build_named(
        "optimizer", cfg.optimizer
    )
    print(f"  Optimizer: {type(optimizer).__name__}")
    print(f"    LR: {optimizer.param_groups[0]['lr']}")
    print(f"    Meta: {getattr(optimizer, '__meta__', {})}")

    # Build scheduler (references optimizer via context)
    scheduler = None
    if cfg.scheduler:
        scheduler = ContainerMixin.build_named("scheduler", cfg.scheduler)
        print(f"  Scheduler: {type(scheduler).__name__}")

    # Show coercion evidence
    print("\n" + "-" * 70)
    print("Type coercion evidence:")
    print(
        f"  cfg.optimizer.data['lr'] = {cfg.optimizer.data.get('lr')} "
        f"(type: {type(cfg.optimizer.data.get('lr')).__name__})"
    )
    print(
        f"  cfg.train.epochs = {cfg.train.epochs} (type: {type(cfg.train.epochs).__name__})"
    )

    # Training
    device = torch.device(cfg.train.device)
    model.to(device)
    criterion.to(device)

    print("\n" + "=" * 70)
    print(f"Starting training on {device}...")
    print("=" * 70)

    for epoch in range(1, cfg.train.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            log_every=cfg.train.log_every,
        )

        if scheduler is not None:
            scheduler.step()

        va_loss, va_acc = validate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f"\nEpoch {epoch}: "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"Final model meta: {getattr(model, '__meta__', {})}")
    print("=" * 70)


if __name__ == "__main__":
    main()
