#!/usr/bin/env python
"""Example 04: PyTorch ResNet18 Training on MNIST with DI Container.

This example demonstrates a real-world use case of the DI Container pattern
for configuring and training a ResNet18 model on MNIST.

Features demonstrated:
1. Model registry with PyTorch nn.Module classes
2. Optimizer and scheduler registries
3. Transform/augmentation registries
4. Full training loop with configuration-driven setup
5. Nested configurations for complex experiments

Requirements:
    pip install torch torchvision

Usage:
    python 04_pytorch_mnist.py
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel, Field
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from registry import BuildCfg, ContainerMixin, FunctionalRegistry, TypeRegistry

# =============================================================================
# Registry Definitions
# =============================================================================


class ModelRegistry(TypeRegistry[object]):
    """Registry for neural network models."""

    pass


class OptimizerRegistry(TypeRegistry[object]):
    """Registry for optimizers."""

    pass


class SchedulerRegistry(TypeRegistry[object]):
    """Registry for learning rate schedulers."""

    pass


class TransformRegistry(FunctionalRegistry):
    """Registry for data transforms."""

    pass


class DatasetRegistry(TypeRegistry[object]):
    """Registry for datasets."""

    pass


class TrainerRegistry(TypeRegistry[object]):
    """Registry for training loops."""

    pass


# Configure repos
ContainerMixin.configure_repos(
    {
        "models": ModelRegistry,
        "optimizers": OptimizerRegistry,
        "schedulers": SchedulerRegistry,
        "transforms": TransformRegistry,
        "datasets": DatasetRegistry,
        "trainers": TrainerRegistry,
        "default": ModelRegistry,
    }
)


# =============================================================================
# Model Definitions
# =============================================================================


class BasicBlock(nn.Module):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18Params(BaseModel):
    """ResNet18 Parameters Schema."""

    num_classes: int = Field(default=10, ge=1)
    in_channels: int = Field(default=1, ge=1)


@ModelRegistry.register_artifact(params_model=ResNet18Params)
class ResNet18(nn.Module):
    """ResNet18 adapted for MNIST (28x28 grayscale images)."""

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super().__init__()
        self.in_planes = 64

        # Modified first layer for 28x28 input (smaller kernel, no maxpool)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class SimpleCNNParams(BaseModel):
    """Simple CNN Parameters Schema."""

    num_classes: int = Field(default=10, ge=1)
    dropout: float = Field(default=0.5, ge=0, le=1)


@ModelRegistry.register_artifact(params_model=SimpleCNNParams)
class SimpleCNN(nn.Module):
    """Simple CNN for MNIST."""

    def __init__(self, num_classes: int = 10, dropout: float = 0.5):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


# =============================================================================
# Optimizer Definitions
# =============================================================================


class SGDParams(BaseModel):
    """SGD Parameters Schema."""

    lr: float = Field(default=0.01, ge=0)
    momentum: float = Field(default=0.9, ge=0, le=1)
    weight_decay: float = Field(default=0, ge=0)


@OptimizerRegistry.register_artifact(params_model=SGDParams)
class SGDOptimizer:
    """Wrapper to create SGD optimizer."""

    def __init__(
        self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0
    ):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

    def create(self, model: nn.Module) -> torch.optim.SGD:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )

    def __repr__(self) -> str:
        return f"SGDOptimizer(lr={self.lr}, momentum={self.momentum})"


class AdamParams(BaseModel):
    """Adam Parameters Schema."""

    lr: float = Field(default=0.001, ge=0)
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = Field(default=0, ge=0)


@OptimizerRegistry.register_artifact(params_model=AdamParams)
class AdamOptimizer:
    """Wrapper to create Adam optimizer."""

    def __init__(
        self,
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0,
    ):
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay

    def create(self, model: nn.Module) -> torch.optim.Adam:
        return torch.optim.Adam(
            model.parameters(),
            lr=self.lr,
            betas=self.betas,
            weight_decay=self.weight_decay,
        )

    def __repr__(self) -> str:
        return f"AdamOptimizer(lr={self.lr})"


# =============================================================================
# Scheduler Definitions
# =============================================================================


@SchedulerRegistry.register_artifact
class StepLRScheduler:
    """Wrapper to create StepLR scheduler."""

    def __init__(self, step_size: int = 10, gamma: float = 0.1):
        self.step_size = step_size
        self.gamma = gamma

    def create(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.StepLR:
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.step_size, gamma=self.gamma
        )

    def __repr__(self) -> str:
        return f"StepLRScheduler(step_size={self.step_size}, gamma={self.gamma})"


@SchedulerRegistry.register_artifact
class CosineScheduler:
    """Wrapper to create CosineAnnealingLR scheduler."""

    def __init__(self, T_max: int = 10, eta_min: float = 0):
        self.T_max = T_max
        self.eta_min = eta_min

    def create(
        self, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler.CosineAnnealingLR:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.T_max, eta_min=self.eta_min
        )

    def __repr__(self) -> str:
        return f"CosineScheduler(T_max={self.T_max})"


# =============================================================================
# Transform Definitions
# =============================================================================


@TransformRegistry.register_artifact
def mnist_train_transform() -> transforms.Compose:
    """Standard MNIST training transforms."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


@TransformRegistry.register_artifact
def mnist_test_transform() -> transforms.Compose:
    """Standard MNIST test transforms."""
    return transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )


@TransformRegistry.register_artifact
def mnist_augmented_transform() -> transforms.Compose:
    """MNIST training transforms with augmentation."""
    return transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )


# =============================================================================
# Trainer Definition
# =============================================================================


@TrainerRegistry.register_artifact
class MNISTTrainer:
    """Complete MNIST training pipeline."""

    def __init__(
        self,
        model_cfg: Any,
        optimizer_cfg: Any,
        scheduler_cfg: Any = None,
        epochs: int = 5,
        batch_size: int = 64,
        log_interval: int = 100,
        data_path: str = "./data",
        ctx: Optional[dict[str, Any]] = None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_interval = log_interval
        self.data_path = data_path
        self.ctx = ctx or {}

        # Store config objects
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        # These will be initialized in setup()
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Any = None
        self.train_loader: Optional[DataLoader[Any]] = None
        self.test_loader: Optional[DataLoader[Any]] = None
        self.device: Optional[torch.device] = None

    def setup(self) -> None:
        """Initialize all components."""
        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = self.model_cfg
        if isinstance(self.model_cfg, nn.Module):
            self.model = self.model_cfg
        assert self.model is not None
        self.model = self.model.to(self.device)
        print(f"Model: {type(self.model).__name__}")

        # Initialize optimizer
        self.optimizer = self.optimizer_cfg.create(self.model)
        print(f"Optimizer: {self.optimizer_cfg}")

        # Initialize scheduler
        if self.scheduler_cfg:
            self.scheduler = self.scheduler_cfg.create(self.optimizer)
            print(f"Scheduler: {self.scheduler_cfg}")

        # Load data
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            self.data_path, train=True, download=True, transform=train_transform
        )
        test_dataset = datasets.MNIST(
            self.data_path, train=False, transform=test_transform
        )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
        print(f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test")

    def train_epoch(self, epoch: int) -> None:
        """Train for one epoch."""
        assert self.model is not None
        assert self.optimizer is not None
        assert self.train_loader is not None
        assert self.device is not None

        self.model.train()
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()

            if batch_idx % self.log_interval == 0:
                print(
                    f"Epoch {epoch} [{batch_idx * len(data)}/{len(self.train_loader.dataset)} "  # type: ignore[arg-type]
                    f"({100.0 * batch_idx / len(self.train_loader):.0f}%)] Loss: {loss.item():.6f}"
                )

    def test(self) -> tuple[float, float]:
        """Evaluate on test set."""
        assert self.model is not None
        assert self.test_loader is not None
        assert self.device is not None

        self.model.eval()
        test_loss = 0.0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.cross_entropy(output, target, reduction="sum").item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        dataset_len = len(self.test_loader.dataset)  # type: ignore[arg-type]
        test_loss /= dataset_len
        accuracy = 100.0 * correct / dataset_len

        print(
            f"\nTest set: Average loss: {test_loss:.4f}, "
            f"Accuracy: {correct}/{dataset_len} ({accuracy:.2f}%)\n"
        )
        return test_loss, accuracy

    def run(self) -> float:
        """Run full training loop."""
        self.setup()

        assert self.optimizer is not None

        best_accuracy = 0.0
        for epoch in range(1, self.epochs + 1):
            print(f"\n{'=' * 50}")
            print(f"Epoch {epoch}/{self.epochs}")
            print("=" * 50)

            self.train_epoch(epoch)
            _, accuracy = self.test()

            if self.scheduler:
                self.scheduler.step()
                print(f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f"New best accuracy: {best_accuracy:.2f}%")

        print(f"\nTraining complete! Best accuracy: {best_accuracy:.2f}%")
        return best_accuracy

    def __repr__(self) -> str:
        return (
            f"MNISTTrainer(\n"
            f"  model={type(self.model_cfg).__name__},\n"
            f"  optimizer={self.optimizer_cfg},\n"
            f"  scheduler={self.scheduler_cfg},\n"
            f"  epochs={self.epochs},\n"
            f"  batch_size={self.batch_size}\n"
            f")"
        )


# =============================================================================
# Main: Build and Run Training from Configuration
# =============================================================================


def main() -> None:
    print("=" * 60)
    print("PyTorch MNIST Training with DI Container")
    print("=" * 60)

    # Clear context for fresh run
    ContainerMixin.clear_context()

    # Print registered components
    print("\nRegistered components:")
    print(f"  Models: {list(ModelRegistry.iter_identifiers())}")
    print(f"  Optimizers: {list(OptimizerRegistry.iter_identifiers())}")
    print(f"  Schedulers: {list(SchedulerRegistry.iter_identifiers())}")

    # Configuration for ResNet18 experiment
    experiment_config = BuildCfg(
        type="MNISTTrainer",
        repo="trainers",
        data={
            "model_cfg": BuildCfg(
                type="ResNet18",
                repo="models",
                data={"num_classes": 10, "in_channels": 1},
            ),
            "optimizer_cfg": BuildCfg(
                type="AdamOptimizer",
                repo="optimizers",
                data={"lr": 0.001, "weight_decay": 1e-4},
            ),
            "scheduler_cfg": BuildCfg(
                type="CosineScheduler",
                repo="schedulers",
                data={"T_max": 5, "eta_min": 1e-6},
            ),
            "epochs": 5,
            "batch_size": 64,
            "log_interval": 100,
        },
        meta={"experiment_name": "resnet18_mnist", "version": "1.0"},
    )

    print("\n" + "=" * 60)
    print("Building trainer from configuration...")
    print("=" * 60)

    # Build the trainer
    trainer = ContainerMixin.build_cfg(experiment_config)
    print(f"\nBuilt: {trainer}")
    print(f"Meta: {getattr(trainer, '__meta__', {})}")

    # Run training
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    final_accuracy = trainer.run()

    print("\n" + "=" * 60)
    print(f"Training complete! Final accuracy: {final_accuracy:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
