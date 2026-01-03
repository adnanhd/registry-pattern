#!/usr/bin/env python
"""Example 01: Registry Pattern Basics.

The Registry Pattern provides a central location for storing and retrieving
artifacts (classes, functions) by name. This enables:

- Decoupling: Code that uses artifacts doesn't need to know their implementations
- Extensibility: New artifacts can be added without modifying existing code
- Configuration-driven: Artifacts can be selected by name from config files

This example demonstrates:
1. Creating registries for classes and functions
2. Registering artifacts with decorators
3. Retrieving and using registered artifacts
4. Strict mode with protocol checking
"""

from __future__ import annotations

from registry import FunctionalRegistry, TypeRegistry

# =============================================================================
# Part 1: Basic Class Registry
# =============================================================================

print("=" * 60)
print("Part 1: Basic Class Registry")
print("=" * 60)


# Create a registry for model classes
class ModelRegistry(TypeRegistry[object]):
    """Registry for ML model classes."""

    pass

    pass


# Register classes using decorator
@ModelRegistry.register_artifact
class LinearModel:
    """A simple linear model."""

    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x):
        return f"LinearModel({self.input_dim} -> {self.output_dim})"


@ModelRegistry.register_artifact
class MLPModel:
    """A multi-layer perceptron."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

    def forward(self, x):
        return f"MLP({self.input_dim} -> {self.hidden_dim} -> {self.output_dim})"


# List registered models
print("\nRegistered models:")
for name in ModelRegistry.iter_identifiers():
    print(f"  - {name}")

# Retrieve and instantiate
model_cls = ModelRegistry.get_artifact("LinearModel")
model = model_cls(input_dim=784, output_dim=10)
print(f"\nCreated: {model.forward(None)}")

# Check existence
print(f"\nHas 'LinearModel': {ModelRegistry.has_identifier('LinearModel')}")
print(f"Has 'ConvModel': {ModelRegistry.has_identifier('ConvModel')}")


# =============================================================================
# Part 2: Function Registry
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Function Registry")
print("=" * 60)


class ActivationRegistry(FunctionalRegistry):
    """Registry for activation functions."""

    pass


@ActivationRegistry.register_artifact
def relu(x: float) -> float:
    """ReLU activation."""
    return max(0.0, x)


@ActivationRegistry.register_artifact
def sigmoid(x: float) -> float:
    """Sigmoid activation."""
    import math

    return 1.0 / (1.0 + math.exp(-x))


@ActivationRegistry.register_artifact
def tanh(x: float) -> float:
    """Tanh activation."""
    import math

    return math.tanh(x)


# Use registered functions
print("\nRegistered activations:")
for name in ActivationRegistry.iter_identifiers():
    fn = ActivationRegistry.get_artifact(name)
    print(f"  - {name}: {name}(0.5) = {fn(0.5):.4f}")


# =============================================================================
# Part 3: Error Handling and Validation
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Error Handling and Validation")
print("=" * 60)


# Try to get non-existent artifact
print("\nTrying to get non-existent model...")
try:
    ModelRegistry.get_artifact("NonExistentModel")
except Exception as e:
    print(f"  Error: {type(e).__name__}")
    print(f"  Message: {str(e)[:80]}...")


# Try to register a duplicate
print("\nTrying to register duplicate model...")
try:

    @ModelRegistry.register_artifact
    class MLPModel:  # Already registered above
        pass
except Exception as e:
    print(f"  Error: {type(e).__name__}")
    print(f"  Duplicates are prevented by default")


# =============================================================================
# Part 4: Unregistration
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Unregistration")
print("=" * 60)

print(f"\nBefore: {list(ModelRegistry.iter_identifiers())}")

ModelRegistry.unregister_identifier("LinearModel")
print(f"After removing LinearModel: {list(ModelRegistry.iter_identifiers())}")


print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
