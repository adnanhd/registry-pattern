#!/usr/bin/env python
"""Example 02: Factory Pattern with Pydantic Validation.

The Factory Pattern builds on the Registry Pattern by adding:

- Configuration-driven instantiation via BuildCfg
- Parameter validation via Pydantic schemas
- Type coercion (e.g., "10" -> 10)
- Extras handling (unknown fields -> meta)

This example demonstrates:
1. Building objects from configuration
2. Auto-extraction of parameter schemas
3. Explicit parameter schemas with validation
4. Type coercion
5. Handling of extra/unknown fields
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from registry import BuildCfg, ContainerMixin, FunctionalRegistry, TypeRegistry

# =============================================================================
# Part 1: Basic Factory Usage
# =============================================================================

print("=" * 60)
print("Part 1: Basic Factory Usage")
print("=" * 60)


class ModelRegistry(TypeRegistry[object]):
    """Registry for model classes."""

    pass


@ModelRegistry.register_artifact
class SimpleModel:
    """A simple model with basic parameters."""

    def __init__(self, hidden_size: int, num_layers: int = 2):
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def __repr__(self):
        return (
            f"SimpleModel(hidden_size={self.hidden_size}, num_layers={self.num_layers})"
        )


# Configure the container
ContainerMixin.configure_repos({"models": ModelRegistry, "default": ModelRegistry})

# Build from configuration
cfg = BuildCfg(
    type="SimpleModel", repo="models", data={"hidden_size": 256, "num_layers": 4}
)
model = ContainerMixin.build_cfg(cfg)
print(f"\nBuilt model: {model}")

# Using default values
cfg_defaults = BuildCfg(
    type="SimpleModel",
    data={"hidden_size": 128},  # num_layers uses default
)
model_defaults = ContainerMixin.build_cfg(cfg_defaults)
print(f"With defaults: {model_defaults}")


# =============================================================================
# Part 2: Auto-Extracted Parameter Schema
# =============================================================================

print("\n" + "=" * 60)
print("Part 2: Auto-Extracted Parameter Schema")
print("=" * 60)


@ModelRegistry.register_artifact
class AutoSchemaModel:
    """Model with auto-extracted schema from __init__."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.dropout = dropout

    def __repr__(self):
        return (
            f"AutoSchemaModel(in={self.input_dim}, out={self.output_dim}, "
            f"act={self.activation}, drop={self.dropout})"
        )


# Check auto-extracted schema
schema = ModelRegistry._get_scheme("AutoSchemaModel")
assert schema is not None, "Schema should be auto-extracted"
print(f"\nAuto-extracted schema fields:")
for name, field in schema.model_fields.items():
    default = field.default if field.default is not None else "required"
    print(f"  - {name}: {field.annotation} = {default}")

# Build with auto-schema
cfg = BuildCfg(
    type="AutoSchemaModel",
    data={"input_dim": 784, "output_dim": 10, "activation": "gelu"},
)
model = ContainerMixin.build_cfg(cfg)
print(f"\nBuilt: {model}")


# =============================================================================
# Part 3: Explicit Parameter Schema with Validation
# =============================================================================

print("\n" + "=" * 60)
print("Part 3: Explicit Parameter Schema with Validation")
print("=" * 60)


class ValidatedModelParams(BaseModel):
    """Explicit schema with validation constraints."""

    hidden_size: int = Field(ge=1, le=4096, description="Hidden layer size")
    num_heads: int = Field(
        ge=1, le=64, default=8, description="Number of attention heads"
    )
    dropout: float = Field(ge=0.0, le=1.0, default=0.1, description="Dropout rate")

    model_config = {"extra": "forbid"}  # Don't allow extra fields


@ModelRegistry.register_artifact(params_model=ValidatedModelParams)
class ValidatedModel:
    """Model with explicit validated parameters."""

    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

    def __repr__(self):
        return f"ValidatedModel(h={self.hidden_size}, heads={self.num_heads}, drop={self.dropout})"


# Valid configuration
cfg = BuildCfg(type="ValidatedModel", data={"hidden_size": 512, "num_heads": 16})
model = ContainerMixin.build_cfg(cfg)
print(f"\nValid config: {model}")


# =============================================================================
# Part 4: Type Coercion
# =============================================================================

print("\n" + "=" * 60)
print("Part 4: Type Coercion")
print("=" * 60)


class CoercionParams(BaseModel):
    """Schema for coercion testing."""

    value_int: int
    value_float: float
    value_bool: bool


@ModelRegistry.register_artifact(params_model=CoercionParams)
class CoercionModel:
    def __init__(self, value_int: int, value_float: float, value_bool: bool):
        self.value_int = value_int
        self.value_float = value_float
        self.value_bool = value_bool

    def __repr__(self):
        return (
            f"CoercionModel(int={self.value_int} [{type(self.value_int).__name__}], "
            f"float={self.value_float} [{type(self.value_float).__name__}], "
            f"bool={self.value_bool} [{type(self.value_bool).__name__}])"
        )


# Pass strings - Pydantic will coerce them
cfg = BuildCfg(
    type="CoercionModel",
    data={
        "value_int": "42",  # string -> int
        "value_float": "3.14",  # string -> float
        "value_bool": "true",  # string -> bool
    },
)
model = ContainerMixin.build_cfg(cfg)
print(f"\nCoerced from strings: {model}")


# =============================================================================
# Part 5: Handling Extra Fields
# =============================================================================

print("\n" + "=" * 60)
print("Part 5: Handling Extra Fields")
print("=" * 60)


class KnownParams(BaseModel):
    """Only known fields."""

    x: int
    y: str = "default"


@ModelRegistry.register_artifact(params_model=KnownParams)
class KnownFieldsModel:
    def __init__(self, x: int, y: str = "default"):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"KnownFieldsModel(x={self.x}, y={self.y!r})"


# Config with extra fields
cfg = BuildCfg(
    type="KnownFieldsModel",
    data={
        "x": 100,
        "y": "hello",
        "unknown_field": "this goes to meta",
        "another_extra": 42,
    },
    meta={"original_meta": "preserved"},
    top_level_extra="also captured",  # Extra at BuildCfg level
)

model = ContainerMixin.build_cfg(cfg)
print(f"\nBuilt model: {model}")
print(f"\nMeta attached to model:")
meta = getattr(model, "__meta__", {})
for key, value in meta.items():
    print(f"  {key}: {value}")


# =============================================================================
# Part 6: Function Factories
# =============================================================================

print("\n" + "=" * 60)
print("Part 6: Function Factories")
print("=" * 60)


class TransformRegistry(FunctionalRegistry):
    """Registry for transform functions."""

    pass


@TransformRegistry.register_artifact
def normalize(mean: tuple, std: tuple):
    """Create a normalize transform config."""
    return {"type": "Normalize", "mean": mean, "std": std}


@TransformRegistry.register_artifact
def resize(size: int):
    """Create a resize transform config."""
    return {"type": "Resize", "size": size}


ContainerMixin.configure_repos(
    {
        "models": ModelRegistry,
        "transforms": TransformRegistry,
        "default": ModelRegistry,
    }
)

# Build function outputs
normalize_cfg = BuildCfg(
    type="normalize",
    repo="transforms",
    data={"mean": (0.485, 0.456, 0.406), "std": (0.229, 0.224, 0.225)},
)
transform = ContainerMixin.build_cfg(normalize_cfg)
print(f"\nBuilt transform: {transform}")


print("\n" + "=" * 60)
print("Done!")
print("=" * 60)
