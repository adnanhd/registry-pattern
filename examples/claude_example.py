#!/usr/bin/env python3
"""
Four Registry Architecture: Object, Function, Type, and Scheme Registries

This implements the complete registry system with:
1. ObjectRegistry - mapping names to object instances
2. FunctionRegistry - mapping names to functions
3. TypeRegistry - mapping names to class types  
4. SchemeRegistry - mapping names to Pydantic models (validation/configuration schemes)
"""

from typing import Any, Dict, Type, Callable, List, Union, get_type_hints
from inspect import signature
from abc import ABC
import yaml
import argparse
from pathlib import Path
from pydantic import BaseModel, create_model, validate_call
from pydantic_core import core_schema


# ==================== Four Registry System ====================


class ObjectRegistry:
    """Registry for mapping names to object instances"""

    def __init__(self):
        self._objects: Dict[str, Any] = {}

    def register(self, name: str, obj: Any) -> None:
        """Register an object instance by name"""
        self._objects[name] = obj

    def get(self, name: str) -> Any:
        """Get a registered object by name"""
        return self._objects[name]

    def all(self) -> Dict[str, Any]:
        """Get all registered objects"""
        return self._objects


class FunctionRegistry:
    """Registry for mapping names to functions"""

    def __init__(self):
        self._functions: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable) -> None:
        """Register a function by name"""
        self._functions[name] = func

    def get(self, name: str) -> Callable:
        """Get a registered function by name"""
        return self._functions[name]

    def all(self) -> Dict[str, Callable]:
        """Get all registered functions"""
        return self._functions


class TypeRegistry:
    """Registry for mapping names to class types"""

    def __init__(self):
        self._types: Dict[str, Type] = {}

    def register(self, name: str, cls: Type) -> None:
        """Register a class type by name"""
        self._types[name] = cls

    def get(self, name: str) -> Type:
        """Get a registered type by name"""
        return self._types[name]

    def all(self) -> Dict[str, Type]:
        """Get all registered types"""
        return self._types


class SchemeRegistry:
    """Registry for mapping names to Pydantic models (validation/configuration schemes)"""

    def __init__(self):
        self._schemes: Dict[str, Type[BaseModel]] = {}

    def register(self, name: str, schema_class: Type[BaseModel]) -> None:
        """Register a Pydantic model schema by name"""
        self._schemes[name] = schema_class

    def get(self, name: str) -> Type[BaseModel]:
        """Get a registered schema by name"""
        return self._schemes[name]

    def all(self) -> Dict[str, Type[BaseModel]]:
        """Get all registered schemas"""
        return self._schemes

    def validate_data(self, scheme_name: str, data: Dict[str, Any]) -> BaseModel:
        """Validate data against a named schema"""
        schema_class = self.get(scheme_name)
        return schema_class(**data)

    def create_instance(self, scheme_name: str, data: Dict[str, Any]) -> BaseModel:
        """Create a validated instance of a schema"""
        return self.validate_data(scheme_name, data)

    def all_schemes_union(self) -> Any:
        """Create a Union type of all registered schemas"""
        if not self._schemes:
            return Any
        return Union[tuple(self._schemes.values())]


# ==================== Master Registry ====================


class MasterRegistry:
    """Master registry containing all four registry types"""

    def __init__(self):
        self.objects = ObjectRegistry()
        self.functions = FunctionRegistry()
        self.types = TypeRegistry()
        self.schemes = SchemeRegistry()

    def summary(self) -> Dict[str, int]:
        """Get a summary of all registries"""
        return {
            "objects": len(self.objects.all()),
            "functions": len(self.functions.all()),
            "types": len(self.types.all()),
            "schemes": len(self.schemes.all()),
        }


# ==================== Config Interface ====================


class ConfigInterfaceMixin(ABC):
    """Mixin for configuration objects that can instantiate, validate, and serialize"""

    type: str
    __type_target__: Type = None

    def instantiate(self) -> Any:
        """Instantiate the target object from this configuration"""
        if self.__type_target__ is None:
            raise NotImplementedError("Target type not set.")
        return self.__type_target__(**self.model_dump())

    def validate(self, obj: Any) -> bool:
        """Validate that an object matches this configuration"""
        return isinstance(obj, self.__type_target__)

    def serialize(self, obj: Any = None) -> dict:
        """Serialize this configuration to a dictionary"""
        return self.model_dump()


# ==================== Enhanced Factory ====================


class Factory:
    """Factory that uses all four registries to create and manage configurations"""

    def __init__(self, master_registry: MasterRegistry):
        self.master_registry = master_registry
        self.type_registry = master_registry.types
        self.scheme_registry = master_registry.schemes

    def get_config_class(self, name: str) -> Type[BaseModel]:
        """Get or create a configuration class for a registered type"""
        # First check if we already have a scheme registered
        try:
            return self.scheme_registry.get(name)
        except KeyError:
            pass

        # If not, create one from the type registry
        typ = self.type_registry.get(name)
        model = self._create_base_model(typ)
        self.scheme_registry.register(name, model)
        return model

    def _create_base_model(self, typ: Type) -> Type[BaseModel]:
        """Create a Pydantic model from a class type's __init__ signature"""
        sig = signature(typ.__init__)
        hints = get_type_hints(typ.__init__)
        fields = {}

        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annotation = hints.get(name)
            if annotation is None:
                raise TypeError(f"Missing annotation for '{name}' in {typ}")
            fields[name] = (annotation, ...)

        model_cls = create_model(
            f"{typ.__name__}Config",
            __base__=(BaseModel, ConfigInterfaceMixin),
            **fields,
        )
        model_cls.__type_target__ = typ
        return model_cls

    def create_config_instance(self, name: str, data: dict) -> BaseModel:
        """Create a configuration instance from data"""
        config_cls = self.get_config_class(name)
        return config_cls.model_validate(data)

    def register_custom_scheme(self, name: str, schema_class: Type[BaseModel]):
        """Register a custom scheme (bypassing auto-generation)"""
        self.scheme_registry.register(name, schema_class)

    def all_configs_union(self) -> Any:
        """Get a Union type of all configuration schemes"""
        return self.scheme_registry.all_schemes_union()

    def dump_schemas(self, path: Path):
        """Dump all schemas to JSON files"""
        path.mkdir(exist_ok=True)
        for name, validator in self.scheme_registry.all().items():
            schema_path = path / f"{name}_schema.json"
            schema_path.write_text(validator.model_json_schema_json(indent=2))


# ==================== Example Usage ====================


# Mock model for demonstration
class MockModel:
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = [0.5] * (input_dim * output_dim)

    def train_step(self, x: List[float]):
        return sum(self.weights) * sum(x)

    def __repr__(self):
        return f"MockModel({self.input_dim}→{self.output_dim})"


# Setup the four-registry system
def setup_system():
    """Setup the complete four-registry system"""

    # Create master registry
    master_registry = MasterRegistry()

    # Register some objects
    master_registry.objects.register("default_model", MockModel(2, 3))

    # Register some functions
    master_registry.functions.register("relu", lambda x: max(0, x))
    master_registry.functions.register("sigmoid", lambda x: 1 / (1 + abs(x)))

    # Register types
    master_registry.types.register("linear", MockModel)

    # Create factory
    factory = Factory(master_registry)

    # Optional: register custom scheme
    class LinearModelConfig(BaseModel, ConfigInterfaceMixin):
        type: str = "linear"
        input_dim: int
        output_dim: int

        def instantiate(self):
            return MockModel(self.input_dim, self.output_dim)

        def validate(self, obj):
            return (
                isinstance(obj, MockModel)
                and obj.input_dim == self.input_dim
                and obj.output_dim == self.output_dim
            )

    factory.register_custom_scheme("linear", LinearModelConfig)

    return master_registry, factory


# Training pipeline using the registry system
@validate_call
def train(cfg: BaseModel, data: List[float]):
    """Training function that works with any configuration scheme"""
    model = cfg.instantiate()
    if not cfg.validate(model):
        raise ValueError("❌ Model doesn't match its config.")
    print(f"✅ Model: {model}")
    output = model.train_step(data)
    print(f"📦 Output: {output:.2f}")
    return output


# CLI interface
def load_config_from_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--data", nargs="+", type=float, required=True, help="Input list"
    )
    parser.add_argument("--save", type=str, help="Optional: path to save config back")
    parser.add_argument(
        "--registry-summary", action="store_true", help="Show registry summary"
    )
    args = parser.parse_args()

    # Setup system
    master_registry, factory = setup_system()

    if args.registry_summary:
        print("📊 Registry Summary:")
        for registry_type, count in master_registry.summary().items():
            print(f"  {registry_type}: {count} items")
        print()

    # Load and process config
    cfg_dict = load_config_from_yaml(Path(args.config))
    type_name = cfg_dict.get("type")
    cfg = factory.create_config_instance(type_name, cfg_dict)
    result = train(cfg, data=args.data)

    if args.save:
        Path(args.save).write_text(yaml.safe_dump(cfg.serialize()))


if __name__ == "__main__":
    main()
