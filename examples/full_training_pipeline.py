# full_training_pipeline.py

from typing import Any, Dict, Type, Callable, List, Union, get_type_hints
from inspect import signature
from abc import ABC
import yaml
import argparse
from pathlib import Path
from pydantic import BaseModel, create_model, validate_call
from pydantic_core import core_schema


# -------------------- Registry --------------------


class TypeRegistry:
    def __init__(self):
        self._types: Dict[str, Type] = {}

    def register(self, name: str, cls: Type) -> None:
        self._types[name] = cls

    def get(self, name: str) -> Type:
        return self._types[name]

    def all(self) -> Dict[str, Type]:
        return self._types


# -------------------- Config Interface --------------------


class ConfigInterfaceMixin(ABC):
    type: str
    __type_target__: Type = None

    def instantiate(self) -> Any:
        if self.__type_target__ is None:
            raise NotImplementedError("Target type not set.")
        return self.__type_target__(**self.model_dump())

    def validate(self, obj: Any) -> bool:
        return isinstance(obj, self.__type_target__)

    def serialize(self, obj: Any = None) -> dict:
        return self.model_dump()


# -------------------- Factory --------------------


class Factory:
    def __init__(self, registry: TypeRegistry):
        self.registry = registry
        self._validators: Dict[str, Type[BaseModel]] = {}
        self._constructors: Dict[str, Callable] = {}

    def register_validator(self, name: str, validator: Type[BaseModel]):
        self._validators[name] = validator

    def get_config_class(self, name: str) -> Type[BaseModel]:
        if name in self._validators:
            return self._validators[name]
        typ = self.registry.get(name)
        model = self._create_base_model(typ)
        self._validators[name] = model
        return model

    def _create_base_model(self, typ: Type) -> Type[BaseModel]:
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
        config_cls = self.get_config_class(name)
        return config_cls.model_validate(data)

    def __getitem__(self, name: str) -> Type[BaseModel]:
        return self.get_config_class(name)

    def all_configs_union(self) -> Any:
        return Union[tuple(self._validators[name] for name in self._validators)]

    def dump_schema(self, path: Path):
        for name, validator in self._validators.items():
            schema_path = path / f"{name}_schema.json"
            schema_path.write_text(validator.model_json_schema_json(indent=2))


# -------------------- Mock model --------------------


class MockModel:
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = [0.5] * (input_dim * output_dim)

    def train_step(self, x: List[float]):
        return sum(self.weights) * sum(x)

    def __repr__(self):
        return f"MockModel({self.input_dim}→{self.output_dim})"


# -------------------- Setup --------------------

registry = TypeRegistry()
registry.register("linear", MockModel)

factory = Factory(registry)


# Optional: custom config
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


factory.register_validator("linear", LinearModelConfig)

# Union of all configs
AllModelConfigs = factory.all_configs_union()

# -------------------- Training pipeline --------------------


@validate_call
def train(cfg: AllModelConfigs, data: List[float]):
    model = cfg.instantiate()
    if not cfg.validate(model):
        raise ValueError("❌ Model doesn't match its config.")
    print(f"✅ Model: {model}")
    output = model.train_step(data)
    print(f"📦 Output: {output:.2f}")
    return output


# -------------------- CLI + YAML loading --------------------


def load_config_from_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--data", nargs="+", type=float, required=True, help="Input list"
    )
    parser.add_argument("--save", type=str, help="Optional: path to save config back")
    args = parser.parse_args()

    cfg_dict = load_config_from_yaml(Path(args.config))
    type_name = cfg_dict.get("type")
    cfg = factory.create_config_instance(type_name, cfg_dict)
    result = train(cfg, data=args.data)

    if args.save:
        Path(args.save).write_text(yaml.safe_dump(cfg.serialize()))


# -------------------- Run --------------------

if __name__ == "__main__":
    main()
