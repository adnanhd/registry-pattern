import inspect
import torch
from typing import Any, Callable, Dict, Type, Tuple
from pydantic import BaseModel, create_model, ValidationError
from pydantic_core import core_schema

# --- Registry using a builder function per type ---
REGISTRY: Dict[str, Tuple[Type, Callable[[Any], Any]]] = {}

def register_type(name: str):
    """
    Decorator to register a type with its builder.
    The default builder expects data to be a dict and calls the type with **data;
    if the type needs special handling (e.g. for variadic arguments), then override.
    """
    def decorator(cls: Type) -> Type:
        def builder(data: Any) -> Any:
            # If data is a list, assume variadic args.
            if isinstance(data, list):
                return cls(*data)
            # Otherwise assume it's a dict.
            return cls(**data)
        REGISTRY[name] = (cls, builder)
        return cls
    return decorator

# Register standard torch modules.
register_type("Linear")(torch.nn.Linear)
register_type("ReLU")(torch.nn.ReLU)
register_type("Sequential")(torch.nn.Sequential)

# --- Custom Module that takes a backbone ---
@register_type("Bok")
class Bok(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)

# --- Dynamic builder for constructor arguments ---
def create_model_factory(t: Type) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model class to validate constructor arguments for type 't'.
    For any parameter whose annotation is a torch.nn.Module (or subclass), we replace the annotation with Any.
    """
    fields = {}
    sig = inspect.signature(t.__init__)
    # Skip 'self' parameter
    for name, param in list(sig.parameters.items())[1:]:
        # Use the parameter's annotation if given, otherwise Any.
        annotation = param.annotation if param.annotation != inspect._empty else Any
        # If the annotation is a subclass of torch.nn.Module, override it with Any.
        try:
            if isinstance(annotation, type) and issubclass(annotation, torch.nn.Module):
                annotation = Any
        except Exception:
            pass

        default = ... if param.default == inspect._empty else param.default
        fields[name] = (annotation, default)
    return create_model(t.__qualname__ + "Builder", **fields)

# --- Helper for recursive build ---
def recursive_build(value: Any) -> Any:
    """
    Recursively traverse the value. If it's a ModelFactory instance, call its build();
    if it is a dict that appears to be a ModelFactory config (has "type" and "data"),
    construct it without re‑validation (using model_construct), then build.
    Otherwise, if it's a list or dict, process its contents.
    """
    if isinstance(value, ModelFactory):
        return value.build()
    elif isinstance(value, dict):
        if "type" in value and "data" in value:
            mf = ModelFactory.model_construct(**value)
            return mf.build()
        else:
            return {k: recursive_build(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [recursive_build(item) for item in value]
    else:
        return value

# --- The unified ModelFactory ---
class ModelFactory(BaseModel):
    type: str
    data: Any
    meta: dict = {}

    def build(self) -> Any:
        if self.type not in REGISTRY:
            raise ValueError(f"Unknown type: {self.type}")
        target_cls, builder = REGISTRY[self.type]
        # Recursively process self.data in case it contains nested configs.
        built_data = recursive_build(self.data)
        return builder(built_data)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def validate_model(value):
            if not isinstance(value, dict):
                raise TypeError("ModelFactory expects a dict")
            type_val = value.get("type")
            if type_val not in REGISTRY:
                raise ValueError(f"Unknown type: {type_val}")
            data_val = value.get("data")
            meta_val = value.get("meta", {})

            target_cls, _ = REGISTRY[type_val]
            Builder = create_model_factory(target_cls)
            try:
                if isinstance(data_val, dict):
                    data_val = Builder(**data_val).model_dump()
                elif isinstance(data_val, list):
                    new_data = []
                    for item in data_val:
                        if isinstance(item, dict) and "type" not in item:
                            new_data.append(Builder(**item).dict())
                        else:
                            new_data.append(item)
                    data_val = new_data
            except Exception:
                pass

            return cls.model_construct(type=type_val, data=data_val, meta=meta_val)
        return core_schema.no_info_plain_validator_function(validate_model)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"title": "ModelFactory", "type": "object"}

# --- Example usage in a nested model ---
class MyModel(BaseModel):
    model: ModelFactory

# --- Example configuration ---
# "Bok" takes a backbone, which is an arbitrary network.
# Here the backbone is a Sequential composed of two Linear layers and one ReLU.
config_data = {
    "model": {
        "type": "Bok",
        "data": {
            "backbone": {
                "type": "Sequential",
                "data": [
                    {
                        "type": "Linear",
                        "data": {"in_features": 10, "out_features": 20},
                        "meta": {"version": "1.0"},
                    },
                    {
                        "type": "ReLU",
                        "data": {"inplace": True},
                        "meta": {"created": "2023-01-01"},
                    },
                    {
                        "type": "Linear",
                        "data": {"in_features": 20, "out_features": 5},
                        "meta": {},
                    },
                ],
                "meta": {"creation_time": "2023-03-05"},
            }
        },
    }
}

if __name__ == "__main__":
    try:
        # Validate the configuration (using model_validate in Pydantic v2).
        instance = MyModel.model_validate(config_data)
        # Build the torch.nn.Module using the unified factory.
        built_model = instance.model.build()
        print(built_model)
    except ValidationError as e:
        print("Validation error:", e)
