import jsonschema
from jsonschema import Draft7Validator, validators

def extend_with_square(validator_class):
    # Define a new validator for the "square" keyword.
    def _validate_square(validator, square, instance, schema):
        if square and isinstance(instance, list) and len(instance) == 2:
            if instance[0] != instance[1]:
                yield jsonschema.ValidationError(
                    "The array must represent a square shape: the first and second elements must be equal."
                )
    # Extend the validator class with the new keyword.
    return validators.extend(validator_class, {"square": _validate_square})

SquareValidator = extend_with_square(Draft7Validator)
import json

tensor_schema = {
    "type": "object",
    "properties": {
        "shape": {
            "type": "array",
            "minItems": 2,
            "maxItems": 2,
            "items": [
                {"type": "integer", "minimum": 1},
                {"type": "integer", "minimum": 1}
            ],
            "square": True  # our custom keyword enforces that the two numbers are equal.
        },
        "dtype": {
            "type": "string",
            "default": "float32"
        }
    },
    "required": ["shape"],
    "additionalProperties": False
}

# For display
print("Tensor schema with square check:")
print(json.dumps(tensor_schema, indent=2))

import torch
from typing import Any, Dict

# Mapping from dtype strings to torch dtypes.
DTYPE_MAP = {
    "float32": torch.float32,
    "float": torch.float32,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
}

def instantiate_tensor(config: Dict[str, Any]) -> torch.Tensor:
    # Validate the configuration using our SquareValidator.
    SquareValidator(tensor_schema).validate(config)
    shape = config["shape"]
    dtype_str = config.get("dtype", "float32")
    dtype = DTYPE_MAP.get(dtype_str.lower(), torch.float32)
    return torch.rand(*shape, dtype=dtype)

# --- Example Usage ---
if __name__ == "__main__":
    # A valid configuration: square shape (e.g., 3x3).
    valid_config = {
        "shape": [3, 3],
        "dtype": "float64"
    }
    tensor_instance = instantiate_tensor(valid_config)
    print("Tensor instance:")
    print(tensor_instance)
    print("Tensor dtype:", tensor_instance.dtype)
    print("Tensor shape:", tensor_instance.shape)

    # An invalid configuration: non-square shape (e.g., 3x4).
    invalid_config = {
        "shape": [3, 4],
        "dtype": "float64"
    }
    try:
        instantiate_tensor(invalid_config)
    except jsonschema.ValidationError as e:
        print("\nValidation error for non-square shape:")
        print(e.message)
