import torch
import json
import jsonschema
from typing import Any, Dict

# Map common dtype strings to torch dtypes.
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

# Define a JSON Schema for a tensor configuration.
tensor_schema = {
    "type": "object",
    "properties": {
        "shape": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 1
        },
        "dtype": {
            "type": "string",
            "default": "float32"
        }
    },
    "required": ["shape"],
    "additionalProperties": False
}

def instantiate_tensor(config: Dict[str, Any]) -> torch.Tensor:
    """
    Validates the config against the JSON Schema and instantiates a torch.Tensor
    using torch.rand. The config must have a "shape" field (list of ints) and
    optionally a "dtype" field (string).
    """
    # Validate the configuration.
    jsonschema.validate(instance=config, schema=tensor_schema)
    
    shape = config["shape"]
    # Use default value if "dtype" is not provided.
    dtype_str = config.get("dtype", "float32")
    dtype = DTYPE_MAP.get(dtype_str.lower(), torch.float32)
    
    # Instantiate the tensor using torch.rand.
    return torch.rand(*shape, dtype=dtype)

# --- Example Usage ---
if __name__ == "__main__":
    # Example configuration for a tensor.
    config = {
        "shape": [3, 3],
        "dtype": "float64"
    }
    
    # Validate and instantiate the tensor.
    tensor_instance = instantiate_tensor(config)
    print("Tensor instance:")
    print(tensor_instance)
    print("Tensor dtype:", tensor_instance.dtype)
    print("Tensor shape:", tensor_instance.shape)
    
    # Optionally, print the JSON Schema.
    print("\nTensor JSON Schema:")
    print(json.dumps(tensor_schema, indent=2))
