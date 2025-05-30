import inspect
from typing_compat import Any, Type, Hashable, Generic, TypeVar, List, ParamSpec
from pydantic import BaseModel, create_model, validate_call

# Assume your existing FunctionalRegistry is imported from your project.
from registry import FunctionalRegistry

# Type variable for keys and for the registered function's return type.
P = ParamSpec("P")
T = TypeVar("T")


def create_model_factory(t: Type) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model class for validating constructor arguments
    for the given type or function 't'.

    The model's fields are generated from the signature of 't'. For each parameter,
    if no annotation is provided, the field type defaults to Any, and if no default
    value is provided, the field is required (using ellipsis).

    Parameters:
        t (Type): The type or function for which to create a builder model.

    Returns:
        Type[BaseModel]: A Pydantic model class representing the parameter schema.
    """
    return create_model(
        t.__qualname__ + "Builder",
        **{
            name: (
                Any if param.annotation == inspect._empty else param.annotation,
                ... if param.default == inspect._empty else param.default,
            )
            for name, param in inspect.signature(t).parameters.items()
        },
    )


class FunctionFactory(FunctionalRegistry[P, T], Generic[P, T]):
    """
    Extension of FunctionalRegistry that implements a function factory pattern.

    In addition to the standard registry functionality, FunctionFactory provides the
    method 'execute_artifact', which validates keyword arguments against a dynamically
    generated Pydantic model (built from the function's signature) and then executes the
    registered function if validation passes.
    """

    @classmethod
    def execute_artifact(cls, key: str, *args: P.args, **kwargs: P.kwargs) -> T:
        """
        Execute the registered function corresponding to the given key using validated parameters.

        This method retrieves the function via the inherited get_registry_item, dynamically
        generates a Pydantic model for validating the function's signature using create_model_factory,
        validates the provided kwargs, and then calls the function with the validated parameters.

        Parameters:
            key (K): The registry key corresponding to the desired function.
            **kwargs (Any): Keyword arguments to be validated and passed to the function.

        Returns:
            T: The result of executing the function.

        Raises:
            pydantic.ValidationError: If the provided kwargs do not match the expected schema.
        """
        # Retrieve the registered function from the registry.
        func = cls.get_registry_item(key)
        # Create a builder model for validating the function's parameters.
        builder_model = create_model_factory(func)
        # Validate the provided kwargs; using model_validate (Pydantic v2) here.
        validated = builder_model.model_validate(kwargs)
        # Call the function with the validated parameters.
        return func(**validated.model_dump())


# -----------------------------------------------------------------------------
# Example usage (for illustration)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import torch
    from typing_extensions import Annotated
    from pydantic import BeforeValidator, WrapValidator, WrapSerializer, InstanceOf

    # Example validation functions and type annotations for torch:
    def validate_positive_int_list(value: List[int]) -> torch.Size:
        if not all(isinstance(dim, int) and dim > 0 for dim in value):
            raise ValueError("All dimensions must be positive integers.")
        return torch.Size(value)

    def validate_torch_size(value: torch.Size, info: Any) -> torch.Size:
        if not isinstance(value, torch.Size):
            raise TypeError(f"Expected torch.Size, got {type(value).__name__}.")
        return value

    def serialize_torch_size(value: torch.Size, handler: Any) -> List[int]:
        return list(handler(value))

    def validate_str_dtype(value: str, info: Any) -> torch.dtype:
        if not isinstance(value, str):
            raise TypeError(f"Expected torch.dtype, got {type(value).__name__}.")
        dtype_mapping = {
            "torch.float32": torch.float32,
            "torch.float": torch.float32,
            "torch.float64": torch.float64,
            "torch.double": torch.float64,
            "torch.float16": torch.float16,
            "torch.half": torch.float16,
            "torch.int32": torch.int32,
            "torch.int": torch.int32,
            "torch.int64": torch.int64,
            "torch.long": torch.int64,
            "torch.int16": torch.int16,
            "torch.short": torch.int16,
            "torch.int8": torch.int8,
            "torch.uint8": torch.uint8,
            "torch.bool": torch.bool,
        }
        if value in dtype_mapping:
            return dtype_mapping[value]
        raise ValueError(f"{value} is not a valid torch dtype.")

    def validate_torch_dtype(value: torch.dtype, info: Any) -> torch.dtype:
        if not isinstance(value, torch.dtype):
            raise TypeError(f"Expected torch.dtype, got {type(value).__name__}.")
        return value

    def serialize_torch_dtype(value: torch.dtype, handler: Any) -> str:
        return handler(str(value))

    TorchDType = Annotated[
        InstanceOf[torch.dtype],
        WrapValidator(validate_torch_dtype),
        BeforeValidator(validate_str_dtype),
        WrapSerializer(serialize_torch_dtype),
    ]

    TorchSize = Annotated[
        InstanceOf[torch.Size],
        WrapValidator(validate_torch_size),
        BeforeValidator(validate_positive_int_list),
        WrapSerializer(serialize_torch_size),
    ]

    # Define a concrete FunctionFactory for creating tensors.
    class TorchCreator(FunctionFactory[[TorchSize, TorchDType], torch.Tensor]):
        pass

    # Register functions using the factory.
    @TorchCreator.register_function
    @validate_call
    def zeros(shape: TorchSize, dtype: TorchDType) -> torch.Tensor:
        """Create a tensor of zeros with the given shape and dtype."""
        return torch.zeros(size=shape, dtype=dtype)

    @TorchCreator.register_function
    @validate_call
    def ones(shape: TorchSize, dtype: TorchDType) -> torch.Tensor:
        """Create a tensor of ones with the given shape and dtype."""
        return torch.ones(size=shape, dtype=dtype)

    # Example: Execute a function using the factory.
    # The keyword arguments will be validated against the dynamically generated model.
    tensor1 = TorchCreator.execute_artifact(
        "zeros", shape=[2, 3], dtype="torch.float32"
    )
    tensor2 = TorchCreator.execute_artifact("ones", shape=[2, 3], dtype="torch.int")
    print("zeros tensor:", tensor1)
    print("ones tensor:", tensor2)
