import inspect
import weakref
import sys
import copy

import json
from typing import (
    Any,
    Callable,
    Hashable,
    Generic,
    TypeVar,
    Type,
    Dict,
    List,
    Union,
    Optional,
)
from argparse import Namespace

import yaml  # Requires PyYAML installed: pip install pyyaml
from pydantic import BaseModel, create_model, TypeAdapter
from pydantic_core import core_schema

# Assume your existing TypeRegistry is imported from your project.
from registry import TypeRegistry, ConfigRegistry
from registry.mixin import MutableMappingValidatorMixin

# Type variable for registered artifact types.
T = TypeVar("T", bound=Type)


def create_model_factory(t: Type) -> Type[BaseModel]:
    """
    Dynamically creates a Pydantic model class for validating constructor arguments
    for the given type 't'.
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


# -----------------------------------------------------------------------------
# Engine conversion functions
# -----------------------------------------------------------------------------


def python_factory(input_data: Any) -> dict:
    """
    Default factory: expects a dict (or kwargs) and returns it as-is.
    """
    if not isinstance(input_data, dict):
        raise ValueError("For 'dict' factory, input must be a dictionary.")
    from copy import deepcopy

    return deepcopy(input_data)


def json_factory(input_data: Any) -> dict:
    """
    Expects a JSON string and converts it to a dictionary.
    """
    if not isinstance(input_data, str):
        raise ValueError("For 'json' factory, input must be a JSON string.")
    return json.loads(input_data)


def yaml_factory(input_data: Any) -> dict:
    """
    Expects a YAML string and converts it to a dictionary.
    """
    if not isinstance(input_data, str):
        raise ValueError("For 'yaml' factory, input must be a YAML string.")
    return yaml.safe_load(input_data)


def argparse_factory(input_data: Any) -> dict:
    """
    Expects an argparse.Namespace instance and converts it to a dictionary.
    """
    if not isinstance(input_data, Namespace):
        raise ValueError("For 'argparse' factory, input must be an argparse.Namespace.")
    return vars(input_data)


# -----------------------------------------------------------------------------
# Extended TypeFactory with multiple instantiation factories and Pydantic integration.
# -----------------------------------------------------------------------------


class TypeFactory(MutableMappingValidatorMixin[Hashable, Callable[[Any], Dict]]):
    """
    Extension of TypeRegistry that implements a factory pattern with support for
    multiple input "factories". In addition, this class provides methods for unregistering
    factories and clearing the registry, as well as a custom Pydantic schema definition.
    """

    # Registry for conversion factories.
    _factories: Dict[str, Callable[..., Dict[str, Any]]]

    @classmethod
    def _get_mapping(cls) -> Dict[str, Callable[..., Dict[str, Any]]]:
        return cls._factories

    @classmethod
    def _get_builder(cls, key: Hashable) -> Type:
        raise NotImplementedError(f"Builder is not implemented for key '{key}'.")

    @classmethod
    def __init_subclass__(
        cls,
        pydantic: bool = False,
        registry: Optional[
            Union[Callable[[Hashable], Type], Type, TypeRegistry]
        ] = None,
        **kwargs,
    ):
        super().__init_subclass__(**kwargs)
        cls._factories = {}
        if pydantic:

            class MyConfReg(ConfigRegistry):
                pass

            cls._registry = MyConfReg
        else:
            cls._registry = None

        if issubclass(registry, TypeRegistry):
            cls._get_builder = staticmethod(lambda key: registry.get_artifact(key))

    @classmethod
    def unregister_factory(cls, key: Hashable) -> None:
        """
        Unregister (remove) a registered class from the repository.

        Parameters:
            key (Hashable): The key corresponding to the registered class.

        Raises:
            KeyError: If the key is not found in the repository.
        """
        cls.unregister_artifact(key)

    @classmethod
    def clear_factories(cls) -> None:
        """
        Clear all registered classes from the repository.
        """
        cls.clear_artifacts()

    @classmethod
    def register_factory(
        cls, name: str, converter: Callable[[Any], Dict]
    ) -> Callable[[Any], Dict]:
        """
        Register a new instantiation factory.

        Parameters:
            name (str): The name of the factory.
            converter (Callable[[Any], dict]): A function that converts the raw input
                                into a dictionary of kwargs.
        Returns:
            Callable[[Any], Dict]: The converter function.
        """
        cls.register_artifact(name, converter)
        return converter

    @classmethod
    def instantiate_class(
        cls,
        type: str,
        data: Dict,
        /,
        meta: Optional[Dict] = None,
        factory: str = "dict",
    ) -> Any:
        """
        Instantiate an artifact from the registered type using validated parameters.

        The 'factory' parameter specifies the conversion method to use. By default,
        the 'dict' factory expects kwargs (i.e. a dict). For other factories, provide
        the input data via the 'data' argument.

        Parameters:
            key (Hashable): The registry key corresponding to the desired type.
            factory (str): The name of the instantiation factory. Default is 'dict'.
            data (Any): The input data for the factory. If None, 'kwargs' is used.
            **kwargs (Any): Keyword arguments for the 'dict' factory.

        Returns:
            An instance of the registered type.
        """
        # Select the appropriate conversion factory.
        config_factory = cls.get_artifact(factory)

        # Convert the input data into a kwargs dict.
        parsed_kwargs = config_factory(data)

        # Create a config dictionary with the parsed kwargs and other metadata.
        config = dict(type=type, data=data, meta=meta)

        # Instantiate and return the object
        return cls._validate_instance(config)

    # --- Helper methods for Pydantic integration ---

    @classmethod
    def _validate_instance(cls, data: Dict) -> Any:
        """
        Validate a dictionary of parameters and instantiate the corresponding
        registered class. This method expects the input dictionary to contain
        a 'type' key that indicates which registered class to use.

        Parameters:
            **kwds: Keyword arguments, must include a 'type' key.

        Returns:
            An instance of the registered type.

        Raises:
            ValueError: If the 'type' key is missing.
        """
        if cls._registry is None:
            raise RuntimeError(f"{cls.__name__} does not support validation.")

        if cls._registry.has_artifact_key(data):
            return data
        elif isinstance(data, dict) and "type" in data.keys():
            # Retrieve the registered type and instantiate the object.
            type = cls._get_builder(data["type"])
            instance = type(**data["data"])

            # Create a Pydantic model for the serialization
            config_model = create_model_factory(type)
            config = config_model.model_validate(data["data"])
            cls._registry.register_instance(instance, config)

            # Return the instanced object.
            return instance
        else:
            raise ValueError("Input data must be a dict containing a 'type' key.")

    @classmethod
    def _serialize_instance(cls, instance: Any) -> BaseModel:
        """
        Serialize an instance of the registered type to a dictionary.
        """
        if cls._registry is None:
            raise RuntimeError(f"{cls.__name__} does not support serialization.")
        else:
            return cls._registry.get_artifact(instance)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Callable
    ) -> core_schema.CoreSchema:
        """
        Provide a custom Pydantic core schema for this factory type.
        This enables you to use the factory as a Pydantic type so that when a dict is
        provided, it is first validated via `validate_instance`, and when serializing,
        the registry item is used.

        Parameters:
            source_type: The type for which the schema is being generated.
            handler: A function that returns the underlying Pydantic schema for a given type.

        Returns:
            A Pydantic JSON/Python schema.
        """
        # py_schema = handler(source_type)
        py_schema = core_schema.no_info_plain_validator_function(
            lambda val: cls._validate_instance(val),
        )

        dict_schema = core_schema.no_info_after_validator_function(
            lambda kwds: cls.get_artifact("dict")(kwds),
            schema=core_schema.dict_schema(),
        )
        json_schema = core_schema.no_info_after_validator_function(
            lambda kwds: cls.get_artifact("json")(kwds),
            schema=core_schema.json_schema(),
        )

        js_schema = core_schema.no_info_after_validator_function(
            lambda kwds: cls._validate_instance(kwds),
            schema=core_schema.union_schema([dict_schema, json_schema]),
        )
        js_schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
            lambda val, nxt: nxt(cls._serialize_instance(val))
        )
        return core_schema.json_or_python_schema(
            json_schema=js_schema, python_schema=py_schema
        )


# -----------------------------------------------------------------------------
# Example usage (for illustration)
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Register MyClass under its qualified name.
    class MyTypeRegistry(TypeRegistry):
        pass  # Inherits _repository from TypeFactory.

    # Example class to register.
    @MyTypeRegistry.register_class
    class MyClass:
        def __init__(self, a: int, b: int = 0):
            self.a = a
            self.b = b

        def compute(self) -> int:
            return self.a + self.b

    # Create a concrete TypeFactory for MyClass.
    class MyClassFactory(TypeFactory, registry=MyTypeRegistry, pydantic=True):
        pass  # Inherits _repository from TypeFactory.

    MyClassFactory.register_factory("dict", python_factory)
    MyClassFactory.register_factory("json", json_factory)
    MyClassFactory.register_factory("yaml", yaml_factory)
    MyClassFactory.register_factory("argparse", argparse_factory)

    # -------------------------------------------------------------------------
    # Using the default 'dict' factory with kwargs.
    dict_input = dict(a=10, b=5)
    instance1 = MyClassFactory.instantiate_class(
        "MyClass", dict_input, factory="dict"
    )
    print("Python factory compute result:", instance1.compute())

    # -------------------------------------------------------------------------
    # Using the 'json' factory: input is a JSON string.
    json_input = '{"a": 20, "b": 7}'
    instance2 = MyClassFactory.instantiate_class(
        "MyClass", data=json_input, factory="json"
    )
    print("JSON factory compute result:", instance2.compute())

    # -------------------------------------------------------------------------
    # Using the 'yaml' factory: input is a YAML string.
    yaml_input = "a: 30\nb: 3"
    instance3 = MyClassFactory.instantiate_class(
        "MyClass", data=yaml_input, factory="yaml"
    )
    print("YAML factory compute result:", instance3.compute())

    # -------------------------------------------------------------------------
    # Using the 'argparse' factory: input is an argparse.Namespace.
    ns = Namespace(a=40, b=2)
    instance4 = MyClassFactory.instantiate_class("MyClass", factory="argparse", data=ns)
    print("Argparse factory compute result:", instance4.compute())

    # -------------------------------------------------------------------------
    print("Before unregistering, factories:", list(MyClassFactory.iter_artifact_keys()))
    # Unregister MyClass and then clear the entire factory.
    print("Before unregistering, registry:", list(MyTypeRegistry.iter_artifact_keys()))
    MyTypeRegistry.unregister_class("MyClass")
    print("After unregistering, registry:", list(MyTypeRegistry.iter_artifact_keys()))

    # Re-register and then clear all factories.
    MyTypeRegistry.register_class(MyClass)
    # MyTypeRegistry.clear_artifacts()
    # print("After clearing, registry:", list(MyTypeRegistry.iter_artifact_keys()))
    ta = TypeAdapter(MyClassFactory)
