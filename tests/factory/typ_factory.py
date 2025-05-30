from dataclasses import dataclass
from functools import partial
from functools import update_wrapper
from functools import wraps
from inspect import signature
from typing import Generic
from typing import TypeVar

from pydantic import InstanceOf
from pydantic import validate_call
from pydantic_core import core_schema
from registry import ObjectConfigMap
from registry import TypeRegistry
from typing_extensions import Callable

Obj = TypeVar("Obj")


@dataclass
class TypeFactory(ObjectConfigMap[Obj], Generic[Obj]):
    registry: InstanceOf[TypeRegistry[Obj]]
    coercion: bool = True

    def validate_builder(self, type: str, *args, **kwds) -> Callable[..., Obj]:
        builder: Callable[..., Obj] = self.registry.get_registry_item(type)
        if len(args) != 0 or len(kwds) != 0:
            builder = partial(builder, *args, **kwds)

        if self.coercion:
            # Extract the signature from the __init__ method of torch.nn.Linear
            original_signature = signature(builder.__init__)

            # Create a new signature by removing 'self'
            parameters = original_signature.parameters.values()
            parameters = list(parameters)[1:]
            new_signature = original_signature.replace(parameters=parameters)

            # Define the wrapper function
            @wraps(builder)
            def builder_wrapper(*args, **kwargs) -> Obj:
                return builder(*args, **kwargs)

            # Assign the modified signature to the wrapper function
            setattr(builder_wrapper, "__signature__", new_signature)
            update_wrapper(builder_wrapper, builder.__init__)

            # Apply Pydantic's validate_call
            return validate_call(builder_wrapper)

        return builder

    def validate_instance(self, type: str, **kwds) -> Obj:
        instance = self.validate_builder(type)(**kwds)
        self.__class__.register_instance(instance, {"type": type, **kwds})
        return instance

    def __get_pydantic_core_schema__(self, source_type, handler):
        assert isinstance(self, TypeRegistry)
        py_schema = handler(source_type)
        js_schema = core_schema.no_info_after_validator_function(
            lambda kwds: self.validate_instance(**kwds),
            schema=core_schema.dict_schema(),
        )
        js_schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
            lambda val, nxt: nxt(self.get_registry_item(val))
        )
        return core_schema.json_or_python_schema(
            json_schema=js_schema, python_schema=py_schema
        )
