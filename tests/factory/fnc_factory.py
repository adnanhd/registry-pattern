from dataclasses import dataclass
from functools import partial
from typing import Callable
from typing import Generic
from typing import TypeVar
from typing import Union

from pydantic import InstanceOf
from pydantic import validate_call
from pydantic_core import core_schema
from registry import FunctionalRegistry
from registry import ObjectConfigMap
from typing_extensions import Callable
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R")


@dataclass
class FunctionalFactory(ObjectConfigMap[Callable[P, R]], Generic[P, R]):
    registry: InstanceOf[FunctionalRegistry[P, R]]
    coercion: bool = True

    def validate_functional(
        self, type: str, **kwds
    ) -> Union[Callable[P, R], partial[R]]:
        functional: Callable[P, R] = self.registry.get_registry_item(type)
        if len(kwds) != 0:
            functional = partial(functional, **kwds)  # type: ignore
        if self.coercion:
            functional = validate_call(functional)
        if not self.__class__.has_registry_key(functional):
            functional = self.__class__.register_instance(
                functional, {"type": type, **kwds}
            )
        return functional

    def __get_pydantic_core_schema__(self, source_type, handler):
        assert isinstance(self, FunctionalFactory)
        py_schema = handler(source_type)
        js_schema = core_schema.no_info_after_validator_function(
            lambda kwds: self.validate_functional(**kwds),
            schema=core_schema.dict_schema(),
        )
        js_schema["serialization"] = core_schema.wrap_serializer_function_ser_schema(
            lambda val, nxt: nxt(self.get_registry_item(val))
        )
        return core_schema.json_or_python_schema(
            json_schema=js_schema, python_schema=py_schema
        )
