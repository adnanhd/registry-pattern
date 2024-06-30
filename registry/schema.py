from pydantic_core import CoreSchema, core_schema, SchemaValidator, SchemaSerializer
from typing import Any, get_type_hints, Callable
from pydantic import BaseModel, TypeAdapter, validate_call, ConfigDict

from pydantic.plugin._schema_validator import create_schema_validator
from pydantic._internal import _generate_schema, _typing_extra
from pydantic._internal._config import ConfigWrapper
from pydantic._internal._validate_call import ValidateCallWrapper
from pydantic_numpy import NpNDArray
import inspect


def test_func(function: Callable[..., Any], config: ConfigDict | None = None, validate_return: bool = False):
    return ValidateCallWrapper(function, config, validate_return)


from typing import Annotated