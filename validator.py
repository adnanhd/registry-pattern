import dataclasses
from typing import Any, Annotated
from pydantic_core import CoreSchema
from pydantic import GetJsonSchemaHandler, GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue

import sys

# `slots` is available on Python >= 3.10
if sys.version_info >= (3, 10):
    slots_true = {'slots': True}
else:
    slots_true = {}


@dataclasses.dataclass(**slots_true)
class CoreSchemaValidator:
    """Use a fixed CoreSchema, avoiding interference from outward annotations."""

    core_schema: CoreSchema
    js_schema: JsonSchemaValue | None = None
    js_core_schema: CoreSchema | None = None
    js_schema_update: JsonSchemaValue | None = None

    def __get_pydantic_json_schema__(self, _schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        if self.js_schema is not None:
            return self.js_schema
        js_schema = handler(self.js_core_schema or self.core_schema)
        if self.js_schema_update is not None:
            js_schema.update(self.js_schema_update)
        return js_schema

    def __get_pydantic_core_schema__(self, _source_type: Any, _handler: GetCoreSchemaHandler) -> CoreSchema:
        return self.core_schema

    @classmethod
    def __class_getitem__(cls, key):
        schema_type, core_schema = key
        assert isinstance(schema_type, type)
        assert isinstance(core_schema, dict)
        return Annotated[schema_type, cls(core_schema=core_schema)]
