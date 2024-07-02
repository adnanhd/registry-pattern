from dataclasses import dataclass
from typing import Any

@dataclass
class _BaseTypedDictAnnotation:
    
    def build_model(self):
        raise NotImplementedError
    
    @classmethod
    def validate_config(cls, values: Any) -> '_BaseTypedDictAnnotation':
        raise NotImplementedError
    
    @classmethod
    def validate_string(cls, values: Any) -> '_BaseTypedDictAnnotation':
        raise NotImplementedError
    
    @staticmethod
    def validate_semicolon_in_type(values: Any, handler: Any) -> dict[str, Any]:
        raise NotImplementedError