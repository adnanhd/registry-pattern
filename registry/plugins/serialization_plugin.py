# registry/plugins/serialization_plugin.py
"""Serialization plugin for saving/loading registries."""

try:
    import orjson
    import yaml
    import msgpack
    SERIALIZATION_AVAILABLE = True
except ImportError:
    SERIALIZATION_AVAILABLE = False
    orjson = yaml = msgpack = None

import json
import pickle
from pathlib import Path
from typing import Type, Any, Dict, Union, Optional, List
from .base import BaseRegistryPlugin
from ..core._validator import ValidationError

class SerializationPlugin(BaseRegistryPlugin):
    """Plugin for registry serialization."""
    
    @property
    def name(self) -> str:
        return "serialization"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def dependencies(self) -> List[str]:
        return []  # Basic JSON/pickle don't require extra deps
    
    def _on_initialize(self) -> None:
        if self.registry_class:
            self._add_serialization_methods()
    
    def _add_serialization_methods(self) -> None:
        """Add serialization methods to the registry."""
        
        def save_to_json(filepath: Union[str, Path], use_orjson: bool = True) -> None:
            """Save registry to JSON file."""
            filepath = Path(filepath)
            data = _serialize_registry()
            
            try:
                if use_orjson and SERIALIZATION_AVAILABLE and orjson:
                    with open(filepath, 'wb') as f:
                        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
                else:
                    with open(filepath, 'w') as f:
                        json.dump(data, f, indent=2, default=str)
            except Exception as e:
                raise ValidationError(
                    f"Failed to save registry to JSON: {e}",
                    suggestions=[
                        "Check file permissions",
                        "Ensure directory exists",
                        "Verify data is JSON serializable"
                    ],
                    context={"filepath": str(filepath), "registry_name": self.registry_class.__name__}
                )
        
        def load_from_json(filepath: Union[str, Path], use_orjson: bool = True) -> None:
            """Load registry from JSON file."""
            filepath = Path(filepath)
            
            try:
                if use_orjson and SERIALIZATION_AVAILABLE and orjson:
                    with open(filepath, 'rb') as f:
                        data = orjson.loads(f.read())
                else:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                
                _deserialize_registry(data)
            except Exception as e:
                raise ValidationError(
                    f"Failed to load registry from JSON: {e}",
                    suggestions=[
                        "Check file exists and is readable",
                        "Verify JSON format is valid",
                        "Ensure data structure matches registry format"
                    ],
                    context={"filepath": str(filepath), "registry_name": self.registry_class.__name__}
                )
        
        def save_to_yaml(filepath: Union[str, Path]) -> None:
            """Save registry to YAML file."""
            if not SERIALIZATION_AVAILABLE or not yaml:
                raise ImportError("PyYAML required. Install with: pip install registry[serialization-plugin]")
            
            filepath = Path(filepath)
            data = _serialize_registry()
            
            try:
                with open(filepath, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            except Exception as e:
                raise ValidationError(
                    f"Failed to save registry to YAML: {e}",
                    suggestions=["Check file permissions", "Ensure directory exists"],
                    context={"filepath": str(filepath), "registry_name": self.registry_class.__name__}
                )
        
        def load_from_yaml(filepath: Union[str, Path]) -> None:
            """Load registry from YAML file."""
            if not SERIALIZATION_AVAILABLE or not yaml:
                raise ImportError("PyYAML required. Install with: pip install registry[serialization-plugin]")
            
            filepath = Path(filepath)
            
            try:
                with open(filepath, 'r') as f:
                    data = yaml.safe_load(f)
                _deserialize_registry(data)
            except Exception as e:
                raise ValidationError(
                    f"Failed to load registry from YAML: {e}",
                    suggestions=["Check file exists", "Verify YAML format"],
                    context={"filepath": str(filepath), "registry_name": self.registry_class.__name__}
                )
        
        def save_to_pickle(filepath: Union[str, Path]) -> None:
            """Save registry to pickle file."""
            filepath = Path(filepath)
            data = _serialize_registry()
            
            try:
                with open(filepath, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                raise ValidationError(
                    f"Failed to save registry to pickle: {e}",
                    suggestions=["Check file permissions", "Ensure data is picklable"],
                    context={"filepath": str(filepath), "registry_name": self.registry_class.__name__}
                )
        
        def load_from_pickle(filepath: Union[str, Path]) -> None:
            """Load registry from pickle file."""
            filepath = Path(filepath)
            
            try:
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                _deserialize_registry(data)
            except Exception as e:
                raise ValidationError(
                    f"Failed to load registry from pickle: {e}",
                    suggestions=["Check file exists", "Ensure pickle format is valid"],
                    context={"filepath": str(filepath), "registry_name": self.registry_class.__name__}
                )
        
        def _serialize_registry() -> Dict[str, Any]:
            """Serialize registry data."""
            # Get the registry mapping safely
            try:
                mapping = self.registry_class._get_mapping()
            except Exception:
                # If we can't get mapping, return empty data
                return {
                    "registry_name": self.registry_class.__name__,
                    "registry_type": self.registry_class.__class__.__name__,
                    "data": {},
                    "metadata": {"size": 0, "strict": False, "abstract": False}
                }
            
            serialized_data = {}
            
            for k, v in mapping.items():
                try:
                    # Simple conversion to string, skip complex objects
                    if isinstance(k, (str, int, float, bool, type(None))):
                        key_str = str(k)
                    elif hasattr(k, '__name__'):
                        key_str = k.__name__
                    else:
                        key_str = f"object_{id(k)}"
                    
                    if isinstance(v, (str, int, float, bool, type(None))):
                        value_str = str(v)
                    elif hasattr(v, '__name__'):
                        value_str = v.__name__
                    else:
                        value_str = f"object_{id(v)}"
                    
                    serialized_data[key_str] = value_str
                except Exception:
                    # Skip problematic items
                    continue
            
            return {
                "registry_name": self.registry_class.__name__,
                "registry_type": self.registry_class.__class__.__name__,
                "data": serialized_data,
                "metadata": {
                    "size": len(serialized_data),
                    "strict": getattr(self.registry_class, '_strict', False),
                    "abstract": getattr(self.registry_class, '_abstract', False),
                }
            }
        
        def _deserialize_registry(data: Dict[str, Any]) -> None:
            """Deserialize registry data."""
            if "data" not in data:
                raise ValidationError("Invalid registry data format")
            
            # Clear existing data
            try:
                self.registry_class._clear_mapping()
            except Exception:
                pass
            
            # Load data (this is simplified - you might need more sophisticated deserialization)
            for key, value in data["data"].items():
                try:
                    self.registry_class.register_artifact(key, value)
                except Exception as e:
                    # Log warning but continue
                    import logging
                    logging.warning(f"Failed to deserialize {key}: {e}")
        
        # Add methods to registry class as unbound functions
        setattr(self.registry_class, 'save_to_json', save_to_json)
        setattr(self.registry_class, 'load_from_json', load_from_json)
        setattr(self.registry_class, 'save_to_yaml', save_to_yaml)
        setattr(self.registry_class, 'load_from_yaml', load_from_yaml)
        setattr(self.registry_class, 'save_to_pickle', save_to_pickle)
        setattr(self.registry_class, 'load_from_pickle', load_from_pickle)
        setattr(self.registry_class, '_serialize_registry', _serialize_registry)
        setattr(self.registry_class, '_deserialize_registry', _deserialize_registry)
