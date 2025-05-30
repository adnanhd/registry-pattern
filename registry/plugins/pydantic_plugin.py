# registry/plugins/pydantic_plugin.py
"""Pydantic validation plugin."""

from typing import Type, Any, Dict, List
from .base import BaseRegistryPlugin
from ..core._validator import ValidationError, ConformanceError

try:
    from pydantic import BaseModel, ValidationError as PydanticValidationError # type: ignore
    from pydantic.fields import FieldInfo   # type: ignore

    PYDANTIC_AVAILABLE = True

    class PydanticPlugin(BaseRegistryPlugin):
        """Plugin for Pydantic model validation."""

        @property
        def name(self) -> str:
            return "pydantic"

        @property
        def version(self) -> str:
            return "1.0.0"

        @property
        def dependencies(self) -> List[str]:
            return ["pydantic"]

        def _on_initialize(self) -> None:
            if not PYDANTIC_AVAILABLE:
                raise ImportError("Pydantic is required for this plugin. Install with: pip install registry[pydantic-plugin]")

            # Extend registry with Pydantic validation methods
            if self.registry_class:
                self._add_pydantic_methods()

        def _add_pydantic_methods(self) -> None:
            """Add Pydantic-specific validation methods to the registry."""

            @classmethod
            def validate_pydantic_model(cls, model_class: Type[BaseModel]) -> Type[BaseModel]:
                """Validate that a class is a Pydantic model."""
                if not (isinstance(model_class, type) and issubclass(model_class, BaseModel)):
                    suggestions = [
                        "Ensure the class inherits from pydantic.BaseModel",
                        "Check that pydantic is properly installed",
                        "Use 'from pydantic import BaseModel' in your class definition",
                    ]
                    context = {
                        "expected_type": "pydantic.BaseModel",
                        "actual_type": type(model_class).__name__,
                        "artifact_name": getattr(model_class, "__name__", str(model_class)),
                    }
                    raise ConformanceError(
                        f"Class {getattr(model_class, '__name__', model_class)} is not a Pydantic model",
                        suggestions,
                        context
                    )
                return model_class

            @classmethod
            def validate_with_pydantic(cls, instance: Any, model_class: Type[BaseModel]) -> Any:
                """Validate an instance using a Pydantic model."""
                try:
                    if isinstance(instance, dict):
                        return model_class(**instance)
                    elif isinstance(instance, BaseModel):
                        return model_class.model_validate(instance.model_dump())
                    else:
                        return model_class.model_validate(instance)
                except PydanticValidationError as e:
                    suggestions = [
                        "Check that all required fields are provided",
                        "Verify field types match the model definition",
                        "Use model.model_dump() to see expected format",
                    ]
                    context = {
                        "model_class": model_class.__name__,
                        "validation_errors": str(e),
                    }
                    raise ValidationError(
                        f"Pydantic validation failed: {e}",
                        suggestions,
                        context
                    )

            # Add methods to the registry class
            setattr(self.registry_class, 'validate_pydantic_model', validate_pydantic_model)
            setattr(self.registry_class, 'validate_with_pydantic', validate_with_pydantic)

except ImportError:
    PYDANTIC_AVAILABLE = False
