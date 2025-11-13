r"""Factory pattern mixin for recursive artifact instantiation.

Provides:
  - Scheme storage for artifacts (local or remote)
  - Recursive factorization from dicts, config files, and RPC
  - Extensible engine registries for different instantiation methods
"""

from __future__ import annotations

import inspect
import json
import logging
from pathlib import Path
from typing import Any, ClassVar, Dict, Generic, Hashable, Optional, TypeVar, Union

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from ..utils import ValidationError
from .validator import MutableValidatorMixin

logger = logging.getLogger(__name__)

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")


class RegistryFactorizorMixin(
    MutableValidatorMixin[KeyType, ValType], Generic[KeyType, ValType]
):
    """Mixin for factory-pattern instantiation with recursive validation.

    Responsibilities:
        - Store pydantic schemes for registered artifacts
        - Factorize artifacts from dicts, files, or network
        - Recursively validate and instantiate nested dependencies
    """

    _schemetry: ClassVar[Dict[str, type[BaseModel]]]  # scheme storage
    _scheme_storage_proxy: ClassVar[
        Optional[Any]
    ] = None  # remote storage if configured

    @classmethod
    def __init_subclass__(
        cls,
        scheme_namespace: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init_subclass__(**kwargs)

        if scheme_namespace is None:
            cls._schemetry = {}
            cls._scheme_storage_proxy = None
        else:
            # Use remote storage for schemes
            from ..storage import RemoteStorageProxy

            cls._scheme_storage_proxy = RemoteStorageProxy[str, Dict[str, Any]](
                namespace=scheme_namespace
            )
            cls._schemetry = {}  # Local cache

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Initialized FactorizorMixin for %s (scheme_namespace=%s)",
                cls.__name__,
                scheme_namespace,
            )

    @classmethod
    def _save_scheme(cls, identifier: str, scheme: type[BaseModel]) -> None:
        """Save a pydantic scheme to storage."""
        cls._schemetry[identifier] = scheme

        if cls._scheme_storage_proxy is not None:
            # Serialize scheme as json schema
            schema_dict = scheme.model_json_schema()
            cls._scheme_storage_proxy[identifier] = schema_dict

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Saved scheme for %s", identifier)

    @classmethod
    def _get_scheme(cls, identifier: str) -> type[BaseModel]:
        """Retrieve a pydantic scheme from storage."""
        if identifier in cls._schemetry:
            return cls._schemetry[identifier]

        if cls._scheme_storage_proxy is not None:
            # Try remote storage
            try:
                schema_dict = cls._scheme_storage_proxy[identifier]
                # Note: reconstructing pydantic models from json schema is complex
                # For now, we rely on local cache
                raise ValidationError(
                    f"Scheme {identifier} not in local cache",
                    ["Ensure artifact is registered in this process"],
                    {"identifier": identifier},
                )
            except Exception:
                pass

        raise ValidationError(
            f"Scheme not found: {identifier}",
            ["Register the artifact first", "Check the identifier spelling"],
            {"identifier": identifier},
        )

    @classmethod
    def _extract_scheme_from_artifact(cls, artifact: Any) -> type[BaseModel]:
        """Extract or create a pydantic scheme from an artifact.

        For functions: use signature parameters
        For classes: use __init__ signature (excluding self)
        """
        from inspect import Parameter, _empty, signature

        from pydantic import create_model

        # Determine if it's a class or function
        if inspect.isclass(artifact):
            sig = signature(artifact.__init__)
            params = list(sig.parameters.values())
            # Remove 'self'
            if params and params[0].name == "self":
                params = params[1:]
        elif callable(artifact):
            sig = signature(artifact)
            params = list(sig.parameters.values())
        else:
            raise ValidationError(
                "Artifact must be a class or callable",
                ["Pass a function or class"],
                {"artifact_type": type(artifact).__name__},
            )

        # Build pydantic fields from signature
        fields: Dict[str, Any] = {}
        for param in params:
            if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
                continue

            annotation = param.annotation if param.annotation is not _empty else Any
            default = param.default if param.default is not _empty else ...
            fields[param.name] = (annotation, default)

        # Create pydantic model
        model_name = f"{getattr(artifact, '__name__', 'Artifact')}Scheme"
        return create_model(model_name, **fields)  # type: ignore

    @classmethod
    def _inject_pydantic_handler(cls, target: Any, identifier: str) -> None:
        """Inject __get_pydantic_core_schema__ and __get_pydantic_json_schema__ for Pydantic v2."""
        from pydantic_core import core_schema

        def __get_pydantic_core_schema__(source_type, handler):
            """Allow Pydantic to validate this type from dict or instance."""

            def validate_from_dict_or_instance(value: Any):
                # Already an instance? Return as-is
                if isinstance(value, source_type):
                    return value
                # Dict? Factorize it recursively
                if isinstance(value, dict):
                    return cls.factorize_artifact(identifier, **value)
                # Invalid type
                raise ValueError(
                    f"Expected {source_type.__name__} instance or dict, "
                    f"got {type(value).__name__}"
                )

            # Use union schema: either instance or dict
            python_schema = core_schema.union_schema(
                [
                    core_schema.is_instance_schema(source_type),
                    core_schema.chain_schema(
                        [
                            core_schema.dict_schema(),
                            core_schema.no_info_plain_validator_function(
                                validate_from_dict_or_instance
                            ),
                        ]
                    ),
                ]
            )

            return core_schema.with_info_plain_validator_function(
                lambda value, _: validate_from_dict_or_instance(value),
                serialization=core_schema.plain_serializer_function_ser_schema(
                    lambda instance: instance,
                    return_schema=core_schema.any_schema(),
                ),
            )

        def __get_pydantic_json_schema__(core_schema_obj, handler):
            """Provide JSON schema representation for this type."""
            # Get the scheme for this type to generate proper JSON schema
            try:
                scheme = cls._get_scheme(identifier)
                # Use the scheme's JSON schema as our JSON schema
                return handler(scheme.__pydantic_core_schema__)
            except Exception:
                # Fallback: return object schema
                return {"type": "object"}

        # Inject as classmethods for classes
        if inspect.isclass(target):
            setattr(
                target,
                "__get_pydantic_core_schema__",
                classmethod(__get_pydantic_core_schema__),
            )
            setattr(
                target,
                "__get_pydantic_json_schema__",
                staticmethod(__get_pydantic_json_schema__),
            )
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Injected pydantic handlers into %s", target.__name__)

    @classmethod
    def _internalize_artifact(cls, value: Any) -> ValType:
        """Override to save schemes and inject pydantic handlers when registering."""
        # Call parent validation
        artifact = super()._internalize_artifact(value)

        # Extract and save scheme, inject pydantic handler
        try:
            identifier = str(cls._identifier_of(artifact))
            scheme = cls._extract_scheme_from_artifact(artifact)
            cls._save_scheme(identifier, scheme)

            # Inject pydantic handler for recursive validation
            cls._inject_pydantic_handler(artifact, identifier)

        except Exception as e:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Could not extract scheme: %s", e)

        return artifact

    @classmethod
    def _recursive_factorize(cls, annotation: Any, value: Any) -> Any:
        """Recursively factorize nested artifacts.

        If annotation is a registered type and value is a dict,
        factorize that type from the dict.
        """
        # Get origin and args for generic types
        from typing import get_args, get_origin

        origin = get_origin(annotation)

        # Handle Optional[X] -> Union[X, None]
        if origin is Union:
            args = get_args(annotation)
            # Try each arg that isn't None
            for arg in args:
                if arg is type(None):
                    continue
                try:
                    return cls._recursive_factorize(arg, value)
                except Exception:
                    continue
            return value

        # If it's a registered type and value is a dict, factorize it
        if inspect.isclass(annotation) and isinstance(value, dict):
            try:
                # Check if this type is registered in any registry
                identifier = str(getattr(annotation, "__name__", annotation))
                if identifier in cls._schemetry or cls.has_identifier(identifier):
                    return cls.factorize_artifact(identifier, **value)
            except Exception as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("Could not recursive factorize %s: %s", annotation, e)

        return value

    @classmethod
    def factorize_artifact(
        cls,
        type: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Factorize (instantiate) an artifact with recursive validation.

        Args:
            type: Identifier of the registered artifact
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Instantiated artifact

        Raises:
            ValidationError: If validation or instantiation fails
        """
        # Get the artifact and scheme
        artifact = cls.get_artifact(type)
        scheme = cls._get_scheme(type)

        # If args provided, convert to kwargs using parameter names
        if args:
            sig = inspect.signature(
                artifact.__init__ if inspect.isclass(artifact) else artifact
            )
            params = list(sig.parameters.values())
            if inspect.isclass(artifact) and params and params[0].name == "self":
                params = params[1:]

            for i, (param, arg) in enumerate(zip(params, args)):
                if param.name not in kwargs:
                    kwargs[param.name] = arg

        # Recursively factorize nested artifacts
        sig = inspect.signature(
            artifact.__init__ if inspect.isclass(artifact) else artifact
        )
        params = list(sig.parameters.values())
        if inspect.isclass(artifact) and params and params[0].name == "self":
            params = params[1:]

        for param in params:
            if param.name in kwargs and param.annotation is not inspect.Parameter.empty:
                kwargs[param.name] = cls._recursive_factorize(
                    param.annotation, kwargs[param.name]
                )

        # Validate with pydantic
        try:
            validated = scheme(**kwargs)
        except PydanticValidationError as e:
            raise ValidationError(
                f"Validation failed for {type}: {e}",
                ["Check argument types and required fields"],
                {"type": type, "errors": str(e)},
            ) from e

        # Instantiate
        config_dict = (
            validated.model_dump()
            if hasattr(validated, "model_dump")
            else validated.dict()
        )

        if inspect.isclass(artifact):
            return artifact(**config_dict)
        else:
            return artifact(**config_dict)

    @classmethod
    def factorize_from_file(
        cls,
        type: str,
        filepath: Union[str, Path],
        engine: Optional[str] = None,
    ) -> Any:
        """Factorize an artifact from a config file.

        Args:
            type: Identifier of the registered artifact
            filepath: Path to config file
            engine: Optional engine name (auto-detected from extension if None)

        Returns:
            Instantiated artifact
        """
        from ..engines import ConfigFileEngine

        filepath = Path(filepath)

        if not filepath.exists():
            raise ValidationError(
                f"Config file not found: {filepath}",
                ["Check the file path"],
                {"filepath": str(filepath)},
            )

        # Auto-detect engine from extension
        if engine is None:
            engine = filepath.suffix.lstrip(".")

        # Load config using engine
        try:
            loader = ConfigFileEngine.get_artifact(engine)
            config = loader(filepath)
        except Exception as e:
            raise ValidationError(
                f"Failed to load config from {filepath}: {e}",
                [
                    f"Ensure {engine} engine is registered",
                    "Check file format is valid",
                ],
                {"filepath": str(filepath), "engine": engine},
            ) from e

        # Factorize from loaded config
        return cls.factorize_artifact(type, **config)

    @classmethod
    def factorize_from_socket(
        cls,
        type: str,
        socket_config: Dict[str, Any],
        protocol: str = "rpc",
    ) -> Any:
        """Factorize an artifact via network/RPC.

        Args:
            type: Identifier of the registered artifact
            socket_config: Configuration for socket connection
            protocol: Protocol name (default: "rpc")

        Returns:
            Instantiated artifact
        """
        from ..engines import SocketEngine

        try:
            handler = SocketEngine.get_artifact(protocol)
            config = handler(type, socket_config)
        except Exception as e:
            raise ValidationError(
                f"Failed to factorize via {protocol}: {e}",
                [
                    f"Ensure {protocol} engine is registered",
                    "Check socket configuration",
                ],
                {"protocol": protocol, "type": type},
            ) from e

        return cls.factorize_artifact(type, **config)
