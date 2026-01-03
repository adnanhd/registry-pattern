r"""DI Container / IoC framework for recursive object graph construction.

This module provides:
  - `BuildCfg`: Pydantic model for the config envelope (type/repo/data/meta)
  - `is_build_cfg`: Check if a value looks like a BuildCfg dict
  - `normalize_cfg`: Normalize a raw dict into BuildCfg format

The BuildCfg envelope schema:
  - type: str - identifier of the artifact to build
  - repo: str - which registry to use (default: "default")
  - data: dict - kwargs to pass to the builder
  - meta: dict - metadata attached to built objects

Unknown top-level keys are moved to meta._extra_cfg_keys.
Unknown data keys (not in builder signature) go to meta._unused_data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Type, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger(__name__)

__all__ = [
    "BuildCfg",
    "is_build_cfg",
    "normalize_cfg",
]


class BuildCfg(BaseModel):
    """Config envelope for DI container.

    Attributes:
        type: Identifier of the artifact to build.
        repo: Registry namespace to use (default: "default").
        data: Keyword arguments for the builder.
        meta: Metadata to attach to built object.

    Unknown top-level keys are automatically moved to meta._extra_cfg_keys.
    """

    model_config = ConfigDict(extra="allow")

    type: str
    repo: str = "default"
    data: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def move_extra_to_meta(self) -> "BuildCfg":
        """Move unknown top-level keys to meta._extra_cfg_keys."""
        extras = getattr(self, "__pydantic_extra__", None) or {}
        if extras:
            # Ensure meta is a mutable copy
            if not isinstance(self.meta, dict):
                self.meta = dict(self.meta)
            self.meta.setdefault("_extra_cfg_keys", {}).update(dict(extras))
            # Clear extras from the model
            try:
                extras.clear()
            except Exception:
                pass
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "Moved %d extra keys to meta._extra_cfg_keys",
                    len(self.meta.get("_extra_cfg_keys", {})),
                )
        return self

    def with_unused_data(self, unused: Dict[str, Any]) -> "BuildCfg":
        """Return a copy with unused data merged into meta._unused_data."""
        if not unused:
            return self
        new_meta = dict(self.meta)
        new_meta.setdefault("_unused_data", {}).update(unused)
        return self.model_copy(update={"meta": new_meta})


def is_build_cfg(value: Any) -> bool:
    """Check if a value looks like a BuildCfg dict.

    Args:
        value: Any value to check.

    Returns:
        True if value is a dict with a "type" key (string).
    """
    if isinstance(value, BuildCfg):
        return True
    if isinstance(value, dict):
        return isinstance(value.get("type"), str)
    return False


def normalize_cfg(
    cfg: Union[Dict[str, Any], BuildCfg],
    *,
    context: Optional[Dict[str, Any]] = None,
) -> BuildCfg:
    """Normalize a raw dict or BuildCfg into a validated BuildCfg.

    Args:
        cfg: Raw config dict or BuildCfg instance.
        context: Optional validation context to pass to Pydantic.

    Returns:
        Validated BuildCfg instance.

    Raises:
        ValidationError: If cfg doesn't match the BuildCfg schema.
    """
    if isinstance(cfg, BuildCfg):
        return cfg
    return BuildCfg.model_validate(cfg, context=context)
