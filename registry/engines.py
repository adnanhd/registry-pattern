r"""Config file loaders.

``ConfigFileEngine`` is a registry mapping file extensions to loader
callables. The ``yaml`` loader requires the ``[yaml]`` extra; the rest
(json, toml, xml) are stdlib only.

Usage::

    from registry.engines import ConfigFileEngine
    cfg = ConfigFileEngine.get_artifact("yaml")(Path("config.yaml"))
"""

from __future__ import annotations

import json as json_lib
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

import yaml as yaml_lib

# TOML: use tomllib (Python 3.11+) or tomli
if sys.version_info >= (3, 11):
    import tomllib as tomli
else:
    import tomli

from .fnc_registry import FunctionalRegistry

logger = logging.getLogger(__name__)

__all__ = ["ConfigFileEngine"]


class ConfigFileEngine(FunctionalRegistry[[Path], Dict[str, Any]]):
    """Registry for config file loaders.

    Each registered function takes a filepath and returns a config dict.
    Signature: ``(filepath: Path) -> Dict[str, Any]``.
    """


@ConfigFileEngine.register_artifact
def json(filepath: Path) -> Dict[str, Any]:
    """Load config from JSON file."""
    with open(filepath, "r") as f:
        return json_lib.load(f)


@ConfigFileEngine.register_artifact
def yaml(filepath: Path) -> Dict[str, Any]:
    """Load config from YAML file. Requires the ``yaml`` extra."""
    with open(filepath, "r") as f:
        return yaml_lib.safe_load(f)


@ConfigFileEngine.register_artifact
def yml(filepath: Path) -> Dict[str, Any]:
    """Load config from YML file (alias for yaml)."""
    return yaml(filepath)


@ConfigFileEngine.register_artifact
def toml(filepath: Path) -> Dict[str, Any]:
    """Load config from TOML file."""
    with open(filepath, "rb") as f:
        return tomli.load(f)


@ConfigFileEngine.register_artifact
def xml(filepath: Path) -> Dict[str, Any]:
    """Load config from XML file.

    Converts XML to dict structure. Attributes become keys prefixed with '@'.
    Text content becomes '_text' key.
    """

    def element_to_dict(elem: ET.Element) -> Dict[str, Any]:
        result: Dict[str, Any] = {}

        # Attributes with @ prefix
        for key, value in elem.attrib.items():
            result[f"@{key}"] = value

        # Children
        children: Dict[str, list] = {}
        for child in elem:
            child_data = element_to_dict(child)
            if child.tag in children:
                children[child.tag].append(child_data)
            else:
                children[child.tag] = [child_data]
        for tag, items in children.items():
            result[tag] = items[0] if len(items) == 1 else items

        # Text
        if elem.text and elem.text.strip():
            if result:
                result["_text"] = elem.text.strip()
            else:
                return elem.text.strip()  # type: ignore[return-value]
        return result

    tree = ET.parse(filepath)
    root = tree.getroot()
    return {root.tag: element_to_dict(root)}
