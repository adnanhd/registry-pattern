r"""Engine registries for extensible artifact factorization.

Provides two registries:

- ConfigFileEngine: maps file extensions to loader functions
- SocketEngine: maps protocols to RPC/network handlers

Usage::

    @ConfigFileEngine.register_artifact
    def json(filepath: Path) -> dict:
        import json
        return json.load(filepath.open())

    @SocketEngine.register_artifact
    def rpc(type: str, config: dict) -> dict:
        # RPC implementation
        return response_dict
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from .fnc_registry import FunctionalRegistry

logger = logging.getLogger(__name__)

__all__ = ["ConfigFileEngine", "SocketEngine"]


class ConfigFileEngine(FunctionalRegistry[[Path], Dict[str, Any]]):
    """Registry for config file loaders.

    Each registered function takes a filepath and returns a config dict.
    Signature: (filepath: Path) -> Dict[str, Any]
    """


class SocketEngine(FunctionalRegistry[[str, Dict[str, Any]], Dict[str, Any]]):
    """Registry for socket/network handlers.

    Each registered function takes a type and socket config, returns config dict.
    Signature: (type: str, socket_config: Dict[str, Any]) -> Dict[str, Any]
    """


# ============================================================================
# Default Config File Engines
# ============================================================================


@ConfigFileEngine.register_artifact
def json(filepath: Path) -> Dict[str, Any]:
    """Load config from JSON file."""
    import json as json_lib

    with open(filepath, "r") as f:
        return json_lib.load(f)


@ConfigFileEngine.register_artifact
def yaml(filepath: Path) -> Dict[str, Any]:
    """Load config from YAML file."""
    try:
        import yaml as yaml_lib
    except ImportError:
        raise ImportError("PyYAML not installed. Install with: pip install pyyaml")
    with open(filepath, "r") as f:
        return yaml_lib.safe_load(f)


@ConfigFileEngine.register_artifact
def yml(filepath: Path) -> Dict[str, Any]:
    """Load config from YML file (alias for yaml)."""
    return yaml(filepath)


@ConfigFileEngine.register_artifact
def toml(filepath: Path) -> Dict[str, Any]:
    """Load config from TOML file."""
    try:
        import tomli
    except ImportError:
        # Try stdlib tomllib (Python 3.11+)
        try:
            import tomllib as tomli  # type: ignore[import-not-found]
        except ImportError:
            raise ImportError(
                "TOML library not installed. Install with: pip install tomli"
            )
    with open(filepath, "rb") as f:
        return tomli.load(f)


# ============================================================================
# Default Socket Engines
# ============================================================================


@SocketEngine.register_artifact
def rpc(type: str, socket_config: Dict[str, Any]) -> Dict[str, Any]:
    """RPC-based factorization handler.

    Socket config format::

        {
            "host": "localhost",
            "port": 5000,
            "timeout": 10.0,
        }
    """
    try:
        import rpyc
    except ImportError:
        raise ImportError("RPyC not installed. Install with: pip install rpyc")

    host = socket_config.get("host", "localhost")
    port = socket_config.get("port", 18861)
    timeout = socket_config.get("timeout", 10.0)

    try:
        conn = rpyc.connect(host, port, config={"sync_request_timeout": timeout})
        result = conn.root.get_config(type)
        conn.close()
        return result
    except Exception as e:
        logger.error("RPC factorization failed: %s", e)
        raise


@SocketEngine.register_artifact
def http(type: str, socket_config: Dict[str, Any]) -> Dict[str, Any]:
    """HTTP-based factorization handler.

    Socket config format::

        {
            "url": "http://example.com/api/config",
            "method": "POST",  # optional, default GET
            "headers": {...},  # optional
            "timeout": 10.0,   # optional
        }
    """
    import requests

    url = socket_config.get("url")
    if not url:
        raise ValueError("Socket config must include 'url'")

    method = socket_config.get("method", "GET").upper()
    headers = socket_config.get("headers", {})
    timeout = socket_config.get("timeout", 10.0)

    if method == "GET":
        response = requests.get(
            url,
            params={"type": type},
            headers=headers,
            timeout=timeout,
        )
    else:  # POST
        response = requests.post(
            url,
            json={"type": type},
            headers=headers,
            timeout=timeout,
        )

    response.raise_for_status()
    return response.json()
