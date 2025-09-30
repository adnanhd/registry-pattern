r"""Remote storage proxy and thread-safe local storage for registries."""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import math
import os
import pickle
import re
import weakref
from collections.abc import ItemsView, KeysView, ValuesView
from threading import RLock
from typing import Any, Dict, Generic, Hashable, Iterator, MutableMapping, TypeVar
from urllib.parse import quote

import requests

from .utils import RegistryKeyError, ValidationError

logger = logging.getLogger(__name__)

__all__ = ["RemoteStorageProxy", "ThreadSafeLocalStorage"]

KeyType = TypeVar("KeyType", bound=Hashable)
ValType = TypeVar("ValType")


def _is_valid_hostname(name: str) -> bool:
    _HOST_LABEL_RE = re.compile(r"^(?!-)[A-Za-z0-9-]{1,63}(?<!-)$")

    # RFC 1035/1123-ish: total <= 253, labels 1..63, alnum/hyphen, no leading/trailing hyphen.
    if len(name) > 253:
        return False
    if name.endswith("."):
        name = name[:-1]
    labels = name.split(".")
    return all(_HOST_LABEL_RE.match(lbl) for lbl in labels)


def _normalize_host_for_url(host: str) -> str:
    """
    Return host normalized for inclusion in http://<host>:<port>
    Wrap IPv6 in [] and leave hostnames/IPv4 untouched.
    """
    raw = host.strip()
    # Reject schemes and pathy junk early
    if "://" in raw:
        raise ValidationError(
            f"host must be a bare hostname or IP, not a URL: '{host}'",
            ["Use values like 'localhost', 'example.com', '10.0.0.5', or '::1'"],
            {"host": host},
        )
    if "/" in raw or " " in raw:
        raise ValidationError(
            f"host contains invalid characters: '{host}'",
            ["Do not include paths or spaces; provide only a hostname or IP."],
            {"host": host},
        )
    # Allow user-provided brackets for IPv6, but normalize to single pair
    unbracketed = raw[1:-1] if (raw.startswith("[") and raw.endswith("]")) else raw

    # Try IP first
    try:
        ip = ipaddress.ip_address(unbracketed)
        if ip.version == 6:
            return f"[{ip.compressed}]"
        return ip.compressed  # IPv4 normalized
    except ValueError:
        pass

    # Fallback to hostname validation (also allow 'localhost')
    if unbracketed.lower() == "localhost" or _is_valid_hostname(unbracketed):
        return unbracketed
    raise ValidationError(
        f"Invalid host: '{host}'",
        [
            "Use a valid hostname (e.g., 'api.example.com') or IP (e.g., '192.168.1.10' or '::1')."
        ],
        {"host": host},
    )


def _validate_namespace(namespace: str) -> str:
    _NAMESPACE_SHAPE_RE = re.compile(r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*$")
    _MAX_NAMESPACE_LENGTH = 253

    if not isinstance(namespace, str):
        raise ValidationError(
            f"namespace must be a string, got {type(namespace).__name__}",
            ["Use dot-delimited namespace: 'my.app.namespace'"],
            {"namespace_type": type(namespace).__name__},
        )

    namespace = namespace.strip()

    if (
        not namespace
        or ".." in namespace
        or namespace.startswith(".")
        or namespace.endswith(".")
        or not _NAMESPACE_SHAPE_RE.fullmatch(namespace)
    ):
        suggestions = ["Use valid dot-delimited alpha-numeric path: 'my.app.namespace'"]
        detail = f"Invalid namespace: '{namespace}'"

        if not namespace:
            detail = "Namespace cannot be empty"
        if not _NAMESPACE_SHAPE_RE.fullmatch(namespace):
            detail = "Namespace cannot contain non-alphanumeric characters"
        if namespace.startswith("."):
            detail = "Namespace cannot start with '.' or '..'"
        if namespace.endswith("."):
            detail = "Namespace cannot end with '.' or '..'"
        if ".." in namespace:
            detail = "Namespace cannot contain '..'"

        raise ValidationError(
            detail,
            suggestions,
            {"namespace": namespace},
        )

    return namespace


def _validate_port(port: int) -> int:
    if isinstance(port, bool):
        # bool is a subclass of int; explicitly reject it
        raise ValidationError(
            f"port must be int in [1, 65535], got bool",
            ["Provide a numeric TCP port like 80, 443, 5555."],
            {"port": port},
        )
    if not isinstance(port, int):
        raise ValidationError(
            f"port must be int in [1, 65535], got {type(port).__name__}",
            ["Cast to int and try again."],
            {"port_type": type(port).__name__},
        )
    if not (1 <= port <= 65535):
        raise ValidationError(
            f"port out of range: {port}",
            ["Choose a TCP port between 1 and 65535."],
            {"port": port},
        )
    return port


def _validate_timeout(timeout: float) -> float:
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        raise ValidationError(
            f"timeout must be a positive finite number, got {type(timeout).__name__}",
            ["Use a small number of seconds like 5, 10, or 30."],
            {"timeout_type": type(timeout).__name__},
        )
    t = float(timeout)
    if not math.isfinite(t) or t <= 0.0:
        raise ValidationError(
            f"timeout must be positive and finite, got {timeout}",
            ["Pick something > 0 and not inf/NaN."],
            {"timeout": timeout},
        )
    # Optional sanity cap to catch accidental minutes-as-seconds typos
    if t > 300:
        raise ValidationError(
            f"timeout suspiciously large: {timeout} seconds",
            [
                "Did you mean seconds? Keep it under 300 unless you enjoy hanging clients."
            ],
            {"timeout": timeout},
        )
    return t


class SerializationError(ValidationError):
    """Raised when object serialization/deserialization fails."""


def _serialize_value(value: Any) -> Dict[str, Any]:
    """Serialize a value for network transmission."""
    if isinstance(value, weakref.ref):
        referent = value()
        if referent is None:
            raise SerializationError(
                "Cannot serialize dead weakref",
                ["Keep strong reference to object before syncing"],
                {"operation": "serialize_weakref"},
            )
        return {
            "type": "weakref",
            "data": _serialize_value(referent)["data"],
            "encoding": _serialize_value(referent)["encoding"],
        }

    try:
        json_str = json.dumps(value)
        return {"type": "json", "data": json_str, "encoding": "json"}
    except (TypeError, ValueError):
        pass

    try:
        pickled = pickle.dumps(value)
        encoded = base64.b64encode(pickled).decode("ascii")
        return {"type": "pickle", "data": encoded, "encoding": "base64"}
    except Exception as e:
        raise SerializationError(
            f"Failed to serialize value: {e}",
            ["Ensure object is pickleable"],
            {"value_type": type(value).__name__},
        ) from e


def _deserialize_value(serialized: Dict[str, Any]) -> Any:
    """Deserialize a value from network transmission."""
    value_type = serialized.get("type")
    data = serialized.get("data")

    if value_type == "json":
        return json.loads(data)
    elif value_type == "pickle":
        decoded = base64.b64decode(data)
        return pickle.loads(decoded)
    elif value_type == "weakref":
        encoding = serialized.get("encoding")
        inner_data = {
            "type": "pickle" if encoding == "base64" else "json",
            "data": data,
        }
        obj = _deserialize_value(inner_data)
        try:
            return weakref.ref(obj)
        except TypeError:
            return obj
    else:
        raise SerializationError(
            f"Unknown serialization type: {value_type}",
            ["Check server/client version compatibility"],
            {"type": value_type},
        )


class ThreadSafeLocalStorage(
    MutableMapping[KeyType, ValType], Generic[KeyType, ValType]
):
    """Thread-safe local storage with single-write multi-read semantics."""

    def __init__(self):
        self._storage: Dict[KeyType, ValType] = {}
        self._lock = RLock()

    def __getitem__(self, key: KeyType) -> ValType:
        with self._lock:
            return self._storage[key]

    def __setitem__(self, key: KeyType, value: ValType) -> None:
        with self._lock:
            self._storage[key] = value

    def __delitem__(self, key: KeyType) -> None:
        with self._lock:
            del self._storage[key]

    def __iter__(self) -> Iterator[KeyType]:
        with self._lock:
            return iter(list(self._storage.keys()))

    def __len__(self) -> int:
        with self._lock:
            return len(self._storage)

    def __contains__(self, key: object) -> bool:
        with self._lock:
            return key in self._storage

    def clear(self) -> None:
        with self._lock:
            self._storage.clear()

    def keys(self) -> KeysView[KeyType]:
        with self._lock:
            return self._storage.keys()

    def values(self) -> ValuesView[ValType]:
        with self._lock:
            return self._storage.values()

    def items(self) -> ItemsView[KeyType, ValType]:
        with self._lock:
            return self._storage.items()


class RemoteStorageProxy(MutableMapping[KeyType, ValType], Generic[KeyType, ValType]):
    """HTTP client proxy for remote registry storage."""

    def __init__(
        self,
        namespace: str,
        host: str = os.getenv("REGISTRY_SERVER_HOST", "localhost"),
        port: int = int(os.getenv("REGISTRY_SERVER_PORT", "8001")),
        timeout: float = float(os.getenv("REGISTRY_SERVER_TIMEOUT", "5.0")),
    ):
        """Initialize remote storage proxy.

        Args:
            namespace: Dot-delimited namespace (e.g., "my.app.models")
            host: Server hostname
            port: Server port
            timeout: Request timeout in seconds
        """
        valid_namespace = _validate_namespace(namespace)
        normalized_host = _normalize_host_for_url(host)
        valid_port = _validate_port(port)
        valid_timeout = _validate_timeout(timeout)

        self.namespace = valid_namespace
        self.host = normalized_host
        self.port = valid_port
        self.timeout = valid_timeout
        self.base_url = f"http://{self.host}:{self.port}"

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RemoteStorageProxy: namespace=%s, url=%s",
                self.namespace,
                self.base_url,
            )

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any] | None:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=self.timeout, **kwargs)
            response.raise_for_status()
            if response.status_code == 204:
                return None
            return response.json()
        except requests.exceptions.Timeout as e:
            raise ValidationError(
                f"Registry server timeout: {self.base_url}",
                [
                    "Check if server is running",
                    "Start server: python -m registry.server",
                ],
                {"namespace": self.namespace},
            ) from e
        except requests.exceptions.ConnectionError as e:
            raise ValidationError(
                f"Cannot connect to registry server: {self.base_url}",
                ["Start server: python -m registry.server"],
                {"namespace": self.namespace},
            ) from e
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise RegistryKeyError(
                    "Key not found in remote registry",
                    ["Verify key exists"],
                    {"namespace": self.namespace},
                ) from e
            raise ValidationError(
                f"Registry server error: {e}",
                ["Check server logs"],
                {"namespace": self.namespace, "status": e.response.status_code},
            ) from e

    def __getitem__(self, key: KeyType) -> ValType:
        key_serialized = _serialize_value(key)
        key_json = json.dumps(key_serialized)
        key_encoded = quote(key_json)

        endpoint = f"/registry/{self.namespace}/get/{key_encoded}"
        response = self._request("GET", endpoint)

        if response is None:
            raise RegistryKeyError(
                f"Key not found: {key}",
                ["Check key exists"],
                {"namespace": self.namespace, "key": str(key)},
            )

        return _deserialize_value(response["value"])

    def __setitem__(self, key: KeyType, value: ValType) -> None:
        key_serialized = _serialize_value(key)
        value_serialized = _serialize_value(value)

        endpoint = f"/registry/{self.namespace}/set"
        self._request(
            "POST", endpoint, json={"key": key_serialized, "value": value_serialized}
        )

    def __delitem__(self, key: KeyType) -> None:
        key_serialized = _serialize_value(key)
        key_json = json.dumps(key_serialized)
        key_encoded = quote(key_json)

        endpoint = f"/registry/{self.namespace}/delete/{key_encoded}"
        self._request("DELETE", endpoint)

    def __iter__(self) -> Iterator[KeyType]:
        endpoint = f"/registry/{self.namespace}/keys"
        response = self._request("GET", endpoint)
        if response is None:
            return iter([])
        keys = response.get("keys", [])
        return iter(_deserialize_value(k) for k in keys)

    def __len__(self) -> int:
        endpoint = f"/registry/{self.namespace}/length"
        response = self._request("GET", endpoint)
        if response is None:
            return 0
        return response.get("length", 0)

    def __contains__(self, key: object) -> bool:
        key_serialized = _serialize_value(key)
        key_json = json.dumps(key_serialized)
        key_encoded = quote(key_json)

        endpoint = f"/registry/{self.namespace}/contains/{key_encoded}"
        response = self._request("GET", endpoint)
        if response is None:
            return False
        return response.get("contains", False)

    def clear(self) -> None:
        endpoint = f"/registry/{self.namespace}/clear"
        self._request("DELETE", endpoint)

    def __repr__(self) -> str:
        return f"<RemoteStorageProxy(namespace='{self.namespace}')>"
