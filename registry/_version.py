"""Version and system information for registry-pattern.

This module provides version information and system diagnostics useful for:
- Bug reports and error reporting
- Debugging environment issues
- Compatibility checks

Usage:
    from registry import __version__, get_version_info, print_version_info

    # Simple version string
    print(__version__)  # "0.4.0"

    # Full version info dict
    info = get_version_info()

    # Print formatted version info (for bug reports)
    print_version_info()

CLI Usage:
    python -m registry --version
    python -m registry info
"""

from __future__ import annotations

import platform
import sys
from typing import Any, Dict, Optional

__version__ = "0.4.0"

# Version components
VERSION_MAJOR = 0
VERSION_MINOR = 4
VERSION_PATCH = 0
VERSION_SUFFIX = ""  # e.g., "alpha", "beta", "rc1"


def get_version() -> str:
    """Get the version string.

    Returns:
        Version string in format "X.Y.Z" or "X.Y.Z-suffix".
    """
    if VERSION_SUFFIX:
        return f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}-{VERSION_SUFFIX}"
    return f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}"


def get_python_info() -> Dict[str, str]:
    """Get Python interpreter information.

    Returns:
        Dict with Python version, implementation, and path.
    """
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
        "prefix": sys.prefix,
    }


def get_platform_info() -> Dict[str, str]:
    """Get platform/OS information.

    Returns:
        Dict with OS, architecture, and machine info.
    """
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }


def get_dependency_versions() -> Dict[str, Optional[str]]:
    """Get versions of key dependencies.

    Returns:
        Dict mapping package names to version strings (or None if not installed).
    """
    deps: Dict[str, Optional[str]] = {}

    # Pydantic
    try:
        import pydantic

        deps["pydantic"] = pydantic.__version__
    except ImportError:
        deps["pydantic"] = None

    # Pydantic core
    try:
        import pydantic_core

        deps["pydantic_core"] = pydantic_core.__version__
    except ImportError:
        deps["pydantic_core"] = None

    # typing_extensions
    try:
        import typing_extensions

        deps["typing_extensions"] = getattr(typing_extensions, "__version__", "unknown")
    except ImportError:
        deps["typing_extensions"] = None

    # PyTorch (optional)
    try:
        import torch

        deps["torch"] = torch.__version__
    except ImportError:
        deps["torch"] = None

    # torchvision (optional)
    try:
        import torchvision

        deps["torchvision"] = torchvision.__version__
    except ImportError:
        deps["torchvision"] = None

    return deps


def get_version_info() -> Dict[str, Any]:
    """Get comprehensive version and system information.

    Returns:
        Dict with version, python, platform, and dependency info.

    Example:
        >>> info = get_version_info()
        >>> print(info["registry_pattern"])
        "0.4.0"
        >>> print(info["python"]["version"])
        "3.11.5"
    """
    return {
        "registry_pattern": __version__,
        "python": get_python_info(),
        "platform": get_platform_info(),
        "dependencies": get_dependency_versions(),
    }


def format_version_info(info: Optional[Dict[str, Any]] = None) -> str:
    """Format version info as a human-readable string with aligned colons.

    Args:
        info: Version info dict from get_version_info(). If None, fetches it.

    Returns:
        Formatted multi-line string suitable for bug reports.
    """
    if info is None:
        info = get_version_info()

    lines = [
        f"registry-pattern: {info['registry_pattern']}",
        "",
        "Python:",
    ]

    # Python info with aligned colons
    py_info = info["python"]
    py_fields = [
        ("Version", py_info["version"]),
        ("Implementation", py_info["implementation"]),
        ("Executable", py_info["executable"]),
    ]
    py_width = max(len(f[0]) for f in py_fields)
    for label, value in py_fields:
        lines.append(f"  {label:<{py_width}} : {value}")

    lines.append("")
    lines.append("Platform:")

    # Platform info with aligned colons
    plat_info = info["platform"]
    plat_fields = [
        ("System", plat_info["system"]),
        ("Release", plat_info["release"]),
        ("Machine", plat_info["machine"]),
    ]
    plat_width = max(len(f[0]) for f in plat_fields)
    for label, value in plat_fields:
        lines.append(f"  {label:<{plat_width}} : {value}")

    lines.append("")
    lines.append("Dependencies:")

    # Dependencies with aligned colons
    deps = info["dependencies"]
    dep_items = [(pkg, ver if ver else "not installed") for pkg, ver in deps.items()]
    dep_width = max(len(d[0]) for d in dep_items)
    for pkg, ver in dep_items:
        lines.append(f"  {pkg:<{dep_width}} : {ver}")

    return "\n".join(lines)


def print_version_info() -> None:
    """Print version and system information to stdout.

    Useful for including in bug reports or debugging environment issues.

    Example:
        >>> from registry import print_version_info
        >>> print_version_info()
        registry-pattern: 0.4.0

        Python:
          Version        : 3.11.5
          Implementation : CPython
          ...
    """
    print(format_version_info())


def get_debug_info() -> str:
    """Get debug information string for error messages.

    Returns a compact single-line string suitable for including in error messages.

    Returns:
        Compact debug info string.

    Example:
        >>> get_debug_info()
        "registry-pattern=0.4.0 python=3.11.5 pydantic=2.5.0 platform=Linux"
    """
    info = get_version_info()
    deps = info["dependencies"]

    parts = [
        f"registry-pattern={info['registry_pattern']}",
        f"python={info['python']['version']}",
    ]

    # Add key dependencies if installed
    if deps.get("pydantic"):
        parts.append(f"pydantic={deps['pydantic']}")
    if deps.get("torch"):
        parts.append(f"torch={deps['torch']}")

    parts.append(f"platform={info['platform']['system']}")

    return " ".join(parts)


# For backwards compatibility
version = __version__
VERSION = __version__
