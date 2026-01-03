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

import importlib
import importlib.util
import platform
import subprocess
import sys
from typing import Any, Dict, Optional

# Version components
VERSION_MAJOR = 0
VERSION_MINOR = 4
VERSION_PATCH = 0
VERSION_SUFFIX = ""  # e.g., "alpha", "beta", "rc1"

__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}{'-' + VERSION_SUFFIX if VERSION_SUFFIX else ''}"


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


def _get_pip_package_version(package_name: str) -> Optional[str]:
    """Get package version via pip show.

    Args:
        package_name: Name of the package to query.

    Returns:
        Version string or None if not found.
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if line.startswith("Version:"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return None


def _get_package_version(
    module_name: str, package_name: Optional[str] = None
) -> Optional[str]:
    """Get package version, trying __version__ first, then pip show.

    Args:
        module_name: Name of the module to import.
        package_name: Package name for pip (defaults to module_name).

    Returns:
        Version string or None if not installed.
    """
    if package_name is None:
        package_name = module_name

    # Use importlib.util.find_spec to check if module exists without importing
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None

    # Module exists, try to get version via import
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", None)
    if version:
        return version

    # Fallback to pip show
    return _get_pip_package_version(package_name)


def get_dependency_versions() -> Dict[str, Optional[str]]:
    """Get versions of key dependencies.

    Returns:
        Dict mapping package names to version strings (or None if not installed).
    """
    deps: Dict[str, Optional[str]] = {}

    deps["pydantic"] = _get_package_version("pydantic")
    deps["pydantic_core"] = _get_package_version("pydantic_core", "pydantic-core")
    deps["typing_extensions"] = _get_package_version("typing_extensions")
    deps["torch"] = _get_package_version("torch")
    deps["torchvision"] = _get_package_version("torchvision")

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

    # Python info with aligned colons
    py_info = info["python"]
    py_fields = [
        ("Version", py_info["version"]),
        ("Implementation", py_info["implementation"]),
        ("Executable", py_info["executable"]),
    ]

    # Platform info with aligned colons
    plat_info = info["platform"]
    plat_fields = [
        ("System", plat_info["system"]),
        ("Release", plat_info["release"]),
        ("Machine", plat_info["machine"]),
    ]

    # Dependencies with aligned colons
    deps = info["dependencies"]
    dep_items = [(pkg, ver if ver else "not installed") for pkg, ver in deps.items()]

    # Calculate max widths for alignment
    py_width = max(len(f[0]) for f in py_fields)
    plat_width = max(len(f[0]) for f in plat_fields)
    dep_width = max(len(d[0]) for d in dep_items)
    width = max(py_width, plat_width, dep_width)

    # Python info
    lines = [
        f"registry-pattern: {info['registry_pattern']}",
        "",
        "Python:",
    ]
    for label, value in py_fields:
        lines.append(f"  {label:>{width}} : {value}")

    # Platform info
    lines.append("")
    lines.append("Platform:")

    for label, value in plat_fields:
        lines.append(f"  {label:>{width}} : {value}")

    # Dependencies
    lines.append("")
    lines.append("Dependencies:")

    for pkg, ver in dep_items:
        lines.append(f"  {pkg:>{width}} : {ver}")

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
