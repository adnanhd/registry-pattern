"""Coverage for registry._version: version strings, package probing, debug info.

Exercises the public helpers (get_version, get_version_info, format/print,
get_debug_info) plus the private package-probing helpers across their found /
not-found / exception branches.
"""

from __future__ import annotations

import pytest

from registry import _version
from registry._version import (
    __version__,
    _get_package_version,
    _get_pip_package_version,
    format_version_info,
    get_debug_info,
    get_dependency_versions,
    get_platform_info,
    get_python_info,
    get_version,
    get_version_info,
    print_version_info,
)


# -- get_version -------------------------------------------------------------


def test_get_version_no_suffix() -> None:
    expected = f"{_version.VERSION_MAJOR}.{_version.VERSION_MINOR}.{_version.VERSION_PATCH}"
    assert get_version() == expected
    assert get_version() == __version__


def test_get_version_with_suffix(monkeypatch) -> None:
    monkeypatch.setattr(_version, "VERSION_SUFFIX", "rc1")
    assert get_version() == (
        f"{_version.VERSION_MAJOR}.{_version.VERSION_MINOR}."
        f"{_version.VERSION_PATCH}-rc1"
    )


# -- info dicts --------------------------------------------------------------


def test_get_python_info_keys() -> None:
    info = get_python_info()
    for key in ("version", "implementation", "executable", "prefix"):
        assert key in info


def test_get_platform_info_keys() -> None:
    info = get_platform_info()
    for key in ("system", "release", "version", "machine", "processor"):
        assert key in info


def test_get_dependency_versions_has_pydantic() -> None:
    deps = get_dependency_versions()
    assert "pydantic" in deps
    # pydantic is a hard dependency, so it must resolve to a version string.
    assert deps["pydantic"]


def test_get_version_info_structure() -> None:
    info = get_version_info()
    assert info["registry_pattern"] == __version__
    assert "python" in info
    assert "platform" in info
    assert "dependencies" in info


# -- formatting / printing ---------------------------------------------------


def test_format_version_info_default() -> None:
    text = format_version_info()
    assert "registry-pattern:" in text
    assert "Python:" in text
    assert "Platform:" in text
    assert "Dependencies:" in text


def test_format_version_info_explicit_info() -> None:
    info = get_version_info()
    text = format_version_info(info)
    assert info["registry_pattern"] in text


def test_print_version_info(capsys) -> None:
    print_version_info()
    out = capsys.readouterr().out
    assert "registry-pattern:" in out


# -- get_debug_info ----------------------------------------------------------


def test_get_debug_info_contains_version_and_platform() -> None:
    debug = get_debug_info()
    assert f"registry-pattern={__version__}" in debug
    assert "python=" in debug
    assert "platform=" in debug
    # pydantic is installed, so its marker must appear.
    assert "pydantic=" in debug


# -- private probing helpers -------------------------------------------------


def test_pip_version_unknown_package_returns_none() -> None:
    assert _get_pip_package_version("registry-pattern-no-such-pkg-zzz") is None


def test_pip_version_subprocess_error_returns_none(monkeypatch) -> None:
    def _boom(*args, **kwargs):
        raise OSError("subprocess unavailable")

    monkeypatch.setattr(_version.subprocess, "run", _boom)
    assert _get_pip_package_version("anything") is None


def test_package_version_missing_module_returns_none() -> None:
    assert _get_package_version("registry_no_such_module_zzz") is None


def test_package_version_found_has_version() -> None:
    # pydantic exposes __version__, so this resolves without a pip fallback.
    assert _get_package_version("pydantic")
