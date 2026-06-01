# Configuration file for the Sphinx documentation builder.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from registry._version import __version__  # noqa: E402

project = "registry-pattern"
copyright = "2024-2026, Adnan Harun Dogan"
author = "Adnan Harun Dogan"
release = __version__
version = __version__

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.mermaid",
]

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "show-inheritance": True,
}

# Fix for ParamSpec and typing_extensions issues
autodoc_type_aliases = {
    "ParamSpec": "typing_extensions.ParamSpec",
}


# Skip problematic members that cause AttributeError
def skip_paramspec(app, what, name, obj, skip, options):
    """Skip ParamSpec and other problematic typing constructs."""
    import sys

    if sys.version_info < (3, 9):
        # Skip ParamSpec in Python < 3.9 to avoid autodoc errors
        if hasattr(obj, "__class__"):
            class_name = obj.__class__.__name__
            if class_name in ("ParamSpec", "_ParamSpecMeta", "TypeVar"):
                return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_paramspec)


html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../../../"))

# Suppress warnings for known issues
suppress_warnings = ["app.add_directive"]
