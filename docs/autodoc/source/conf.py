# Configuration file for Sphinx documentation builder
project = "Registry"
copyright = "2024"
author = "Research Team"
release = "0.2.0"

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    'sphinxcontrib.mermaid'
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


html_theme = "alabaster"
html_static_path = ["_static"]

import os
import sys

sys.path.insert(0, os.path.abspath("../../../"))

# Suppress warnings for known issues
suppress_warnings = ["app.add_directive"]
