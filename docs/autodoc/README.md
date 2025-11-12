# Registry Sphinx Autodoc Setup

Automatic documentation generation from source code docstrings using Sphinx.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Build HTML documentation:
```bash
make html
```

3. View documentation:
```bash
open build/html/index.html
```

## Structure

- `source/conf.py` — Sphinx configuration
- `source/index.rst` — Main index
- `source/modules.rst` — Module listing
- `source/registry.rst` — Auto-documented registry modules
- `Makefile` — Build automation
- `requirements.txt` — Python dependencies

## Files Generated

Documentation is automatically extracted from:
- `registry/__init__.py`
- `registry/__main__.py`
- `registry/engines.py`
- `registry/fnc_registry.py`
- `registry/typ_registry.py`
- `registry/sch_registry.py`
- `registry/storage.py`
- `registry/utils.py`
- `registry/mixin/accessor.py`
- `registry/mixin/mutator.py`
- `registry/mixin/validator.py`
- `registry/mixin/factorizor.py`

## Configuration

Edit `source/conf.py` to customize:
- Theme (default: alabaster)
- Extensions
- Output format
- Build options

## Output

Generated HTML documentation in: `build/html/`

Serve locally:
```bash
python -m http.server --directory build/html 8000
# Then open http://localhost:8000
```
