# Sphinx autodoc

Auto-generated API reference for `registry-pattern`. Extracts docstrings
from the source tree and renders them through Sphinx + `sphinx.ext.autodoc`.

## Build

```bash
pip install -r requirements.txt
make html
open build/html/index.html
```

## Layout

- `source/conf.py`     -- Sphinx configuration (project name, theme, extensions)
- `source/index.rst`   -- top-level index
- `source/modules.rst` -- module listing
- `source/registry.rst`-- auto-documented `registry` package
- `source/api.rst`     -- public API reference
- `Makefile`           -- build automation
- `requirements.txt`   -- doc-build deps (`sphinx`, `sphinx-rtd-theme`, etc.)

## Documented modules

Autodoc picks up every public submodule of `registry`:

- `registry.typ_registry`  -- `TypeRegistry`
- `registry.fnc_registry`  -- `FunctionalRegistry`
- `registry.factory`       -- recursive `build()` + `serialize()`
- `registry.container`     -- `BuildCfg`, envelope helpers
- `registry.schema`        -- Pydantic schema derivation
- `registry.type_guard`    -- `Buildable[T]` annotation
- `registry.markers`       -- assertion markers (`AssertEq`, `AssertMin`, ...)
- `registry.validators`    -- validator mediums + `ValidationError`
- `registry.meters`        -- observability meters (write into `meta`)
- `registry.reporters`     -- observability reporters (ship externally)
- `registry.storage`       -- storage backend protocol
- `registry.engines`       -- registration engines
- `registry.resolve_cache` -- name -> class resolution cache
- `registry.utils`         -- shared helpers
- `registry.experimental.torch_compat` -- optional PyTorch shims

## Configuration

Edit `source/conf.py` to customize the theme, extensions, or output
format. The Sphinx config pulls the project version from
`registry/_version.py` so the docs and the package stay in sync.

## Serve locally

```bash
python -m http.server --directory build/html 8000
# open http://localhost:8000
```
