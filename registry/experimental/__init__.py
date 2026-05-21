"""Experimental extensions -- domain-specific helpers built on the core API.

Modules under ``registry.experimental`` are NOT imported by ``import registry``
and carry no stability guarantees. They package commonly-needed extensions
(torch markers, torch.profiler meter, TensorBoard reporter, ...) so users
don't reinvent them.

Each submodule pins its own optional dependency in ``pyproject.toml``::

    pip install 'registry-pattern[torch]'   # registry.experimental.torch_compat
"""
