registry-pattern
================

Name-based class/function registries plus a recursive factory that builds
object graphs from JSON-friendly envelopes. Pydantic-validated, observable,
hierarchical, and stdlib-only at the core.

Overview
--------

The library is two cooperating pieces:

- **Registries** -- ``TypeRegistry[T]`` and ``FunctionalRegistry`` let classes
  and functions register themselves by name under a dotted ``repo`` path with
  ``@MyReg.register_artifact``.
- **Factory** -- a single recursive :func:`registry.build` consumes a
  ``BuildCfg``-shaped envelope ``{type, repo, data, meta}`` (or a class, or a
  string name), validates the kwargs against the target's Pydantic schema,
  recurses on nested envelopes, resolves ``$ref`` strings against sibling and
  ctx scope, invokes the target, and runs registry hooks. The symmetric
  :func:`registry.serialize` handles the outbound side.

Around those sit tree-shaped sub-registrars (``post_init`` / ``serialize_meta``
hooks inherited via Python inheritance), an annotated marker contract
(``registry.markers``), and an observability split between *meters* (which
write measurements into the envelope's ``meta``) and *reporters* (which ship
events to external sinks).

Installation
------------

.. code-block:: bash

   pip install registry-pattern

The core depends only on ``pydantic`` and ``typing-extensions``. Optional
extras:

.. code-block:: bash

   pip install 'registry-pattern[yaml]'    # ConfigFileEngine.yaml loader
   pip install 'registry-pattern[otel]'    # OpenTelemetryReporter
   pip install 'registry-pattern[torch]'   # deps for examples/torch_compat.py
   pip install 'registry-pattern[all]'     # everything above + docs / dev

Quick start
-----------

.. code-block:: python

   import torch.nn as nn
   from registry import TypeRegistry, build


   class ModelRegistry(TypeRegistry[nn.Module]):
       pass


   @ModelRegistry.register_artifact
   class MLP(nn.Module):
       def __init__(self, hidden: int = 128) -> None:
           super().__init__()
           self.hidden = hidden


   model = build({"type": "MLP", "data": {"hidden": 256}})
   assert isinstance(model, MLP) and model.hidden == 256

Other ways to build the same object:

.. code-block:: python

   build(MLP, {"hidden": 256}, validator="python")   # class + kwargs dict
   build(MLP, "hidden: 256\n", validator="yaml")       # class + raw YAML
   build("MLP", {"hidden": 256})                        # string name + kwargs

And the round-trip back out:

.. code-block:: python

   from registry import serialize

   env = serialize(model, serializator="python")   # -> {"type": ..., "data": ...}
   yaml = serialize(model, serializator="yaml")      # YAML of the envelope
   json = serialize(model, serializator="json")      # JSON of the envelope

Documentation
-------------

- :doc:`api` -- the public entry points re-exported from the top-level
  ``registry`` package.
- :doc:`modules` -- per-module reference for the full package.

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Contents:

   api
   modules

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
