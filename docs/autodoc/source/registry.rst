registry package
================

Submodules
----------

registry.\_\_main\_\_ module
----------------------------

.. automodule:: registry.__main__
   :members:
   :undoc-members:
   :show-inheritance:

registry.container module
-------------------------

The ``BuildCfg`` envelope (``{type, repo, data, meta}``) plus the
``is_build_cfg`` / ``normalize_cfg`` helpers that route every envelope
detection through Pydantic ``model_validate``.

.. automodule:: registry.container
   :members:
   :undoc-members:
   :show-inheritance:

registry.factory module
-----------------------

The ``build`` pipeline (recurse, validate, pre-call, invoke, post-init,
meta-schema), the ``serialize`` round-trip, and the ``register_ref_scheme``
extension point for ``$scheme://`` references inside config trees.

.. automodule:: registry.factory
   :members:
   :undoc-members:
   :show-inheritance:

registry.typ\_registry module
-----------------------------

.. automodule:: registry.typ_registry
   :members:
   :undoc-members:
   :show-inheritance:

registry.fnc\_registry module
-----------------------------

.. automodule:: registry.fnc_registry
   :members:
   :undoc-members:
   :show-inheritance:

registry.type\_guard module
---------------------------

Pydantic type guard ``Buildable[T]`` accepting either an instance of ``T``
or a ``BuildCfg`` that builds to ``T``.

.. automodule:: registry.type_guard
   :members:
   :undoc-members:
   :show-inheritance:

registry.schema module
----------------------

.. automodule:: registry.schema
   :members:
   :undoc-members:
   :show-inheritance:

registry.validators module
--------------------------

.. automodule:: registry.validators
   :members:
   :undoc-members:
   :show-inheritance:

registry.markers module
-----------------------

.. automodule:: registry.markers
   :members:
   :undoc-members:
   :show-inheritance:

registry.meters module
----------------------

.. automodule:: registry.meters
   :members:
   :undoc-members:
   :show-inheritance:

registry.reporters module
-------------------------

.. automodule:: registry.reporters
   :members:
   :undoc-members:
   :show-inheritance:

registry.engines module
-----------------------

.. automodule:: registry.engines
   :members:
   :undoc-members:
   :show-inheritance:

registry.storage module
-----------------------

.. automodule:: registry.storage
   :members:
   :undoc-members:
   :show-inheritance:

registry.utils module
---------------------

.. automodule:: registry.utils
   :members:
   :undoc-members:
   :show-inheritance:

registry.mixin package
----------------------

.. automodule:: registry.mixin
   :members:
   :undoc-members:
   :show-inheritance:

registry.mixin.accessor module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: registry.mixin.accessor
   :members:
   :undoc-members:
   :show-inheritance:

registry.mixin.mutator module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: registry.mixin.mutator
   :members:
   :undoc-members:
   :show-inheritance:

registry.mixin.validator module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: registry.mixin.validator
   :members:
   :undoc-members:
   :show-inheritance:

registry.experimental package
-----------------------------

.. automodule:: registry.experimental.torch_compat
   :members:
   :undoc-members:
   :show-inheritance:
