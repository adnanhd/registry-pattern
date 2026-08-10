"""ArtifactOf[Registrar] + pydantic validator composition.

Demonstrates the pydantic-native linker: the base type builds (instance-or-config)
and downstream ``Annotated`` metadata are plain after-validators, including a
cross-argument dependency that reads the already-validated sibling from
``info.data``.
"""

from typing import Annotated, Union

import pytest
from pydantic import AfterValidator, BaseModel, ValidationInfo

from registry import TypeRegistry
from registry.integrations.pydantic import ArtifactOf


class Widget:
    def __init__(self, size: int) -> None:
        self.size = size


class Gadget:
    def __init__(self, widget: Widget) -> None:
        self.widget = widget


class WidgetRegistry(TypeRegistry[Widget], repo="widgets"):
    pass


class GadgetRegistry(TypeRegistry[Gadget], repo="gadgets"):
    pass


class Gizmo:
    def __init__(self, power: int) -> None:
        self.power = power


class GizmoRegistry(TypeRegistry[Gizmo], repo="gizmos"):
    pass


class Model:
    def __init__(self, n: int) -> None:
        self._params = [f"w{i}" for i in range(n)]

    def parameters(self) -> list:
        return list(self._params)


class Opt:
    def __init__(self, params: list, lr: float) -> None:
        self.params = list(params)
        self.lr = lr


class ModelRegistry(TypeRegistry[Model], repo="aof_models"):
    pass


class OptRegistry(TypeRegistry[Opt], repo="aof_opts"):
    pass


WidgetRegistry.register_artifact(Widget)
GadgetRegistry.register_artifact(Gadget)
GizmoRegistry.register_artifact(Gizmo)
ModelRegistry.register_artifact(Model)
OptRegistry.register_artifact(Opt)


def test_accepts_live_instance() -> None:
    class Cfg(BaseModel):
        w: ArtifactOf[WidgetRegistry]

    w = Widget(10)
    assert Cfg(w=w).w is w


def test_builds_from_config() -> None:
    class Cfg(BaseModel):
        w: ArtifactOf[WidgetRegistry]

    built = Cfg(w={"type": "Widget", "data": {"size": 20}}).w
    assert isinstance(built, Widget)
    assert built.size == 20


def test_rejects_wrong_type() -> None:
    class Cfg(BaseModel):
        w: ArtifactOf[WidgetRegistry]

    with pytest.raises(Exception):
        Cfg(w="not a widget")


def test_composes_value_after_validator() -> None:
    def bump(w: Widget) -> Widget:
        w.size += 1
        return w

    class Cfg(BaseModel):
        w: Annotated[ArtifactOf[WidgetRegistry], AfterValidator(bump)]

    # build (size=5) THEN the after-validator (bump -> 6): the base builds, the
    # Annotated metadata runs after.
    assert Cfg(w={"type": "Widget", "data": {"size": 5}}).w.size == 6


def test_cross_arg_dependency_via_info_data() -> None:
    # A BoundTo-style cross-argument check: the gadget's widget must be the
    # already-validated sibling ``w``. Reads it from ``info.data`` (populated
    # because ``w`` is declared before ``g``).
    def bound_to_w(g: Gadget, info: ValidationInfo) -> Gadget:
        sibling = info.data.get("w")
        if sibling is not None and g.widget is not sibling:
            raise ValueError("gadget.widget is not the sibling 'w'")
        return g

    class Cfg(BaseModel):
        w: ArtifactOf[WidgetRegistry]
        g: Annotated[ArtifactOf[GadgetRegistry], AfterValidator(bound_to_w)]

    w = Widget(3)
    ok = Cfg(w=w, g=Gadget(w))  # g bound to the same w -> passes
    assert ok.g.widget is ok.w

    with pytest.raises(Exception):
        Cfg(w=Widget(3), g=Gadget(Widget(99)))  # g bound to a different widget


def test_serialize_python_returns_artifact() -> None:
    class Cfg(BaseModel):
        w: ArtifactOf[WidgetRegistry]

    dumped = Cfg(w={"type": "Widget", "data": {"size": 7}}).model_dump()  # python mode
    assert isinstance(dumped["w"], Widget)
    assert dumped["w"].size == 7


def test_serialize_json_returns_config() -> None:
    class Cfg(BaseModel):
        w: ArtifactOf[WidgetRegistry]

    dumped = Cfg(w={"type": "Widget", "data": {"size": 7}}).model_dump(mode="json")
    assert isinstance(dumped["w"], dict)
    assert dumped["w"]["type"] == "Widget"
    # round-trips: the emitted config rebuilds the same artifact
    assert Cfg(w=dumped["w"]).w.size == 7


def test_rejects_non_registry_parameter() -> None:
    with pytest.raises(TypeError):
        ArtifactOf[int]  # a builtin, not a registry
    with pytest.raises(TypeError):
        ArtifactOf[Widget]  # a plain class, not a registry
    with pytest.raises(TypeError):
        ArtifactOf[Widget(1)]  # an instance, not a registry class


def test_union_tuple_syntax_accepts_either() -> None:
    class Cfg(BaseModel):
        x: ArtifactOf[WidgetRegistry, GizmoRegistry]

    assert isinstance(Cfg(x=Widget(1)).x, Widget)  # instance of the first
    assert isinstance(Cfg(x=Gizmo(9)).x, Gizmo)  # instance of the second
    assert isinstance(Cfg(x={"type": "Widget", "data": {"size": 2}}).x, Widget)
    assert isinstance(Cfg(x={"type": "Gizmo", "data": {"power": 3}}).x, Gizmo)


def test_union_explicit_syntax_accepts_either() -> None:
    class Cfg(BaseModel):
        x: Union[ArtifactOf[WidgetRegistry], ArtifactOf[GizmoRegistry]]

    assert isinstance(Cfg(x=Widget(1)).x, Widget)
    assert isinstance(Cfg(x=Gizmo(9)).x, Gizmo)
    assert isinstance(Cfg(x={"type": "Gizmo", "data": {"power": 3}}).x, Gizmo)


def test_union_rejects_foreign_value() -> None:
    class Cfg(BaseModel):
        x: ArtifactOf[WidgetRegistry, GizmoRegistry]

    with pytest.raises(Exception):
        Cfg(x="neither")


def test_union_empty_raises() -> None:
    with pytest.raises(TypeError):
        ArtifactOf[()]  # no registrar


def test_sibling_injection_via_ref_ctx() -> None:
    # The optimizer is BUILT over the sibling model's parameters via a
    # ``$model.parameters()`` ref. The model field, validated just before the
    # optimizer, is threaded into the factory build scope through ``info.data``,
    # so the ref resolves against the live sibling -- the pydantic-native form of
    # torchestrator's optimizer-over-model.parameters() build dependency.
    class Bundle(BaseModel):
        model: ArtifactOf[ModelRegistry]
        optimizer: ArtifactOf[OptRegistry]

    b = Bundle(
        model={"type": "Model", "data": {"n": 3}},
        optimizer={
            "type": "Opt",
            "data": {"params": "$model.parameters()", "lr": 0.1},
        },
    )
    assert isinstance(b.optimizer, Opt)
    assert b.optimizer.params == b.model.parameters()  # built over the sibling
    assert b.optimizer.lr == 0.1


def test_sibling_injection_with_live_model_instance() -> None:
    # Same, but the sibling model is passed as a LIVE instance (not a config).
    # info.data still carries it, so the optimizer config's ref resolves.
    class Bundle(BaseModel):
        model: ArtifactOf[ModelRegistry]
        optimizer: ArtifactOf[OptRegistry]

    m = Model(2)
    b = Bundle(
        model=m,
        optimizer={
            "type": "Opt",
            "data": {"params": "$model.parameters()", "lr": 0.5},
        },
    )
    assert b.model is m
    assert b.optimizer.params == m.parameters()
