"""End-to-end integration tests for the factory pipeline.

Exercises every layer in combination:

  - hierarchical repo tree with cumulative ``post_init`` via Python inheritance
  - explicit ``meta_schema`` enforcement on sub-registries
  - implicit ``meta_schema`` derived from ``Annotated[..., compute]`` markers
  - ``Annotated[..., validate]`` markers firing before invocation
  - nested envelopes resolving via ``$ref`` against ctx and sibling scope
  - per-medium build (``validator="python"`` / ``"yaml"`` / ``"argparse"``)
  - meters writing into ``meta`` so meta_schema sees them at validation time

No torch dependency -- everything uses plain Python objects so tests run in
any environment that has pydantic.
"""

from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, ConfigDict

from registry import (
    FunctionalRegistry,
    LifetimeMeter,
    TypeRegistry,
    attach_meter,
    build,
    detach_meter,
    meters,
    resolve,
    serialize,
    validate,
)
from registry.meters import FactoryMeter


# =============================================================================
# Tree of registrars: Models -> {CNNModels, Pretrained -> ImagenetPretrained}
# =============================================================================


class Models(TypeRegistry[object], repo="myapp.models"):
    """Top-level model family: every registered class must survive a NaN-equivalent check."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["family_seen"] = "models"
        if getattr(instance, "_broken", False):
            raise ValueError("model marked _broken=True")


class CNNModels(Models, repo="myapp.models.cnn"):
    """CNN sub-family: requires a kernel_size attribute on the built instance."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        super().post_init(instance, meta)
        meta["family_cnn"] = True
        if not hasattr(instance, "kernel_size"):
            raise ValueError("cnn family: instance lacks kernel_size")


class PretrainedMeta(BaseModel):
    """Explicit meta_schema: every Pretrained build must populate these."""

    model_config = ConfigDict(extra="allow")
    family_seen:   str
    checksum:      str
    verified:      bool


class Pretrained(Models, repo="myapp.models.pretrained"):
    """Pretrained sub-family: enforces checksum + verified via meta_schema."""

    meta_schema = PretrainedMeta

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        super().post_init(instance, meta)
        wh = getattr(instance, "_weights_hash", None)
        if wh is None:
            raise ValueError("pretrained: instance lacks _weights_hash")
        meta["checksum"] = wh
        meta["verified"] = True


class ImagenetPretrained(Pretrained, CNNModels, repo="myapp.models.pretrained.imagenet"):
    """Pretrained CNN trained on ImageNet: cumulative checks from BOTH parents."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        # Calls Pretrained first via MRO, then CNNModels.
        super().post_init(instance, meta)
        meta["dataset"] = "imagenet"


# =============================================================================
# Annotated markers (validate + compute) -- no torch
# =============================================================================


@dataclass(frozen=True)
class MustBe:
    """validate marker: value must equal expected."""
    expected: Any

    def validate(self, value: Any, kwargs: dict[str, Any], ctx: dict[str, Any]) -> None:
        if value != self.expected:
            raise ValueError(f"must be {self.expected!r}, got {value!r}")


@dataclass(frozen=True)
class Doubled:
    """compute marker: writes 2*value into meta[name]."""
    name: str

    def compute(self, value: float) -> float:
        return value * 2


@dataclass(frozen=True)
class Recorded:
    """compute marker: writes str(value) into meta[name]."""
    name: str

    def compute(self, value: Any) -> str:
        return str(value)


# =============================================================================
# Function-level sub-registry tree
# =============================================================================


class Steps(FunctionalRegistry, repo="myapp.steps"):
    pass


class TrainingSteps(Steps, repo="myapp.steps.training"):
    """Training-step sub-family: cross-arg invariants enforced before invocation."""

    @classmethod
    def pre_call(cls, target, kwargs, ctx, meta):
        # All training steps must have a numeric lr > 0
        lr = kwargs.get("lr", None)
        if lr is None or lr <= 0:
            raise ValueError(f"training step: lr must be positive, got {lr!r}")
        meta["lr_validated"] = True


# =============================================================================
# Registered artifacts (single class in multiple sub-registries)
# =============================================================================


class _Net:
    """Plain Python stand-in for nn.Module."""
    _weights_hash = "sha256:net-baseline-001"
    _broken = False

    def __init__(self, kernel_size: int = 3, channels: int = 16) -> None:
        self.kernel_size = kernel_size
        self.channels = channels


class _NetMissingHash(_Net):
    _weights_hash = None  # type: ignore[assignment]


class _BrokenNet(_Net):
    _broken = True


CNNModels.register_artifact(_Net)
Pretrained.register_artifact(_Net)
ImagenetPretrained.register_artifact(_Net)
Pretrained.register_artifact(_NetMissingHash)
CNNModels.register_artifact(_BrokenNet)


@TrainingSteps.register_artifact
def train_step(
    lr:       Annotated[float, MustBe(0.001), Recorded("learning_rate")],
    momentum: Annotated[float, Doubled("momentum_doubled")] = 0.9,
) -> dict[str, float]:
    """Plain function. lr is doubly-marked (validate + compute); momentum has compute only."""
    return {"loss": 0.5, "lr": lr, "momentum": momentum}


# =============================================================================
# Tests
# =============================================================================


def test_repo_paths_populated() -> None:
    assert Models.repo == "myapp.models"
    assert CNNModels.repo == "myapp.models.cnn"
    assert Pretrained.repo == "myapp.models.pretrained"
    assert ImagenetPretrained.repo == "myapp.models.pretrained.imagenet"
    assert Steps.repo == "myapp.steps"
    assert TrainingSteps.repo == "myapp.steps.training"


def test_cnn_build_runs_inherited_then_local_post_init() -> None:
    cfg: dict[str, Any] = {
        "type": "_Net", "data": {"kernel_size": 5}, "meta": {},
    }
    build(cfg, repo="myapp.models.cnn")
    assert cfg["meta"]["family_seen"] == "models"     # Models.post_init
    assert cfg["meta"]["family_cnn"] is True           # CNNModels.post_init


def test_pretrained_build_enforces_meta_schema() -> None:
    cfg: dict[str, Any] = {
        "type": "_Net", "data": {"kernel_size": 5}, "meta": {},
    }
    build(cfg, repo="myapp.models.pretrained")
    # meta_schema enforced -> all required keys present
    assert cfg["meta"]["family_seen"] == "models"
    assert cfg["meta"]["checksum"] == "sha256:net-baseline-001"
    assert cfg["meta"]["verified"] is True


def test_pretrained_build_fails_when_post_init_doesnt_supply_required_meta() -> None:
    """_NetMissingHash has _weights_hash=None -> post_init raises BEFORE meta_schema."""
    cfg: dict[str, Any] = {
        "type": "_NetMissingHash", "data": {"kernel_size": 5}, "meta": {},
    }
    with pytest.raises(ValueError, match="lacks _weights_hash"):
        build(cfg, repo="myapp.models.pretrained")


def test_imagenet_build_runs_full_inheritance_chain() -> None:
    """Both Pretrained AND CNNModels checks fire via cooperative super()."""
    cfg: dict[str, Any] = {
        "type": "_Net", "data": {"kernel_size": 3}, "meta": {},
    }
    build(cfg, repo="myapp.models.pretrained.imagenet")
    # All four post_init levels left their mark
    assert cfg["meta"]["family_seen"] == "models"     # Models
    assert cfg["meta"]["family_cnn"] is True           # CNNModels (via MRO)
    assert cfg["meta"]["checksum"]                     # Pretrained
    assert cfg["meta"]["dataset"] == "imagenet"        # ImagenetPretrained


def test_resolve_tree_prefix_finds_specific_subregistry() -> None:
    reg, art = resolve("_Net", repo="myapp.models.pretrained.imagenet")
    assert reg is ImagenetPretrained
    assert art is _Net


def test_resolve_tree_ambiguous_without_repo() -> None:
    # _Net is in CNNModels, Pretrained, ImagenetPretrained -> 3 matches
    with pytest.raises(KeyError, match="ambiguous"):
        resolve("_Net")


def test_resolve_tree_ambiguous_under_broader_prefix() -> None:
    # Under "myapp.models" prefix, _Net is in 3 registries
    with pytest.raises(KeyError, match="ambiguous"):
        resolve("_Net", repo="myapp.models")


def test_broken_net_fails_in_cnn_registry() -> None:
    cfg: dict[str, Any] = {"type": "_BrokenNet", "data": {}, "meta": {}}
    with pytest.raises(ValueError, match="_broken"):
        build(cfg, repo="myapp.models.cnn")


def test_annotated_validate_marker_fires_before_invocation() -> None:
    """train_step has MustBe(0.001) on lr -- 0.002 should reject."""
    cfg: dict[str, Any] = {"type": "train_step", "data": {"lr": 0.002}, "meta": {}}
    with pytest.raises(ValueError, match="must be 0.001"):
        build(cfg)


def test_annotated_compute_writes_meta() -> None:
    cfg: dict[str, Any] = {"type": "train_step", "data": {"lr": 0.001, "momentum": 0.9}, "meta": {}}
    build(cfg)
    assert cfg["meta"]["learning_rate"] == "0.001"     # Recorded
    assert cfg["meta"]["momentum_doubled"] == 1.8      # Doubled


def test_subregistry_pre_call_hook_fires() -> None:
    """TrainingSteps.pre_call enforces lr > 0 before invocation."""
    cfg: dict[str, Any] = {"type": "train_step", "data": {"lr": 0.001}, "meta": {}}
    build(cfg, repo="myapp.steps.training")
    assert cfg["meta"]["lr_validated"] is True


def test_implicit_meta_schema_from_compute_markers_validates_types() -> None:
    """Derived meta_schema enforces learning_rate: str and momentum_doubled: float."""
    cfg: dict[str, Any] = {"type": "train_step", "data": {"lr": 0.001}, "meta": {}}
    out = build(cfg)
    assert out == {"loss": 0.5, "lr": 0.001, "momentum": 0.9}
    assert isinstance(cfg["meta"]["learning_rate"], str)
    assert isinstance(cfg["meta"]["momentum_doubled"], float)


def test_validate_alone_does_not_instantiate() -> None:
    """validate() returns kwargs without invoking target."""
    # Use the class directly to avoid the cross-repo ambiguity of "_Net".
    kwargs = validate(_Net, {"kernel_size": 9}, validator="python")
    assert kwargs == {"kernel_size": 9, "channels": 16}  # default applied
    # No instance constructed: would fire post_init / meta_schema if it did


def test_serialize_round_trip_through_python_medium() -> None:
    """build then serialize back, expect a BuildCfg-shaped envelope."""
    inst = build("_Net", {"kernel_size": 7, "channels": 32}, validator="python",
                 repo="myapp.models.cnn")
    env = serialize(inst, serializator="python", repo="myapp.models.cnn")
    assert env["type"] == "_Net"
    assert env["data"] == {"kernel_size": 7, "channels": 32}
    assert isinstance(env["meta"], dict)


def test_serialize_meta_hook_cascades_via_super() -> None:
    """A `serialize_meta` classmethod on a parent registry fires for all
    children via Python inheritance + cooperative super()."""

    class _MetaModels(TypeRegistry[object], repo="meta_demo"):
        @classmethod
        def serialize_meta(cls, instance: Any, meta: dict[str, Any]) -> None:
            meta["family"] = "meta_demo"
            meta["channels"] = instance.channels

    class _MetaCNN(_MetaModels, repo="meta_demo.cnn"):
        @classmethod
        def serialize_meta(cls, instance: Any, meta: dict[str, Any]) -> None:
            super().serialize_meta(instance, meta)
            meta["axis"] = "cnn"

    class _Inst:
        def __init__(self, channels: int = 8) -> None:
            self.channels = channels

    _MetaCNN.register_artifact(_Inst)
    inst = _MetaCNN.get_artifact("_Inst")(channels=11)
    env = serialize(inst, serializator="python", repo="meta_demo.cnn")
    assert env["data"] == {"channels": 11}
    # Parent + child both contributed:
    assert env["meta"] == {"family": "meta_demo", "channels": 11, "axis": "cnn"}


def test_yaml_medium_end_to_end() -> None:
    inst = build("_Net", "kernel_size: 5\nchannels: 8\n", validator="yaml",
                 repo="myapp.models.cnn")
    assert (inst.kernel_size, inst.channels) == (5, 8)


def test_argparse_medium_end_to_end() -> None:
    ns = Namespace(kernel_size=11, channels=4)
    inst = build("_Net", ns, validator="argparse", repo="myapp.models.cnn")
    assert (inst.kernel_size, inst.channels) == (11, 4)


def test_meter_fires_alongside_post_init_and_markers() -> None:
    """LifetimeMeter writes meta BEFORE meta_schema enforcement.

    Pretrained's meta_schema allows extra fields (`extra="allow"`), so the
    meter's `lifetime_seconds` survives the validation pass.
    """
    attach_meter(LifetimeMeter())
    try:
        cfg: dict[str, Any] = {"type": "_Net", "data": {"kernel_size": 5}, "meta": {}}
        build(cfg, repo="myapp.models.pretrained")
        assert "lifetime_seconds" in cfg["meta"]
        assert cfg["meta"]["checksum"]
    finally:
        detach_meter("lifetime")


def test_nested_envelope_across_subregistries_with_meta_propagation() -> None:
    """Nested build: outer is a TrainingSteps function, inner is a CNN model.

    Each level's hooks/markers fire at its OWN envelope's meta. The outer envelope
    sees its own meta (markers + pre_call); the inner envelope (cfg["data"]["net"])
    has its own meta filled by Pretrained.post_init.
    """
    @TrainingSteps.register_artifact
    def use_net(net: Any, lr: Annotated[float, MustBe(0.001), Recorded("learning_rate")]) -> dict:
        return {"net_kernel": net.kernel_size, "lr": lr}

    cfg: dict[str, Any] = {
        "type": "use_net",
        "data": {
            "net": {
                "type": "_Net", "repo": "myapp.models.pretrained",
                "data": {"kernel_size": 7}, "meta": {},
            },
            "lr": 0.001,
        },
        "meta": {},
    }
    out = build(cfg, repo="myapp.steps.training")

    assert out["net_kernel"] == 7

    # Inner envelope's meta -- filled by Pretrained.post_init
    inner_meta = cfg["data"]["net"]["meta"]
    assert inner_meta["checksum"] == "sha256:net-baseline-001"
    assert inner_meta["verified"] is True

    # Outer envelope's meta -- markers + pre_call hook
    outer_meta = cfg["meta"]
    assert outer_meta["lr_validated"] is True             # pre_call
    assert outer_meta["learning_rate"] == "0.001"          # Recorded marker


def test_dollar_ref_resolves_built_inner_into_outer_kwargs() -> None:
    """$ref pulls a built sibling's attribute into another sibling's data."""
    @Steps.register_artifact
    def take_kernel(kernel: int) -> int:
        return kernel * 10

    cfg: dict[str, Any] = {
        "type": "take_kernel",
        "data": {
            # _Net is registered in 3 repos -- pick one explicitly
            "net": {"type": "_Net", "repo": "myapp.models.cnn",
                    "data": {"kernel_size": 9}, "meta": {}},
            "kernel": "$net.kernel_size",
        },
        "meta": {},
    }
    out = build(cfg, repo="myapp.steps")
    assert out == 90


# =============================================================================
# Deep nested builds: each level instantiates from a distinct registrar.
# Verifies the recursive pipeline traverses arbitrary depth and every
# level's post_init hook fires against its OWN envelope's meta.
# =============================================================================


class _LvlA(TypeRegistry[object], repo="depth.a"):
    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["lvl"] = "a"


class _LvlB(TypeRegistry[object], repo="depth.b"):
    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["lvl"] = "b"


class _LvlC(FunctionalRegistry, repo="depth.c"):
    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["lvl"] = "c"


class _LvlD(TypeRegistry[object], repo="depth.d"):
    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["lvl"] = "d"


class _LvlE(FunctionalRegistry, repo="depth.e"):
    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["lvl"] = "e"


@_LvlA.register_artifact
class _NodeA:
    def __init__(self, child: Any, label: str = "a") -> None:
        self.child = child
        self.label = label


@_LvlB.register_artifact
class _NodeB:
    def __init__(self, child: Any, n: int = 0) -> None:
        self.child = child
        self.n = n


@_LvlC.register_artifact
def _wrap_c(child: Any, factor: int = 1) -> dict[str, Any]:
    return {"wrapped": child, "factor": factor}


@_LvlD.register_artifact
class _NodeD:
    def __init__(self, child: Any) -> None:
        self.child = child


@_LvlE.register_artifact
def _leaf(value: int = 0) -> int:
    return value


def test_three_level_deep_build_mixed_registrars() -> None:
    """L1 (TypeRegistry) -> L2 (FunctionalRegistry) -> L3 (TypeRegistry leaf)."""
    cfg: dict[str, Any] = {
        "type": "_NodeA", "repo": "depth.a", "meta": {},
        "data": {
            "label": "root",
            "child": {
                "type": "_wrap_c", "repo": "depth.c", "meta": {},
                "data": {
                    "factor": 7,
                    "child": {
                        "type": "_NodeD", "repo": "depth.d", "meta": {},
                        "data": {"child": "terminal"},
                    },
                },
            },
        },
    }
    out = build(cfg)

    # Structural traversal
    assert out.label == "root"
    assert out.child["factor"] == 7
    assert out.child["wrapped"].child == "terminal"

    # Each level's post_init wrote into its OWN envelope's meta
    assert cfg["meta"]["lvl"] == "a"
    assert cfg["data"]["child"]["meta"]["lvl"] == "c"
    assert cfg["data"]["child"]["data"]["child"]["meta"]["lvl"] == "d"


def test_four_level_deep_build_distinct_registrars() -> None:
    """A -> B -> C -> E (leaf), four distinct registries."""
    cfg: dict[str, Any] = {
        "type": "_NodeA", "repo": "depth.a", "meta": {},
        "data": {
            "child": {
                "type": "_NodeB", "repo": "depth.b", "meta": {},
                "data": {
                    "n": 2,
                    "child": {
                        "type": "_wrap_c", "repo": "depth.c", "meta": {},
                        "data": {
                            "factor": 3,
                            "child": {
                                "type": "_leaf", "repo": "depth.e", "meta": {},
                                "data": {"value": 99},
                            },
                        },
                    },
                },
            },
        },
    }
    out = build(cfg)

    assert out.child.n == 2
    assert out.child.child["factor"] == 3
    assert out.child.child["wrapped"] == 99   # leaf function returned the int

    # Verify each level's post_init fired against its OWN envelope's meta
    meta_a = cfg["meta"]
    meta_b = cfg["data"]["child"]["meta"]
    meta_c = cfg["data"]["child"]["data"]["child"]["meta"]
    meta_e = cfg["data"]["child"]["data"]["child"]["data"]["child"]["meta"]
    assert (meta_a["lvl"], meta_b["lvl"], meta_c["lvl"], meta_e["lvl"]) == ("a", "b", "c", "e")


def test_five_level_deep_build_all_registrars() -> None:
    """A -> B -> C -> D -> E, every level different registrar; FunctionalRegistry
    nodes interleaved with TypeRegistry nodes."""
    cfg: dict[str, Any] = {
        "type": "_NodeA", "repo": "depth.a", "meta": {},
        "data": {
            "child": {
                "type": "_NodeB", "repo": "depth.b", "meta": {},
                "data": {
                    "n": 1,
                    "child": {
                        "type": "_wrap_c", "repo": "depth.c", "meta": {},
                        "data": {
                            "factor": 5,
                            "child": {
                                "type": "_NodeD", "repo": "depth.d", "meta": {},
                                "data": {
                                    "child": {
                                        "type": "_leaf", "repo": "depth.e", "meta": {},
                                        "data": {"value": 42},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    }
    out = build(cfg)

    # Walk the constructed graph
    assert out.child.n == 1
    assert out.child.child["factor"] == 5
    assert out.child.child["wrapped"].child == 42

    # Every level's post_init fired in its OWN envelope's meta
    levels = []
    node = cfg
    while True:
        levels.append(node["meta"].get("lvl"))
        # find the nested envelope key
        data = node["data"]
        next_node = None
        for v in data.values():
            if isinstance(v, dict) and "type" in v and "data" in v:
                next_node = v
                break
        if next_node is None:
            break
        node = next_node

    assert levels == ["a", "b", "c", "d", "e"]


def test_deep_build_ref_pulls_value_from_outer_sibling() -> None:
    """At depth 4, a sibling uses ``$ref`` to read from a peer at the same level."""
    cfg: dict[str, Any] = {
        "type": "_NodeA", "repo": "depth.a", "meta": {},
        "data": {
            "label": "outer",
            "child": {
                "type": "_NodeB", "repo": "depth.b", "meta": {},
                "data": {
                    "n": 13,
                    "child": {
                        "type": "_wrap_c", "repo": "depth.c", "meta": {},
                        "data": {
                            # $ref reads from this envelope's sibling scope
                            "factor": "$n",
                            "child": {
                                "type": "_NodeD", "repo": "depth.d", "meta": {},
                                "data": {"child": "leaf"},
                            },
                        },
                    },
                },
            },
        },
    }
    out = build(cfg)
    assert out.child.child["factor"] == 13   # $n -> 13 (from _NodeB sibling)
