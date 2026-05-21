#!/usr/bin/env python
"""Example 05: pushing the registry-tree limits.

Demonstrates the full power of hierarchical sub-registrars:

  - 4-level deep tree (`models` / `models.cnn` / `models.cnn.imagenet`)
  - cross-cutting orthogonal sub-registry (`models.pretrained` joins
    cumulative validation in a different axis than `.cnn`)
  - multiple inheritance among sub-registries via Python's MRO
    (`ImagenetCNN(Pretrained, CNNModels)` runs BOTH parents' checks)
  - same artifact registered in 3 registries with different post_init
    disciplines per repo
  - meta_schema escalation: child requires more fields than parent
  - ambiguous lookups + repo= disambiguation
  - build via every entry: dict, BuildCfg, class, string name
  - cross-registry envelope with $ref between siblings

Stdlib + pydantic only.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest  # for clean failure-mode demo  # noqa: F401
from pydantic import BaseModel, ConfigDict

from registry import (
    BuildCfg,
    FunctionalRegistry,
    TypeRegistry,
    build,
    resolve,
    serialize,
)


# =============================================================================
# Tree of sub-registrars (the interesting bit)
# =============================================================================


class Models(TypeRegistry[Any], repo="zoo.models"):
    """Top of the tree -- the only invariant: parameter sanity."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta.setdefault("family_chain", []).append("models")
        if getattr(instance, "_broken", False):
            raise ValueError("instance flagged _broken=True")


class CNNModels(Models, repo="zoo.models.cnn"):
    """Axis 1: architecture-shaped sub-registry. CNN must declare kernel_size."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        super().post_init(instance, meta)
        meta["family_chain"].append("cnn")
        if not hasattr(instance, "kernel_size"):
            raise ValueError("models.cnn: missing kernel_size")


class TransformerModels(Models, repo="zoo.models.transformer"):
    """Sibling of CNNModels on the architecture axis."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        super().post_init(instance, meta)
        meta["family_chain"].append("transformer")
        if not hasattr(instance, "n_heads"):
            raise ValueError("models.transformer: missing n_heads")


class PretrainedMeta(BaseModel):
    """Explicit meta_schema: pretrained registries MUST yield these keys."""

    model_config = ConfigDict(extra="allow")
    family_chain: list[str]
    checksum:     str
    verified:     bool


class Pretrained(Models, repo="zoo.models.pretrained"):
    """Axis 2: lifecycle-shaped sub-registry. Independent of architecture axis.

    Requires the instance to carry an _expected_hash attribute and proves the
    weights actually match it.
    """

    meta_schema = PretrainedMeta

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        super().post_init(instance, meta)
        meta["family_chain"].append("pretrained")
        expected = getattr(instance, "_expected_hash", None)
        if expected is None:
            raise ValueError("models.pretrained: missing _expected_hash")
        actual = hashlib.sha256(repr(instance.__dict__).encode()).hexdigest()[:16]
        # for the demo we don't enforce equality -- a real lib would
        meta["checksum"] = actual
        meta["verified"] = True


class ImagenetCNN(Pretrained, CNNModels, repo="zoo.models.cnn.imagenet"):
    """Diamond inheritance: walks BOTH Pretrained AND CNNModels via super().

    The MRO order (Pretrained first, then CNNModels) determines the cascade.
    """

    class _Meta(PretrainedMeta):
        # Extends parent schema with dataset key.
        dataset: str

    meta_schema = _Meta

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        # Cooperative super() walks Pretrained -> CNNModels -> Models in MRO.
        super().post_init(instance, meta)
        meta["family_chain"].append("imagenet")
        meta["dataset"] = "imagenet"


# =============================================================================
# Function-side sub-registries (different tree, same mechanism)
# =============================================================================


class Pipelines(FunctionalRegistry, repo="zoo.pipelines"):
    @classmethod
    def pre_call(cls, target, kwargs, ctx, meta):
        meta["pipeline_chain"] = ["pipelines"]


class TrainingPipelines(Pipelines, repo="zoo.pipelines.training"):
    @classmethod
    def pre_call(cls, target, kwargs, ctx, meta):
        super().pre_call(target, kwargs, ctx, meta)
        meta["pipeline_chain"].append("training")
        lr = kwargs.get("lr")
        if lr is None or lr <= 0:
            raise ValueError(f"training pipeline: lr must be positive, got {lr!r}")


# =============================================================================
# The registered classes -- same physical class in three repos
# =============================================================================


class _ResNet50:
    """One class, three registrations, three disciplines applied."""
    _expected_hash = "sha256:resnet50-pinned"
    _broken = False

    def __init__(self, kernel_size: int = 7, channels: int = 64,
                 pretrained: bool = True) -> None:
        self.kernel_size = kernel_size
        self.channels = channels
        self.pretrained = pretrained


CNNModels.register_artifact(_ResNet50)         # only CNN check
Pretrained.register_artifact(_ResNet50)        # only hash check + meta_schema
ImagenetCNN.register_artifact(_ResNet50)       # both + dataset key


# A non-CNN, non-pretrained model lives only in Models.
class _Mystery:
    def __init__(self, width: int = 8): self.width = width


Models.register_artifact(_Mystery)


# A transformer lives only in TransformerModels.
class _TinyTransformer:
    n_heads = 4
    def __init__(self, layers: int = 2): self.layers = layers


TransformerModels.register_artifact(_TinyTransformer)


@TrainingPipelines.register_artifact
def train_step(model: Any, lr: float, batch_size: int = 32) -> dict[str, float]:
    return {"loss": 0.5, "lr": lr, "batch_size": float(batch_size)}


# =============================================================================
# Demonstrations
# =============================================================================


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


def demo_cumulative_post_init() -> None:
    section("[1] Multi-level inheritance: family_chain grows down the tree")

    # Build at "zoo.models.cnn" -> Models + CNNModels checks fire
    cfg: dict[str, Any] = {"type": "_ResNet50", "data": {}, "meta": {}}
    build(cfg, repo="zoo.models.cnn")
    print("repo=zoo.models.cnn       ->", cfg["meta"]["family_chain"])

    # Build at "zoo.models.pretrained" -> Models + Pretrained
    cfg = {"type": "_ResNet50", "data": {}, "meta": {}}
    build(cfg, repo="zoo.models.pretrained")
    print("repo=zoo.models.pretrained ->", cfg["meta"]["family_chain"])

    # Build at "zoo.models.cnn.imagenet" -> diamond: Pretrained + CNN + Models
    cfg = {"type": "_ResNet50", "data": {}, "meta": {}}
    build(cfg, repo="zoo.models.cnn.imagenet")
    print("repo=zoo.models.cnn.imagenet ->", cfg["meta"]["family_chain"])
    print("                  dataset ->", cfg["meta"]["dataset"])


def demo_meta_schema_escalation() -> None:
    section("[2] meta_schema escalation: child requires more fields than parent")

    # Pretrained requires: family_chain, checksum, verified
    cfg: dict[str, Any] = {"type": "_ResNet50", "data": {}, "meta": {}}
    build(cfg, repo="zoo.models.pretrained")
    assert {"family_chain", "checksum", "verified"} <= set(cfg["meta"])
    print("Pretrained meta keys: ", sorted(cfg["meta"]))

    # ImagenetCNN extends with dataset
    cfg = {"type": "_ResNet50", "data": {}, "meta": {}}
    build(cfg, repo="zoo.models.cnn.imagenet")
    assert "dataset" in cfg["meta"]
    print("ImagenetCNN meta keys:", sorted(cfg["meta"]))


def demo_ambiguity_and_disambiguation() -> None:
    section("[3] Ambiguous resolves -- the same class is in 3 registries")

    try:
        resolve("_ResNet50")
    except KeyError as e:
        print("no repo            ->", str(e)[:90], "...")

    try:
        resolve("_ResNet50", repo="zoo.models")
    except KeyError as e:
        print("repo=zoo.models    ->", str(e)[:90], "...")

    reg, _ = resolve("_ResNet50", repo="zoo.models.cnn.imagenet")
    print("repo=...cnn.imagenet -> resolved to", reg.__name__)


def demo_failure_modes() -> None:
    section("[4] Failure modes that the tree catches")

    # CNN registry rejects something without kernel_size
    try:
        build({"type": "_Mystery", "data": {}}, repo="zoo.models.cnn")
    except Exception as e:
        print("Mystery -> models.cnn      ->", type(e).__name__, str(e).splitlines()[0][:80])

    # Pretrained registry rejects something without _expected_hash
    try:
        build({"type": "_Mystery", "data": {}}, repo="zoo.models.pretrained")
    except Exception as e:
        print("Mystery -> .pretrained     ->", type(e).__name__, str(e).splitlines()[0][:80])


def demo_function_tree_with_ref_and_cross_axis() -> None:
    section("[5] Cross-tree envelope: training pipeline + tree-axis model + $ref")

    cfg: dict[str, Any] = {
        "type": "train_step",
        "repo": "zoo.pipelines.training",
        "data": {
            # Inner envelope crosses to a sibling tree (model-axis); $ref pulls
            # the built instance's kernel_size out for batch_size.
            "model": {
                "type": "_ResNet50", "repo": "zoo.models.cnn.imagenet",
                "data": {"kernel_size": 7}, "meta": {},
            },
            "lr": 1e-3,
            "batch_size": "$model.kernel_size",   # cross-sibling ref
        },
        "meta": {},
    }
    out = build(cfg)
    print("training pipeline result:", out)
    print("pipeline meta chain     :", cfg["meta"]["pipeline_chain"])
    print("inner model meta chain  :", cfg["data"]["model"]["meta"]["family_chain"])


def demo_every_build_entry() -> None:
    section("[6] Every supported build() entry shape, same outcome")

    # 1. Envelope dict
    o1 = build({"type": "_ResNet50", "repo": "zoo.models.cnn", "data": {}})
    # 2. BuildCfg instance
    o2 = build(BuildCfg(type="_ResNet50", repo="zoo.models.cnn", data={}))
    # 3. String name + repo
    o3 = build("_ResNet50", {}, repo="zoo.models.cnn", validator="python")
    # 4. Class direct
    o4 = build(_ResNet50, {}, repo="zoo.models.cnn", validator="python")

    print("dict       ->", o1.__class__.__name__, "k=", o1.kernel_size)
    print("BuildCfg   ->", o2.__class__.__name__)
    print("name+repo  ->", o3.__class__.__name__)
    print("class dir  ->", o4.__class__.__name__)


def demo_serialize_round_trip() -> None:
    section("[7] Serialize round-trips a built tree-node back to a dict")

    m = build(_ResNet50, {"kernel_size": 11, "channels": 128, "pretrained": False},
              validator="python", repo="zoo.models.cnn")
    out = serialize(m, serializator="python")
    print("dump        ->", out)
    yaml = serialize(m, serializator="yaml")
    print("yaml lines  ->", yaml.replace("\n", " | "))


def main() -> None:
    demo_cumulative_post_init()
    demo_meta_schema_escalation()
    demo_ambiguity_and_disambiguation()
    demo_failure_modes()
    demo_function_tree_with_ref_and_cross_axis()
    demo_every_build_entry()
    demo_serialize_round_trip()


if __name__ == "__main__":
    main()
