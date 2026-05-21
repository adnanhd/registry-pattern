"""Tests for hierarchical repo paths and tree-shaped registrar lookup."""

from typing import Any

import pytest

from registry import TypeRegistry, build, resolve


class Models(TypeRegistry[Any], repo="models"):
    pass


class ImagenetPretrained(TypeRegistry[Any], repo="models.imagenet_pretrained"):
    """Stricter sub-registrar: registered classes get a hash check."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        # Tag instance so test can see the strict hook fired.
        meta["pretrained_validated"] = True


class Losses(TypeRegistry[Any], repo="losses"):
    pass


class _ResNet:
    def __init__(self, layers: int = 50, pretrained: bool = False) -> None:
        self.layers = layers
        self.pretrained = pretrained


# Same class object lives in BOTH registries.
Models.register_artifact(_ResNet)
ImagenetPretrained.register_artifact(_ResNet)


@pytest.fixture(autouse=True)
def _reset_registries():
    yield
    # No cleanup needed for module-level registrations; tests don't mutate them.


def test_repo_path_set_correctly() -> None:
    assert Models.repo == "models"
    assert ImagenetPretrained.repo == "models.imagenet_pretrained"
    assert Losses.repo == "losses"


def test_resolve_without_repo_finds_all_matches_ambiguous() -> None:
    with pytest.raises(KeyError, match="ambiguous"):
        resolve("_ResNet")  # registered in two repos


def test_resolve_exact_repo_picks_one() -> None:
    reg, art = resolve("_ResNet", repo="models")
    assert reg is Models
    reg2, art2 = resolve("_ResNet", repo="models.imagenet_pretrained")
    assert reg2 is ImagenetPretrained


def test_resolve_prefix_repo_disambiguates_when_only_one_under_prefix() -> None:
    # Drop one of the two, the other should resolve with bare prefix
    Models.unregister_identifier("_ResNet")
    try:
        reg, art = resolve("_ResNet", repo="models")
        # only "models.imagenet_pretrained" remains under "models"; prefix-match finds it
        assert reg is ImagenetPretrained
    finally:
        Models.register_artifact(_ResNet)


def test_resolve_unrelated_repo_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        resolve("_ResNet", repo="losses")


def test_build_with_repo_invokes_strict_post_init() -> None:
    cfg: dict[str, Any] = {"type": "_ResNet", "data": {"layers": 50}, "meta": {}}
    build(cfg, repo="models.imagenet_pretrained")
    assert cfg["meta"]["pretrained_validated"] is True


def test_build_with_repo_skips_strict_when_broader() -> None:
    cfg: dict[str, Any] = {"type": "_ResNet", "data": {"layers": 50}, "meta": {}}
    build(cfg, repo="models")  # Models has no post_init -> meta stays empty
    assert "pretrained_validated" not in cfg["meta"]


def test_build_kwarg_overrides_envelope_repo() -> None:
    # Envelope says "models" but the kwarg promotes to the stricter sub-repo.
    cfg: dict[str, Any] = {
        "type": "_ResNet", "repo": "models", "data": {"layers": 18}, "meta": {},
    }
    build(cfg, repo="models.imagenet_pretrained")
    assert cfg["meta"]["pretrained_validated"] is True


def test_build_explicit_class_with_repo() -> None:
    cfg_dict: dict[str, Any] = {}
    obj = build(_ResNet, {"layers": 34}, validator="python", repo="models.imagenet_pretrained")
    assert obj.layers == 34
    # meta is attached to instance
    assert obj.__meta__["pretrained_validated"] is True
