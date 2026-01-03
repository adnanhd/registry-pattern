#!/usr/bin/env python
"""Tests for deeply nested BuildCfg envelope configurations.

This module tests recursive building of nested configurations at various depths,
ensuring proper validation, type coercion, and meta propagation through the
entire object graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from pydantic import BaseModel, Field

from registry import BuildCfg, ContainerMixin, TypeRegistry

# =============================================================================
# Test Registries and Components
# =============================================================================


class Level1Registry(TypeRegistry[object]):
    """Registry for level 1 components."""

    pass


class Level2Registry(TypeRegistry[object]):
    """Registry for level 2 components."""

    pass


class Level3Registry(TypeRegistry[object]):
    """Registry for level 3 components."""

    pass


class Level4Registry(TypeRegistry[object]):
    """Registry for level 4 components."""

    pass


class Level5Registry(TypeRegistry[object]):
    """Registry for level 5 components."""

    pass


# Configure repos
@pytest.fixture(autouse=True)
def setup_repos():
    """Setup registries before each test."""
    # Clear any existing registrations
    for registry in [
        Level1Registry,
        Level2Registry,
        Level3Registry,
        Level4Registry,
        Level5Registry,
    ]:
        registry.clear_artifacts()

    ContainerMixin.clear_context()
    ContainerMixin.configure_repos(
        {
            "level1": Level1Registry,
            "level2": Level2Registry,
            "level3": Level3Registry,
            "level4": Level4Registry,
            "level5": Level5Registry,
            "default": Level1Registry,
        }
    )

    # Register components
    register_all_components()

    yield

    # Cleanup
    ContainerMixin.clear_context()


def register_all_components():
    """Register all test components."""

    # Level 5 - Deepest (leaf nodes)
    @Level5Registry.register_artifact
    class Leaf:
        def __init__(self, value: int, name: str = "leaf"):
            self.value = value
            self.name = name

        def __repr__(self):
            return f"Leaf({self.name}={self.value})"

    @Level5Registry.register_artifact
    class LeafWithMeta:
        def __init__(self, data: str):
            self.data = data

        def __repr__(self):
            return f"LeafWithMeta({self.data!r})"

    # Level 4 - Contains level 5
    @Level4Registry.register_artifact
    class Container4:
        def __init__(self, child: Any, multiplier: int = 1):
            self.child = child
            self.multiplier = multiplier

        def __repr__(self):
            return f"Container4(child={self.child}, mult={self.multiplier})"

    @Level4Registry.register_artifact
    class MultiChild4:
        def __init__(self, children: List[Any]):
            self.children = children

        def __repr__(self):
            return f"MultiChild4({len(self.children)} children)"

    # Level 3 - Contains level 4
    @Level3Registry.register_artifact
    class Container3:
        def __init__(self, nested: Any, label: str = "l3"):
            self.nested = nested
            self.label = label

        def __repr__(self):
            return f"Container3({self.label}, nested={self.nested})"

    @Level3Registry.register_artifact
    class DualContainer3:
        def __init__(self, left: Any, right: Any):
            self.left = left
            self.right = right

        def __repr__(self):
            return f"DualContainer3(left={self.left}, right={self.right})"

    # Level 2 - Contains level 3
    @Level2Registry.register_artifact
    class Container2:
        def __init__(self, inner: Any, config: Optional[Dict[str, Any]] = None):
            self.inner = inner
            self.config = config or {}

        def __repr__(self):
            return f"Container2(inner={self.inner})"

    @Level2Registry.register_artifact
    class ContextAware2:
        def __init__(self, component: Any, ctx: dict = None):
            self.component = component
            self.ctx = ctx or {}

        def __repr__(self):
            return f"ContextAware2(component={self.component}, ctx_keys={list(self.ctx.keys())})"

    # Level 1 - Top level, contains level 2
    @Level1Registry.register_artifact
    class Root:
        def __init__(self, payload: Any, version: str = "1.0"):
            self.payload = payload
            self.version = version

        def __repr__(self):
            return f"Root(v{self.version}, payload={self.payload})"

    @Level1Registry.register_artifact
    class MultiRoot:
        def __init__(self, components: List[Any], name: str = "multi"):
            self.components = components
            self.name = name

        def __repr__(self):
            return f"MultiRoot({self.name}, {len(self.components)} components)"


# =============================================================================
# Tests: 4-Level Deep Nesting
# =============================================================================


class TestFourLevelNesting:
    """Test configurations nested 4 levels deep."""

    def test_four_level_nested_config(self):
        """Test building a 4-level deep nested configuration."""
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="Container3",
                            repo="level3",
                            data={
                                "nested": BuildCfg(
                                    type="Container4",
                                    repo="level4",
                                    data={
                                        "child": BuildCfg(
                                            type="Leaf",
                                            repo="level5",
                                            data={"value": 42, "name": "deepest"},
                                        ),
                                        "multiplier": 2,
                                    },
                                ),
                                "label": "middle",
                            },
                        )
                    },
                ),
                "version": "2.0",
            },
            meta={"experiment": "deep_nesting"},
        )

        result = ContainerMixin.build_cfg(cfg)

        # Verify structure
        assert result.version == "2.0"
        assert result.payload.inner.label == "middle"
        assert result.payload.inner.nested.multiplier == 2
        assert result.payload.inner.nested.child.value == 42
        assert result.payload.inner.nested.child.name == "deepest"

        # Verify meta propagation
        assert getattr(result, "__meta__", {}).get("experiment") == "deep_nesting"

    def test_four_level_with_dict_syntax(self):
        """Test 4-level nesting using dict syntax instead of BuildCfg."""
        cfg = BuildCfg.model_validate(
            {
                "type": "Root",
                "repo": "level1",
                "data": {
                    "payload": {
                        "type": "Container2",
                        "repo": "level2",
                        "data": {
                            "inner": {
                                "type": "Container3",
                                "repo": "level3",
                                "data": {
                                    "nested": {
                                        "type": "Container4",
                                        "repo": "level4",
                                        "data": {
                                            "child": {
                                                "type": "Leaf",
                                                "repo": "level5",
                                                "data": {"value": 99},
                                            }
                                        },
                                    }
                                },
                            }
                        },
                    }
                },
            }
        )

        result = ContainerMixin.build_cfg(cfg)
        assert result.payload.inner.nested.child.value == 99


class TestFiveLevelNesting:
    """Test configurations nested 5 levels deep."""

    def test_five_level_deep_chain(self):
        """Test building a 5-level deep nested chain."""
        # L1 -> L2 -> L3 -> L4 -> L5 (leaf)
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="Container3",
                            repo="level3",
                            data={
                                "nested": BuildCfg(
                                    type="Container4",
                                    repo="level4",
                                    data={
                                        "child": BuildCfg(
                                            type="Leaf",
                                            repo="level5",
                                            data={"value": 100, "name": "level5_leaf"},
                                        )
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        # Navigate the full chain
        leaf = result.payload.inner.nested.child
        assert leaf.value == 100
        assert leaf.name == "level5_leaf"


class TestBranchingNesting:
    """Test configurations with branching (multiple children at same level)."""

    def test_dual_branch_deep_nesting(self):
        """Test nested config with two branches, each going deep."""
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="DualContainer3",
                            repo="level3",
                            data={
                                "left": BuildCfg(
                                    type="Container4",
                                    repo="level4",
                                    data={
                                        "child": BuildCfg(
                                            type="Leaf",
                                            repo="level5",
                                            data={"value": 1, "name": "left_leaf"},
                                        )
                                    },
                                ),
                                "right": BuildCfg(
                                    type="Container4",
                                    repo="level4",
                                    data={
                                        "child": BuildCfg(
                                            type="Leaf",
                                            repo="level5",
                                            data={"value": 2, "name": "right_leaf"},
                                        )
                                    },
                                ),
                            },
                        )
                    },
                )
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        # Verify both branches
        assert result.payload.inner.left.child.value == 1
        assert result.payload.inner.left.child.name == "left_leaf"
        assert result.payload.inner.right.child.value == 2
        assert result.payload.inner.right.child.name == "right_leaf"

    def test_list_of_nested_configs(self):
        """Test a list of nested configurations at various depths."""
        cfg = BuildCfg(
            type="MultiRoot",
            repo="level1",
            data={
                "name": "list_test",
                "components": [
                    BuildCfg(
                        type="Container2",
                        repo="level2",
                        data={
                            "inner": BuildCfg(
                                type="Container3",
                                repo="level3",
                                data={
                                    "nested": BuildCfg(
                                        type="Leaf",
                                        repo="level5",
                                        data={"value": i * 10},
                                    )
                                },
                            )
                        },
                    )
                    for i in range(3)
                ],
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        assert len(result.components) == 3
        for i, comp in enumerate(result.components):
            # The nested chain is: Container2 -> Container3 -> Leaf
            # Note: Container3 expects Container4, but we're using Leaf directly
            # This tests flexible nesting
            assert comp.inner.nested.value == i * 10

    def test_multi_child_with_list_of_leaves(self):
        """Test MultiChild4 containing a list of leaf nodes."""
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="Container3",
                            repo="level3",
                            data={
                                "nested": BuildCfg(
                                    type="MultiChild4",
                                    repo="level4",
                                    data={
                                        "children": [
                                            BuildCfg(
                                                type="Leaf",
                                                repo="level5",
                                                data={"value": v, "name": f"leaf_{v}"},
                                            )
                                            for v in [10, 20, 30, 40]
                                        ]
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        children = result.payload.inner.nested.children
        assert len(children) == 4
        for i, child in enumerate(children):
            expected_value = (i + 1) * 10
            assert child.value == expected_value
            assert child.name == f"leaf_{expected_value}"


class TestContextAcrossNesting:
    """Test context injection across nested levels."""

    def test_context_available_at_all_levels(self):
        """Test that context is accessible at any nesting level."""
        # Pre-populate context
        ContainerMixin._ctx["shared_value"] = 999
        ContainerMixin._ctx["shared_config"] = {"key": "value"}

        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="ContextAware2",
                    repo="level2",
                    data={
                        "component": BuildCfg(
                            type="Leaf", repo="level5", data={"value": 50}
                        )
                    },
                )
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        # ContextAware2 should have received the ctx
        assert result.payload.ctx.get("shared_value") == 999
        assert result.payload.ctx.get("shared_config") == {"key": "value"}
        assert result.payload.component.value == 50

    def test_build_named_then_reference(self):
        """Test building named objects that can be referenced later."""
        # Build and name a leaf first
        leaf_cfg = BuildCfg(
            type="Leaf", repo="level5", data={"value": 777, "name": "named_leaf"}
        )
        ContainerMixin.build_named("my_leaf", leaf_cfg)

        # Now build something that accesses context
        cfg = BuildCfg(
            type="ContextAware2",
            repo="level2",
            data={
                "component": BuildCfg(
                    type="Leaf",
                    repo="level5",
                    data={"value": 0},  # dummy
                )
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        # Should be able to access the named leaf from ctx
        assert "my_leaf" in result.ctx
        assert result.ctx["my_leaf"].value == 777


class TestMetaPropagation:
    """Test that meta is properly attached at each nesting level."""

    def test_meta_on_each_level(self):
        """Test that each level gets its own meta attached."""
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="Container3",
                            repo="level3",
                            data={
                                "nested": BuildCfg(
                                    type="Leaf",
                                    repo="level5",
                                    data={"value": 1},
                                    meta={"level": 5, "name": "leaf_meta"},
                                )
                            },
                            meta={"level": 3},
                        )
                    },
                    meta={"level": 2},
                )
            },
            meta={"level": 1, "top": True},
        )

        result = ContainerMixin.build_cfg(cfg)

        # Check meta at each level
        assert getattr(result, "__meta__", {}).get("level") == 1
        assert getattr(result, "__meta__", {}).get("top") is True
        assert getattr(result.payload, "__meta__", {}).get("level") == 2
        assert getattr(result.payload.inner, "__meta__", {}).get("level") == 3
        assert getattr(result.payload.inner.nested, "__meta__", {}).get("level") == 5
        assert (
            getattr(result.payload.inner.nested, "__meta__", {}).get("name")
            == "leaf_meta"
        )


class TestTypeCoercionAcrossLevels:
    """Test that type coercion works at all nesting levels."""

    def test_string_to_int_at_deep_level(self):
        """Test that string values are coerced to int at any depth."""
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="Container3",
                            repo="level3",
                            data={
                                "nested": BuildCfg(
                                    type="Container4",
                                    repo="level4",
                                    data={
                                        "child": BuildCfg(
                                            type="Leaf",
                                            repo="level5",
                                            data={
                                                "value": "123"
                                            },  # String should be coerced
                                        ),
                                        "multiplier": "5",  # String should be coerced
                                    },
                                )
                            },
                        )
                    },
                )
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        # Values should be integers, not strings
        assert result.payload.inner.nested.child.value == 123
        assert isinstance(result.payload.inner.nested.child.value, int)
        assert result.payload.inner.nested.multiplier == 5
        assert isinstance(result.payload.inner.nested.multiplier, int)


class TestExtrasHandlingAcrossLevels:
    """Test that extra/unknown fields are handled correctly at all levels."""

    def test_extras_captured_at_each_level(self):
        """Test that unknown fields at any level go to that level's meta."""
        cfg = BuildCfg(
            type="Root",
            repo="level1",
            data={
                "payload": BuildCfg(
                    type="Container2",
                    repo="level2",
                    data={
                        "inner": BuildCfg(
                            type="Leaf",
                            repo="level5",
                            data={
                                "value": 1,
                                "unknown_leaf_field": "should_go_to_meta",
                            },
                        ),
                        "unknown_container2_field": 42,
                    },
                ),
                "unknown_root_field": "root_extra",
            },
        )

        result = ContainerMixin.build_cfg(cfg)

        # Check that unknown fields ended up in meta._unused_data
        root_meta = getattr(result, "__meta__", {})
        assert (
            root_meta.get("_unused_data", {}).get("unknown_root_field") == "root_extra"
        )

        payload_meta = getattr(result.payload, "__meta__", {})
        assert (
            payload_meta.get("_unused_data", {}).get("unknown_container2_field") == 42
        )

        leaf_meta = getattr(result.payload.inner, "__meta__", {})
        assert (
            leaf_meta.get("_unused_data", {}).get("unknown_leaf_field")
            == "should_go_to_meta"
        )
