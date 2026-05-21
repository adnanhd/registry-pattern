"""Unit tests for ``registry.experimental.torch_compat`` -- markers, meter, reporter.

All tests skip cleanly when torch is not installed.
"""

from __future__ import annotations

from typing import Annotated, Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from registry import (  # noqa: E402 -- after importorskip
    FunctionalRegistry,
    TypeRegistry,
    attach_meter,
    attach_reporter,
    build,
    detach_meter,
    detach_reporter,
)
from registry.experimental.torch_compat import (  # noqa: E402
    BoundTo,
    Checksum,
    Device,
    EffectiveLr,
    InputShape,
    InputShapeMatches,
    MatchesInputShape,
    MatchesOutputShape,
    MatchesTargetShape,
    NumParams,
    OutputShape,
    SameDeviceAs,
    TensorBoardReporter,
    TorchProfilerMeter,
    VerifyChecksum,
    device_of,
    hash_state_dict,
)


# =============================================================================
# Helpers
# =============================================================================


def test_device_of_handles_tensor_module_optimizer_and_tuple() -> None:
    t = torch.zeros(2, 2)
    assert str(device_of(t)).startswith("cpu")

    m = nn.Linear(4, 2)
    assert str(device_of(m)).startswith("cpu")

    opt = torch.optim.SGD(m.parameters(), lr=1e-2)
    assert str(device_of(opt)).startswith("cpu")

    assert str(device_of((t, t))).startswith("cpu")
    # Empty / non-tensor passthrough
    assert device_of("cpu") == "cpu"


def test_hash_state_dict_is_stable() -> None:
    m = nn.Linear(4, 2)
    h1 = hash_state_dict(m.state_dict())
    h2 = hash_state_dict(m.state_dict())
    assert h1 == h2
    assert h1.startswith("sha256:")


# =============================================================================
# Validate markers
# =============================================================================


class _Steps(FunctionalRegistry, repo="tc.steps"):
    pass


def test_same_device_as_passes_when_matching() -> None:
    @_Steps.register_artifact
    def step1(model: Annotated[nn.Module, SameDeviceAs("device")], device: str) -> str:
        return device

    out = build({
        "type": "step1", "repo": "tc.steps", "data": {
            "model": nn.Linear(4, 2),
            "device": "cpu",
        },
    })
    assert out == "cpu"


def test_same_device_as_raises_on_mismatch() -> None:
    @_Steps.register_artifact
    def step2(model: Annotated[nn.Module, SameDeviceAs("device")], device: str) -> str:
        return device

    with pytest.raises(ValueError, match="device mismatch"):
        build({
            "type": "step2", "repo": "tc.steps", "data": {
                "model": nn.Linear(4, 2),
                "device": "cuda",   # model is on cpu
            },
        })


def test_bound_to_passes_for_real_pairing() -> None:
    @_Steps.register_artifact
    def step3(model: nn.Module, opt: Annotated[torch.optim.Optimizer, BoundTo("model")]) -> int:
        return sum(1 for _ in opt.param_groups[0]["params"])

    model = nn.Linear(4, 2)
    out = build({
        "type": "step3", "repo": "tc.steps", "data": {
            "model": model,
            "opt": torch.optim.SGD(model.parameters(), lr=1e-2),
        },
    })
    assert out > 0


def test_bound_to_raises_when_optimizer_holds_other_params() -> None:
    @_Steps.register_artifact
    def step4(model: nn.Module, opt: Annotated[torch.optim.Optimizer, BoundTo("model")]) -> str:
        return "ok"

    other = nn.Linear(8, 8)
    with pytest.raises(ValueError, match="not bound"):
        build({
            "type": "step4", "repo": "tc.steps", "data": {
                "model": nn.Linear(4, 2),
                "opt": torch.optim.SGD(other.parameters(), lr=1e-2),
            },
        })


def test_verify_checksum_passes_when_expected() -> None:
    m = nn.Linear(4, 2)
    expected = hash_state_dict(m.state_dict())

    @_Steps.register_artifact
    def step5(model: Annotated[nn.Module, VerifyChecksum(expected)]) -> str:
        return "ok"

    assert build({"type": "step5", "repo": "tc.steps", "data": {"model": m}}) == "ok"


def test_verify_checksum_raises_on_mismatch() -> None:
    m = nn.Linear(4, 2)

    @_Steps.register_artifact
    def step6(model: Annotated[nn.Module, VerifyChecksum("sha256:deadbeef")]) -> str:
        return "ok"

    with pytest.raises(ValueError, match="checksum mismatch"):
        build({"type": "step6", "repo": "tc.steps", "data": {"model": m}})


def test_input_shape_matches_strict_compares_tensor_to_in_features() -> None:
    @_Steps.register_artifact
    def step7(model: nn.Module, x: Annotated[torch.Tensor, InputShapeMatches("model")]) -> int:
        return x.shape[1]

    assert build({"type": "step7", "repo": "tc.steps", "data": {
        "model": nn.Linear(4, 2),
        "x": torch.zeros(8, 4),
    }}) == 4

    with pytest.raises(ValueError, match="in_features"):
        build({"type": "step7", "repo": "tc.steps", "data": {
            "model": nn.Linear(4, 2),
            "x": torch.zeros(8, 7),
        }})


# =============================================================================
# Shape-contract markers (need __meta__["input_shape" | "output_shape"])
# =============================================================================


class _ShapeReg(TypeRegistry[Any], repo="tc.shaped"):
    """Registers models that declare input/output shapes in __meta__ via post_init."""

    @classmethod
    def post_init(cls, instance: Any, meta: dict[str, Any]) -> None:
        meta["input_shape"] = getattr(instance, "_input_shape", None)
        meta["output_shape"] = getattr(instance, "_output_shape", None)
        meta["target_shape"] = getattr(instance, "_target_shape", None)


@_ShapeReg.register_artifact
class _ShapedModel:
    _input_shape = (4,)
    _output_shape = (2,)

    def __init__(self) -> None:
        pass


@_ShapeReg.register_artifact
class _ShapedCriterion:
    _input_shape = (2,)
    _target_shape = (1,)

    def __init__(self) -> None:
        pass


def test_matches_input_shape_validates_against_peer_meta() -> None:
    @_Steps.register_artifact
    def step8(model: Any, batch: Annotated[torch.Tensor, MatchesInputShape("model")]) -> int:
        return batch.shape[1]

    model = build({"type": "_ShapedModel", "repo": "tc.shaped", "data": {}})
    assert build({
        "type": "step8", "repo": "tc.steps",
        "data": {"model": model, "batch": torch.zeros(16, 4)},
    }) == 4

    with pytest.raises(ValueError, match="input shape mismatch"):
        build({
            "type": "step8", "repo": "tc.steps",
            "data": {"model": model, "batch": torch.zeros(16, 7)},
        })


def test_matches_output_shape_links_criterion_input_to_model_output() -> None:
    @_Steps.register_artifact
    def step9(model: Any, criterion: Annotated[Any, MatchesOutputShape("model")]) -> str:
        return "ok"

    model = build({"type": "_ShapedModel", "repo": "tc.shaped", "data": {}})
    criterion = build({"type": "_ShapedCriterion", "repo": "tc.shaped", "data": {}})
    assert build({
        "type": "step9", "repo": "tc.steps",
        "data": {"model": model, "criterion": criterion},
    }) == "ok"


def test_matches_target_shape_validates_batch_target_against_criterion_meta() -> None:
    @_Steps.register_artifact
    def step10(criterion: Any, batch: Annotated[tuple, MatchesTargetShape("criterion")]) -> str:
        return "ok"

    criterion = build({"type": "_ShapedCriterion", "repo": "tc.shaped", "data": {}})
    assert build({
        "type": "step10", "repo": "tc.steps",
        "data": {"criterion": criterion, "batch": (torch.zeros(8, 2), torch.zeros(8, 1))},
    }) == "ok"

    with pytest.raises(ValueError, match="target shape mismatch"):
        build({
            "type": "step10", "repo": "tc.steps",
            "data": {
                "criterion": criterion,
                "batch": (torch.zeros(8, 2), torch.zeros(8, 5)),   # bad target dim
            },
        })


# =============================================================================
# Compute markers
# =============================================================================


def test_checksum_writes_sha_prefix_into_meta() -> None:
    @_Steps.register_artifact
    def step11(model: Annotated[nn.Module, Checksum("hash")]) -> str:
        return "ok"

    cfg: dict[str, Any] = {"type": "step11", "repo": "tc.steps", "meta": {},
                            "data": {"model": nn.Linear(4, 2)}}
    build(cfg)
    assert cfg["meta"]["hash"].startswith("sha256:")


def test_num_params_writes_parameter_count() -> None:
    @_Steps.register_artifact
    def step12(model: Annotated[nn.Module, NumParams("n")]) -> str:
        return "ok"

    m = nn.Linear(4, 2)  # 4*2 + 2 = 10 params
    cfg: dict[str, Any] = {"type": "step12", "repo": "tc.steps", "meta": {},
                            "data": {"model": m}}
    build(cfg)
    assert cfg["meta"]["n"] == 10


def test_effective_lr_writes_optimizer_lr() -> None:
    @_Steps.register_artifact
    def step13(opt: Annotated[torch.optim.Optimizer, EffectiveLr("lr")]) -> str:
        return "ok"

    m = nn.Linear(4, 2)
    cfg: dict[str, Any] = {"type": "step13", "repo": "tc.steps", "meta": {},
                            "data": {"opt": torch.optim.SGD(m.parameters(), lr=0.123)}}
    build(cfg)
    assert cfg["meta"]["lr"] == pytest.approx(0.123)


def test_device_compute_marker_writes_device_string() -> None:
    @_Steps.register_artifact
    def step14(t: Annotated[torch.Tensor, Device("dev")]) -> str:
        return "ok"

    cfg: dict[str, Any] = {"type": "step14", "repo": "tc.steps", "meta": {},
                            "data": {"t": torch.zeros(2)}}
    build(cfg)
    assert cfg["meta"]["dev"].startswith("cpu")


def test_input_shape_compute_marker_writes_tensor_shape() -> None:
    @_Steps.register_artifact
    def step15(x: Annotated[torch.Tensor, InputShape("shape")]) -> str:
        return "ok"

    cfg: dict[str, Any] = {"type": "step15", "repo": "tc.steps", "meta": {},
                            "data": {"x": torch.zeros(8, 4, 3)}}
    build(cfg)
    assert cfg["meta"]["shape"] == (4, 3)


def test_output_shape_compute_marker_smoke() -> None:
    @_Steps.register_artifact
    def step16(y: Annotated[torch.Tensor, OutputShape("out")]) -> str:
        return "ok"

    cfg: dict[str, Any] = {"type": "step16", "repo": "tc.steps", "meta": {},
                            "data": {"y": torch.zeros(2, 5)}}
    build(cfg)
    assert cfg["meta"]["out"] == (5,)


# =============================================================================
# Meter
# =============================================================================


def test_torch_profiler_meter_writes_profile_keys() -> None:
    attach_meter(TorchProfilerMeter(top_n=3))
    try:
        @_Steps.register_artifact
        def step17(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
            return model(x)

        cfg: dict[str, Any] = {
            "type": "step17", "repo": "tc.steps", "meta": {},
            "data": {"model": nn.Linear(4, 2), "x": torch.zeros(8, 4)},
        }
        build(cfg)
        assert "torch_profile_self_cpu_time_total_us" in cfg["meta"]
        assert "torch_profile_top_ops" in cfg["meta"]
        assert isinstance(cfg["meta"]["torch_profile_top_ops"], list)
    finally:
        detach_meter("torch_profiler")


def test_torch_profiler_meter_cleans_up_on_error() -> None:
    attach_meter(TorchProfilerMeter())
    try:
        @_Steps.register_artifact
        def step18(model: nn.Module) -> torch.Tensor:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            build({"type": "step18", "repo": "tc.steps",
                    "data": {"model": nn.Linear(4, 2)}})
    finally:
        detach_meter("torch_profiler")


# =============================================================================
# Reporter
# =============================================================================


def test_tensorboard_reporter_writes_scalars_for_numeric_meta(tmp_path: Any) -> None:
    pytest.importorskip("torch.utils.tensorboard")

    log_dir = tmp_path / "runs"
    reporter = TensorBoardReporter(log_dir=str(log_dir))
    attach_reporter(reporter)
    try:
        @_Steps.register_artifact
        def step19(x: Annotated[torch.Tensor, InputShape("shape")]) -> int:
            return int(x.numel())

        cfg: dict[str, Any] = {
            "type": "step19", "repo": "tc.steps", "meta": {"loss": 0.5},
            "data": {"x": torch.zeros(8, 4)},
        }
        build(cfg)

        # Force flush / inspect that some event file got written
        files = list(log_dir.iterdir())
        assert files, "TensorBoardReporter should have produced at least one event file"
    finally:
        detach_reporter("tensorboard")
