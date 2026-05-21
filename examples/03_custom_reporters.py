#!/usr/bin/env python
"""Custom reporters: how to extend FactoryReporter for W&B, TensorBoard, etc.

The library ships JournalReporter / HTTPDashboardReporter / OpenTelemetryReporter
out of the box. For everything else (Weights & Biases, TensorBoard, MLflow,
ClearML, Neptune, ...) you just subclass FactoryReporter and attach it.

This example shows two such custom reporters. Neither runs without its dep:

    pip install wandb tensorboard

If you skip the deps, this file still imports fine -- the imports are inside
the reporter classes' __init__.
"""

from __future__ import annotations

from typing import Any

from registry import FactoryReporter, FunctionalRegistry, attach_reporter, build


# =============================================================================
# WandB reporter -- one run per top-level build, meta fields become metrics
# =============================================================================


class WandbReporter(FactoryReporter):
    """Logs each top-level build to a Weights & Biases run.

    On build start a new W&B run opens; meta dict at on_built is logged as
    metrics; on_error sets the run status to 'failed' and closes it. Nested
    builds reuse the parent's run (single run per training invocation).
    """

    name: str = "wandb"

    def __init__(
        self,
        project: str,
        entity: str | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        import wandb

        self._wandb = wandb
        self._project: str = project
        self._entity: str | None = entity
        self._config: dict[str, Any] = config or {}
        self._run: Any | None = None
        self._depth: int = 0

    def on_build_start(self, *, cfg, ctx, meta) -> None:
        self._depth += 1
        if self._depth == 1:
            self._run = self._wandb.init(
                project=self._project,
                entity=self._entity,
                config={**self._config, "registry.type": getattr(cfg, "type", "?")},
                reinit=True,
            )

    def on_built(self, *, target, result, meta, ctx) -> None:
        if self._run is not None:
            metrics = {k: v for k, v in meta.items() if isinstance(v, (int, float, bool))}
            if metrics:
                self._wandb.log(metrics)
        self._depth -= 1
        if self._depth == 0 and self._run is not None:
            self._wandb.finish()
            self._run = None

    def on_error(self, *, cfg, exc, ctx, meta) -> None:
        if self._run is not None:
            self._wandb.log({"error": str(exc), "error_type": type(exc).__name__})
            self._wandb.finish(exit_code=1)
            self._run = None
        self._depth = 0


# =============================================================================
# TensorBoard reporter -- writes meta fields as scalars
# =============================================================================


# (TensorBoardReporter lives in registry.experimental.torch_compat -- see
# the [torch] extra. This file focuses on WandB as a downstream extension
# example since wandb is not torch-specific.)


# =============================================================================
# Demo wiring -- run if you have wandb / tensorboard installed
# =============================================================================


def main() -> None:
    class StepRegistry(FunctionalRegistry):
        pass

    @StepRegistry.register_artifact
    def fake_step(epoch: int) -> dict[str, float]:
        return {"loss": 1.0 / (epoch + 1), "acc": 0.5 + 0.05 * epoch}

    # WandB: custom extension defined inline above.
    try:
        attach_reporter(WandbReporter(project="registry-demo"))
        print("wandb reporter attached")
    except ImportError:
        print("wandb not installed; skipping WandbReporter")

    # TensorBoard: lives in the library, just import + attach.
    try:
        from registry.experimental.torch_compat import TensorBoardReporter

        attach_reporter(TensorBoardReporter(log_dir="runs/demo"))
        print("tensorboard reporter attached")
    except ImportError:
        print("torch / tensorboard not installed; skipping TensorBoardReporter")

    # Loop is the consumer's; reporters fire on every build.
    for epoch in range(3):
        build({"type": "fake_step", "data": {"epoch": epoch}, "meta": {}})


if __name__ == "__main__":
    main()
