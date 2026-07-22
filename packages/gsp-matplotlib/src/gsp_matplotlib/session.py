"""GSP session implementation for the Matplotlib reference provider."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from gsp import Scene
from gsp.backends import SessionRequest
from gsp.protocol import View2D

from .capabilities import capability_snapshot
from .layout import resolve_matplotlib_layout_snapshot
from .protocol_renderer import MatplotlibProtocolRenderResult, render_protocol_scene_with_layout


class _MatplotlibLiveView2DBinding:
    """Synchronize native axes limits with one canonical session-owned View2D."""

    def __init__(
        self,
        *,
        result: MatplotlibProtocolRenderResult,
        scene: Scene,
        view: View2D,
    ) -> None:
        self.result = result
        self.scene = scene
        self.view = view
        self.revision_index = 1
        self.view2d_revision = "view-rev:matplotlib-live-1"
        self.view_snapshot_id = (
            result.view_snapshot_id or "view-snapshot:matplotlib-live-1"
        )
        self._applying_canonical_view = False
        self._closed = False
        self._callback_ids = (
            result.axes.callbacks.connect("xlim_changed", self._on_native_limits),
            result.axes.callbacks.connect("ylim_changed", self._on_native_limits),
        )
        object.__setattr__(result, "view_snapshot_id", self.view_snapshot_id)

    @property
    def closed(self) -> bool:
        return self._closed

    def apply_canonical_view(self, view: View2D) -> None:
        """Apply accepted canonical state without recursively accepting callbacks."""
        if self._closed:
            raise RuntimeError("live View2D binding is closed")
        if view.id != self.view.id or view.panel_id != self.view.panel_id:
            raise ValueError("canonical View2D target does not match live binding")
        self._applying_canonical_view = True
        try:
            self.result.axes.set_xlim(view.x_range)
            self.result.axes.set_ylim(view.y_range)
        finally:
            self._applying_canonical_view = False
        self._accept_view(view)

    def close(self) -> None:
        if self._closed:
            return
        for callback_id in self._callback_ids:
            self.result.axes.callbacks.disconnect(callback_id)
        self._closed = True

    def _on_native_limits(self, _axes: Any) -> None:
        if self._closed or self._applying_canonical_view:
            return
        axes = self.result.axes
        x0, x1 = axes.get_xlim()
        y0, y1 = axes.get_ylim()
        self._accept_view(
            replace(
                self.view,
                x_range=(float(x0), float(x1)),
                y_range=(float(y0), float(y1)),
            )
        )

    def _accept_view(self, view: View2D) -> None:
        if view == self.view:
            return
        self.revision_index += 1
        self.view = view
        self.view2d_revision = f"view-rev:matplotlib-live-{self.revision_index}"
        self.view_snapshot_id = f"view-snapshot:matplotlib-live-{self.revision_index}"
        layout_snapshot = resolve_matplotlib_layout_snapshot(
            self.result.figure,
            self.result.axes,
            snapshot_id=f"layout:matplotlib-live-{self.revision_index}",
            view=view,
            axis_guides=self.scene.axis_guides,
            panel_text_guides=self.scene.panel_text_guides,
        )
        object.__setattr__(self.result, "layout_snapshot", layout_snapshot)
        object.__setattr__(self.result, "view_snapshot_id", self.view_snapshot_id)


class MatplotlibSession:
    backend_name = "matplotlib"

    def __init__(self, *, request: SessionRequest) -> None:
        self.request = request
        self.capabilities = capability_snapshot()
        self._diagnostics: list[str] = []
        self._results: list[MatplotlibProtocolRenderResult] = []
        self._view2d_bindings: dict[Any, _MatplotlibLiveView2DBinding] = {}
        self._closed = False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")

    def render(
        self,
        scene: Scene,
        *,
        target: str | Path | None = None,
        output_dpi: float | None = None,
        **savefig_kwargs: Any,
    ) -> MatplotlibProtocolRenderResult:
        self._require_open()
        if not isinstance(scene, Scene):
            raise TypeError("render() requires a gsp.Scene")
        result = render_protocol_scene_with_layout(
            visuals=scene.visuals,
            view=scene.view2d,
            view3d=scene.view3d,
            axis_guides=scene.axis_guides,
            panel_text_guides=scene.panel_text_guides,
            colorbar_guides=scene.colorbar_guides,
            color_scales={item.id: item for item in scene.color_scales},
            transform_resources={item.id: item for item in scene.transforms},
            canvas_size=scene.canvas_size,
            output_dpi=output_dpi,
        )
        self._results.append(result)
        if scene.view2d is not None:
            self._view2d_bindings[result.axes] = _MatplotlibLiveView2DBinding(
                result=result,
                scene=scene,
                view=scene.view2d,
            )
        if target is not None:
            result.figure.savefig(target, **savefig_kwargs)
        return result

    def display(self, scene: Scene, **kwargs: Any) -> MatplotlibProtocolRenderResult:
        return self.render(scene, **kwargs)

    def run(self) -> None:
        self._require_open()
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        if self._closed:
            return
        import matplotlib.pyplot as plt

        for binding in self._view2d_bindings.values():
            binding.close()
        for result in self._results:
            plt.close(result.figure)
        self._closed = True

    def __enter__(self) -> "MatplotlibSession":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
