"""Native axes synchronization with canonical session-owned View2D state."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from gsp import Scene
from gsp.protocol import View2D

from .layout import resolve_matplotlib_layout_snapshot
from .protocol_renderer import MatplotlibProtocolRenderResult
from ._panel_state import _replace_resolved_panel, _set_panel_view_snapshot_id


class _MatplotlibLiveView2DBinding:
    """Synchronize native axes limits with one canonical session-owned View2D."""

    def __init__(
        self,
        *,
        result: MatplotlibProtocolRenderResult,
        scene: Scene,
        view: View2D,
        axes: Any,
    ) -> None:
        self.result = result
        self.scene = scene
        self.view = view
        self.axes = axes
        self.revision_index = 1
        self.view2d_revision = "view-rev:matplotlib-live-1"
        self.view_snapshot_id = (
            result.view_snapshot_id_for_panel(view.panel_id) or "view-snapshot:matplotlib-live-1"
        )
        self._applying_canonical_view = False
        self._closed = False
        self._callback_ids = (
            axes.callbacks.connect("xlim_changed", self._on_native_limits),
            axes.callbacks.connect("ylim_changed", self._on_native_limits),
        )
        _set_panel_view_snapshot_id(result, view.panel_id, self.view_snapshot_id)

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
            self.axes.set_xlim(view.x_range)
            self.axes.set_ylim(view.y_range)
        finally:
            self._applying_canonical_view = False
        self._accept_view(view)

    def close(self) -> None:
        if self._closed:
            return
        for callback_id in self._callback_ids:
            self.axes.callbacks.disconnect(callback_id)
        self._closed = True

    def _on_native_limits(self, _axes: Any) -> None:
        if self._closed or self._applying_canonical_view:
            return
        axes = self.axes
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
        if not self.result.layout_was_consumed:
            panel_snapshot = resolve_matplotlib_layout_snapshot(
                self.result.figure,
                self.axes,
                snapshot_id=f"layout:matplotlib-live-{self.revision_index}",
                panel_id=view.panel_id,
                view=view,
                axis_guides=tuple(
                    guide for guide in self.scene.axis_guides if guide.view_id == view.id
                ),
                panel_text_guides=tuple(
                    guide
                    for guide in self.scene.panel_text_guides
                    if guide.panel_id == view.panel_id
                ),
                panel_rect_px=self.result.layout_snapshot.panel(view.panel_id).panel_rect_px,
            )
            object.__setattr__(
                self.result,
                "layout_snapshot",
                _replace_resolved_panel(self.result.layout_snapshot, panel_snapshot),
            )
        _set_panel_view_snapshot_id(self.result, view.panel_id, self.view_snapshot_id)
