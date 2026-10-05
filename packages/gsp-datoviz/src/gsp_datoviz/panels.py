"""Retained panel activation and aggregate layout snapshots."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from gsp.protocol import (
    PanelLayoutIntent,
    PixelOrigin,
    RenderTarget,
    ResolvedLayoutSnapshot,
    View2D,
    View3D,
    resolve_panel_layout_intent,
)

from ._state import (
    DatovizRetainedView3DUpdateStats,
    _DatovizRetainedPanelState,
)
from .layout import (
    _consumed_layout_native_panel_bounds,
    _create_panel,
    _preflight_consumed_layout_panel_api,
    _resolve_datoviz_partial_layout_snapshot,
    _validate_consumed_perspective_aspect,
    _validate_renderer_consumed_layout_view,
)
from .native_api import (
    _configure_datoviz_view3d_camera,
    _configure_ndc_panel_view2d,
    _set_panel_background_color,
)

if TYPE_CHECKING:
    from .protocol_renderer import DatovizV04ProtocolRenderer


def _store_active_panel_state(self: DatovizV04ProtocolRenderer) -> None:
    self._panel_states[self.panel_id] = _DatovizRetainedPanelState(
        panel=self.panel,
        view=self.view,
        view3d=self.view3d,
        panel_bounds=self.panel_bounds,
        consumed_layout_snapshot=self.consumed_layout_snapshot,
        retained_view2d_position_uploads=self.retained_view2d_position_uploads,
        retained_view3d_meshes=self.retained_view3d_meshes,
        retained_view3d_texts=self.retained_view3d_texts,
        retained_view3d_update_stats=self.retained_view3d_update_stats,
        view2d_axis_state=self.view2d_axis_state,
        native_view3d_camera=self.native_view3d_camera,
    )


def activate_panel(self: DatovizV04ProtocolRenderer, panel_id: str) -> None:
    """Select one retained panel as the target of subsequent adapter operations."""
    if panel_id == self.panel_id:
        self._store_active_panel_state()
        return
    # Preserve the direct-renderer single-panel API, where QueryRequest historically used
    # an arbitrary scene panel id because the renderer itself had no scene identity.
    if len(self._panel_states) == 1:
        return
    self._store_active_panel_state()
    try:
        state = self._panel_states[panel_id]
    except KeyError as exc:
        raise ValueError(f"renderer has no retained panel {panel_id!r}") from exc
    self.panel_id = panel_id
    self.panel = state.panel
    self.view = state.view
    self.view3d = state.view3d
    self.panel_bounds = state.panel_bounds
    self.consumed_layout_snapshot = state.consumed_layout_snapshot
    self.retained_view2d_position_uploads = state.retained_view2d_position_uploads
    self.retained_view3d_meshes = state.retained_view3d_meshes
    self.retained_view3d_texts = state.retained_view3d_texts
    self.retained_view3d_update_stats = state.retained_view3d_update_stats
    self.view2d_axis_state = state.view2d_axis_state
    self.native_view3d_camera = state.native_view3d_camera


def add_retained_panel(
    self: DatovizV04ProtocolRenderer,
    *,
    panel_id: str,
    view: View2D | None,
    view3d: View3D | None,
    panel_layout: PanelLayoutIntent,
    consumed_layout_snapshot: ResolvedLayoutSnapshot | None = None,
) -> Any:
    """Create another typed panel in this renderer's retained figure."""
    if panel_id in self._panel_states:
        raise ValueError(f"renderer already contains panel {panel_id!r}")
    if view is not None and view3d is not None:
        raise ValueError("a retained panel accepts either View2D or View3D, not both")
    if view is not None and view.panel_id != panel_id:
        raise ValueError("View2D panel_id does not match retained panel")
    if view3d is not None and view3d.panel_id != panel_id:
        raise ValueError("View3D panel_id does not match retained panel")
    if consumed_layout_snapshot is not None:
        _preflight_consumed_layout_panel_api(self.dvz)
        _validate_renderer_consumed_layout_view(consumed_layout_snapshot, view=view, view3d=view3d)
        _validate_consumed_perspective_aspect(consumed_layout_snapshot, view3d)
        panel_bounds = _consumed_layout_native_panel_bounds(consumed_layout_snapshot)
    else:
        target = RenderTarget(
            logical_width_px=self.resolved_canvas.canvas_width_px,
            logical_height_px=self.resolved_canvas.canvas_height_px,
            device_scale=self.resolved_canvas.framebuffer_per_canvas_px,
            dpi=self.resolved_canvas.output_dpi,
            pixel_origin=PixelOrigin.TOP_LEFT,
            query_coordinate_space="plot",
        )
        resolved = tuple(
            item
            for item in resolve_panel_layout_intent(panel_layout, target)
            if item.panel_id == panel_id
        )
        if len(resolved) != 1:
            raise ValueError("panel_layout must resolve the retained panel exactly once")
        rect = resolved[0].panel_rect_px
        panel_bounds = (
            rect.x / target.logical_width_px,
            rect.y / target.logical_height_px,
            rect.width / target.logical_width_px,
            rect.height / target.logical_height_px,
        )

    self._store_active_panel_state()
    self.panel_id = panel_id
    self.view = view
    self.view3d = view3d
    self.panel_bounds = panel_bounds
    self.consumed_layout_snapshot = consumed_layout_snapshot
    self.retained_view2d_position_uploads = []
    self.retained_view3d_meshes = []
    self.retained_view3d_texts = []
    self.retained_view3d_update_stats = DatovizRetainedView3DUpdateStats()
    self.view2d_axis_state = None
    self.native_view3d_camera = None
    self.panel = _create_panel(self.dvz, self.figure, panel_bounds)
    _set_panel_background_color(self.dvz, self.panel, self.background_rgba8)
    _configure_ndc_panel_view2d(self.dvz, self.panel)
    if view is not None:
        self.apply_datoviz_data_view2d(view)
    if view3d is not None:
        self.native_view3d_camera = _configure_datoviz_view3d_camera(self.dvz, self.panel, view3d)
        self.retained_view3d_update_stats.view_projection_uniform_updates += 1
    self._store_active_panel_state()
    return self.panel


def set_authoritative_scene_layout_snapshot(
    self: DatovizV04ProtocolRenderer, snapshot: ResolvedLayoutSnapshot | None
) -> None:
    self._authoritative_scene_layout_snapshot = snapshot


def resolve_partial_layout_snapshot(
    self: DatovizV04ProtocolRenderer, *, snapshot_id_prefix: str = "layout:datoviz"
) -> ResolvedLayoutSnapshot:
    """Map Datoviz-reported panel frame fields into a partial GSP snapshot.

    The adapter only copies geometry and identities reported by Datoviz. Missing guide/query
    semantics remain explicit diagnostics and are intentionally not synthesized here.
    """
    if self._authoritative_scene_layout_snapshot is not None:
        return self._authoritative_scene_layout_snapshot
    if len(self._panel_states) > 1:
        active_panel_id = self.panel_id
        snapshots: list[ResolvedLayoutSnapshot] = []
        for panel_id in self._panel_states:
            self.activate_panel(panel_id)
            snapshots.append(
                _resolve_datoviz_partial_layout_snapshot(
                    self.dvz,
                    self.panel,
                    resolved_canvas=self.resolved_canvas,
                    panel_id=panel_id,
                    view_id=(
                        self.view.id
                        if self.view is not None
                        else self.view3d.id
                        if self.view3d is not None
                        else None
                    ),
                    snapshot_id_prefix=snapshot_id_prefix,
                )
            )
        self.activate_panel(active_panel_id)
        return ResolvedLayoutSnapshot(
            snapshot_id=f"{snapshot_id_prefix}:multi-panel",
            render_target=snapshots[0].render_target,
            panels=tuple(snapshot.only_panel() for snapshot in snapshots),
        )
    if self.consumed_layout_snapshot is not None:
        return self.consumed_layout_snapshot
    return _resolve_datoviz_partial_layout_snapshot(
        self.dvz,
        self.panel,
        resolved_canvas=self.resolved_canvas,
        panel_id=(
            self.view.panel_id
            if self.view is not None
            else self.view3d.panel_id
            if self.view3d is not None
            else "panel:default"
        ),
        view_id=(
            self.view.id
            if self.view is not None
            else self.view3d.id
            if self.view3d is not None
            else None
        ),
        snapshot_id_prefix=snapshot_id_prefix,
    )


def authoritative_layout_snapshot(
    self: DatovizV04ProtocolRenderer,
) -> ResolvedLayoutSnapshot | None:
    """Return the consumed GSP layout retained independently of the native panel."""
    return self._authoritative_scene_layout_snapshot or self.consumed_layout_snapshot
