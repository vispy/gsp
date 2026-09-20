"""GSP session implementation for the Matplotlib reference provider."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from gsp import Scene
from gsp.backends import SessionRequest
from gsp.protocol import (
    AdaptationOutcome,
    GUIDE_QUERY_PAYLOAD_KIND,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    PointVisual,
    QueryCoordinateSpace,
    QueryRequest,
    QueryResult,
    QueryScope,
    ResolvedLayoutSnapshot,
    CanvasSize,
    ClipScope,
    TextVisual,
    VIEW3D_QUERY_PAYLOAD_KIND,
    View2D,
    View3D,
    resolve_panel_layout_intent,
    resolve_view3d_projection_snapshot,
)

from .capabilities import capability_snapshot
from .layout import resolve_matplotlib_layout_snapshot
from .layout_query import query_resolved_layout_guides
from .protocol_query import (
    QueryVisualEntry,
    query_view3d_ray_context,
    query_visuals,
    unsupported_query_result,
)
from .protocol_renderer import MatplotlibProtocolRenderResult, render_protocol_scene_with_layout
from .scoped_query import query_scoped_scene


_QUERYABLE_VISUAL_TYPES = (
    PointVisual,
    MarkerVisual,
    ImageVisual,
    TextVisual,
    MeshVisual,
)


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


class MatplotlibSession:
    backend_name = "matplotlib"

    def __init__(self, *, request: SessionRequest) -> None:
        self.request = request
        self.capabilities = capability_snapshot()
        self._diagnostics: list[str] = []
        self._results: list[MatplotlibProtocolRenderResult] = []
        self._scene_results: dict[str, tuple[Scene, MatplotlibProtocolRenderResult]] = {}
        self._latest_scene_id: str | None = None
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
        layout_snapshot: ResolvedLayoutSnapshot | None = None,
        **savefig_kwargs: Any,
    ) -> MatplotlibProtocolRenderResult:
        self._require_open()
        if not isinstance(scene, Scene):
            raise TypeError("render() requires a gsp.Scene")
        unsupported_clip_scopes = {
            attachment.clip_scope
            for attachment in scene.attachments
            if attachment.clip_scope is not ClipScope.PLOT
        }
        if unsupported_clip_scopes:
            raise ValueError(
                "Matplotlib adapter does not support attachment clip scopes "
                f"{sorted(scope.value for scope in unsupported_clip_scopes)!r}"
            )
        _validate_consumed_layout_scene(scene, layout_snapshot)
        panel_results: list[MatplotlibProtocolRenderResult] = []
        figure = None
        for panel in scene.panels:
            active_view = scene.primary_view_for_panel(panel.id)
            panel_result = render_protocol_scene_with_layout(
                visuals=scene.visuals_for_panel(panel.id),
                view=active_view if isinstance(active_view, View2D) else None,
                view3d=active_view if isinstance(active_view, View3D) else None,
                axis_guides=tuple(
                    guide
                    for guide in scene.axis_guides
                    if active_view is not None and guide.view_id == active_view.id
                ),
                panel_text_guides=tuple(
                    guide for guide in scene.panel_text_guides if guide.panel_id == panel.id
                ),
                colorbar_guides=tuple(
                    guide for guide in scene.colorbar_guides if guide.panel_id == panel.id
                ),
                color_scales={item.id: item for item in scene.color_scales},
                transform_resources={item.id: item for item in scene.transforms},
                canvas_size=scene.canvas_size,
                output_dpi=output_dpi,
                layout_snapshot=(
                    _snapshot_for_panel(layout_snapshot, panel.id)
                    if layout_snapshot is not None
                    else None
                ),
                panel_id=panel.id,
                panel_layout=scene.panel_layout,
                figure=figure,
            )
            figure = panel_result.figure
            panel_results.append(panel_result)

        result = _combine_panel_results(panel_results, consumed=layout_snapshot is not None)
        self._results.append(result)
        self._scene_results[scene.id] = (scene, result)
        self._latest_scene_id = scene.id
        for view2d in scene.views2d:
            axes = result.axes_for_panel(view2d.panel_id)
            self._view2d_bindings[axes] = _MatplotlibLiveView2DBinding(
                result=result,
                scene=scene,
                view=view2d,
                axes=axes,
            )
        if target is not None:
            savefig_kwargs.setdefault("dpi", result.figure.dpi)
            result.figure.savefig(target, **savefig_kwargs)
        return result

    def display(
        self,
        scene: Scene,
        *,
        layout_snapshot: ResolvedLayoutSnapshot | None = None,
        **kwargs: Any,
    ) -> MatplotlibProtocolRenderResult:
        return self.render(scene, layout_snapshot=layout_snapshot, **kwargs)

    def query(
        self,
        request: QueryRequest,
        *,
        scene_id: str | None = None,
    ) -> QueryResult:
        """Query the latest render of one scene without exposing renderer objects."""
        self._require_open()
        if not isinstance(request, QueryRequest):
            raise TypeError("query() requires a QueryRequest")
        scene, result = self._query_target(scene_id)
        if request.panel_id not in _scene_panel_ids(scene):
            return unsupported_query_result(
                request,
                f"panel {request.panel_id!r} is not present in scene {scene.id!r}",
            )

        layout_snapshot_id = result.layout_snapshot_id
        effective_request = replace(
            request,
            layout_snapshot_id=(
                request.layout_snapshot_id
                if request.layout_snapshot_id is not None
                else layout_snapshot_id
            ),
            view_snapshot_id=(
                request.view_snapshot_id
                if request.view_snapshot_id is not None
                else result.view_snapshot_id_for_panel(request.panel_id)
            ),
        )
        panel_layout = _snapshot_for_panel(result.layout_snapshot, request.panel_id)
        panel_bounds = _matplotlib_panel_bounds(result, request.panel_id)
        active_view = scene.primary_view_for_panel(request.panel_id)

        if (
            isinstance(active_view, View3D)
            and VIEW3D_QUERY_PAYLOAD_KIND in effective_request.requested_extension_payload_kinds
        ):
            snapshot = resolve_view3d_projection_snapshot(active_view, layout_snapshot=panel_layout)
            return query_view3d_ray_context(
                effective_request,
                active_view,
                snapshot,
                panel_bounds=panel_bounds,
                layout_snapshot=panel_layout,
            )

        guide_extension_request = effective_request.scope is QueryScope.GUIDES and set(
            effective_request.requested_extension_payload_kinds
        ).issubset({GUIDE_QUERY_PAYLOAD_KIND})
        if effective_request.requested_extension_payload_kinds and not guide_extension_request:
            return unsupported_query_result(
                effective_request,
                "Matplotlib public panel query does not support extension payloads: "
                f"{effective_request.requested_extension_payload_kinds}",
            )

        if effective_request.scope in (QueryScope.DATA, QueryScope.ALL_RENDERED):
            unsupported = tuple(
                type(visual).__name__
                for visual in scene.visuals_for_panel(request.panel_id)
                if not isinstance(visual, _QUERYABLE_VISUAL_TYPES)
            )
            if unsupported:
                return unsupported_query_result(
                    effective_request,
                    "Matplotlib public query does not support rendered visual "
                    f"families: {unsupported}",
                )

        decision = self.capabilities.adapt_query_request(effective_request)
        if decision.outcome is not AdaptationOutcome.ACCEPT:
            return unsupported_query_result(
                effective_request,
                decision.diagnostic or "Matplotlib query request is unsupported",
            )

        entries = tuple(
            QueryVisualEntry(visual, z_order=scene.attachment_for_visual(visual.id).z_order)
            for visual in scene.visuals_for_panel(request.panel_id)
            if isinstance(visual, _QUERYABLE_VISUAL_TYPES)
        )
        if effective_request.scope is QueryScope.DATA:
            return query_visuals(
                effective_request,
                entries,
                panel_bounds=(
                    panel_bounds
                    if effective_request.coordinate_space is QueryCoordinateSpace.PANEL
                    else None
                ),
                color_scales={item.id: item for item in scene.color_scales},
                view=active_view if isinstance(active_view, View2D) else None,
                transform_resources={item.id: item for item in scene.transforms},
            )
        if effective_request.scope is QueryScope.GUIDES:
            return query_resolved_layout_guides(effective_request, panel_layout)
        return query_scoped_scene(
            effective_request,
            visual_entries=entries,
            view=active_view if isinstance(active_view, View2D) else None,
            layout_snapshot=panel_layout,
            panel_bounds=(
                panel_bounds
                if effective_request.coordinate_space is QueryCoordinateSpace.PANEL
                else None
            ),
        )

    def _query_target(self, scene_id: str | None) -> tuple[Scene, MatplotlibProtocolRenderResult]:
        target = self._latest_scene_id if scene_id is None else scene_id
        if target is None:
            raise RuntimeError("query() requires a rendered scene")
        try:
            return self._scene_results[target]
        except KeyError as exc:
            raise RuntimeError(
                f"query() scene {target!r} has not been rendered by this session"
            ) from exc

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


def _scene_panel_ids(scene: Scene) -> frozenset[str]:
    return frozenset(panel.id for panel in scene.panels)


def _matplotlib_panel_bounds(
    result: MatplotlibProtocolRenderResult,
    panel_id: str,
) -> tuple[float, float, float, float]:
    rect = result.layout_snapshot.panel(panel_id).plot_rect_px
    return (rect.x, rect.x + rect.width, rect.y, rect.y + rect.height)


def _snapshot_for_panel(snapshot: ResolvedLayoutSnapshot, panel_id: str) -> ResolvedLayoutSnapshot:
    """Project one panel out of a scene-wide snapshot for singular helpers."""
    return ResolvedLayoutSnapshot(
        snapshot_id=snapshot.snapshot_id,
        render_target=snapshot.render_target,
        panels=(snapshot.panel(panel_id),),
    )


def _combine_panel_results(
    panel_results: list[MatplotlibProtocolRenderResult], *, consumed: bool
) -> MatplotlibProtocolRenderResult:
    if not panel_results:
        raise ValueError("Matplotlib rendering requires at least one panel")
    first = panel_results[0]
    render_target = first.layout_snapshot.render_target
    if any(result.layout_snapshot.render_target != render_target for result in panel_results[1:]):
        raise ValueError("Matplotlib panel passes resolved inconsistent render targets")
    snapshot_ids = {result.layout_snapshot.snapshot_id for result in panel_results}
    snapshot_id = next(iter(snapshot_ids)) if len(snapshot_ids) == 1 else "layout:matplotlib"
    snapshot = ResolvedLayoutSnapshot(
        snapshot_id=snapshot_id,
        render_target=render_target,
        panels=tuple(result.layout_snapshot.only_panel() for result in panel_results),
    )
    panel_axes = tuple(
        (result.layout_snapshot.only_panel().panel_id, result.axes) for result in panel_results
    )
    panel_view_snapshot_ids = tuple(
        (result.layout_snapshot.only_panel().panel_id, result.view_snapshot_id)
        for result in panel_results
    )
    panel_view3d_projection_snapshots = tuple(
        (
            result.layout_snapshot.only_panel().panel_id,
            result.view3d_projection_snapshot,
        )
        for result in panel_results
    )
    return MatplotlibProtocolRenderResult(
        figure=first.figure,
        axes=first.axes,
        layout_snapshot=snapshot,
        resolved_canvas=first.resolved_canvas,
        view_snapshot_id=first.view_snapshot_id if len(panel_results) == 1 else None,
        view3d_projection_snapshot=(
            first.view3d_projection_snapshot if len(panel_results) == 1 else None
        ),
        layout_was_consumed=consumed,
        panel_axes=panel_axes,
        panel_view_snapshot_ids=panel_view_snapshot_ids,
        panel_view3d_projection_snapshots=panel_view3d_projection_snapshots,
    )


def _replace_resolved_panel(
    current: ResolvedLayoutSnapshot, replacement: ResolvedLayoutSnapshot
) -> ResolvedLayoutSnapshot:
    panel = replacement.only_panel()
    return ResolvedLayoutSnapshot(
        snapshot_id=replacement.snapshot_id,
        render_target=current.render_target,
        panels=tuple(
            panel if existing.panel_id == panel.panel_id else existing
            for existing in current.panels
        ),
    )


def _set_panel_view_snapshot_id(
    result: MatplotlibProtocolRenderResult,
    panel_id: str,
    snapshot_id: str | None,
) -> None:
    updated = tuple(
        (candidate, snapshot_id if candidate == panel_id else existing)
        for candidate, existing in result.panel_view_snapshot_ids
    )
    object.__setattr__(result, "panel_view_snapshot_ids", updated)
    if len(updated) == 1:
        object.__setattr__(result, "view_snapshot_id", snapshot_id)


def _validate_consumed_layout_scene(
    scene: Scene, layout_snapshot: ResolvedLayoutSnapshot | None
) -> None:
    if layout_snapshot is None:
        return
    if not isinstance(layout_snapshot, ResolvedLayoutSnapshot):
        raise TypeError("layout_snapshot must be a ResolvedLayoutSnapshot")
    _validate_scene_canvas_target(scene.canvas_size, layout_snapshot)
    scene_panel_ids = tuple(panel.id for panel in scene.panels)
    snapshot_panel_ids = tuple(panel.panel_id for panel in layout_snapshot.panels)
    if set(scene_panel_ids) != set(snapshot_panel_ids):
        raise ValueError("layout_snapshot panels do not match the consumed scene panels")
    for panel in scene.panels:
        active_view = scene.primary_view_for_panel(panel.id)
        resolved_panel = layout_snapshot.panel(panel.id)
        if active_view is not None:
            if resolved_panel.view_id != active_view.id:
                raise ValueError("layout_snapshot view_id does not match the active scene view")
        elif resolved_panel.view_id is not None:
            raise ValueError("viewless scene panel cannot consume a view-bound layout_snapshot")
    expected = resolve_panel_layout_intent(scene.panel_layout, layout_snapshot.render_target)
    expected_rects = {panel.panel_id: panel.panel_rect_px for panel in expected}
    if any(
        expected_rects.get(panel.id) != layout_snapshot.panel(panel.id).panel_rect_px
        for panel in scene.panels
    ):
        raise ValueError("layout_snapshot panel_rect_px does not match the scene panel allocation")
    if scene.axis_guides or scene.colorbar_guides:
        raise ValueError(
            "consumed Matplotlib layout does not prove native axis/colorbar guide geometry"
        )
    for panel in scene.panels:
        panel_guides = tuple(
            guide for guide in scene.panel_text_guides if guide.panel_id == panel.id
        )
        title_guides = tuple(guide for guide in panel_guides if guide.role.value == "title")
        if len(title_guides) != len(panel_guides) or len(title_guides) > 1:
            raise ValueError("consumed Matplotlib layout supports at most one title per panel")
        title_box_ids = {box.guide_id for box in layout_snapshot.panel(panel.id).title_boxes}
        if title_guides and title_guides[0].id not in title_box_ids:
            raise ValueError("supplied title guide requires matching resolved title geometry")


def _validate_scene_canvas_target(
    canvas_size: CanvasSize | None, snapshot: ResolvedLayoutSnapshot
) -> None:
    if canvas_size is None:
        return
    target = snapshot.render_target
    resolved = canvas_size.resolve(
        output_dpi=canvas_size.reference_dpi * target.device_scale,
        device_scale=target.device_scale,
    )
    if (
        resolved.canvas_width_px != target.logical_width_px
        or resolved.canvas_height_px != target.logical_height_px
        or resolved.framebuffer_width != target.framebuffer_width_px
        or resolved.framebuffer_height != target.framebuffer_height_px
        or resolved.device_scale_x != target.device_scale
        or resolved.device_scale_y != target.device_scale
    ):
        raise ValueError(
            "scene canvas policy does not resolve to the consumed layout render target"
        )
