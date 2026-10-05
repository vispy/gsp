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
    ClipScope,
    TextVisual,
    VIEW3D_QUERY_PAYLOAD_KIND,
    View2D,
    View3D,
    resolve_view3d_projection_snapshot,
)

from ._live_view2d import _MatplotlibLiveView2DBinding
from ._point_update import update_point_collection
from ._native_point_query import native_point_query
from ._panel_state import (
    _combine_panel_results,
    _matplotlib_panel_bounds,
    _scene_panel_ids,
    _snapshot_for_panel,
    _validate_consumed_layout_scene,
)
from .capabilities import capability_snapshot
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


class MatplotlibSession:
    backend_name = "matplotlib"

    def __init__(self, *, request: SessionRequest) -> None:
        self.request = request
        self.capabilities = capability_snapshot()
        self._diagnostics: list[str] = []
        self._results: list[MatplotlibProtocolRenderResult] = []
        self._scene_results: dict[str, tuple[Scene, MatplotlibProtocolRenderResult]] = {}
        self._latest_scene_id: str | None = None
        self._scene_revisions: dict[str, int] = {}
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
        import matplotlib.pyplot as plt

        figure = plt.figure()
        bindings: dict[Any, _MatplotlibLiveView2DBinding] = {}
        try:
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
                    visual_z_orders={item.visual_id: item.z_order for item in scene.attachments},
                )
                figure = panel_result.figure
                panel_results.append(panel_result)

            result = _combine_panel_results(panel_results, consumed=layout_snapshot is not None)
            for view2d in scene.views2d:
                axes = result.axes_for_panel(view2d.panel_id)
                bindings[axes] = _MatplotlibLiveView2DBinding(
                    result=result,
                    scene=scene,
                    view=view2d,
                    axes=axes,
                )
            if target is not None:
                savefig_kwargs.setdefault("dpi", result.figure.dpi)
                result.figure.savefig(target, **savefig_kwargs)
        except BaseException:
            for binding in bindings.values():
                binding.close()
            plt.close(figure)
            raise
        self._results.append(result)
        self._scene_results[scene.id] = (scene, result)
        self._scene_revisions[scene.id] = self._scene_revisions.get(scene.id, -1) + 1
        self._latest_scene_id = scene.id
        self._view2d_bindings.update(bindings)
        return result

    def scene_revision(self, scene_id: str | None = None) -> int:
        """Return the revision of the latest successful render/update of a scene."""
        self._require_open()
        scene, _ = self._query_target(scene_id)
        return self._scene_revisions[scene.id]

    def update_point(self, visual: PointVisual, *, scene_id: str | None = None) -> int:
        """Update point positions, colors and sizes without recreating graphics resources."""
        self._require_open()
        if not isinstance(visual, PointVisual):
            raise TypeError("update_point() requires a PointVisual")
        scene, result = self._query_target(scene_id)
        binding = (
            self._view2d_bindings.get(
                result.axes_for_panel(scene.attachment_for_visual(visual.id).panel_id)
            )
            if any(item.id == visual.id for item in scene.visuals)
            else None
        )
        updated = update_point_collection(
            scene, result, visual, view=binding.view if binding is not None else None
        )
        self._scene_results[scene.id] = (updated, result)
        for binding in self._view2d_bindings.values():
            if binding.result is result:
                binding.scene = updated
        self._scene_revisions[scene.id] += 1
        result.figure.canvas.draw_idle()
        return self._scene_revisions[scene.id]

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
        binding = self._view2d_bindings.get(result.axes_for_panel(request.panel_id))
        if binding is not None:
            active_view = binding.view

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

        point_only = all(
            isinstance(visual, PointVisual) for visual in scene.visuals_for_panel(request.panel_id)
        )
        # Native point footprints cover both coordinate spaces. Keep the global
        # capability conservative for mixed scenes using standalone evaluators.
        decision_request = effective_request
        if (
            point_only
            and effective_request.scope is QueryScope.DATA
            and not effective_request.requested_extension_payload_kinds
        ):
            decision_request = replace(
                effective_request, scope=QueryScope.DATA, coordinate_space=QueryCoordinateSpace.DATA
            )
        decision = self.capabilities.adapt_query_request(decision_request)
        if decision.outcome is not AdaptationOutcome.ACCEPT:
            return unsupported_query_result(
                effective_request,
                decision.diagnostic or "Matplotlib query request is unsupported",
            )

        entries = tuple(
            QueryVisualEntry(
                visual,
                z_order=scene.attachment_for_visual(visual.id).z_order,
                native_query=(
                    native_point_query(
                        scene,
                        result,
                        visual,
                        active_view if isinstance(active_view, View2D) else None,
                    )
                    if isinstance(visual, PointVisual)
                    else None
                ),
            )
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
