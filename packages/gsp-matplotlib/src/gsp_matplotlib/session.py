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
    TextVisual,
    VIEW3D_QUERY_PAYLOAD_KIND,
    View2D,
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
        self._scene_results[scene.id] = (scene, result)
        self._latest_scene_id = scene.id
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
                else result.view_snapshot_id
            ),
        )
        panel_bounds = _matplotlib_panel_bounds(result)

        if (
            scene.view3d is not None
            and VIEW3D_QUERY_PAYLOAD_KIND
            in effective_request.requested_extension_payload_kinds
        ):
            snapshot = resolve_view3d_projection_snapshot(
                scene.view3d, layout_snapshot_id=layout_snapshot_id
            )
            return query_view3d_ray_context(
                effective_request,
                scene.view3d,
                snapshot,
                panel_bounds=panel_bounds,
            )

        guide_extension_request = (
            effective_request.scope is QueryScope.GUIDES
            and set(effective_request.requested_extension_payload_kinds).issubset(
                {GUIDE_QUERY_PAYLOAD_KIND}
            )
        )
        if (
            effective_request.requested_extension_payload_kinds
            and not guide_extension_request
        ):
            return unsupported_query_result(
                effective_request,
                "Matplotlib public panel query does not support extension payloads: "
                f"{effective_request.requested_extension_payload_kinds}",
            )

        if effective_request.scope in (QueryScope.DATA, QueryScope.ALL_RENDERED):
            unsupported = tuple(
                type(visual).__name__
                for visual in scene.visuals
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
            QueryVisualEntry(visual, z_order=index)
            for index, visual in enumerate(scene.visuals)
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
                view=scene.view2d,
                transform_resources={item.id: item for item in scene.transforms},
            )
        if effective_request.scope is QueryScope.GUIDES:
            return query_resolved_layout_guides(
                effective_request, result.layout_snapshot
            )
        return query_scoped_scene(
            effective_request,
            visual_entries=entries,
            view=scene.view2d,
            layout_snapshot=result.layout_snapshot,
            panel_bounds=(
                panel_bounds
                if effective_request.coordinate_space is QueryCoordinateSpace.PANEL
                else None
            ),
        )

    def _query_target(
        self, scene_id: str | None
    ) -> tuple[Scene, MatplotlibProtocolRenderResult]:
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
    panel_ids = {panel.id for panel in scene.panels}
    if scene.view2d is not None:
        panel_ids.add(scene.view2d.panel_id)
    if scene.view3d is not None:
        panel_ids.add(scene.view3d.panel_id)
    return frozenset(panel_ids)


def _matplotlib_panel_bounds(
    result: MatplotlibProtocolRenderResult,
) -> tuple[float, float, float, float]:
    rect = result.layout_snapshot.panel_rect_px
    return (rect.x, rect.x + rect.width, rect.y, rect.y + rect.height)
