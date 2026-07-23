"""Backend-neutral session wrapper for the Datoviz v0.4 renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsp import Scene
from gsp.backends import SessionRequest
from gsp.protocol import (
    AdaptationOutcome,
    AxisDimension,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    PathVisual,
    PixelVisual,
    PointVisual,
    PrimitiveVisual,
    QueryRequest,
    QueryResult,
    QueryStatus,
    SegmentVisual,
    SphereVisual,
    TextVisual,
    TickSpecKind,
    VIEW3D_QUERY_PAYLOAD_KIND,
    VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY,
    VectorVisual,
)

from .capabilities import datoviz_v04_capability_snapshot
from .protocol_renderer import DatovizV04ProtocolRenderer, import_datoviz_v04


class DatovizSession:
    backend_name = "datoviz"

    def __init__(self, *, request: SessionRequest) -> None:
        self.request = request
        self._dvz = import_datoviz_v04()
        self.capabilities = datoviz_v04_capability_snapshot(self._dvz)
        self._diagnostics: list[str] = []
        self._renderers: list[DatovizV04ProtocolRenderer] = []
        self._renderer_scenes: dict[int, Scene] = {}
        self._scene_renderers: dict[str, tuple[Scene, DatovizV04ProtocolRenderer]] = {}
        self._latest_scene_id: str | None = None
        self._interactive_view2d_renderers: set[int] = set()
        self._interactive_view3d_renderers: set[int] = set()
        self._closed = False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def render(
        self,
        scene: Scene,
        *,
        target: str | Path | None = None,
        **kwargs: Any,
    ) -> DatovizV04ProtocolRenderer:
        self._require_open()
        if not isinstance(scene, Scene):
            raise TypeError("render() requires a gsp.Scene")
        if kwargs:
            raise TypeError(f"unsupported Datoviz render options: {sorted(kwargs)!r}")
        renderer = self._build_renderer(scene)
        self._renderers.append(renderer)
        self._renderer_scenes[id(renderer)] = scene
        self._scene_renderers[scene.id] = (scene, renderer)
        self._latest_scene_id = scene.id
        if target is not None:
            Path(target).write_bytes(renderer.capture_png_bytes())
        return renderer

    def display(
        self,
        scene: Scene,
        *,
        block: bool = True,
        frame_count: int = 1,
        **kwargs: Any,
    ) -> DatovizV04ProtocolRenderer:
        if kwargs:
            raise TypeError(f"unsupported Datoviz display options: {sorted(kwargs)!r}")
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        renderer = self.render(scene)
        self._enable_interactive_view2d(renderer, scene)
        self._enable_interactive_view3d(renderer, scene)
        if block:
            renderer.show(frame_count=frame_count)
        return renderer

    def query(
        self,
        request: QueryRequest,
        *,
        scene_id: str | None = None,
    ) -> QueryResult:
        """Query the latest live renderer for one session-rendered scene."""
        self._require_open()
        if not isinstance(request, QueryRequest):
            raise TypeError("query() requires a QueryRequest")
        scene, renderer = self._query_target(scene_id)
        if request.panel_id not in _scene_panel_ids(scene):
            return _unsupported_query_result(
                request,
                f"panel {request.panel_id!r} is not present in scene {scene.id!r}",
            )

        if (
            scene.view3d is not None
            and VIEW3D_QUERY_PAYLOAD_KIND
            in request.requested_extension_payload_kinds
        ):
            return renderer.query_view3d_ray_context(
                request, layout_snapshot_id="layout:datoviz-session"
            )

        unsupported = tuple(
            type(visual).__name__
            for visual in scene.visuals
            if not isinstance(visual, PointVisual)
        )
        if unsupported:
            return _unsupported_query_result(
                request,
                "Datoviz public panel query supports only point-only scenes; "
                f"unproven rendered visual families: {unsupported}",
            )

        decision = self.capabilities.adapt_query_request(request)
        if decision.outcome is not AdaptationOutcome.ACCEPT:
            return _unsupported_query_result(
                request,
                decision.diagnostic or "Datoviz query request is unsupported",
            )
        return renderer.query_panel(request)

    def _query_target(
        self, scene_id: str | None
    ) -> tuple[Scene, DatovizV04ProtocolRenderer]:
        target = self._latest_scene_id if scene_id is None else scene_id
        if target is None:
            raise RuntimeError("query() requires a rendered scene")
        try:
            return self._scene_renderers[target]
        except KeyError as exc:
            raise RuntimeError(
                f"query() scene {target!r} has not been rendered by this session"
            ) from exc

    def run(self) -> None:
        self._require_open()
        if not self._renderers:
            raise RuntimeError("run() requires a rendered scene")
        renderer = self._renderers[-1]
        scene = self._renderer_scenes[id(renderer)]
        self._enable_interactive_view2d(renderer, scene)
        self._enable_interactive_view3d(renderer, scene)
        renderer.show(frame_count=0)

    def close(self) -> None:
        if self._closed:
            return
        for renderer in reversed(self._renderers):
            renderer.close()
        self._closed = True

    def __enter__(self) -> "DatovizSession":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")

    def _enable_interactive_view2d(
        self, renderer: DatovizV04ProtocolRenderer, scene: Scene
    ) -> None:
        renderer_id = id(renderer)
        if renderer_id in self._interactive_view2d_renderers:
            return
        if scene.view2d is None or scene.view3d is not None:
            return
        renderer.enable_gsp_view2d_navigation(scene.view2d)
        self._interactive_view2d_renderers.add(renderer_id)

    def _enable_interactive_view3d(
        self, renderer: DatovizV04ProtocolRenderer, scene: Scene
    ) -> None:
        renderer_id = id(renderer)
        if renderer_id in self._interactive_view3d_renderers:
            return
        if scene.view3d is None or scene.view2d is not None:
            return
        if not self.capabilities.supports_view3d_capability(
            VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY
        ):
            return
        renderer.enable_gsp_view3d_navigation(scene.view3d)
        self._interactive_view3d_renderers.add(renderer_id)

    def _build_renderer(self, scene: Scene) -> DatovizV04ProtocolRenderer:
        renderer = DatovizV04ProtocolRenderer(
            dvz=self._dvz,
            color_scales={item.id: item for item in scene.color_scales},
            texture_resources={item.id: item for item in scene.textures},
            canvas_size=scene.canvas_size,
            view=None if scene.axis_guides else scene.view2d,
            view3d=scene.view3d,
            transform_resources={item.id: item for item in scene.transforms},
        )
        try:
            self._configure_guides(renderer, scene)
            for visual in _canonical_visual_emission_order(scene.visuals):
                _add_visual(renderer, visual)
            for guide in scene.colorbar_guides:
                renderer.add_colorbar_guide(guide)
        except Exception:
            renderer.close()
            raise
        return renderer

    @staticmethod
    def _configure_guides(renderer: DatovizV04ProtocolRenderer, scene: Scene) -> None:
        if not scene.axis_guides:
            return
        if scene.view2d is None:
            raise ValueError("axis guides require Scene.view2d")
        x_guide = next(
            (guide for guide in scene.axis_guides if guide.dimension is AxisDimension.X),
            None,
        )
        y_guide = next(
            (guide for guide in scene.axis_guides if guide.dimension is AxisDimension.Y),
            None,
        )
        if x_guide is None or y_guide is None:
            raise ValueError("Datoviz axis rendering requires both X and Y guides")
        explicit = (
            x_guide.tick_spec.kind is TickSpecKind.EXPLICIT
            or y_guide.tick_spec.kind is TickSpecKind.EXPLICIT
        )
        renderer.configure_view2d_axes(
            scene.view2d,
            x_label=x_guide.label_text,
            y_label=y_guide.label_text,
            grid=x_guide.grid_visible or y_guide.grid_visible,
            backend_auto_ticks=not explicit,
            x_tick_values=x_guide.tick_spec.explicit_values,
            x_tick_labels=x_guide.tick_spec.explicit_labels,
            y_tick_values=y_guide.tick_spec.explicit_values,
            y_tick_labels=y_guide.tick_spec.explicit_labels,
        )


def _add_visual(renderer: DatovizV04ProtocolRenderer, visual: object) -> None:
    if isinstance(visual, PointVisual):
        renderer.add_point_visual(visual)
    elif isinstance(visual, PixelVisual):
        renderer.add_pixel_visual(visual)
    elif isinstance(visual, SphereVisual):
        renderer.add_sphere_visual(visual)
    elif isinstance(visual, VectorVisual):
        renderer.add_vector_visual(visual)
    elif isinstance(visual, PrimitiveVisual):
        renderer.add_primitive_visual(visual)
    elif isinstance(visual, MarkerVisual):
        renderer.add_marker_visual(visual)
    elif isinstance(visual, SegmentVisual):
        renderer.add_segment_visual(visual)
    elif isinstance(visual, PathVisual):
        renderer.add_path_visual(visual)
    elif isinstance(visual, ImageVisual):
        renderer.add_image_visual(visual)
    elif isinstance(visual, TextVisual):
        renderer.add_text_visual(visual)
    elif isinstance(visual, MeshVisual):
        renderer.add_mesh_visual(visual)
    else:
        raise TypeError(f"unsupported protocol visual: {type(visual).__name__}")


def _canonical_visual_emission_order(
    visuals: tuple[object, ...],
) -> tuple[object, ...]:
    """Emit geometry first, then stable overlay text ordered by semantic z-order."""
    geometry = tuple(visual for visual in visuals if not isinstance(visual, TextVisual))
    text = tuple(
        visual
        for _index, visual in sorted(
            (
                (index, visual)
                for index, visual in enumerate(visuals)
                if isinstance(visual, TextVisual)
            ),
            key=lambda item: (item[1].z_order, item[0]),
        )
    )
    return (*geometry, *text)


def _scene_panel_ids(scene: Scene) -> frozenset[str]:
    panel_ids = {panel.id for panel in scene.panels}
    if scene.view2d is not None:
        panel_ids.add(scene.view2d.panel_id)
    if scene.view3d is not None:
        panel_ids.add(scene.view3d.panel_id)
    return frozenset(panel_ids)


def _unsupported_query_result(
    request: QueryRequest, diagnostic: str
) -> QueryResult:
    return QueryResult(
        request_id=request.id,
        status=QueryStatus.UNSUPPORTED,
        hit=False,
        panel_coordinate=request.coordinate,
        diagnostic=diagnostic,
        layout_snapshot_id=request.layout_snapshot_id,
        view_snapshot_id=request.view_snapshot_id,
    )
