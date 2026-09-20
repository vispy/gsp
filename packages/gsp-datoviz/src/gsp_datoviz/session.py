"""Backend-neutral session wrapper for the Datoviz v0.4 renderer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsp import Scene
from gsp.backends import SessionRequest
from gsp.protocol import (
    AdaptationOutcome,
    AxisDimension,
    CanvasSize,
    ClipScope,
    GuideQueryPolicy,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    PathVisual,
    PanelTextRole,
    PixelVisual,
    PointVisual,
    PrimitiveVisual,
    QueryRequest,
    QueryResult,
    QueryStatus,
    ResolvedLayoutSnapshot,
    SegmentVisual,
    SphereVisual,
    TextVisual,
    TickSpecKind,
    VIEW3D_QUERY_PAYLOAD_KIND,
    VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY,
    VectorVisual,
    View2D,
    View3D,
    resolve_panel_layout_intent,
)

from .capabilities import datoviz_v04_capability_snapshot
from .protocol_renderer import DatovizV04ProtocolRenderer, import_datoviz_v04


_DATOVIZ_ITEM_QUERY_VISUAL_TYPES = (
    PointVisual,
    PixelVisual,
    MarkerVisual,
    SphereVisual,
    VectorVisual,
    SegmentVisual,
    PathVisual,
    PrimitiveVisual,
    MeshVisual,
    ImageVisual,
)


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
        layout_snapshot: ResolvedLayoutSnapshot | None = None,
        **kwargs: Any,
    ) -> DatovizV04ProtocolRenderer:
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
                "Datoviz adapter does not support attachment clip scopes "
                f"{sorted(scope.value for scope in unsupported_clip_scopes)!r}"
            )
        _validate_consumed_layout_scene(scene, layout_snapshot)
        if layout_snapshot is not None and scene.panel_text_guides:
            self._diagnostics.append("panel_text_title_unsupported_no_public_renderer_path")
        if kwargs:
            raise TypeError(f"unsupported Datoviz render options: {sorted(kwargs)!r}")
        renderer = (
            self._build_renderer(scene)
            if layout_snapshot is None
            else self._build_renderer(scene, layout_snapshot=layout_snapshot)
        )
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
        layout_snapshot: ResolvedLayoutSnapshot | None = None,
        **kwargs: Any,
    ) -> DatovizV04ProtocolRenderer:
        if kwargs:
            raise TypeError(f"unsupported Datoviz display options: {sorted(kwargs)!r}")
        if frame_count < 1:
            raise ValueError("frame_count must be positive")
        renderer = self.render(scene, layout_snapshot=layout_snapshot)
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

        panel_view = scene.primary_view_for_panel(request.panel_id)
        if isinstance(panel_view, View3D) and (
            VIEW3D_QUERY_PAYLOAD_KIND in request.requested_extension_payload_kinds
        ):
            layout_getter = getattr(renderer, "authoritative_layout_snapshot", None)
            effective_layout = layout_getter() if layout_getter is not None else None
            return renderer.query_view3d_ray_context(
                request,
                layout_snapshot_id=(
                    effective_layout.snapshot_id
                    if effective_layout is not None
                    else "layout:datoviz-session"
                ),
            )

        if not any(
            isinstance(visual, _DATOVIZ_ITEM_QUERY_VISUAL_TYPES)
            for visual in scene.visuals_for_panel(request.panel_id)
        ):
            return _unsupported_query_result(
                request,
                "Datoviz scene contains no visual family with qualified native item queries",
            )

        decision = self.capabilities.adapt_query_request(request)
        if decision.outcome is not AdaptationOutcome.ACCEPT:
            return _unsupported_query_result(
                request,
                decision.diagnostic or "Datoviz query request is unsupported",
            )
        return renderer.query_panel(request)

    def _query_target(self, scene_id: str | None) -> tuple[Scene, DatovizV04ProtocolRenderer]:
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
        if not scene.views2d:
            return
        for index, view in enumerate(scene.views2d):
            if len(scene.views2d) == 1:
                renderer.enable_gsp_view2d_navigation(view)
            else:
                renderer.enable_gsp_view2d_navigation(
                    view, controller_id=f"nav:datoviz-live:{index}"
                )
        self._interactive_view2d_renderers.add(renderer_id)

    def _enable_interactive_view3d(
        self, renderer: DatovizV04ProtocolRenderer, scene: Scene
    ) -> None:
        renderer_id = id(renderer)
        if renderer_id in self._interactive_view3d_renderers:
            return
        if not scene.views3d:
            return
        if not self.capabilities.supports_view3d_capability(
            VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY
        ):
            return
        for index, view in enumerate(scene.views3d):
            if len(scene.views3d) == 1:
                renderer.enable_gsp_view3d_navigation(view)
            else:
                renderer.enable_gsp_view3d_navigation(
                    view, controller_id=f"nav:datoviz-live-3d:{index}"
                )
        self._interactive_view3d_renderers.add(renderer_id)

    def _build_renderer(
        self,
        scene: Scene,
        *,
        layout_snapshot: ResolvedLayoutSnapshot | None = None,
    ) -> DatovizV04ProtocolRenderer:
        first_panel = scene.panels[0]
        first_view = scene.primary_view_for_panel(first_panel.id)
        first_layout = (
            _layout_snapshot_for_panel(layout_snapshot, first_panel.id)
            if layout_snapshot is not None
            else None
        )
        panel_bounds = _normalized_plot_bounds(first_layout) if first_layout is not None else None
        renderer = DatovizV04ProtocolRenderer(
            dvz=self._dvz,
            color_scales={item.id: item for item in scene.color_scales},
            texture_resources={item.id: item for item in scene.textures},
            canvas_size=(
                _canvas_size_for_consumed_layout(layout_snapshot)
                if layout_snapshot is not None
                else scene.canvas_size
            ),
            view=first_view
            if isinstance(first_view, View2D) and not _panel_has_axes(scene, first_panel.id)
            else None,
            view3d=first_view if isinstance(first_view, View3D) else None,
            transform_resources={item.id: item for item in scene.transforms},
            panel_bounds=panel_bounds,
            panel_id=first_panel.id,
            panel_layout=scene.panel_layout,
            consumed_layout_snapshot=first_layout,
        )
        try:
            for index, panel in enumerate(scene.panels):
                view = scene.primary_view_for_panel(panel.id)
                panel_layout = (
                    _layout_snapshot_for_panel(layout_snapshot, panel.id)
                    if layout_snapshot is not None
                    else None
                )
                if index:
                    renderer.add_retained_panel(
                        panel_id=panel.id,
                        view=(
                            view
                            if isinstance(view, View2D) and not _panel_has_axes(scene, panel.id)
                            else None
                        ),
                        view3d=view if isinstance(view, View3D) else None,
                        panel_layout=scene.panel_layout,
                        consumed_layout_snapshot=panel_layout,
                    )
                self._configure_guides(renderer, scene, panel.id)
                for visual in _canonical_visual_emission_order(scene.visuals_for_panel(panel.id)):
                    _add_visual(renderer, visual)
                for guide in scene.colorbar_guides:
                    if guide.panel_id == panel.id:
                        renderer.add_colorbar_guide(guide)
            set_layout = getattr(renderer, "set_authoritative_scene_layout_snapshot", None)
            if set_layout is not None:
                set_layout(layout_snapshot)
            activate_panel = getattr(renderer, "activate_panel", None)
            if activate_panel is not None:
                activate_panel(first_panel.id)
        except Exception:
            renderer.close()
            raise
        return renderer

    @staticmethod
    def _configure_guides(
        renderer: DatovizV04ProtocolRenderer, scene: Scene, panel_id: str
    ) -> None:
        view = scene.primary_view_for_panel(panel_id)
        guides = (
            tuple(guide for guide in scene.axis_guides if guide.view_id == view.id)
            if isinstance(view, View2D)
            else ()
        )
        if not guides:
            return
        if not isinstance(view, View2D):
            raise ValueError("axis guides require a View2D on their panel")
        view2d = view
        x_guide = next(
            (guide for guide in guides if guide.dimension is AxisDimension.X),
            None,
        )
        y_guide = next(
            (guide for guide in guides if guide.dimension is AxisDimension.Y),
            None,
        )
        if x_guide is None or y_guide is None:
            raise ValueError("Datoviz axis rendering requires both X and Y guides")
        explicit = (
            x_guide.tick_spec.kind is TickSpecKind.EXPLICIT
            or y_guide.tick_spec.kind is TickSpecKind.EXPLICIT
        )
        renderer.configure_view2d_axes(
            view2d,
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
    return frozenset(panel.id for panel in scene.panels)


def _panel_has_axes(scene: Scene, panel_id: str) -> bool:
    view = scene.primary_view_for_panel(panel_id)
    return isinstance(view, View2D) and any(guide.view_id == view.id for guide in scene.axis_guides)


def _layout_snapshot_for_panel(
    snapshot: ResolvedLayoutSnapshot, panel_id: str
) -> ResolvedLayoutSnapshot:
    try:
        panel = snapshot.panel(panel_id)
    except ValueError:
        if len(snapshot.panels) != 1:
            raise
        panel = snapshot.only_panel()
    return ResolvedLayoutSnapshot(
        snapshot_id=snapshot.snapshot_id,
        render_target=snapshot.render_target,
        panels=(panel,),
    )


def _unsupported_query_result(request: QueryRequest, diagnostic: str) -> QueryResult:
    return QueryResult(
        request_id=request.id,
        status=QueryStatus.UNSUPPORTED,
        hit=False,
        panel_coordinate=request.coordinate,
        diagnostic=diagnostic,
        layout_snapshot_id=request.layout_snapshot_id,
        view_snapshot_id=request.view_snapshot_id,
    )


def _normalized_plot_bounds(
    snapshot: ResolvedLayoutSnapshot,
) -> tuple[float, float, float, float]:
    target = snapshot.render_target
    plot = snapshot.only_panel().plot_rect_px
    y = (
        plot.y / target.logical_height_px
        if target.pixel_origin.value == "top-left"
        else 1.0 - (plot.y + plot.height) / target.logical_height_px
    )
    return (
        plot.x / target.logical_width_px,
        y,
        plot.width / target.logical_width_px,
        plot.height / target.logical_height_px,
    )


def _validate_consumed_layout_scene(
    scene: Scene, layout_snapshot: ResolvedLayoutSnapshot | None
) -> None:
    if layout_snapshot is None:
        return
    if not isinstance(layout_snapshot, ResolvedLayoutSnapshot):
        raise TypeError("layout_snapshot must be a ResolvedLayoutSnapshot")
    scene_panel_ids = _scene_panel_ids(scene)
    snapshot_panel_ids = frozenset(panel.panel_id for panel in layout_snapshot.panels)
    singleton_compatibility = len(scene.panels) == len(layout_snapshot.panels) == 1
    if snapshot_panel_ids != scene_panel_ids and not singleton_compatibility:
        raise ValueError("layout_snapshot panels do not match the scene panels")
    for panel in scene.panels:
        active_view = scene.primary_view_for_panel(panel.id)
        resolved_panel = (
            layout_snapshot.only_panel()
            if singleton_compatibility
            else layout_snapshot.panel(panel.id)
        )
        if active_view is not None:
            if resolved_panel.view_id != active_view.id:
                raise ValueError("layout_snapshot view_id does not match the active scene view")
        elif resolved_panel.view_id is not None:
            raise ValueError("viewless scene panel cannot consume a view-bound layout_snapshot")
    expected = resolve_panel_layout_intent(scene.panel_layout, layout_snapshot.render_target)
    expected_by_id = {panel.panel_id: panel for panel in expected}
    if any(
        expected_by_id[panel.id].panel_rect_px
        != (
            layout_snapshot.only_panel().panel_rect_px
            if singleton_compatibility
            else layout_snapshot.panel(panel.id).panel_rect_px
        )
        for panel in scene.panels
    ):
        raise ValueError("layout_snapshot panel_rect_px does not match the scene panel allocation")
    _validate_scene_canvas_target(scene.canvas_size, layout_snapshot)
    if any(
        guide.role is not PanelTextRole.TITLE
        or guide.query_policy is not GuideQueryPolicy.NON_QUERYABLE
        for guide in scene.panel_text_guides
    ):
        raise ValueError("consumed Datoviz layout may omit only non-queryable title guides")
    if scene.axis_guides or scene.colorbar_guides:
        raise ValueError(
            "consumed Datoviz layout does not prove native guide geometry; "
            "axis and colorbar guide geometry remains unproven"
        )


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


def _canvas_size_for_consumed_layout(snapshot: ResolvedLayoutSnapshot) -> CanvasSize:
    target = snapshot.render_target
    return CanvasSize.reference_px(
        target.logical_width_px,
        target.logical_height_px,
        reference_dpi=target.dpi or 96.0,
    ).with_requested_device_scale(target.device_scale)
