"""Panel-scoped native query execution and freshness checks."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from gsp.protocol import (
    GUIDE_QUERY_PAYLOAD_KIND,
    SCALAR_COLOR_QUERY_PAYLOAD_KIND,
    LogicalCoordinateRegion,
    PixelOrigin,
    QueryCoordinateSpace,
    QueryDiagnostic,
    QueryDiagnosticSeverity,
    QueryHitPolicy,
    QueryPayload,
    QueryRequest,
    QueryResult,
    QueryScope,
    QueryStatus,
    View3DDiagnosticCode,
    View3DMeshPickDiagnosticCode,
    View3DMeshTrianglePickPayload,
    View3DMeshTrianglePickRequest,
    classify_logical_coordinate,
    plot_logical_px_to_plot_ndc,
    resolve_view3d_projection_snapshot,
)

from gsp_datoviz.query import (
    DatovizQueryPayload,
    datoviz_query_view3d_ray_context,
    datoviz_v04_query_binding_diagnostics,
    datoviz_v04_query_binding_ready,
    decode_dvz_query_result,
)

from ._state import (
    DVZ_QUERY_HIT_FRONTMOST,
    DVZ_QUERY_PROFILE_UNSUPPORTED,
    DVZ_SCENE_TARGET_FACE,
    DVZ_SCENE_TARGET_ITEM,
    _ScalarVisualData,
)
from .colors import (
    _scalar_payload_for_query_result,
)
from .layout import (
    _datoviz_query_panel_bounds,
    _query_datoviz_panel_frame_guides,
)
from .native_api import (
    _datoviz_query_request_diagnostic,
    _datoviz_request_id,
    _panel_query,
    _query_frame_resolution_ready,
    _scene_poll_query,
    _unsupported_query_result,
    datoviz_v04_panel_frame_guide_query_diagnostics,
    datoviz_v04_panel_frame_guide_query_ready,
    datoviz_v04_panel_frame_snapshot_diagnostics,
)

if TYPE_CHECKING:
    from .protocol_renderer import DatovizV04ProtocolRenderer


def query_panel(
    self: DatovizV04ProtocolRenderer,
    request: QueryRequest,
    *,
    native_target: int = DVZ_SCENE_TARGET_ITEM,
    native_metadata: dict[str, int] | None = None,
) -> QueryResult:
    """Queue and poll one Datoviz panel query for data-scope panel coordinates."""
    self.activate_panel(request.panel_id)
    stale = self._stale_consumed_layout_result(request)
    if stale is not None:
        return stale
    if request.scope in (QueryScope.GUIDES, QueryScope.ALL_RENDERED):
        return self._query_panel_guides(request)

    request_diagnostic = _datoviz_query_request_diagnostic(request)
    if request_diagnostic is not None:
        unsupported = _unsupported_query_result(request, request_diagnostic)
        if unsupported is not None:
            return unsupported
    if not datoviz_v04_query_binding_ready(self.dvz):
        diagnostics = ", ".join(datoviz_v04_query_binding_diagnostics(self.dvz))
        unsupported = _unsupported_query_result(
            request, f"Datoviz query binding is unavailable: {diagnostics}"
        )
        if unsupported is not None:
            return unsupported

    dvz_request = self.dvz.dvz_query_request()
    dvz_request.request_id = _datoviz_request_id(request.id)
    dvz_request.target = native_target
    dvz_request.hit_policy = getattr(self.dvz, "DVZ_QUERY_HIT_FRONTMOST", DVZ_QUERY_HIT_FRONTMOST)
    dvz_request.profile = getattr(
        self.dvz, "DVZ_QUERY_PROFILE_UNSUPPORTED", DVZ_QUERY_PROFILE_UNSUPPORTED
    )

    native_coordinate = self._native_query_coordinate(request.coordinate)
    if native_coordinate is None:
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.MISS,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic="query coordinate is outside the consumed plot viewport",
            layout_snapshot_id=(
                self.consumed_layout_snapshot.snapshot_id
                if self.consumed_layout_snapshot is not None
                else request.layout_snapshot_id
            ),
            view_snapshot_id=request.view_snapshot_id,
        )
    x, y = native_coordinate
    if _panel_query(self.dvz, self.panel, x, y, dvz_request) != 0:
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.FAILED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic="Datoviz panel query enqueue failed",
        )

    if _query_frame_resolution_ready(self.dvz):
        self._ensure_offscreen_view()
        self._render_offscreen_frame()

    raw_result = self.dvz.DvzQueryResult()
    if not _scene_poll_query(self.dvz, self.scene, raw_result):
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.DROPPED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic="Datoviz query produced no resolved result during bounded poll",
        )

    decoded = replace(
        decode_dvz_query_result(raw_result, native_metadata=native_metadata),
        request_id=request.id,
        panel_coordinate=request.coordinate,
        layout_snapshot_id=(
            self.consumed_layout_snapshot.snapshot_id
            if self.consumed_layout_snapshot is not None
            else request.layout_snapshot_id
        ),
    )
    semantic_visual_id = self._semantic_visual_id(int(getattr(raw_result, "visual_id", 0)))
    if decoded.status is QueryStatus.HIT and semantic_visual_id is not None:
        decoded = replace(
            decoded,
            visual_id=semantic_visual_id,
            hits=tuple(replace(hit, visual_id=semantic_visual_id) for hit in decoded.hits),
        )
    return self._decorate_scalar_query_result(decoded, request)


def _semantic_visual_id(self: DatovizV04ProtocolRenderer, native_visual_id: int) -> str | None:
    """Resolve one native Datoviz ID back to the GSP scene visual identity."""
    getter = getattr(self.dvz, "dvz_visual_id", None)
    if not callable(getter):
        return None
    for semantic_id, native_visual in self.visuals.items():
        try:
            if int(getter(native_visual)) == native_visual_id:
                return semantic_id
        except (TypeError, ValueError):
            continue
    return None


def _native_query_coordinate(
    self: DatovizV04ProtocolRenderer, coordinate: tuple[float, float]
) -> tuple[float, float] | None:
    if self.consumed_layout_snapshot is None:
        return (float(coordinate[0]), float(coordinate[1]))
    snapshot = self.consumed_layout_snapshot
    if classify_logical_coordinate(snapshot, coordinate) is not LogicalCoordinateRegion.DATA_PLOT:
        return None
    plot = snapshot.only_panel().plot_rect_px
    x, y = coordinate
    local_y = y - plot.y
    if snapshot.render_target.pixel_origin is PixelOrigin.BOTTOM_LEFT:
        local_y = plot.height - local_y
    return (float(x - plot.x), float(local_y))


def _stale_consumed_layout_result(
    self: DatovizV04ProtocolRenderer, request: QueryRequest
) -> QueryResult | None:
    snapshot = self.consumed_layout_snapshot
    if (
        snapshot is None
        or request.layout_snapshot_id is None
        or request.layout_snapshot_id == snapshot.snapshot_id
    ):
        return None
    return QueryResult(
        request_id=request.id,
        status=QueryStatus.STALE,
        hit=False,
        panel_coordinate=request.coordinate,
        diagnostic="query references a stale consumed layout snapshot",
        layout_snapshot_id=snapshot.snapshot_id,
        view_snapshot_id=request.view_snapshot_id,
    )


def _query_panel_guides(self: DatovizV04ProtocolRenderer, request: QueryRequest) -> QueryResult:
    if request.coordinate_space != QueryCoordinateSpace.PANEL:
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.UNSUPPORTED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic=("Datoviz guide query requires panel logical-pixel coordinates"),
            layout_snapshot_id=request.layout_snapshot_id,
        )
    if request.hit_policy != QueryHitPolicy.FRONTMOST:
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.UNSUPPORTED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic="Datoviz guide query supports frontmost hit policy only",
            layout_snapshot_id=request.layout_snapshot_id,
        )
    unsupported_payloads = tuple(
        kind
        for kind in request.requested_extension_payload_kinds
        if kind != GUIDE_QUERY_PAYLOAD_KIND
    )
    if unsupported_payloads:
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.UNSUPPORTED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic=(
                "Datoviz guide query does not support requested extension "
                f"payloads: {unsupported_payloads}"
            ),
            layout_snapshot_id=request.layout_snapshot_id,
        )
    if not datoviz_v04_panel_frame_guide_query_ready(self.dvz):
        diagnostics = (
            *datoviz_v04_panel_frame_snapshot_diagnostics(self.dvz),
            *datoviz_v04_panel_frame_guide_query_diagnostics(self.dvz),
        )
        prefix = (
            "all-rendered-guides-unsupported"
            if request.scope == QueryScope.ALL_RENDERED
            else "axis-guide-query-unsupported"
        )
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.UNSUPPORTED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic=(
                f"{prefix}: Datoviz panel frame guide hit/readback is "
                "unavailable: " + "; ".join(diagnostics)
            ),
            layout_snapshot_id=request.layout_snapshot_id,
        )
    return _query_datoviz_panel_frame_guides(
        self.dvz,
        self.panel,
        request,
        all_rendered=request.scope == QueryScope.ALL_RENDERED,
    )


def query_view3d_ray_context(
    self: DatovizV04ProtocolRenderer, request: QueryRequest, *, layout_snapshot_id: str
) -> QueryResult:
    """Return a canonical View3D ray-context payload for the current Datoviz panel."""
    self.activate_panel(request.panel_id)
    if self.consumed_layout_snapshot is not None and (
        layout_snapshot_id != self.consumed_layout_snapshot.snapshot_id
        or request.layout_snapshot_id not in (None, self.consumed_layout_snapshot.snapshot_id)
    ):
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.STALE,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic="View3D ray query references a stale consumed layout snapshot",
            layout_snapshot_id=self.consumed_layout_snapshot.snapshot_id,
            view_snapshot_id=request.view_snapshot_id,
        )
    if self.view3d is None:
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.UNSUPPORTED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic=(
                f"{View3DDiagnosticCode.VIEW3D_NOT_SUPPORTED.value}: "
                "Datoviz View3D ray readback requires a renderer View3D"
            ),
            layout_snapshot_id=request.layout_snapshot_id,
            view_snapshot_id=request.view_snapshot_id,
        )
    snapshot = resolve_view3d_projection_snapshot(
        self.view3d,
        layout_snapshot=self.consumed_layout_snapshot,
        layout_snapshot_id=layout_snapshot_id,
    )
    return datoviz_query_view3d_ray_context(
        request,
        self.view3d,
        snapshot,
        panel_bounds=self._query_plot_bounds(),
        layout_snapshot=self.consumed_layout_snapshot,
    )


def _query_plot_bounds(self: DatovizV04ProtocolRenderer) -> tuple[float, float, float, float]:
    if self.consumed_layout_snapshot is not None:
        plot = self.consumed_layout_snapshot.only_panel().plot_rect_px
        return (plot.x, plot.x + plot.width, plot.y, plot.y + plot.height)
    return _datoviz_query_panel_bounds(
        self.resolved_canvas.framebuffer_width,
        self.resolved_canvas.framebuffer_height,
        self.panel_bounds,
    )


def query_view3d_mesh_triangle_pick(
    self: DatovizV04ProtocolRenderer,
    request: View3DMeshTrianglePickRequest,
    *,
    layout_snapshot_id: str,
) -> QueryResult:
    """Use Datoviz public FACE queries for the bounded single-mesh S044 subset."""
    result_id = f"query:{request.view_id}:mesh-pick"

    def finish(
        status: QueryStatus,
        *,
        code: View3DMeshPickDiagnosticCode | None = None,
        message: str | None = None,
        **payload_fields: Any,
    ) -> QueryResult:
        diagnostics = (
            ()
            if code is None
            else (
                QueryDiagnostic(
                    code=code,
                    severity=QueryDiagnosticSeverity.ERROR,
                    message=message,
                ),
            )
        )
        payload = View3DMeshTrianglePickPayload(
            status=status,
            hit=status is QueryStatus.HIT,
            view_id=request.view_id,
            panel_id=request.panel_id or (self.view3d.panel_id if self.view3d else None),
            panel_xy=request.panel_xy,
            diagnostics=diagnostics,
            **payload_fields,
        )
        return QueryResult(
            request_id=result_id,
            status=status,
            hit=status is QueryStatus.HIT,
            visual_id=payload.visual_id,
            panel_coordinate=request.panel_xy,
            extension_payload_kind=payload.kind,
            extension_payload=payload,
            diagnostic=None if code is None else code.value,
            layout_snapshot_id=payload.layout_snapshot_id or layout_snapshot_id,
            view_snapshot_id=payload.view_projection_snapshot_id,
        )

    if self.consumed_layout_snapshot is not None and (
        layout_snapshot_id != self.consumed_layout_snapshot.snapshot_id
        or request.expected_layout_snapshot_id
        not in (None, self.consumed_layout_snapshot.snapshot_id)
    ):
        return finish(
            QueryStatus.STALE,
            code=View3DMeshPickDiagnosticCode.STALE_LAYOUT_SNAPSHOT,
            message="mesh pick references a stale consumed layout snapshot",
            layout_snapshot_id=self.consumed_layout_snapshot.snapshot_id,
        )
    if self.view3d is None:
        return finish(
            QueryStatus.UNSUPPORTED,
            code=View3DMeshPickDiagnosticCode.UNSUPPORTED_BACKEND,
            message="Datoviz mesh picking requires a retained View3D",
        )
    if request.view_id != self.view3d.id:
        return finish(
            QueryStatus.INVALID,
            code=View3DMeshPickDiagnosticCode.INVALID_VIEW_ID,
            message="mesh pick view_id does not match the renderer View3D",
        )
    if request.panel_id not in (None, self.view3d.panel_id):
        return finish(
            QueryStatus.INVALID,
            code=View3DMeshPickDiagnosticCode.INVALID_PANEL_ID,
            message="mesh pick panel_id does not match the renderer View3D",
        )
    projection = resolve_view3d_projection_snapshot(
        self.view3d,
        layout_snapshot=self.consumed_layout_snapshot,
        layout_snapshot_id=layout_snapshot_id,
    )
    common: dict[str, Any] = dict(
        layout_snapshot_id=layout_snapshot_id,
        view_revision=self.view3d.revision,
        view_projection_snapshot_id=projection.view_projection_snapshot_id,
        depth_mode=self.view3d.depth_mode.value,
    )
    if request.expected_view_revision not in (None, self.view3d.revision):
        return finish(
            QueryStatus.STALE,
            code=View3DMeshPickDiagnosticCode.STALE_VIEW_REVISION,
            message="mesh pick references a stale View3D revision",
            **common,
        )
    if request.expected_view_projection_snapshot_id not in (
        None,
        projection.view_projection_snapshot_id,
    ):
        return finish(
            QueryStatus.STALE,
            code=View3DMeshPickDiagnosticCode.STALE_VIEW_PROJECTION_SNAPSHOT,
            message="mesh pick references a stale View3D projection snapshot",
            **common,
        )
    if len(self.visuals) != 1 or len(self.retained_view3d_meshes) != 1:
        return finish(
            QueryStatus.UNSUPPORTED,
            code=View3DMeshPickDiagnosticCode.UNSUPPORTED_SCENE_OCCLUDER,
            message=(
                "Datoviz FACE queries are strict only when exactly one retained "
                "DATA-space MeshVisual is the scene's sole visual"
            ),
            **common,
        )
    if self.consumed_layout_snapshot is not None:
        region = classify_logical_coordinate(self.consumed_layout_snapshot, request.panel_xy)
        if region is not LogicalCoordinateRegion.DATA_PLOT:
            return finish(
                QueryStatus.INVALID,
                code=View3DMeshPickDiagnosticCode.INVALID_OUTSIDE_PANEL,
                message="mesh pick coordinate is outside the consumed plot viewport",
                **common,
            )
        plot_ndc = plot_logical_px_to_plot_ndc(self.consumed_layout_snapshot, request.panel_xy)
    else:
        x0, x1, y0, y1 = self._query_plot_bounds()
        x, y = request.panel_xy
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            return finish(
                QueryStatus.INVALID,
                code=View3DMeshPickDiagnosticCode.INVALID_OUTSIDE_PANEL,
                message="mesh pick coordinate is outside the renderer plot viewport",
                **common,
            )
        plot_ndc = (
            -1.0 + 2.0 * (x - x0) / (x1 - x0),
            1.0 - 2.0 * (y - y0) / (y1 - y0),
        )

    native_metadata: dict[str, int] = {}
    native = query_panel(
        self,
        QueryRequest(
            id=result_id,
            panel_id=self.view3d.panel_id,
            coordinate=request.panel_xy,
            coordinate_space=QueryCoordinateSpace.PANEL,
            requested_payload=(QueryPayload.IDENTITY,),
            layout_snapshot_id=layout_snapshot_id,
            view_snapshot_id=projection.view_projection_snapshot_id,
        ),
        native_target=getattr(self.dvz, "DVZ_SCENE_TARGET_FACE", DVZ_SCENE_TARGET_FACE),
        native_metadata=native_metadata,
    )
    if native.status not in (QueryStatus.HIT, QueryStatus.MISS) or (
        native.status is QueryStatus.HIT
        and not isinstance(native.extension_payload, DatovizQueryPayload)
    ):
        return finish(
            QueryStatus.UNSUPPORTED,
            code=View3DMeshPickDiagnosticCode.UNSUPPORTED_NATIVE_STATE_ONLY,
            message="Datoviz FACE query did not return decodable public face identity",
            plot_ndc_xy=plot_ndc,
            **common,
        )
    freshness_serial = native_metadata.get("freshness_serial", 0)
    if freshness_serial <= 0:
        return finish(
            QueryStatus.UNSUPPORTED,
            code=View3DMeshPickDiagnosticCode.UNSUPPORTED_NATIVE_STATE_ONLY,
            message="Datoviz FACE query did not return a positive native freshness serial",
            plot_ndc_xy=plot_ndc,
            **common,
        )
    pick_scene_snapshot_id = f"pick-scene:datoviz-{freshness_serial}"
    if request.expected_pick_scene_snapshot_id not in (None, pick_scene_snapshot_id):
        return finish(
            QueryStatus.STALE,
            code=View3DMeshPickDiagnosticCode.STALE_PICK_SCENE_SNAPSHOT,
            message="mesh pick references a stale Datoviz scene snapshot",
            pick_scene_snapshot_id=pick_scene_snapshot_id,
            plot_ndc_xy=plot_ndc,
            **common,
        )
    if native.status is QueryStatus.MISS:
        return finish(
            QueryStatus.MISS,
            plot_ndc_xy=plot_ndc,
            pick_scene_snapshot_id=pick_scene_snapshot_id,
            **common,
        )
    native_payload = native.extension_payload
    assert isinstance(native_payload, DatovizQueryPayload)
    return finish(
        QueryStatus.HIT,
        plot_ndc_xy=plot_ndc,
        pick_scene_snapshot_id=pick_scene_snapshot_id,
        visual_id=native.visual_id,
        visual_type="MeshVisual",
        primitive_kind="triangle",
        primitive_index=native_payload.face_id,
        **common,
    )


def _decorate_scalar_query_result(
    self: DatovizV04ProtocolRenderer, result: QueryResult, request: QueryRequest
) -> QueryResult:
    """Attach exact S026 scalar payloads from retained protocol scene data."""
    if result.status != QueryStatus.HIT:
        return result

    wants_scalar_payload = (
        SCALAR_COLOR_QUERY_PAYLOAD_KIND in request.requested_extension_payload_kinds
    )
    metadata = self._scalar_metadata_for_query_result(result)
    if metadata is None:
        if wants_scalar_payload:
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.UNSUPPORTED,
                hit=False,
                panel_coordinate=request.coordinate,
                diagnostic=(
                    "scalar_query_source_unavailable: Datoviz query hit could not "
                    "be matched to a retained scalar-colored point or image visual"
                ),
            )
        return result

    payload = _scalar_payload_for_query_result(metadata, result)
    if payload is None:
        if wants_scalar_payload:
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.UNSUPPORTED,
                hit=False,
                panel_coordinate=request.coordinate,
                diagnostic=(
                    "scalar_query_source_unavailable: Datoviz query hit did not "
                    "include a usable point item id or image texel id"
                ),
            )
        return result

    return replace(
        result,
        displayed_rgba=payload.displayed_rgba,
        value=payload.source_value,
        extension_payload_kind=SCALAR_COLOR_QUERY_PAYLOAD_KIND,
        extension_payload=payload,
    )


def _scalar_metadata_for_query_result(
    self: DatovizV04ProtocolRenderer, result: QueryResult
) -> _ScalarVisualData | None:
    if not self.scalar_visuals:
        return None
    if result.visual_id in self.scalar_visuals:
        return self.scalar_visuals[result.visual_id]

    candidates = [
        metadata
        for metadata in self.scalar_visuals.values()
        if metadata.visual_family == result.visual_family
    ]
    if len(candidates) == 1:
        return candidates[0]
    return None
