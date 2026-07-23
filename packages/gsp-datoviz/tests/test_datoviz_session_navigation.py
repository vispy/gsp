from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import gsp
from gsp.protocol import (
    AdaptationDecision,
    AdaptationOutcome,
    Camera3D,
    CoordinateSpace,
    ImageVisual,
    MarkerShape,
    MarkerVisual,
    MeshVisual,
    OrthographicProjection3D,
    PathVisual,
    PixelVisual,
    PointVisual,
    PrimitiveTopology,
    PrimitiveVisual,
    QueryCoordinateSpace,
    QueryPayload,
    QueryRequest,
    QueryResult,
    QueryScope,
    QueryStatus,
    SegmentVisual,
    SphereVisual,
    TextVisual,
    VectorVisual,
    VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY,
    VIEW3D_QUERY_PAYLOAD_KIND,
    View2D,
    View3D,
)
from gsp_datoviz.session import DatovizSession
import gsp_datoviz.session as session_module


class _FakeRenderer:
    def __init__(self, view: View2D | View3D) -> None:
        self.view = view
        self.enable_calls: list[View2D | None] = []
        self.enable_view3d_calls: list[View3D | None] = []
        self.show_calls: list[int] = []
        self.closed = False
        self.query_calls: list[QueryRequest] = []
        self.ray_query_calls: list[tuple[QueryRequest, str]] = []

    def enable_gsp_view2d_navigation(self, view: View2D | None = None) -> object:
        self.enable_calls.append(view)
        return object()

    def enable_gsp_view3d_navigation(self, view: View3D | None = None) -> object:
        self.enable_view3d_calls.append(view)
        return object()

    def show(self, *, frame_count: int) -> None:
        self.show_calls.append(frame_count)

    def close(self) -> None:
        self.closed = True

    def query_panel(self, request: QueryRequest) -> QueryResult:
        self.query_calls.append(request)
        if request.coordinate != (20.0, 20.0):
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.MISS,
                hit=False,
                panel_coordinate=request.coordinate,
            )
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.HIT,
            hit=True,
            panel_coordinate=request.coordinate,
            visual_id="visual:point",
            item_id=0,
            data_coordinate=(0.0, 0.0),
            displayed_rgba=(1.0, 0.0, 0.0, 1.0),
        )

    def query_view3d_ray_context(
        self, request: QueryRequest, *, layout_snapshot_id: str
    ) -> QueryResult:
        self.ray_query_calls.append((request, layout_snapshot_id))
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.MISS,
            hit=False,
            panel_coordinate=request.coordinate,
        )


class _FakeCapabilities:
    def __init__(
        self, *, live_view3d: bool = False, query_outcome: AdaptationOutcome = AdaptationOutcome.ACCEPT
    ) -> None:
        self.live_view3d = live_view3d
        self.query_outcome = query_outcome

    def supports_view3d_capability(self, capability: str) -> bool:
        return (
            capability == VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY
            and self.live_view3d
        )

    def adapt_query_request(self, request: QueryRequest) -> AdaptationDecision:
        return AdaptationDecision(
            self.query_outcome,
            diagnostic=(
                "runtime query capability rejected request"
                if self.query_outcome is not AdaptationOutcome.ACCEPT
                else None
            ),
        )


def _session(
    renderer: _FakeRenderer,
    *,
    live_view3d: bool = False,
    query_outcome: AdaptationOutcome = AdaptationOutcome.ACCEPT,
) -> DatovizSession:
    session = object.__new__(DatovizSession)
    session.request = None  # type: ignore[assignment]
    session._dvz = object()  # type: ignore[assignment]
    session.capabilities = _FakeCapabilities(  # type: ignore[assignment]
        live_view3d=live_view3d, query_outcome=query_outcome
    )
    session._diagnostics = []
    session._renderers = []
    session._renderer_scenes = {}
    session._scene_renderers = {}
    session._latest_scene_id = None
    session._interactive_view2d_renderers = set()
    session._interactive_view3d_renderers = set()
    session._closed = False
    session._build_renderer = lambda scene: renderer  # type: ignore[assignment,method-assign,return-value]
    return session


def _scene() -> gsp.Scene:
    return gsp.Scene(
        id="scene:live-view2d",
        visuals=(
            PointVisual(
                id="visual:point",
                positions=np.asarray([[0.0, 0.0]], dtype=np.float32),
                colors=np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
            ),
        ),
        view2d=View2D(
            id="view:main",
            panel_id="panel:main",
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        ),
    )


def _mesh3d_scene() -> gsp.Scene:
    return gsp.Scene(
        id="scene:mesh3d",
        visuals=(
            MeshVisual(
                id="visual:mesh3d",
                positions=np.asarray(
                    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
                color=np.asarray([70, 130, 220, 255], dtype=np.uint8),
                coordinate_space=CoordinateSpace.DATA,
            ),
        ),
        view3d=View3D(
            id="view:main",
            panel_id="panel:main",
            camera=Camera3D(
                eye=(3.0, 3.0, 3.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
            ),
            projection=OrthographicProjection3D(),
        ),
    )


def test_interactive_2d_display_enables_canonical_navigation_exactly_once() -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    session = _session(renderer)

    assert session.display(scene, block=False) is renderer  # type: ignore[comparison-overlap]
    session.run()

    assert renderer.enable_calls == [scene.view2d]
    assert renderer.show_calls == [0]

    session.close()
    assert renderer.closed


def test_run_after_noninteractive_render_enables_navigation_once() -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    session = _session(renderer)

    session.render(scene)
    assert renderer.enable_calls == []

    session.run()

    assert renderer.enable_calls == [scene.view2d]


def test_offscreen_render_does_not_enable_interactive_navigation(tmp_path: Any) -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    renderer.capture_png_bytes = lambda: b"png"  # type: ignore[attr-defined]
    session = _session(renderer)

    session.render(scene, target=tmp_path / "frame.png")

    assert renderer.enable_calls == []


def test_offscreen_render_accepts_static_view3d_mesh_scene(tmp_path: Any) -> None:
    scene = _mesh3d_scene()
    assert scene.view3d is not None
    renderer = _FakeRenderer(scene.view3d)
    renderer.capture_png_bytes = lambda: b"png"  # type: ignore[attr-defined]
    session = _session(renderer, live_view3d=True)
    target = tmp_path / "mesh3d.png"

    session.render(scene, target=target)

    assert target.read_bytes() == b"png"
    assert renderer.enable_calls == []
    assert renderer.enable_view3d_calls == []


def test_interactive_view3d_is_enabled_only_from_advertised_session_capability() -> None:
    scene = _mesh3d_scene()
    assert scene.view3d is not None

    static_renderer = _FakeRenderer(scene.view3d)
    static_session = _session(static_renderer)
    static_session.display(scene, block=False)
    static_session.run()
    assert static_renderer.enable_view3d_calls == []

    live_renderer = _FakeRenderer(scene.view3d)
    live_session = _session(live_renderer, live_view3d=True)
    live_session.display(scene, block=False)
    live_session.run()

    assert live_renderer.enable_view3d_calls == [scene.view3d]
    assert live_renderer.show_calls == [0]


def test_scene_emits_geometry_then_stably_z_ordered_overlay_text(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, str]] = []

    class RecordingRenderer:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def add_point_visual(self, visual: PointVisual) -> None:
            calls.append(("point", visual.id))

        def add_text_visual(self, visual: TextVisual) -> None:
            calls.append(("text", visual.id))

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        session_module, "DatovizV04ProtocolRenderer", RecordingRenderer
    )
    session = object.__new__(DatovizSession)
    session._dvz = object()  # type: ignore[assignment]
    scene = gsp.Scene(
        id="scene:text-order",
        visuals=(
            TextVisual(
                id="text:z5",
                texts=("z5",),
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                coordinate_space=CoordinateSpace.NDC,
                z_order=5,
            ),
            PointVisual(
                id="point:geometry",
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                colors=np.array([[255, 255, 255, 255]], dtype=np.uint8),
                coordinate_space=CoordinateSpace.NDC,
            ),
            TextVisual(
                id="text:z1-first",
                texts=("first",),
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                coordinate_space=CoordinateSpace.NDC,
                z_order=1,
            ),
            TextVisual(
                id="text:z1-second",
                texts=("second",),
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                coordinate_space=CoordinateSpace.NDC,
                z_order=1,
            ),
        ),
    )

    session._build_renderer(scene)

    assert calls == [
        ("point", "point:geometry"),
        ("text", "text:z1-first"),
        ("text", "text:z1-second"),
        ("text", "text:z5"),
    ]


def test_public_datoviz_query_routes_to_live_renderer_and_checks_lifecycle() -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    session = _session(renderer)
    request = QueryRequest(
        id="query:point",
        panel_id="panel:main",
        coordinate=(20.0, 20.0),
        coordinate_space=QueryCoordinateSpace.PANEL,
        requested_payload=(QueryPayload.IDENTITY,),
    )

    with np.testing.assert_raises_regex(RuntimeError, "requires a rendered scene"):
        session.query(request)
    session.render(scene)
    result = session.query(request, scene_id=scene.id)

    assert result.status is QueryStatus.HIT
    assert result.data_coordinate == (0.0, 0.0)
    assert result.displayed_rgba == (1.0, 0.0, 0.0, 1.0)
    miss = session.query(
        QueryRequest(
            id="query:miss",
            panel_id="panel:main",
            coordinate=(0.0, 0.0),
            requested_payload=(QueryPayload.IDENTITY,),
        )
    )
    assert miss.status is QueryStatus.MISS
    assert renderer.query_calls[0] == request
    assert len(renderer.query_calls) == 2
    with np.testing.assert_raises_regex(RuntimeError, "has not been rendered"):
        session.query(request, scene_id="scene:unknown")
    session.close()
    with np.testing.assert_raises_regex(RuntimeError, "session is closed"):
        session.query(request)


def test_public_datoviz_query_routes_proven_view3d_ray_path() -> None:
    scene = _mesh3d_scene()
    assert scene.view3d is not None
    renderer = _FakeRenderer(scene.view3d)
    session = _session(renderer)
    request = QueryRequest(
        id="query:ray",
        panel_id="panel:main",
        coordinate=(20.0, 20.0),
        coordinate_space=QueryCoordinateSpace.PANEL,
        requested_payload=(QueryPayload.IDENTITY,),
        requested_extension_payload_kinds=(VIEW3D_QUERY_PAYLOAD_KIND,),
    )

    session.render(scene)
    result = session.query(request)

    assert result.status is QueryStatus.MISS
    assert renderer.ray_query_calls == [(request, "layout:datoviz-session")]


def test_public_datoviz_query_targets_latest_explicit_and_replaced_scene_render() -> None:
    first = _scene()
    second = gsp.Scene(
        id="scene:second",
        visuals=first.visuals,
        view2d=first.view2d,
    )
    first_renderer = _FakeRenderer(first.view2d)  # type: ignore[arg-type]
    second_renderer = _FakeRenderer(second.view2d)  # type: ignore[arg-type]
    replacement_renderer = _FakeRenderer(first.view2d)  # type: ignore[arg-type]
    renderers = iter((first_renderer, second_renderer, replacement_renderer))
    session = _session(first_renderer)
    session._build_renderer = lambda scene: next(renderers)  # type: ignore[assignment,method-assign,return-value]
    request = QueryRequest(
        id="query:target",
        panel_id="panel:main",
        coordinate=(20.0, 20.0),
        requested_payload=(QueryPayload.IDENTITY,),
    )

    session.render(first)
    session.render(second)
    session.query(request)
    session.query(request, scene_id=first.id)
    session.render(first)
    session.query(request, scene_id=first.id)

    assert second_renderer.query_calls == [request]
    assert first_renderer.query_calls == [request]
    assert replacement_renderer.query_calls == [request]


def _unproven_visuals() -> tuple[Any, ...]:
    rgba = np.asarray([[255, 0, 0, 255]], dtype=np.uint8)
    point = np.asarray([[0.0, 0.0]], dtype=np.float32)
    return (
        PixelVisual("visual:pixel", point, rgba, coordinate_space=CoordinateSpace.NDC),
        SphereVisual(
            "visual:sphere",
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
            1.0,
            rgba,
        ),
        VectorVisual(
            "visual:vector",
            point,
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            rgba,
            coordinate_space=CoordinateSpace.NDC,
        ),
        PrimitiveVisual(
            "visual:primitive",
            PrimitiveTopology.POINT_LIST,
            point,
            rgba,
            coordinate_space=CoordinateSpace.NDC,
        ),
        MarkerVisual(
            "visual:marker",
            point,
            MarkerShape.DISC,
            fill_colors=rgba,
        ),
        SegmentVisual(
            "visual:segment",
            point,
            np.asarray([[1.0, 1.0]], dtype=np.float32),
            rgba,
            1.0,
        ),
        PathVisual(
            "visual:path",
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            (2,),
            rgba,
            1.0,
        ),
        ImageVisual(
            "visual:image",
            np.zeros((2, 2, 4), dtype=np.uint8),
            (-1.0, 1.0, -1.0, 1.0),
        ),
        TextVisual(
            "visual:text",
            ("text",),
            point,
            CoordinateSpace.NDC,
        ),
        MeshVisual(
            "visual:mesh",
            np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.asarray([[0, 1, 2]], dtype=np.uint32),
            CoordinateSpace.NDC,
            color=np.asarray([255, 0, 0, 255], dtype=np.uint8),
        ),
    )


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("visual", _unproven_visuals(), ids=lambda visual: type(visual).__name__)
def test_public_datoviz_query_rejects_every_unproven_family_and_mixed_scene(
    visual: Any, mixed: bool
) -> None:
    base = _scene()
    visuals = ((base.visuals[0], visual) if mixed else (visual,))
    view3d = _mesh3d_scene().view3d if isinstance(visual, SphereVisual) else None
    view2d = None if view3d is not None else base.view2d
    scene = gsp.Scene(
        id=f"scene:unsupported:{type(visual).__name__.lower()}:{int(mixed)}",
        visuals=visuals,
        view2d=view2d,
        view3d=view3d,
    )
    renderer = _FakeRenderer(view3d or view2d)  # type: ignore[arg-type]
    session = _session(renderer)
    session.render(scene)

    result = session.query(
        QueryRequest(
            id="query:unsupported",
            panel_id="panel:main",
            coordinate=(20.0, 20.0),
            requested_payload=(QueryPayload.IDENTITY,),
        )
    )

    assert result.status is QueryStatus.UNSUPPORTED
    assert result.diagnostic is not None
    assert type(visual).__name__ in result.diagnostic
    assert renderer.query_calls == []


def test_public_datoviz_query_returns_structured_capability_rejection() -> None:
    scene = _scene()
    renderer = _FakeRenderer(scene.view2d)  # type: ignore[arg-type]
    session = _session(renderer, query_outcome=AdaptationOutcome.REJECT)
    session.render(scene)

    result = session.query(
        QueryRequest(
            id="query:rejected",
            panel_id="panel:main",
            coordinate=(20.0, 20.0),
            scope=QueryScope.DATA,
            requested_payload=(QueryPayload.IDENTITY,),
        )
    )

    assert result.status is QueryStatus.UNSUPPORTED
    assert result.diagnostic == "runtime query capability rejected request"
    assert renderer.query_calls == []
