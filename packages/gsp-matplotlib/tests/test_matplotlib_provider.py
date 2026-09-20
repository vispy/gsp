from __future__ import annotations

import subprocess
import struct
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import gsp
from conformance.p038_support import resolved_single_panel_fixture, single_panel_scene
from gsp.backends import SessionRequest
from gsp.protocol import (
    AdaptationOutcome,
    AxisDimension,
    AxisGuide,
    AxisSide,
    Camera3D,
    CanvasSize,
    CoordinateSpace,
    ExplicitPanelLayoutV1,
    GUIDE_QUERY_PAYLOAD_KIND,
    MeshVisual,
    NormalizedRenderTargetRect,
    LogicalPixelRect,
    Orbit3DPayload,
    Pan3DPayload,
    PerspectiveProjection3D,
    PointVisual,
    Panel,
    PanelPlacement,
    PanelTextGuide,
    PanelTextRole,
    PixelOrigin,
    QueryCoordinateSpace,
    QueryPayload,
    QueryRequest,
    QueryScope,
    QueryStatus,
    RenderTarget,
    ResolvedLayoutSnapshot,
    ResolvedPanelLayout,
    SphereVisual,
    VIEW3D_QUERY_PAYLOAD_KIND,
    View2D,
    View3D,
    VisualAttachment,
    Zoom3DPayload,
    orbit_view3d,
    pan_view3d,
    zoom_view3d,
)
from gsp_matplotlib.session import MatplotlibSession


def test_matplotlib_plugin_is_discoverable_without_eager_matplotlib_import() -> None:
    code = """
import sys
import gsp

assert "matplotlib.pyplot" not in sys.modules
infos = {item.name: item for item in gsp.discover_backends()}
assert "matplotlib" in infos
assert infos["matplotlib"].available is None
assert "matplotlib.pyplot" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_matplotlib_renders_multi_panel_scene_with_scene_wide_layout() -> None:
    panels = (Panel("panel:left"), Panel("panel:right"))
    scene = gsp.Scene(
        id="scene:multi",
        panels=panels,
        panel_layout=ExplicitPanelLayoutV1(
            (
                PanelPlacement(
                    panels[0].id,
                    NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0),
                ),
                PanelPlacement(
                    panels[1].id,
                    NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0),
                ),
            )
        ),
    )
    session = MatplotlibSession(request=SessionRequest())
    result = session.render(scene)

    assert tuple(panel.panel_id for panel in result.layout_snapshot.panels) == (
        "panel:left",
        "panel:right",
    )
    assert result.axes_for_panel("panel:left") is not result.axes_for_panel("panel:right")
    assert len(result.figure.axes) == 2
    assert result.layout_snapshot.panel("panel:left").panel_rect_px.width == pytest.approx(
        result.layout_snapshot.panel("panel:right").panel_rect_px.width
    )
    session.close()


def test_matplotlib_routes_panel_visuals_guides_and_queries_by_identity() -> None:
    panels = (Panel("panel:left"), Panel("panel:right"))
    views = (
        View2D(
            id="view:left",
            panel_id=panels[0].id,
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        ),
        View2D(
            id="view:right",
            panel_id=panels[1].id,
            x_range=(9.0, 11.0),
            y_range=(-1.0, 1.0),
        ),
    )
    visuals = (
        PointVisual(
            id="visual:left",
            positions=np.asarray([[0.0, 0.0]], dtype=np.float32),
            colors=np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
            sizes=10.0,
            coordinate_space=CoordinateSpace.DATA,
        ),
        PointVisual(
            id="visual:right",
            positions=np.asarray([[10.0, 0.0]], dtype=np.float32),
            colors=np.asarray([[0, 0, 255, 255]], dtype=np.uint8),
            sizes=10.0,
            coordinate_space=CoordinateSpace.DATA,
        ),
    )
    scene = gsp.Scene(
        id="scene:routed-multi",
        panels=panels,
        panel_layout=ExplicitPanelLayoutV1(
            (
                PanelPlacement(panels[0].id, NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
                PanelPlacement(panels[1].id, NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
            )
        ),
        visuals=visuals,
        views2d=views,
        attachments=(
            VisualAttachment(visuals[0].id, panels[0].id, views[0].id),
            VisualAttachment(visuals[1].id, panels[1].id, views[1].id),
        ),
        axis_guides=(
            AxisGuide(
                id="guide:left-x",
                view_id=views[0].id,
                dimension=AxisDimension.X,
                side=AxisSide.BOTTOM,
                label_text="Left X",
            ),
            AxisGuide(
                id="guide:right-x",
                view_id=views[1].id,
                dimension=AxisDimension.X,
                side=AxisSide.BOTTOM,
                label_text="Right X",
            ),
        ),
        panel_text_guides=(
            PanelTextGuide(
                id="guide:left-title",
                panel_id=panels[0].id,
                role=PanelTextRole.TITLE,
                text="Left",
            ),
            PanelTextGuide(
                id="guide:right-title",
                panel_id=panels[1].id,
                role=PanelTextRole.TITLE,
                text="Right",
            ),
        ),
    )

    session = MatplotlibSession(request=SessionRequest())
    result = session.render(scene)

    left_axes = result.axes_for_panel(panels[0].id)
    right_axes = result.axes_for_panel(panels[1].id)
    assert left_axes.get_title() == "Left"
    assert right_axes.get_title() == "Right"
    assert left_axes.get_xlabel() == "Left X"
    assert right_axes.get_xlabel() == "Right X"
    assert len(left_axes.collections) == len(right_axes.collections) == 1
    assert result.layout_snapshot.panel(panels[0].id).view_id == views[0].id
    assert result.layout_snapshot.panel(panels[1].id).view_id == views[1].id
    assert {box.guide_id for box in result.layout_snapshot.panel(panels[0].id).title_boxes} == {
        "guide:left-title"
    }
    assert {box.guide_id for box in result.layout_snapshot.panel(panels[1].id).title_boxes} == {
        "guide:right-title"
    }

    left_hit = session.query(QueryRequest("query:left", panels[0].id, (0.0, 0.0)))
    right_hit = session.query(QueryRequest("query:right", panels[1].id, (10.0, 0.0)))
    assert left_hit.status is QueryStatus.HIT
    assert left_hit.visual_id == visuals[0].id
    assert right_hit.status is QueryStatus.HIT
    assert right_hit.visual_id == visuals[1].id
    right_title = result.layout_snapshot.panel(panels[1].id).title_boxes[0].rect_px
    right_guide_hit = session.query(
        QueryRequest(
            "query:right-title",
            panels[1].id,
            (
                right_title.x + right_title.width / 2.0,
                right_title.y + right_title.height / 2.0,
            ),
            coordinate_space=QueryCoordinateSpace.PANEL,
            scope=QueryScope.GUIDES,
            requested_extension_payload_kinds=(GUIDE_QUERY_PAYLOAD_KIND,),
        )
    )
    assert right_guide_hit.status is QueryStatus.HIT
    assert right_guide_hit.visual_id == "guide:right-title"
    session.close()


def test_matplotlib_routes_mixed_2d_and_3d_panel_views() -> None:
    panels = (Panel("panel:2d"), Panel("panel:3d"))
    view2d = View2D(id="view:2d", panel_id=panels[0].id)
    view3d = View3D(
        id="view:3d",
        panel_id=panels[1].id,
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=PerspectiveProjection3D(),
    )
    scene = gsp.Scene(
        id="scene:mixed-views",
        panels=panels,
        panel_layout=ExplicitPanelLayoutV1(
            (
                PanelPlacement(panels[0].id, NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
                PanelPlacement(panels[1].id, NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
            )
        ),
        views2d=(view2d,),
        views3d=(view3d,),
    )

    session = MatplotlibSession(request=SessionRequest())
    result = session.render(scene)
    plot = result.layout_snapshot.panel(panels[1].id).plot_rect_px
    ray = session.query(
        QueryRequest(
            id="query:mixed-ray",
            panel_id=panels[1].id,
            coordinate=(plot.x + plot.width / 2.0, plot.y + plot.height / 2.0),
            coordinate_space=QueryCoordinateSpace.PANEL,
            requested_payload=(QueryPayload.IDENTITY,),
            requested_extension_payload_kinds=(VIEW3D_QUERY_PAYLOAD_KIND,),
        )
    )

    assert result.layout_snapshot.panel(panels[0].id).view_id == view2d.id
    assert result.layout_snapshot.panel(panels[1].id).view_id == view3d.id
    assert result.view_snapshot_id_for_panel(panels[1].id) is not None
    assert result.view3d_projection_snapshot_for_panel(panels[0].id) is None
    assert result.view3d_projection_snapshot_for_panel(panels[1].id) is not None
    assert ray.status is QueryStatus.HIT
    assert ray.view_snapshot_id == result.view_snapshot_id_for_panel(panels[1].id)
    session.close()


def test_matplotlib_consumes_multi_panel_layout_as_one_authoritative_snapshot() -> None:
    panels = (Panel("panel:left"), Panel("panel:right"))
    views = (
        View2D(id="view:left", panel_id=panels[0].id),
        View2D(id="view:right", panel_id=panels[1].id),
    )
    scene = gsp.Scene(
        id="scene:consume-multi",
        panels=panels,
        panel_layout=ExplicitPanelLayoutV1(
            (
                PanelPlacement(panels[0].id, NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
                PanelPlacement(panels[1].id, NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
            )
        ),
        views2d=views,
        canvas_size=CanvasSize.pixel_exact(400, 200),
    )
    target = RenderTarget(400, 200, pixel_origin=PixelOrigin.TOP_LEFT)
    consumed = ResolvedLayoutSnapshot(
        snapshot_id="layout:consume-multi",
        render_target=target,
        panels=(
            ResolvedPanelLayout(
                panel_id=panels[0].id,
                panel_rect_px=LogicalPixelRect(0, 0, 200, 200),
                plot_rect_px=LogicalPixelRect(20, 10, 160, 170),
                view_id=views[0].id,
            ),
            ResolvedPanelLayout(
                panel_id=panels[1].id,
                panel_rect_px=LogicalPixelRect(200, 0, 200, 200),
                plot_rect_px=LogicalPixelRect(220, 10, 160, 170),
                view_id=views[1].id,
            ),
        ),
    )

    session = MatplotlibSession(request=SessionRequest())
    result = session.render(scene, layout_snapshot=consumed)

    assert result.layout_was_consumed is True
    assert result.layout_snapshot == consumed
    assert result.axes_for_panel(panels[0].id).get_position().bounds == pytest.approx(
        (0.05, 0.1, 0.4, 0.85)
    )
    assert result.axes_for_panel(panels[1].id).get_position().bounds == pytest.approx(
        (0.55, 0.1, 0.4, 0.85)
    )
    session.close()


def test_matplotlib_session_renders_gsp_scene() -> None:
    scene = single_panel_scene(
        id="scene:test",
        visuals=(
            PointVisual(
                id="visual:points",
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
                sizes=8.0,
            ),
        ),
        view2d=View2D(
            id="view:main",
            panel_id="panel:main",
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        ),
    )
    with gsp.open_session("matplotlib", require={"visual.points"}) as session:
        result = session.render(scene)
        assert result.axes.get_xlim() == (-1.0, 1.0)


def test_matplotlib_session_consumed_layout_validates_scene_target_before_render() -> None:
    panel = Panel(
        id="panel:consumed",
    )
    view = View2D(id="view:consumed", panel_id=panel.id)
    scene = single_panel_scene(
        id="scene:consumed",
        panels=(panel,),
        panel_layout=ExplicitPanelLayoutV1(
            (
                PanelPlacement(
                    panel.id,
                    NormalizedRenderTargetRect(0.1, 0.2, 0.8, 0.6),
                ),
            )
        ),
        view2d=view,
        canvas_size=CanvasSize.pixel_exact(320, 240),
    )
    layout = resolved_single_panel_fixture(
        snapshot_id="layout:consumed",
        render_target=RenderTarget(640, 480, pixel_origin=PixelOrigin.TOP_LEFT),
        panel_rect_px=LogicalPixelRect(64, 96, 512, 288),
        plot_rect_px=LogicalPixelRect(100, 120, 440, 220),
        view_id=view.id,
    )
    session = MatplotlibSession(request=SessionRequest())

    with pytest.raises(ValueError, match="scene canvas policy"):
        session.render(scene, layout_snapshot=layout)

    assert session._results == []
    session.close()


def test_matplotlib_session_captures_static_perspective_mesh(tmp_path: Path) -> None:
    scene = single_panel_scene(
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
            projection=PerspectiveProjection3D(),
        ),
    )
    target = tmp_path / "mesh3d.png"

    with gsp.open_session("matplotlib", require={"output.file", "visual.mesh"}) as session:
        result = session.render(scene, target=target)

    assert target.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert len(result.axes.collections) == 1


def test_matplotlib_session_preserves_pixel_exact_png_dimensions(tmp_path: Path) -> None:
    scene = single_panel_scene(
        id="scene:pixel-exact",
        canvas_size=CanvasSize.pixel_exact(800, 600),
    )
    target = tmp_path / "pixel-exact.png"

    with gsp.open_session("matplotlib", require={"output.file"}) as session:
        session.render(scene, target=target)

    header = target.read_bytes()[:24]
    assert struct.unpack(">II", header[16:24]) == (800, 600)


def test_matplotlib_session_deterministically_rerenders_revised_view3d_state() -> None:
    mesh = MeshVisual(
        id="visual:camera-mesh",
        positions=np.asarray(
            [[-0.8, -0.4, 0.2], [0.9, -0.2, -0.1], [-0.1, 0.7, 0.6]],
            dtype=np.float32,
        ),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        color=np.asarray([70, 130, 220, 255], dtype=np.uint8),
        coordinate_space=CoordinateSpace.DATA,
    )
    home = View3D(
        id="view:main",
        panel_id="panel:main",
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=PerspectiveProjection3D(
            fov_y_degrees=45.0,
            near_far=(0.1, 100.0),
        ),
    )
    orbited = orbit_view3d(
        home,
        Orbit3DPayload(delta_yaw_radians=0.25, delta_pitch_radians=-0.1),
    )
    panned = pan_view3d(
        orbited,
        Pan3DPayload(delta_view_right=0.3, delta_view_up=-0.15),
    )
    zoomed = zoom_view3d(panned, Zoom3DPayload(scale=1.4))
    reset = replace(
        zoomed,
        camera=home.camera,
        projection=home.projection,
        revision=zoomed.revision + 1,
    )
    views = (home, orbited, panned, zoomed, reset)

    projected: list[np.ndarray] = []
    with MatplotlibSession(request=SessionRequest(require=frozenset({"visual.mesh"}))) as session:
        for view in views:
            result = session.render(
                single_panel_scene(
                    id=f"scene:camera-{view.revision}",
                    visuals=(mesh,),
                    view3d=view,
                )
            )
            projected.append(np.asarray(result.axes.collections[0].get_paths()[0].vertices)[:3])
        repeated = session.render(
            single_panel_scene(
                id="scene:camera-repeat",
                visuals=(mesh,),
                view3d=zoomed,
            )
        )
        repeated_projection = np.asarray(repeated.axes.collections[0].get_paths()[0].vertices)[:3]

    assert [view.revision for view in views] == [0, 1, 2, 3, 4]
    assert not np.allclose(projected[0], projected[1])
    assert not np.allclose(projected[1], projected[2])
    assert not np.allclose(projected[2], projected[3])
    np.testing.assert_allclose(projected[0], projected[4])
    np.testing.assert_allclose(projected[3], repeated_projection)


def _query_scene(scene_id: str, x: float) -> gsp.Scene:
    return single_panel_scene(
        id=scene_id,
        visuals=(
            PointVisual(
                id=f"visual:{scene_id.split(':')[-1]}",
                positions=np.asarray([[x, 0.0]], dtype=np.float32),
                colors=np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
                sizes=4.0,
            ),
        ),
        view2d=View2D(id=f"view:{scene_id.split(':')[-1]}", panel_id="panel:main"),
    )


def test_public_query_routes_latest_and_explicit_scene_ids() -> None:
    first = _query_scene("scene:first", 0.0)
    second = _query_scene("scene:second", 10.0)
    with MatplotlibSession(request=SessionRequest()) as session:
        session.render(first)
        session.render(second)

        latest = session.query(
            QueryRequest(id="query:latest", panel_id="panel:main", coordinate=(10.0, 0.0))
        )
        explicit = session.query(
            QueryRequest(id="query:first", panel_id="panel:main", coordinate=(0.0, 0.0)),
            scene_id=first.id,
        )

    assert latest.status is QueryStatus.HIT
    assert latest.visual_id == "visual:second"
    assert explicit.status is QueryStatus.HIT
    assert explicit.visual_id == "visual:first"


def test_public_query_session_state_errors_are_clear() -> None:
    request = QueryRequest(id="query:state", panel_id="panel:main", coordinate=(0.0, 0.0))
    session = MatplotlibSession(request=SessionRequest())
    with np.testing.assert_raises_regex(RuntimeError, "requires a rendered scene"):
        session.query(request)
    session.render(_query_scene("scene:known", 0.0))
    with np.testing.assert_raises_regex(RuntimeError, "has not been rendered"):
        session.query(request, scene_id="scene:unknown")
    session.close()
    with np.testing.assert_raises_regex(RuntimeError, "session is closed"):
        session.query(request)


def test_public_query_preserves_hit_miss_and_existing_payloads() -> None:
    scene = _query_scene("scene:payload", 0.0)
    with MatplotlibSession(request=SessionRequest()) as session:
        render = session.render(scene)
        hit = session.query(
            QueryRequest(id="query:hit", panel_id="panel:main", coordinate=(0.0, 0.0))
        )
        miss = session.query(
            QueryRequest(id="query:miss", panel_id="panel:main", coordinate=(20.0, 20.0))
        )

    assert hit.status is QueryStatus.HIT
    assert hit.item_id == 0
    assert hit.data_coordinate == (0.0, 0.0)
    assert hit.displayed_rgba == (1.0, 0.0, 0.0, 1.0)
    assert hit.layout_snapshot_id == render.layout_snapshot_id
    assert hit.view_snapshot_id == render.view_snapshot_id
    assert miss.status is QueryStatus.MISS
    assert miss.layout_snapshot_id == render.layout_snapshot_id
    assert miss.view_snapshot_id == render.view_snapshot_id


def test_public_query_preserves_explicit_snapshot_ids() -> None:
    scene = _query_scene("scene:explicit-snapshots", 0.0)
    request = QueryRequest(
        id="query:explicit-snapshots",
        panel_id="panel:main",
        coordinate=(0.0, 0.0),
        layout_snapshot_id="layout:explicit-stale",
        view_snapshot_id="view-snapshot:explicit-stale",
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        session.render(scene)
        result = session.query(request)

    assert result.layout_snapshot_id == request.layout_snapshot_id
    assert result.view_snapshot_id == request.view_snapshot_id


@pytest.mark.parametrize("scope", [QueryScope.DATA, QueryScope.GUIDES])
def test_public_query_rejects_unwired_extension_payload(scope: QueryScope) -> None:
    scene = _query_scene("scene:unknown-extension", 0.0)
    request = QueryRequest(
        id="query:unknown-extension",
        panel_id="panel:main",
        coordinate=(0.0, 0.0),
        coordinate_space=(
            QueryCoordinateSpace.PANEL if scope is QueryScope.GUIDES else QueryCoordinateSpace.DATA
        ),
        scope=scope,
        requested_extension_payload_kinds=("unknown.extension",),
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        decision = session.capabilities.adapt_query_request(request)
        session.render(scene)
        result = session.query(request)

    assert decision.outcome is AdaptationOutcome.REJECT
    assert result.status is QueryStatus.UNSUPPORTED
    assert result.diagnostic is not None
    assert "unknown.extension" in result.diagnostic


def test_public_guide_query_negotiates_known_guide_extension() -> None:
    scene = _query_scene("scene:guide-extension", 0.0)
    request = QueryRequest(
        id="query:guide-extension",
        panel_id="panel:main",
        coordinate=(0.0, 0.0),
        coordinate_space=QueryCoordinateSpace.PANEL,
        scope=QueryScope.GUIDES,
        requested_extension_payload_kinds=(GUIDE_QUERY_PAYLOAD_KIND,),
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        decision = session.capabilities.adapt_query_request(request)
        session.render(scene)
        result = session.query(request)

    assert decision.outcome is AdaptationOutcome.ACCEPT
    assert result.status in (QueryStatus.HIT, QueryStatus.MISS)


def test_matplotlib_public_capabilities_do_not_claim_incoherent_all_rendered_scope() -> None:
    scene = _query_scene("scene:all-rendered", 0.0)
    request = QueryRequest(
        id="query:all-rendered",
        panel_id="panel:main",
        coordinate=(0.0, 0.0),
        scope=QueryScope.ALL_RENDERED,
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        assert session.capabilities.supports_query_scope(QueryScope.DATA)
        assert session.capabilities.supports_query_scope(QueryScope.GUIDES)
        assert not session.capabilities.supports_query_scope(QueryScope.ALL_RENDERED)
        session.render(scene)
        result = session.query(request)

    assert result.status is QueryStatus.UNSUPPORTED


def test_public_query_capability_and_unproven_s065_visual_are_structured() -> None:
    sphere_scene = single_panel_scene(
        id="scene:sphere",
        visuals=(
            SphereVisual(
                id="visual:sphere",
                positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
                radii=1.0,
                colors=np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
            ),
        ),
        view3d=View3D(
            id="view:sphere",
            panel_id="panel:main",
            camera=Camera3D(
                eye=(3.0, 3.0, 3.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
            ),
            projection=PerspectiveProjection3D(),
        ),
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        assert session.capabilities.supports_query_scope(QueryScope.DATA)
        session.render(sphere_scene)
        unsupported = session.query(
            QueryRequest(id="query:sphere", panel_id="panel:main", coordinate=(0.0, 0.0))
        )

    assert unsupported.status is QueryStatus.UNSUPPORTED
    assert unsupported.diagnostic is not None
    assert "SphereVisual" in unsupported.diagnostic


def test_public_query_routes_proven_matplotlib_view3d_ray_path() -> None:
    scene = single_panel_scene(
        id="scene:ray",
        view3d=View3D(
            id="view:ray",
            panel_id="panel:main",
            camera=Camera3D(
                eye=(3.0, 3.0, 3.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
            ),
            projection=PerspectiveProjection3D(),
        ),
    )
    request = QueryRequest(
        id="query:ray",
        panel_id="panel:main",
        coordinate=(320.0, 240.0),
        coordinate_space=QueryCoordinateSpace.PANEL,
        requested_payload=(QueryPayload.IDENTITY,),
        requested_extension_payload_kinds=(VIEW3D_QUERY_PAYLOAD_KIND,),
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        session.render(scene)
        result = session.query(request)
        stale = session.query(
            replace(
                request,
                id="query:ray-stale",
                view_snapshot_id="view-snapshot:explicit-stale",
            )
        )

    assert result.status is QueryStatus.HIT
    assert result.extension_payload_kind == VIEW3D_QUERY_PAYLOAD_KIND
    assert stale.status is QueryStatus.STALE
    assert stale.view_snapshot_id != "view-snapshot:explicit-stale"
