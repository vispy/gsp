from __future__ import annotations

import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import gsp
from gsp.backends import SessionRequest
from gsp.protocol import (
    AdaptationOutcome,
    Camera3D,
    CoordinateSpace,
    GUIDE_QUERY_PAYLOAD_KIND,
    MeshVisual,
    Orbit3DPayload,
    Pan3DPayload,
    PerspectiveProjection3D,
    PointVisual,
    QueryCoordinateSpace,
    QueryPayload,
    QueryRequest,
    QueryScope,
    QueryStatus,
    SphereVisual,
    VIEW3D_QUERY_PAYLOAD_KIND,
    View2D,
    View3D,
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


def test_matplotlib_session_renders_gsp_scene() -> None:
    scene = gsp.Scene(
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


def test_matplotlib_session_captures_static_perspective_mesh(tmp_path: Path) -> None:
    scene = gsp.Scene(
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
    with MatplotlibSession(
        request=SessionRequest(require=frozenset({"visual.mesh"}))
    ) as session:
        for view in views:
            result = session.render(
                gsp.Scene(
                    id=f"scene:camera-{view.revision}",
                    visuals=(mesh,),
                    view3d=view,
                )
            )
            projected.append(
                np.asarray(result.axes.collections[0].get_paths()[0].vertices)[:3]
            )
        repeated = session.render(
            gsp.Scene(
                id="scene:camera-repeat",
                visuals=(mesh,),
                view3d=zoomed,
            )
        )
        repeated_projection = np.asarray(
            repeated.axes.collections[0].get_paths()[0].vertices
        )[:3]

    assert [view.revision for view in views] == [0, 1, 2, 3, 4]
    assert not np.allclose(projected[0], projected[1])
    assert not np.allclose(projected[1], projected[2])
    assert not np.allclose(projected[2], projected[3])
    np.testing.assert_allclose(projected[0], projected[4])
    np.testing.assert_allclose(projected[3], repeated_projection)


def _query_scene(scene_id: str, x: float) -> gsp.Scene:
    return gsp.Scene(
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
            QueryRequest(
                id="query:latest", panel_id="panel:main", coordinate=(10.0, 0.0)
            )
        )
        explicit = session.query(
            QueryRequest(
                id="query:first", panel_id="panel:main", coordinate=(0.0, 0.0)
            ),
            scene_id=first.id,
        )

    assert latest.status is QueryStatus.HIT
    assert latest.visual_id == "visual:second"
    assert explicit.status is QueryStatus.HIT
    assert explicit.visual_id == "visual:first"


def test_public_query_session_state_errors_are_clear() -> None:
    request = QueryRequest(
        id="query:state", panel_id="panel:main", coordinate=(0.0, 0.0)
    )
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
            QueryRequest(
                id="query:hit", panel_id="panel:main", coordinate=(0.0, 0.0)
            )
        )
        miss = session.query(
            QueryRequest(
                id="query:miss", panel_id="panel:main", coordinate=(20.0, 20.0)
            )
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
            QueryCoordinateSpace.PANEL
            if scope is QueryScope.GUIDES
            else QueryCoordinateSpace.DATA
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
    sphere_scene = gsp.Scene(
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
            QueryRequest(
                id="query:sphere", panel_id="panel:main", coordinate=(0.0, 0.0)
            )
        )

    assert unsupported.status is QueryStatus.UNSUPPORTED
    assert unsupported.diagnostic is not None
    assert "SphereVisual" in unsupported.diagnostic


def test_public_query_routes_proven_matplotlib_view3d_ray_path() -> None:
    scene = gsp.Scene(
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
