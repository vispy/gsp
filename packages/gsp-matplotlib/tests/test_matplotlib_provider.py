from __future__ import annotations

import subprocess
import sys
from dataclasses import replace

import numpy as np

import gsp
from gsp.backends import SessionRequest
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    MeshVisual,
    Orbit3DPayload,
    Pan3DPayload,
    PerspectiveProjection3D,
    PointVisual,
    View2D,
    View3D,
    Zoom3DPayload,
    orbit_view3d,
    pan_view3d,
    zoom_view3d,
)
from gsp_matplotlib.session import MatplotlibSession


def test_matplotlib_plugin_is_discoverable_without_eager_matplotlib_import():
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


def test_matplotlib_session_renders_gsp_scene():
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


def test_matplotlib_session_captures_static_perspective_mesh(tmp_path):
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


def test_matplotlib_session_deterministically_rerenders_revised_view3d_state():
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
                np.array(result.axes.collections[0].get_paths()[0].vertices[:3])
            )
        repeated = session.render(
            gsp.Scene(
                id="scene:camera-repeat",
                visuals=(mesh,),
                view3d=zoomed,
            )
        )
        repeated_projection = np.array(
            repeated.axes.collections[0].get_paths()[0].vertices[:3]
        )

    assert [view.revision for view in views] == [0, 1, 2, 3, 4]
    assert not np.allclose(projected[0], projected[1])
    assert not np.allclose(projected[1], projected[2])
    assert not np.allclose(projected[2], projected[3])
    np.testing.assert_allclose(projected[0], projected[4])
    np.testing.assert_allclose(projected[3], repeated_projection)
