import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from gsp.protocol import (
    Camera3D,
    OrthographicProjection3D,
    PerspectiveProjection3D,
    SphereVisual,
    View3D,
)
from gsp_matplotlib.capabilities import capability_snapshot
from gsp_matplotlib.protocol_renderer import render_sphere_visual


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:1",
        camera=Camera3D(
            eye=(0.0, 0.0, 5.0), target=(0.0, 0.0, 0.0), up=(0.0, 1.0, 0.0)
        ),
        projection=OrthographicProjection3D(
            xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), near_far=(0.1, 20.0)
        ),
    )


def test_matplotlib_sphere_projects_data_radii_and_orders_near_last() -> None:
    figure, axes = plt.subplots(figsize=(4, 4), dpi=100)
    try:
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        visual = SphereVisual(
            id="sphere:adapted",
            positions=np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            radii=np.array([0.5, 1.0], dtype=np.float32),
            colors=np.array([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
        )
        artist = render_sphere_visual(axes, visual, view3d=_view3d())
        assert artist.get_gid() == visual.id
        np.testing.assert_allclose(
            np.asarray(artist.get_offsets(), dtype=np.float64),
            [[0.5, 0.5], [0.5, 0.5]],
        )
        np.testing.assert_allclose(
            artist.get_facecolor(),
            [[1, 0, 0, 1], [0, 0, 1, 1]],
        )
        np.testing.assert_allclose(artist.get_sizes(), [5184.0, 20736.0])
    finally:
        plt.close(figure)


def test_matplotlib_perspective_sphere_near_center_has_larger_projected_size() -> None:
    figure, axes = plt.subplots(figsize=(4, 4), dpi=100)
    try:
        figure.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        view3d = View3D(
            id="view:perspective",
            panel_id="panel:1",
            camera=Camera3D(
                eye=(0.0, 0.0, 5.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 1.0, 0.0),
            ),
            projection=PerspectiveProjection3D(
                fov_y_degrees=45.0,
                near_far=(0.1, 20.0),
            ),
        )
        visual = SphereVisual(
            id="sphere:perspective",
            positions=np.array(
                [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float32
            ),
            radii=0.5,
            colors=np.array([255, 0, 0, 255], dtype=np.uint8),
        )

        artist = render_sphere_visual(axes, visual, view3d=view3d)

        far_area, near_area = artist.get_sizes()
        assert near_area > far_area
    finally:
        plt.close(figure)


def test_matplotlib_sphere_capability_is_adapted_without_analytic_depth() -> None:
    caps = capability_snapshot()
    assert caps.supports_visual("sphere")
    assert caps.supports_view3d_capability("spherevisual.v1")
    assert not caps.supports_view3d_capability("spherevisual.analytic_surface_depth.v1")
    metadata = caps.metadata["spherevisual"]
    assert isinstance(metadata, str)
    assert "analytic surface depth is not claimed" in metadata
    assert "center-depth painter ordering" in metadata
    assert "view-plane approximation" in metadata
