from __future__ import annotations

import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    PerspectiveProjection3D,
    SphereVisual,
    View3D,
)


def _view3d() -> View3D:
    return View3D(
        id="view:spheres",
        panel_id="panel:spheres",
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


def test_spherevisual_scene_preserves_data_radii_and_colors() -> None:
    visual = SphereVisual(
        id="sphere:contract",
        positions=np.array(
            [[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32
        ),
        radii=np.array([0.25, 0.75], dtype=np.float32),
        colors=np.array(
            [[255, 0, 0, 255], [0, 128, 255, 128]], dtype=np.uint8
        ),
    )
    scene = Scene(id="scene:spheres", visuals=(visual,), view3d=_view3d())

    assert scene.visuals == (visual,)
    assert visual.coordinate_space is CoordinateSpace.DATA
    np.testing.assert_allclose(visual.radius_values(), [0.25, 0.75])
    np.testing.assert_array_equal(
        visual.colors, [[255, 0, 0, 255], [0, 128, 255, 128]]
    )


def test_spherevisual_contract_requires_data_space_and_view3d() -> None:
    visual = SphereVisual(
        id="sphere:view3d",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        radii=0.5,
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="Scene.view3d"):
        Scene(id="scene:missing-view3d", visuals=(visual,))

    with pytest.raises(ValueError, match="CoordinateSpace.DATA"):
        SphereVisual(
            id="sphere:ndc",
            positions=visual.positions,
            radii=visual.radii,
            colors=visual.colors,
            coordinate_space=CoordinateSpace.NDC,
        )
