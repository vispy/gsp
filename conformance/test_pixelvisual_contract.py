from __future__ import annotations

import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    Panel,
    PerspectiveProjection3D,
    PixelVisual,
    View2D,
    View3D,
)


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:3d",
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


def test_pixelvisual_2d_scene_preserves_logical_sizes_and_colors() -> None:
    visual = PixelVisual(
        id="pixel:2d",
        positions=np.array([[-0.5, -0.5], [0.5, 0.5]], dtype=np.float32),
        colors=np.array(
            [[255, 0, 0, 255], [0, 255, 0, 128]], dtype=np.uint8
        ),
        pixel_size_px=np.array([3.0, 7.0], dtype=np.float32),
    )
    scene = Scene(
        id="scene:pixel-2d",
        visuals=(visual,),
        panels=(Panel(id="panel:2d", figure_id="figure:1"),),
        view2d=View2D(id="view:2d", panel_id="panel:2d"),
    )

    assert scene.visuals == (visual,)
    np.testing.assert_array_equal(visual.pixel_size_values(), [3.0, 7.0])
    np.testing.assert_array_equal(
        visual.colors, [[255, 0, 0, 255], [0, 255, 0, 128]]
    )


def test_pixelvisual_3d_contract_requires_data_space_and_view3d() -> None:
    visual = PixelVisual(
        id="pixel:3d",
        positions=np.array([[-0.5, 0.0, 0.0], [0.5, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
        pixel_size_px=5.0,
        coordinate_space=CoordinateSpace.DATA,
    )

    with pytest.raises(ValueError, match="Scene.view3d"):
        Scene(id="scene:missing-view3d", visuals=(visual,))

    scene = Scene(id="scene:pixel-3d", visuals=(visual,), view3d=_view3d())
    assert scene.visuals == (visual,)
    np.testing.assert_array_equal(visual.pixel_size_values(), [5.0, 5.0])

    with pytest.raises(ValueError, match="CoordinateSpace.DATA"):
        Scene(
            id="scene:pixel-3d-ndc",
            visuals=(
                PixelVisual(
                    id="pixel:3d-ndc",
                    positions=visual.positions,
                    colors=visual.colors,
                    coordinate_space=CoordinateSpace.NDC,
                ),
            ),
            view3d=_view3d(),
        )
