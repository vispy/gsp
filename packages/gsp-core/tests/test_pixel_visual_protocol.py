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


def _pixels(**kwargs: object) -> PixelVisual:
    values = {
        "id": "pixel:test",
        "positions": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "colors": np.array([255, 0, 0, 255], dtype=np.uint8),
    }
    values.update(kwargs)
    return PixelVisual(**values)  # type: ignore[arg-type]


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:1",
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


def test_pixel_visual_accepts_uniform_and_per_item_fields() -> None:
    uniform = _pixels(pixel_size_px=2.0)
    assert uniform.pixel_size_values().tolist() == [2.0, 2.0]
    per_item = _pixels(
        colors=np.array(
            [[255, 0, 0, 255], [0, 255, 0, 128]], dtype=np.uint8
        ),
        pixel_size_px=np.array([2.0, 4.0], dtype=np.float32),
    )
    assert per_item.pixel_size_values().tolist() == [2.0, 4.0]


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("positions", np.array([0.0, 1.0], dtype=np.float32), "positions"),
        (
            "positions",
            np.array([[0.0, np.nan]], dtype=np.float32),
            "finite",
        ),
        (
            "colors",
            np.array([[255, 0, 0, 255]], dtype=np.uint8),
            "colors",
        ),
        (
            "colors",
            np.array([1.1, 0.0, 0.0, 1.0], dtype=np.float32),
            "must be in",
        ),
        (
            "colors",
            np.array([np.nan, 0.0, 0.0, 1.0], dtype=np.float32),
            "finite",
        ),
        (
            "pixel_size_px",
            np.array([1.0], dtype=np.float32),
            "length",
        ),
        ("pixel_size_px", 0.0, "positive"),
        ("pixel_size_px", np.inf, "finite"),
    ],
)
def test_pixel_visual_rejects_malformed_fields(
    field: str, value: object, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _pixels(**{field: value})


def test_pixel_visual_scene_view_requirements() -> None:
    pixels2d = _pixels()
    with pytest.raises(ValueError, match="Scene.view2d"):
        Scene(id="scene:missing-2d", visuals=(pixels2d,))
    view2d = View2D(id="view:2d", panel_id="panel:1")
    Scene(
        id="scene:2d",
        visuals=(pixels2d,),
        panels=(Panel(id="panel:1", figure_id="figure:1"),),
        view2d=view2d,
    )

    pixels3d = _pixels(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="Scene.view3d"):
        Scene(id="scene:missing-3d", visuals=(pixels3d,))
    Scene(id="scene:3d", visuals=(pixels3d,), view3d=_view3d())
    with pytest.raises(ValueError, match="CoordinateSpace.DATA"):
        Scene(
            id="scene:ndc3",
            visuals=(
                _pixels(
                    positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
                    colors=np.array([255, 255, 255, 255], dtype=np.uint8),
                    coordinate_space=CoordinateSpace.NDC,
                ),
            ),
            view3d=_view3d(),
        )
