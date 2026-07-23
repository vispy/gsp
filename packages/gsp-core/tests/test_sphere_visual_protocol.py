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


def _sphere(**kwargs: object) -> SphereVisual:
    values = {
        "id": "sphere:test",
        "positions": np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=np.float32),
        "radii": 0.5,
        "colors": np.array([255, 0, 0, 255], dtype=np.uint8),
    }
    values.update(kwargs)
    return SphereVisual(**values)  # type: ignore[arg-type]


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:1",
        camera=Camera3D(eye=(3.0, 3.0, 3.0), target=(0.0, 0.0, 0.0), up=(0.0, 0.0, 1.0)),
        projection=PerspectiveProjection3D(near_far=(0.1, 100.0)),
    )


def test_sphere_visual_accepts_uniform_and_per_item_radii_and_colors() -> None:
    assert _sphere().radius_values().tolist() == [0.5, 0.5]
    visual = _sphere(
        radii=np.array([0.25, 1.5], dtype=np.float64),
        colors=np.array([[255, 0, 0, 255], [0, 0, 255, 128]], dtype=np.uint8),
    )
    assert visual.radius_values().tolist() == [0.25, 1.5]
    assert visual.coordinate_space is CoordinateSpace.DATA


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("positions", np.array([[0.0, 0.0]], dtype=np.float32), r"shape \(N, 3\)"),
        ("positions", np.array([[0.0, np.nan, 0.0]], dtype=np.float32), "finite"),
        ("radii", 0.0, "positive"),
        ("radii", np.array([0.5], dtype=np.float32), "length"),
        ("radii", np.inf, "finite"),
        ("colors", np.array([[255, 0, 0, 255]], dtype=np.uint8), "colors"),
        (
            "colors",
            np.array([1.0, np.nan, 0.0, 1.0], dtype=np.float32),
            "finite",
        ),
        ("coordinate_space", CoordinateSpace.NDC, "CoordinateSpace.DATA"),
    ],
)
def test_sphere_visual_rejects_invalid_inputs(field: str, value: object, error: str) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _sphere(**{field: value})


def test_sphere_visual_requires_scene_view3d() -> None:
    visual = _sphere()
    with pytest.raises(ValueError, match="Scene.view3d"):
        Scene(id="scene:no-view", visuals=(visual,))
    Scene(id="scene:view3d", visuals=(visual,), view3d=_view3d())
