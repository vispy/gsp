"""Scene-level validation for the one-view GSP 0.2 implementation slice."""

import pytest
import numpy as np

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    OrthographicProjection3D,
    TextVisual,
    View2D,
    View3D,
    VisualTransformBinding,
)


def _view2d() -> View2D:
    return View2D(id="view:2d", panel_id="panel:main")


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:main",
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=OrthographicProjection3D(),
    )


def test_scene_rejects_both_active_views() -> None:
    with pytest.raises(ValueError, match="cannot define both view2d and view3d"):
        Scene(id="scene:invalid", view2d=_view2d(), view3d=_view3d())


def test_scene_accepts_viewless_ndc_or_one_active_view() -> None:
    assert Scene(id="scene:viewless").view2d is None
    assert Scene(id="scene:2d", view2d=_view2d()).view2d is not None
    assert Scene(id="scene:3d", view3d=_view3d()).view3d is not None


def test_scene_accepts_text_billboard3d_only_with_data_view3d_contract() -> None:
    billboard = TextVisual(
        id="visual:billboard",
        texts=("origin",),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
    )

    assert Scene(
        id="scene:billboard", visuals=(billboard,), view3d=_view3d()
    ).visuals == (billboard,)
    with pytest.raises(ValueError, match="DATA positions3d require Scene.view3d"):
        Scene(id="scene:no-view", visuals=(billboard,))

    ndc3 = TextVisual(
        id="visual:ndc3",
        texts=("invalid",),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.NDC,
    )
    with pytest.raises(ValueError, match="positions3d require CoordinateSpace.DATA"):
        Scene(id="scene:ndc3", visuals=(ndc3,), view3d=_view3d())

    transformed = TextVisual(
        id="visual:transformed3",
        texts=("invalid",),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
        transform=VisualTransformBinding.from_ref("transform:invalid"),
    )
    with pytest.raises(ValueError, match="does not support a 2D visual transform"):
        Scene(id="scene:transform3", visuals=(transformed,), view3d=_view3d())
