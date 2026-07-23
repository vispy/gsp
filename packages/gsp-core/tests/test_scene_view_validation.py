"""Scene-level validation for the one-view GSP 0.2 implementation slice."""

import pytest

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    OrthographicProjection3D,
    View2D,
    View3D,
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
