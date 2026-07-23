"""S065 TextVisual billboard contract shared across backend profiles."""

import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    FontRole,
    PerspectiveProjection3D,
    TEXT_VISUAL_BILLBOARD3D_CAPABILITY,
    TEXT_VISUAL_BILLBOARD3D_DEPTH_OCCLUSION_CAPABILITY,
    TextAnchorX,
    TextAnchorY,
    TextVisual,
    View3D,
)


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:main",
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=PerspectiveProjection3D(),
    )


def test_billboard_contract_preserves_bounded_text_fields() -> None:
    visual = TextVisual(
        id="visual:labels",
        texts=("ASCII", "Δ café"),
        positions=np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
        rgba=np.array([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8),
        font_size_px=np.array([12, 18], dtype=np.float32),
        font_role=FontRole.MONOSPACE,
        anchor_x=(TextAnchorX.LEFT, TextAnchorX.RIGHT),
        anchor_y=(TextAnchorY.TOP, TextAnchorY.BOTTOM),
        rotation_rad=np.array([0, np.pi / 4], dtype=np.float32),
        z_order=3,
    )

    scene = Scene(id="scene:labels", visuals=(visual,), view3d=_view3d())

    assert scene.visuals == (visual,)
    assert visual.font_size_values().tolist() == [12.0, 18.0]
    assert visual.anchor_x_values() == (TextAnchorX.LEFT, TextAnchorX.RIGHT)
    assert visual.anchor_y_values() == (TextAnchorY.TOP, TextAnchorY.BOTTOM)
    assert visual.z_order == 3


def test_billboard_capability_ids_keep_depth_occlusion_separate() -> None:
    assert TEXT_VISUAL_BILLBOARD3D_CAPABILITY == "textvisual.billboard3d.v1"
    assert (
        TEXT_VISUAL_BILLBOARD3D_DEPTH_OCCLUSION_CAPABILITY
        == "textvisual.billboard3d.depth_occlusion.v1"
    )
    assert TEXT_VISUAL_BILLBOARD3D_CAPABILITY != (
        TEXT_VISUAL_BILLBOARD3D_DEPTH_OCCLUSION_CAPABILITY
    )


def test_billboard_requires_view3d() -> None:
    visual = TextVisual(
        id="visual:label",
        texts=("label",),
        positions=np.array([[0, 0, 0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
    )
    with pytest.raises(ValueError, match="DATA positions3d require Scene.view3d"):
        Scene(id="scene:invalid", visuals=(visual,))
