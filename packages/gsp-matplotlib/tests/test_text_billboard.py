"""Matplotlib S065 screen-facing TextVisual billboard coverage."""

import matplotlib.pyplot as plt
import numpy as np
from typing import cast

from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    FontRole,
    PerspectiveProjection3D,
    TextAnchorX,
    TextAnchorY,
    TextVisual,
    View3D,
)
from gsp_matplotlib.capabilities import capability_snapshot
from gsp_matplotlib.protocol_renderer import render_text_visual


def _view(*, eye: tuple[float, float, float]) -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:main",
        camera=Camera3D(
            eye=eye,
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=PerspectiveProjection3D(),
    )


def _billboard() -> TextVisual:
    return TextVisual(
        id="visual:billboard",
        texts=("ASCII", "Δ café"),
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
        rgba=np.array([[255, 0, 0, 255], [0, 0, 255, 192]], dtype=np.uint8),
        font_size_px=np.array([16.0, 24.0], dtype=np.float32),
        font_role=FontRole.SERIF,
        anchor_x=(TextAnchorX.LEFT, TextAnchorX.RIGHT),
        anchor_y=(TextAnchorY.TOP, TextAnchorY.BOTTOM),
        rotation_rad=np.array([0.0, np.pi / 4], dtype=np.float32),
        z_order=9,
    )


def test_matplotlib_billboard_projects_anchors_and_keeps_logical_size() -> None:
    visual = _billboard()
    fig1, ax1 = plt.subplots(dpi=96)
    fig2, ax2 = plt.subplots(dpi=96)
    try:
        first = render_text_visual(ax1, visual, view3d=_view(eye=(3.0, 3.0, 3.0)))
        second = render_text_visual(ax2, visual, view3d=_view(eye=(-3.0, 3.0, 3.0)))

        assert first[0].get_transform() is ax1.transAxes
        assert first[0].get_text() == "ASCII"
        assert first[1].get_text() == "Δ café"
        assert first[0].get_horizontalalignment() == "left"
        assert first[1].get_horizontalalignment() == "right"
        assert first[0].get_verticalalignment() == "top"
        assert first[1].get_verticalalignment() == "bottom"
        assert first[1].get_rotation() == 45.0
        assert first[0].get_zorder() == 9
        np.testing.assert_allclose(
            cast(tuple[float, float, float, float], first[0].get_color()),
            np.asarray((1.0, 0.0, 0.0, 1.0)),
        )
        np.testing.assert_allclose(
            cast(tuple[float, float, float, float], first[1].get_color()),
            np.asarray((0.0, 0.0, 1.0, 192.0 / 255.0)),
        )
        assert first[0].get_fontfamily() == ["serif"]
        assert first[1].get_fontfamily() == ["serif"]
        np.testing.assert_allclose(
            np.asarray([float(item.get_fontsize()) for item in first]),
            np.asarray([12.0, 18.0]),
        )
        np.testing.assert_allclose(
            np.asarray([float(item.get_fontsize()) for item in first]),
            np.asarray([float(item.get_fontsize()) for item in second]),
        )
        assert tuple(item.get_text() for item in first) == ("ASCII", "Δ café")
        assert tuple(item.get_zorder() for item in first) == (9, 9)
        assert first[1].get_position() != second[1].get_position()
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_matplotlib_advertises_billboard_without_depth_or_glyph_parity() -> None:
    caps = capability_snapshot()

    assert caps.supports_view3d_capability("textvisual.billboard3d.v1")
    assert not caps.supports_view3d_capability(
        "textvisual.billboard3d.depth_occlusion.v1"
    )
    assert caps.font_layout_capability.rasterization_parity is False
