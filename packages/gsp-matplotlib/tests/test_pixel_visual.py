import matplotlib

matplotlib.use("Agg")

import matplotlib.collections
import matplotlib.pyplot as plt
import numpy as np
import pytest

from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    PerspectiveProjection3D,
    PixelVisual,
    VisualTransformBinding,
    View3D,
)
from gsp_matplotlib.capabilities import capability_snapshot
from gsp_matplotlib.protocol_renderer import render_pixel_visual


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


def test_matplotlib_pixel_visual_renders_square_logical_sizes_and_colors() -> None:
    figure, axes = plt.subplots(dpi=100)
    try:
        visual = PixelVisual(
            id="pixel:2d",
            positions=np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            colors=np.array(
                [[255, 0, 0, 255], [0, 255, 0, 128]], dtype=np.uint8
            ),
            pixel_size_px=np.array([4.0, 8.0], dtype=np.float32),
        )
        artist = render_pixel_visual(axes, visual)
        assert isinstance(artist, matplotlib.collections.PathCollection)
        assert artist.get_gid() == "pixel:2d"
        np.testing.assert_allclose(artist.get_sizes(), [8.2944, 33.1776])
        np.testing.assert_allclose(
            artist.get_facecolors(),
            [[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 128 / 255]],
        )
    finally:
        plt.close(figure)


def test_matplotlib_pixel_visual_projects_3d_anchor_as_square_overlay() -> None:
    figure, axes = plt.subplots()
    try:
        visual = PixelVisual(
            id="pixel:3d",
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            colors=np.array([255, 255, 255, 255], dtype=np.uint8),
            pixel_size_px=6.0,
            coordinate_space=CoordinateSpace.DATA,
        )
        artist = render_pixel_visual(axes, visual, view3d=_view3d())
        np.testing.assert_allclose(artist.get_offsets(), [[0.5, 0.5]])
        np.testing.assert_allclose(artist.get_sizes(), [18.6624])
        np.testing.assert_allclose(artist.get_facecolors(), [[1.0, 1.0, 1.0, 1.0]])
    finally:
        plt.close(figure)


def test_matplotlib_pixel_visual_rejects_transform_on_3d_anchors() -> None:
    figure, axes = plt.subplots()
    try:
        visual = PixelVisual(
            id="pixel:3d-transform",
            positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
            colors=np.array([255, 255, 255, 255], dtype=np.uint8),
            transform=VisualTransformBinding.inline_affine(
                np.eye(3, dtype=np.float64)
            ),
        )
        with pytest.raises(NotImplementedError, match="2D transform"):
            render_pixel_visual(axes, visual, view3d=_view3d())
    finally:
        plt.close(figure)


def test_matplotlib_capabilities_advertise_pixel_strict_and_adapted_scope() -> None:
    capabilities = capability_snapshot()
    assert capabilities.supports_visual("pixel")
    assert capabilities.supports_view3d_capability("pixelvisual.v1")
    assert capabilities.supports_view3d_capability(
        "pixelvisual.positions3d.data.view3d.v1"
    )
    assert capabilities.supports_view3d_capability(
        "pixelvisual.exact_logical_size.v1"
    )
    assert "adapted projected-square" in capabilities.metadata["pixelvisual_3d"]
