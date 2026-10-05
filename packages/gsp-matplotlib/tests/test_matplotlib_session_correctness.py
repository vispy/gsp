"""Cross-family stacking and failed-render resource ownership regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conformance.p038_support import single_panel_scene
from gsp.backends import SessionRequest
from gsp.protocol import (
    CoordinateSpace,
    ExplicitPanelLayoutV1,
    ImageVisual,
    MeshShading,
    MeshUVMode,
    MeshVisual,
    NormalizedRenderTargetRect,
    Panel,
    PanelPlacement,
    PointVisual,
    QueryCoordinateSpace,
    QueryRequest,
    Texture2D,
    View2D,
    VisualAttachment,
)
from gsp_matplotlib.protocol_renderer import render_protocol_scene_with_layout
from gsp_matplotlib.session import MatplotlibSession


@pytest.mark.parametrize("image_z", [10, -10, -20])
def test_attachment_stacking_matches_pixels_and_frontmost_query(image_z: int) -> None:
    point = PointVisual(
        id="visual:point",
        positions=np.array([[0.0, 0.0]], dtype=np.float32),
        colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
        sizes=30.0,
        coordinate_space=CoordinateSpace.DATA,
    )
    image = ImageVisual(
        id="visual:image",
        image=np.full((2, 2, 4), [0, 0, 255, 255], dtype=np.uint8),
        extent=(-1.0, 1.0, -1.0, 1.0),
        coordinate_space=CoordinateSpace.DATA,
    )
    view = View2D(id="view:main", panel_id="panel:main", x_range=(-1.0, 1.0), y_range=(-1.0, 1.0))
    scene = single_panel_scene(
        id="scene:stacking",
        visuals=(point, image),
        view2d=view,
        attachments=(
            VisualAttachment(point.id, view.panel_id, view.id, z_order=-10),
            VisualAttachment(image.id, view.panel_id, view.id, z_order=image_z),
        ),
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        result = session.render(scene)
        assert result.axes.collections[0].get_zorder() == -10
        assert result.axes.images[0].get_zorder() == image_z
        x, y = result.axes.transData.transform((0.0, 0.0))
        pixels = np.asarray(result.figure.canvas.buffer_rgba())
        expected_color = [0, 0, 255, 255] if image_z >= -10 else [255, 0, 0, 255]
        np.testing.assert_array_equal(pixels[pixels.shape[0] - int(y), int(x)], expected_color)
        query = session.query(
            QueryRequest(
                id="query:stacking",
                panel_id=view.panel_id,
                coordinate=(0.0, 0.0),
                coordinate_space=QueryCoordinateSpace.DATA,
            )
        )
        assert query.hits[0].visual_id == (image.id if image_z >= -10 else point.id)


def _unsupported_mesh() -> MeshVisual:
    return MeshVisual(
        id="visual:textured",
        positions=np.array([[-0.5, -0.5], [0.5, -0.5], [0.0, 0.5]], dtype=np.float32),
        faces=np.array([[0, 1, 2]], dtype=np.uint32),
        coordinate_space=CoordinateSpace.NDC,
        color=np.array([255, 255, 255, 255], dtype=np.uint8),
        shading=MeshShading.TEXTURE2D_UNLIT,
        texture2d_id="texture:checker",
        uv_mode=MeshUVMode.VERTEX,
        uvs=np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32),
    )


@pytest.mark.parametrize("later_panel", [False, True])
def test_failed_visual_render_closes_figure_and_preserves_previous_scene(later_panel: bool) -> None:
    before = set(plt.get_fignums())
    mesh = _unsupported_mesh()
    scene = single_panel_scene(
        id="scene:failure",
        visuals=(mesh,),
        textures=(Texture2D("texture:checker", np.zeros((1, 1, 4), dtype=np.uint8)),),
    )
    if later_panel:
        scene = replace(
            scene,
            panels=(Panel("panel:first"), Panel("panel:main")),
            panel_layout=ExplicitPanelLayoutV1(
                (
                    PanelPlacement("panel:first", NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
                    PanelPlacement("panel:main", NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
                )
            ),
        )
    with MatplotlibSession(request=SessionRequest()) as session:
        previous = session.render(single_panel_scene(id="scene:previous"))
        live_figures = set(plt.get_fignums())
        with pytest.raises(NotImplementedError, match="texture2d_unlit_unsupported"):
            session.render(scene)
        assert set(plt.get_fignums()) == live_figures
        assert session._query_target(None)[1] is previous
        assert len(session._results) == 1
    assert set(plt.get_fignums()) == before


def test_failed_target_write_closes_figure_and_bindings(tmp_path: Path) -> None:
    before = set(plt.get_fignums())
    view = View2D(id="view:main", panel_id="panel:main", x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    with MatplotlibSession(request=SessionRequest()) as session:
        with pytest.raises(FileNotFoundError):
            session.render(
                single_panel_scene(id="scene:write-failure", view2d=view),
                target=tmp_path / "missing" / "out.png",
            )
        assert set(plt.get_fignums()) == before
        assert not session._view2d_bindings
        with pytest.raises(RuntimeError, match="requires a rendered scene"):
            session._query_target(None)
        session.render(single_panel_scene(id="scene:recovery"))
    assert set(plt.get_fignums()) == before


def test_standalone_failed_render_closes_only_owned_figures() -> None:
    before = set(plt.get_fignums())
    with pytest.raises(NotImplementedError, match="texture2d_unlit_unsupported"):
        render_protocol_scene_with_layout(visuals=(_unsupported_mesh(),))
    assert set(plt.get_fignums()) == before
    figure = plt.figure()
    try:
        with pytest.raises(NotImplementedError, match="texture2d_unlit_unsupported"):
            render_protocol_scene_with_layout(visuals=(_unsupported_mesh(),), figure=figure)
        assert plt.fignum_exists(figure.number)
    finally:
        plt.close(figure)
