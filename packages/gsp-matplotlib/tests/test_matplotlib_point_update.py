"""Retained point updates preserve native resources and semantic query state."""

from __future__ import annotations

from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conformance.p038_support import single_panel_scene
from gsp.backends import SessionRequest
from gsp.protocol import (
    CoordinateSpace,
    PointVisual,
    QueryCoordinateSpace,
    QueryRequest,
    View2D,
    ColorMapId,
    ColorMapRef,
    ColorScale,
    LinearNormalize,
    ScalarColorEncoding,
    ScalarColorSlot,
)
from gsp_matplotlib.color_mapping import map_scalar_values
from gsp_matplotlib.plugin import MatplotlibProvider
from gsp_matplotlib.session import MatplotlibSession


def _point(x: float = -0.5) -> PointVisual:
    return PointVisual(
        id="visual:points",
        positions=np.array([[x, 0.0]], dtype=np.float32),
        colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
        sizes=10.0,
        coordinate_space=CoordinateSpace.DATA,
    )


def _scene(point: PointVisual, scene_id: str = "scene:points"):
    return single_panel_scene(
        id=scene_id,
        visuals=(point,),
        view2d=View2D(
            id="view:main", panel_id="panel:main", x_range=(-1.0, 1.0), y_range=(-1.0, 1.0)
        ),
    )


def _query(session: MatplotlibSession, x: float, scene_id: str | None = None):
    return session.query(
        QueryRequest(
            id="query:point",
            panel_id="panel:main",
            coordinate=(x, 0.0),
            coordinate_space=QueryCoordinateSpace.DATA,
        ),
        scene_id=scene_id,
    )


def test_point_update_preserves_figure_artist_and_updates_queries() -> None:
    original = _point()
    colors = np.array([[0, 0, 255, 255]], dtype=np.uint8)
    positions = np.array([[0.5, 0.0]], dtype=np.float32)
    changed = replace(original, positions=positions, colors=colors, sizes=20.0)
    with MatplotlibSession(request=SessionRequest()) as session:
        result = session.render(_scene(original))
        artist = result.axes.collections[0]
        figures = set(plt.get_fignums())
        previous_areas = artist.get_sizes().copy()
        assert session.scene_revision() == 0
        assert _query(session, -0.5).hit
        assert session.update_point(changed) == 1
        assert session.scene_revision() == 1
        assert result.axes.collections[0] is artist
        assert session._query_target(None)[1] is result
        assert set(plt.get_fignums()) == figures
        np.testing.assert_allclose(artist.get_offsets(), positions)
        np.testing.assert_allclose(artist.get_facecolors(), [[0.0, 0.0, 1.0, 1.0]])
        np.testing.assert_allclose(artist.get_sizes(), previous_areas * 4)
        assert _query(session, 0.5).hit
        assert not _query(session, -0.5).hit
        positions[:] = 0.0
        colors[:] = 0
        stored = session._query_target(None)[0].visuals[0]
        assert not stored.positions.flags.writeable
        assert _query(session, 0.5).hit
        assert session.update_point(_point()) == 2


def test_revisions_are_independent_and_increment_on_rerender() -> None:
    with MatplotlibSession(request=SessionRequest()) as session:
        session.render(_scene(_point(), "scene:first"))
        session.render(_scene(_point(), "scene:second"))
        assert session.scene_revision() == 0
        assert session.update_point(_point(0.5), scene_id="scene:first") == 1
        assert session.scene_revision("scene:first") == 1
        assert session.scene_revision("scene:second") == 0
        assert _query(session, 0.5, "scene:first").hit
        assert _query(session, -0.5, "scene:second").hit
        session.render(_scene(_point(), "scene:first"))
        assert session.scene_revision() == 2


@pytest.mark.parametrize("structural_change", ["count", "dimension", "coordinates"])
def test_rejected_point_update_preserves_native_and_semantic_state(structural_change: str) -> None:
    original = _point()
    if structural_change == "count":
        changed = replace(
            original,
            positions=np.zeros((2, 2), dtype=np.float32),
            colors=np.full((2, 4), 255, dtype=np.uint8),
        )
    elif structural_change == "dimension":
        changed = replace(original, positions=np.zeros((1, 3), dtype=np.float32))
    else:
        changed = replace(original, coordinate_space=CoordinateSpace.NDC)
    with MatplotlibSession(request=SessionRequest()) as session:
        result = session.render(_scene(original))
        offsets = result.axes.collections[0].get_offsets().copy()
        with pytest.raises(ValueError):
            session.update_point(changed)
        assert session.scene_revision() == 0
        np.testing.assert_array_equal(result.axes.collections[0].get_offsets(), offsets)
        assert _query(session, -0.5).hit


def test_point_update_rejects_unknown_scene_visual_and_closed_session() -> None:
    session = MatplotlibSession(request=SessionRequest())
    with pytest.raises(RuntimeError, match="requires a rendered scene"):
        session.update_point(_point())
    session.render(_scene(_point()))
    with pytest.raises(RuntimeError, match="has not been rendered"):
        session.update_point(_point(), scene_id="scene:missing")
    with pytest.raises((ValueError, KeyError)):
        session.update_point(replace(_point(), id="visual:missing"))
    with pytest.raises(TypeError, match="PointVisual"):
        session.update_point(object())  # type: ignore[arg-type]
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.update_point(_point())
    with pytest.raises(RuntimeError, match="closed"):
        session.scene_revision()


def test_point_update_capability_is_discoverable() -> None:
    provider = MatplotlibProvider()
    assert "scene.update.points.v1" in provider.describe().declared_capabilities
    assert "scene.update.points.v1" in provider.probe().capabilities


def test_scalar_point_update_remaps_colors_and_alpha_in_place() -> None:
    scale = ColorScale(
        id="scale:main",
        colormap=ColorMapRef(ColorMapId.VIRIDIS),
        normalize=LinearNormalize(vmin=0.0, vmax=1.0),
    )
    encoding = ScalarColorEncoding(
        slot=ScalarColorSlot.COLOR,
        values=np.array([0.0], dtype=np.float32),
        color_scale_id=scale.id,
    )
    original = replace(_point(), colors=None, color_encoding=encoding)
    changed_encoding = replace(encoding, values=np.array([1.0], dtype=np.float32), alpha=0.5)
    changed = replace(original, color_encoding=changed_encoding)
    with MatplotlibSession(request=SessionRequest()) as session:
        result = session.render(replace(_scene(original), color_scales=(scale,)))
        artist = result.axes.collections[0]
        assert session.update_point(changed) == 1
        assert result.axes.collections[0] is artist
        np.testing.assert_allclose(
            artist.get_facecolors(), map_scalar_values(changed_encoding.values, scale, alpha=0.5)
        )
        with pytest.raises(ValueError, match="scalar"):
            session.update_point(_point())
        assert session.scene_revision() == 1


@pytest.mark.parametrize("dpi", [96.0, 192.0])
@pytest.mark.parametrize("device_scale", [1.0, 2.0])
@pytest.mark.parametrize("coordinate_space", [CoordinateSpace.DATA, CoordinateSpace.NDC])
def test_public_point_footprint_uses_screen_diameter_across_zoom_and_dpi(
    dpi: float, device_scale: float, coordinate_space: CoordinateSpace
) -> None:
    from gsp.protocol import CanvasSize

    point = replace(_point(0.0), sizes=20.0, coordinate_space=coordinate_space)
    scene = replace(
        _scene(point),
        canvas_size=CanvasSize.reference_px(400, 300).with_requested_device_scale(device_scale),
    )
    with MatplotlibSession(request=SessionRequest()) as session:
        result = session.render(scene, output_dpi=dpi)
        artist = result.axes.collections[0]
        for x_range, y_range in [((-1.0, 1.0), (-1.0, 1.0)), ((-100.0, 100.0), (-2.0, 2.0))]:
            result.axes.set_xlim(x_range)
            result.axes.set_ylim(y_range)
            result.figure.canvas.draw()
            center = artist.get_offset_transform().transform(artist.get_offsets())[0]
            px_scale = result.resolved_canvas.framebuffer_per_canvas_px
            target = result.layout_snapshot.render_target
            for offset, expected in [(5.0, True), (16.0, False)]:
                display = center + [offset * px_scale, 0.0]
                data_xy = tuple(result.axes.transData.inverted().transform(display))
                panel_xy = (
                    float(display[0] * target.logical_width_px / result.figure.bbox.width),
                    float(
                        target.logical_height_px
                        - display[1] * target.logical_height_px / result.figure.bbox.height
                    ),
                )
                for query_space, xy in [
                    (QueryCoordinateSpace.DATA, data_xy),
                    (QueryCoordinateSpace.PANEL, panel_xy),
                ]:
                    queried = session.query(
                        QueryRequest(
                            id="query:footprint",
                            panel_id="panel:main",
                            coordinate=xy,
                            coordinate_space=query_space,
                        )
                    )
                    assert queried.hit is expected


def test_point_update_after_navigation_uses_current_view_for_query_payloads() -> None:
    from gsp.protocol import InlineAffineTransform2D, TransformQueryPayload, VisualTransformBinding

    transform = VisualTransformBinding(
        inline=InlineAffineTransform2D(matrix=np.eye(3, dtype=np.float64))
    )
    original = replace(_point(0.0), transform=transform)
    with MatplotlibSession(request=SessionRequest()) as session:
        result = session.render(_scene(original))
        result.axes.set_xlim(0.0, 2.0)
        result.axes.set_ylim(-2.0, 2.0)
        changed = replace(original, positions=np.array([[1.0, 0.0]], dtype=np.float32))
        session.update_point(changed)
        np.testing.assert_allclose(result.axes.collections[0].get_offsets(), [[1.0, 0.0]])
        queried = _query(session, 1.0)
        assert queried.hit
        assert isinstance(queried.extension_payload, TransformQueryPayload)
        assert queried.extension_payload.plot_ndc == (0.0, 0.0)
        assert session._view2d_bindings[result.axes].view.x_range == (0.0, 2.0)
