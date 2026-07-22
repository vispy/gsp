from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import gsp
from gsp.protocol import CoordinateSpace, PointVisual, View2D


def _scene(coordinate_space: CoordinateSpace) -> gsp.Scene:
    return gsp.Scene(
        id=f"scene:{coordinate_space.value}",
        visuals=(
            PointVisual(
                id=f"visual:{coordinate_space.value}",
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
                sizes=8.0,
                coordinate_space=coordinate_space,
            ),
        ),
        view2d=View2D(
            id="view:main",
            panel_id="panel:main",
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        ),
    )


def _point_display_position(result: object) -> np.ndarray:
    axes = result.axes  # type: ignore[attr-defined]
    collection = axes.collections[0]
    offset = np.asarray(collection.get_offsets()[0], dtype=np.float64)
    return np.asarray(collection.get_offset_transform().transform(offset))


def test_data_artist_follows_native_axes_limit_change() -> None:
    with gsp.open_session("matplotlib", require={"visual.points"}) as session:
        result = session.display(_scene(CoordinateSpace.DATA))
        before = _point_display_position(result)

        result.axes.set_xlim(0.0, 2.0)
        after = _point_display_position(result)
        expected = np.asarray(result.axes.transData.transform((0.0, 0.0)))

        assert after[0] == pytest.approx(expected[0])
        assert after[0] < before[0]


def test_ndc_overlay_stays_fixed_during_native_axes_limit_change() -> None:
    with gsp.open_session("matplotlib", require={"visual.points"}) as session:
        result = session.display(_scene(CoordinateSpace.NDC))
        before = _point_display_position(result)

        result.axes.set_xlim(0.0, 2.0)
        result.axes.set_ylim(-4.0, 4.0)

        np.testing.assert_allclose(_point_display_position(result), before)


def test_native_limits_update_session_owned_canonical_view_and_cleanup() -> None:
    session = gsp.open_session("matplotlib", require={"visual.points"})
    result = session.display(_scene(CoordinateSpace.DATA))
    binding = session._view2d_bindings[result.axes]  # type: ignore[attr-defined]
    initial_revision = binding.view2d_revision
    initial_snapshot = binding.view_snapshot_id

    result.axes.set_xlim(3.0, -2.0)
    result.axes.set_ylim(5.0, -7.0)

    assert binding.view.x_range == pytest.approx((3.0, -2.0))
    assert binding.view.y_range == pytest.approx((5.0, -7.0))
    assert binding.view2d_revision != initial_revision
    assert binding.view_snapshot_id != initial_snapshot

    session.close()
    assert binding.closed


def test_canonical_view_application_advances_once_without_callback_recursion() -> None:
    with gsp.open_session("matplotlib", require={"visual.points"}) as session:
        result = session.display(_scene(CoordinateSpace.DATA))
        binding = session._view2d_bindings[result.axes]  # type: ignore[attr-defined]

        binding.apply_canonical_view(
            View2D(
                id="view:main",
                panel_id="panel:main",
                x_range=(-4.0, 2.0),
                y_range=(8.0, -3.0),
            )
        )

        assert binding.revision_index == 2
        assert result.axes.get_xlim() == pytest.approx((-4.0, 2.0))
        assert result.axes.get_ylim() == pytest.approx((8.0, -3.0))
