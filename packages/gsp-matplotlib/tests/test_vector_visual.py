from __future__ import annotations

from typing import cast

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
    VectorAnchor,
    VectorCap,
    VectorVisual,
    View2D,
    View3D,
    project_view3d_data_point,
)
from gsp_matplotlib.capabilities import capability_snapshot
from gsp_matplotlib.protocol_renderer import render_vector_visual


def test_matplotlib_vector_2d_uses_resolved_endpoints_widths_and_caps() -> None:
    figure, axes = plt.subplots(dpi=100)
    visual = VectorVisual(
        id="vector:mpl-2d",
        positions=np.array([[0.0, 0.0], [2.0, 1.0]], dtype=np.float32),
        vectors=np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float32),
        colors=np.array(
            [[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8
        ),
        widths_px=np.array([2.0, 4.0], dtype=np.float32),
        scale=0.5,
        anchor=VectorAnchor.CENTER,
        start_cap=VectorCap.ROUND,
        end_cap=VectorCap.TRIANGLE_OUT,
    )
    artists = render_vector_visual(
        axes, visual, view=View2D(id="view:2d", panel_id="panel:2d")
    )

    lines = artists[0]
    assert isinstance(lines, matplotlib.collections.LineCollection)
    np.testing.assert_allclose(
        lines.get_segments(),
        [[[-0.5, 0.0], [0.5, 0.0]], [[2.0, 0.5], [2.0, 1.5]]],
    )
    np.testing.assert_allclose(
        lines.get_linewidths(),  # type: ignore[attr-defined]
        np.asarray([1.44, 2.88]),  # type: ignore[arg-type]
    )
    assert len(artists) == 5
    assert all(artist.get_gid() == visual.id for artist in artists)
    plt.close(figure)


def test_matplotlib_vector_3d_projects_line_and_declares_adaptation() -> None:
    figure, axes = plt.subplots()
    view3d = View3D(
        id="view:3d",
        panel_id="panel:3d",
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=PerspectiveProjection3D(near_far=(0.1, 100.0)),
    )
    visual = VectorVisual(
        id="vector:mpl-3d",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        colors=np.array([0, 255, 0, 255], dtype=np.uint8),
    )
    artists = render_vector_visual(axes, visual, view3d=view3d)
    assert len(artists) == 2
    axes_box = axes.get_position()
    width, height = figure.get_size_inches() * figure.dpi
    aspect = float(axes_box.width * width / (axes_box.height * height))
    expected_ndc = np.asarray(
        [
            project_view3d_data_point(
                view3d, endpoint, aspect_ratio=aspect
            )[:2]
            for endpoint in ((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        ]
    )
    line_collection = cast(matplotlib.collections.LineCollection, artists[0])
    np.testing.assert_allclose(
        line_collection.get_segments()[0],
        np.asarray(0.5 * (expected_ndc + 1.0)),
    )

    capabilities = capability_snapshot()
    assert capabilities.supports_visual("vector")
    assert capabilities.supports_view3d_capability("vectorvisual.straight.v1")
    assert capabilities.supports_view3d_capability(
        "vectorvisual.positions3d.data.view3d.v1"
    )
    assert "cap rasterization differs" in cast(
        str, capabilities.metadata["vectorvisual"]
    )
    plt.close(figure)


@pytest.mark.parametrize(
    ("cap", "artist_count"),
    [
        (VectorCap.NONE, 1),
        (VectorCap.BUTT, 1),
        (VectorCap.ROUND, 3),
        (VectorCap.TRIANGLE_IN, 3),
        (VectorCap.TRIANGLE_OUT, 3),
        (VectorCap.SQUARE, 3),
    ],
)
def test_matplotlib_vector_maps_every_cap(
    cap: VectorCap, artist_count: int
) -> None:
    figure, axes = plt.subplots()
    visual = VectorVisual(
        id=f"vector:cap:{cap.value}",
        positions=np.array([[0.0, 0.0]], dtype=np.float32),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
        start_cap=cap,
        end_cap=cap,
    )
    artists = render_vector_visual(
        axes, visual, view=View2D(id="view:cap", panel_id="panel:cap")
    )
    assert len(artists) == artist_count
    assert all(artist.get_gid() == visual.id for artist in artists)
    plt.close(figure)


def test_matplotlib_vector_enforces_2d_and_3d_view_gates() -> None:
    figure, axes = plt.subplots()
    visual2d = VectorVisual(
        id="vector:missing-2d",
        positions=np.array([[0.0, 0.0]], dtype=np.float32),
        vectors=np.array([[1.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
    )
    with pytest.raises(NotImplementedError, match="View2D"):
        render_vector_visual(axes, visual2d)

    visual3d = VectorVisual(
        id="vector:missing-3d",
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        vectors=np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
    )
    with pytest.raises(NotImplementedError, match="DATA space and View3D"):
        render_vector_visual(axes, visual3d)
    with pytest.raises(NotImplementedError, match="DATA space and View3D"):
        render_vector_visual(
            axes,
            VectorVisual(
                id="vector:ndc-3d",
                positions=visual3d.positions,
                vectors=visual3d.vectors,
                colors=visual3d.colors,
                coordinate_space=CoordinateSpace.NDC,
            ),
            view3d=View3D(
                id="view:3d-gate",
                panel_id="panel:3d-gate",
                camera=Camera3D(
                    eye=(3.0, 3.0, 3.0),
                    target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0),
                ),
                projection=PerspectiveProjection3D(near_far=(0.1, 100.0)),
            ),
        )
    plt.close(figure)
