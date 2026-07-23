from __future__ import annotations

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
    PrimitiveTopology,
    PrimitiveVisual,
    VisualTransformBinding,
    View2D,
    View3D,
)
from gsp_matplotlib.capabilities import capability_snapshot
from gsp_matplotlib.protocol_renderer import render_primitive_visual


def _visual(topology: PrimitiveTopology, *, indexed: bool) -> PrimitiveVisual:
    positions = np.array(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float32,
    )
    counts = {
        PrimitiveTopology.POINT_LIST: 4,
        PrimitiveTopology.LINE_LIST: 4,
        PrimitiveTopology.LINE_STRIP: 4,
        PrimitiveTopology.TRIANGLE_LIST: 3,
        PrimitiveTopology.TRIANGLE_STRIP: 4,
    }
    count = counts[topology]
    return PrimitiveVisual(
        id=f"primitive:{topology.value}:{indexed}",
        topology=topology,
        positions=positions[:count],
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
        indices=np.arange(count - 1, -1, -1, dtype=np.uint32) if indexed else None,
    )


@pytest.mark.parametrize("topology", list(PrimitiveTopology))
@pytest.mark.parametrize("indexed", [False, True])
def test_matplotlib_maps_every_topology_indexed_and_unindexed(
    topology: PrimitiveTopology, indexed: bool
) -> None:
    figure, axes = plt.subplots()
    visual = _visual(topology, indexed=indexed)
    (artist,) = render_primitive_visual(
        axes,
        visual,
        view=View2D(id="view:2d", panel_id="panel:2d"),
    )
    assert artist.get_gid() == visual.id
    if topology is PrimitiveTopology.POINT_LIST:
        assert isinstance(artist, matplotlib.collections.PathCollection)
    elif topology in (PrimitiveTopology.LINE_LIST, PrimitiveTopology.LINE_STRIP):
        assert isinstance(artist, matplotlib.collections.LineCollection)
    else:
        assert isinstance(artist, matplotlib.collections.PolyCollection)
    plt.close(figure)


def test_matplotlib_primitive_uses_effective_index_stream_and_vertex_colors() -> None:
    figure, axes = plt.subplots()
    visual = PrimitiveVisual(
        id="primitive:effective-stream",
        topology=PrimitiveTopology.LINE_STRIP,
        positions=np.array(
            [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32
        ),
        colors=np.array(
            [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
            ],
            dtype=np.uint8,
        ),
        indices=np.array([2, 0, 1], dtype=np.uint32),
    )
    (artist,) = render_primitive_visual(
        axes,
        visual,
        view=View2D(id="view:stream", panel_id="panel:stream"),
    )
    assert isinstance(artist, matplotlib.collections.LineCollection)
    np.testing.assert_allclose(
        artist.get_segments(),
        [[[1.0, 1.0], [0.0, 0.0]], [[0.0, 0.0], [1.0, 0.0]]],
    )
    np.testing.assert_allclose(
        artist.get_colors(),  # type: ignore[arg-type]
        [[0.5, 0.0, 0.5, 1.0], [0.5, 0.5, 0.0, 1.0]],  # type: ignore[arg-type]
    )
    plt.close(figure)


def test_matplotlib_primitive_adapts_inline_2d_transform() -> None:
    figure, axes = plt.subplots()
    visual = PrimitiveVisual(
        id="primitive:transform",
        topology=PrimitiveTopology.POINT_LIST,
        positions=np.array([[0.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 255, 255, 255], dtype=np.uint8),
        transform=VisualTransformBinding.inline_affine(
            np.array(
                [[1.0, 0.0, 0.25], [0.0, 1.0, -0.5], [0.0, 0.0, 1.0]]
            )
        ),
    )
    (artist,) = render_primitive_visual(
        axes,
        visual,
        view=View2D(id="view:transform", panel_id="panel:transform"),
    )
    assert isinstance(artist, matplotlib.collections.PathCollection)
    np.testing.assert_allclose(
        artist.get_offsets(),  # type: ignore[arg-type]
        [[0.25, -0.5]],  # type: ignore[arg-type]
    )
    plt.close(figure)


def test_matplotlib_projects_3d_primitives_and_declares_explicit_adaptation() -> None:
    figure, axes = plt.subplots()
    visual = PrimitiveVisual(
        id="primitive:3d",
        topology=PrimitiveTopology.TRIANGLE_LIST,
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        ),
        colors=np.array(
            [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]],
            dtype=np.uint8,
        ),
    )
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
    (artist,) = render_primitive_visual(axes, visual, view3d=view3d)
    assert isinstance(artist, matplotlib.collections.PolyCollection)

    capabilities = capability_snapshot()
    assert capabilities.supports_visual("primitive")
    for capability in (
        "primitivevisual.v1",
        "primitivevisual.indexed.v1",
        "primitivevisual.point_list",
        "primitivevisual.line_list",
        "primitivevisual.line_strip",
        "primitivevisual.triangle_list",
        "primitivevisual.triangle_strip",
    ):
        assert capabilities.supports_view3d_capability(capability)
    assert "adaptation" in str(capabilities.metadata["primitivevisual"])
    plt.close(figure)


def test_matplotlib_primitive_view_and_transform_capability_rejection() -> None:
    figure, axes = plt.subplots()
    with pytest.raises(NotImplementedError, match="primitivevisual_view2d_required"):
        render_primitive_visual(
            axes,
            _visual(PrimitiveTopology.POINT_LIST, indexed=False),
        )
    visual3d = PrimitiveVisual(
        id="primitive:missing-3d",
        topology=PrimitiveTopology.POINT_LIST,
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        colors=np.array([255, 255, 255, 255], dtype=np.uint8),
        coordinate_space=CoordinateSpace.NDC,
    )
    with pytest.raises(NotImplementedError, match="primitivevisual_view3d_required"):
        render_primitive_visual(axes, visual3d)
    with pytest.raises(NotImplementedError, match="primitivevisual_transform_unsupported"):
        render_primitive_visual(
            axes,
            PrimitiveVisual(
                id="primitive:transform-3d",
                topology=PrimitiveTopology.POINT_LIST,
                positions=visual3d.positions,
                colors=visual3d.colors,
                coordinate_space=CoordinateSpace.DATA,
                transform=VisualTransformBinding.inline_affine(
                    np.eye(3, dtype=np.float64)
                ),
            ),
            view3d=View3D(
                id="view:transform-3d",
                panel_id="panel:transform-3d",
                camera=Camera3D(
                    eye=(3.0, 3.0, 3.0),
                    target=(0.0, 0.0, 0.0),
                    up=(0.0, 0.0, 1.0),
                ),
                projection=PerspectiveProjection3D(near_far=(0.1, 100.0)),
            ),
        )
    plt.close(figure)
