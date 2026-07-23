from dataclasses import fields
import inspect

import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    PerspectiveProjection3D,
    PrimitiveTopology,
    PrimitiveVisual,
    View2D,
    View3D,
)


def _primitive(**kwargs: object) -> PrimitiveVisual:
    values: dict[str, object] = {
        "id": "primitive:test",
        "topology": PrimitiveTopology.TRIANGLE_LIST,
        "positions": np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32
        ),
        "colors": np.array([255, 0, 0, 255], dtype=np.uint8),
    }
    values.update(kwargs)
    return PrimitiveVisual(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("topology", "positions"),
    [
        (PrimitiveTopology.POINT_LIST, [[0.0, 0.0]]),
        (PrimitiveTopology.LINE_LIST, [[0.0, 0.0], [1.0, 1.0]]),
        (PrimitiveTopology.LINE_STRIP, [[0.0, 0.0], [1.0, 1.0]]),
        (
            PrimitiveTopology.TRIANGLE_LIST,
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ),
        (
            PrimitiveTopology.TRIANGLE_STRIP,
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ),
    ],
)
def test_primitive_visual_accepts_every_bounded_topology(
    topology: PrimitiveTopology, positions: list[list[float]]
) -> None:
    visual = _primitive(
        topology=topology,
        positions=np.asarray(positions, dtype=np.float64),
        colors=np.tile(
            np.array([255, 0, 0, 255], dtype=np.uint8), (len(positions), 1)
        ),
    )
    assert visual.topology is topology
    np.testing.assert_array_equal(
        visual.resolved_vertex_indices(), np.arange(len(positions))
    )


@pytest.mark.parametrize("topology", list(PrimitiveTopology))
def test_primitive_visual_applies_cardinality_after_indices(
    topology: PrimitiveTopology,
) -> None:
    counts = {
        PrimitiveTopology.POINT_LIST: 1,
        PrimitiveTopology.LINE_LIST: 2,
        PrimitiveTopology.LINE_STRIP: 2,
        PrimitiveTopology.TRIANGLE_LIST: 3,
        PrimitiveTopology.TRIANGLE_STRIP: 3,
    }
    count = counts[topology]
    visual = _primitive(
        topology=topology,
        positions=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        indices=np.arange(count, dtype=np.uint16),
        colors=np.array([255, 255, 255, 255], dtype=np.uint8),
    )
    assert visual.index_values() is not None
    assert visual.index_values().dtype == np.uint32  # type: ignore[union-attr]


@pytest.mark.parametrize(
    ("topology", "count"),
    [
        (PrimitiveTopology.POINT_LIST, 0),
        (PrimitiveTopology.LINE_LIST, 3),
        (PrimitiveTopology.LINE_STRIP, 1),
        (PrimitiveTopology.TRIANGLE_LIST, 4),
        (PrimitiveTopology.TRIANGLE_STRIP, 2),
    ],
)
def test_primitive_visual_rejects_malformed_cardinality(
    topology: PrimitiveTopology, count: int
) -> None:
    with pytest.raises(ValueError, match="primitivevisual_invalid_cardinality"):
        _primitive(
            topology=topology,
            positions=np.arange(max(count, 3) * 2, dtype=np.float32).reshape(
                max(count, 3), 2
            ),
            colors=np.array([255, 0, 0, 255], dtype=np.uint8),
            indices=np.arange(count, dtype=np.int32),
        )


@pytest.mark.parametrize(
    ("indices", "diagnostic"),
    [
        (np.array([[0, 1, 2]], dtype=np.int32), "invalid_indices_shape"),
        (np.array([0.0, 1.0, 2.0]), "noninteger_index"),
        (np.array([True, False, True]), "noninteger_index"),
        (np.array([0.0, np.nan, 2.0]), "nonfinite_index"),
        (np.array([0, -1, 2]), "negative_index"),
        (np.array([0, 1, 3]), "index_out_of_range"),
    ],
)
def test_primitive_visual_rejects_invalid_indices(
    indices: np.ndarray, diagnostic: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=f"primitivevisual_{diagnostic}"):
        _primitive(indices=indices)


def test_primitive_visual_validates_dimensions_colors_views_and_topology_type() -> None:
    with pytest.raises(ValueError, match="shape"):
        _primitive(positions=np.zeros((3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="colors"):
        _primitive(colors=np.zeros((2, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="finite"):
        _primitive(colors=np.array([0.0, np.nan, 0.0, 1.0], dtype=np.float32))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        _primitive(colors=np.array([0.0, 2.0, 0.0, 1.0], dtype=np.float32))
    with pytest.raises(TypeError, match="PrimitiveTopology"):
        _primitive(topology="triangle_list")

    visual2d = _primitive()
    Scene(
        id="scene:2d",
        visuals=(visual2d,),
        view2d=View2D(id="view:2d", panel_id="panel:2d"),
    )
    with pytest.raises(ValueError, match="Scene.view2d"):
        Scene(id="scene:missing-2d", visuals=(visual2d,))

    visual3d = _primitive(
        positions=np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            dtype=np.float32,
        )
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
    Scene(id="scene:3d", visuals=(visual3d,), view3d=view3d)
    with pytest.raises(ValueError, match="Scene.view3d"):
        Scene(id="scene:missing-3d", visuals=(visual3d,))
    with pytest.raises(ValueError, match="CoordinateSpace.DATA"):
        Scene(
            id="scene:ndc-3d",
            visuals=(
                _primitive(
                    positions=visual3d.positions,
                    coordinate_space=CoordinateSpace.NDC,
                ),
            ),
            view3d=view3d,
        )


def test_primitive_visual_preserves_effective_index_stream_exactly() -> None:
    visual = _primitive(
        positions=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        indices=np.array([3, 1, 3], dtype=np.int16),
    )
    np.testing.assert_array_equal(visual.index_values(), [3, 1, 3])
    np.testing.assert_array_equal(visual.resolved_vertex_indices(), [3, 1, 3])


def test_primitive_visual_public_surface_has_no_raw_gpu_fields() -> None:
    expected = {
        "id",
        "topology",
        "positions",
        "colors",
        "indices",
        "coordinate_space",
        "transform",
    }
    assert {field.name for field in fields(PrimitiveVisual)} == expected
    assert set(inspect.signature(PrimitiveVisual).parameters) == expected
    forbidden = {
        "shader",
        "pipeline",
        "slot",
        "material",
        "normals",
        "textures",
        "depth",
        "culling",
        "instance",
        "native_handle",
    }
    assert expected.isdisjoint(forbidden)
