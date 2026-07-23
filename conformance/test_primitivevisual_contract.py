import numpy as np

from gsp.protocol import PrimitiveTopology, PrimitiveVisual


def test_primitivevisual_fixture_preserves_topology_and_public_indices() -> None:
    visual = PrimitiveVisual(
        id="primitive:fixture",
        topology=PrimitiveTopology.TRIANGLE_STRIP,
        positions=np.array(
            [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]],
            dtype=np.float32,
        ),
        colors=np.array(
            [
                [255, 0, 0, 255],
                [0, 255, 0, 255],
                [0, 0, 255, 255],
                [255, 255, 255, 255],
            ],
            dtype=np.uint8,
        ),
        indices=np.array([0, 1, 2, 3], dtype=np.uint32),
    )
    assert visual.topology is PrimitiveTopology.TRIANGLE_STRIP
    np.testing.assert_array_equal(visual.index_values(), [0, 1, 2, 3])
