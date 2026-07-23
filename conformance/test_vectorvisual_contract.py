from __future__ import annotations

import numpy as np

from gsp import Scene
from gsp.protocol import (
    VectorAnchor,
    VectorCap,
    VectorVisual,
    View2D,
)


def test_vectorvisual_fixture_preserves_resolved_semantics() -> None:
    visual = VectorVisual(
        id="vector:fixture",
        positions=np.array([[-1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        vectors=np.array([[1.0, 1.0], [-1.0, 1.0]], dtype=np.float32),
        colors=np.array(
            [[255, 64, 32, 255], [32, 128, 255, 192]], dtype=np.uint8
        ),
        widths_px=np.array([2.0, 4.0], dtype=np.float32),
        scale=0.5,
        anchor=VectorAnchor.CENTER,
        start_cap=VectorCap.ROUND,
        end_cap=VectorCap.TRIANGLE_OUT,
    )
    scene = Scene(
        id="scene:vector-fixture",
        visuals=(visual,),
        view2d=View2D(id="view:vector-fixture", panel_id="panel:fixture"),
    )

    tails, heads = visual.endpoint_values()
    np.testing.assert_allclose(tails, [[-1.25, -0.25], [1.25, -0.25]])
    np.testing.assert_allclose(heads, [[-0.75, 0.25], [0.75, 0.25]])
    np.testing.assert_array_equal(visual.width_values(), [2.0, 4.0])
    assert scene.visuals == (visual,)
