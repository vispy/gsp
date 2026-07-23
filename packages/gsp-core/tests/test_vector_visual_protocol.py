import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    PerspectiveProjection3D,
    VectorAnchor,
    VectorCap,
    VectorVisual,
    View2D,
    View3D,
)


def _vectors(**kwargs: object) -> VectorVisual:
    values: dict[str, object] = {
        "id": "vector:test",
        "positions": np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
        "vectors": np.array([[2.0, 0.0], [0.0, -1.0]], dtype=np.float32),
        "colors": np.array([255, 0, 0, 255], dtype=np.uint8),
    }
    values.update(kwargs)
    return VectorVisual(**values)  # type: ignore[arg-type]


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:3d",
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=PerspectiveProjection3D(near_far=(0.1, 100.0)),
    )


@pytest.mark.parametrize(
    ("anchor", "expected_tail", "expected_head"),
    [
        (VectorAnchor.TAIL, [0.0, 0.0], [4.0, 0.0]),
        (VectorAnchor.CENTER, [-2.0, 0.0], [2.0, 0.0]),
        (VectorAnchor.HEAD, [-4.0, 0.0], [0.0, 0.0]),
    ],
)
def test_vector_anchor_and_scale_resolve_canonical_endpoints(
    anchor: VectorAnchor, expected_tail: list[float], expected_head: list[float]
) -> None:
    visual = _vectors(
        positions=np.array([[0.0, 0.0]], dtype=np.float32),
        vectors=np.array([[2.0, 0.0]], dtype=np.float32),
        scale=2.0,
        anchor=anchor,
    )
    tails, heads = visual.endpoint_values()
    np.testing.assert_allclose(tails[0], expected_tail)
    np.testing.assert_allclose(heads[0], expected_head)


def test_vector_visual_accepts_caps_widths_colors_and_dimensions() -> None:
    visual = _vectors(
        widths_px=np.array([2.0, 5.0], dtype=np.float32),
        start_cap=VectorCap.ROUND,
        end_cap=VectorCap.SQUARE,
        colors=np.array(
            [[255, 0, 0, 255], [0, 255, 0, 128]], dtype=np.uint8
        ),
    )
    np.testing.assert_array_equal(visual.width_values(), [2.0, 5.0])
    Scene(
        id="scene:2d",
        visuals=(visual,),
        view2d=View2D(id="view:2d", panel_id="panel:2d"),
    )

    visual3d = _vectors(
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
        vectors=np.array([[0.0, 0.0, 1.0]], dtype=np.float64),
        colors=np.array([0.0, 0.5, 1.0, 1.0], dtype=np.float32),
    )
    Scene(id="scene:3d", visuals=(visual3d,), view3d=_view3d())
    with pytest.raises(ValueError, match="Scene.view3d"):
        Scene(id="scene:missing-3d", visuals=(visual3d,))
    with pytest.raises(ValueError, match="CoordinateSpace.DATA"):
        Scene(
            id="scene:ndc-3d",
            visuals=(
                _vectors(
                    positions=visual3d.positions,
                    vectors=visual3d.vectors,
                    coordinate_space=CoordinateSpace.NDC,
                ),
            ),
            view3d=_view3d(),
        )
    with pytest.raises(ValueError, match="Scene.view2d"):
        Scene(id="scene:missing-2d", visuals=(visual,))


@pytest.mark.parametrize("cap", list(VectorCap))
def test_vector_visual_preserves_every_cap(cap: VectorCap) -> None:
    visual = _vectors(start_cap=cap, end_cap=cap)
    assert visual.start_cap is cap
    assert visual.end_cap is cap


@pytest.mark.parametrize(
    ("anchor", "expected_tail", "expected_head"),
    [
        (VectorAnchor.TAIL, [1.0, 2.0, 3.0], [3.0, 0.0, 4.0]),
        (VectorAnchor.CENTER, [0.0, 3.0, 2.5], [2.0, 1.0, 3.5]),
        (VectorAnchor.HEAD, [-1.0, 4.0, 2.0], [1.0, 2.0, 3.0]),
    ],
)
def test_vector_visual_resolves_3d_endpoints_exactly(
    anchor: VectorAnchor,
    expected_tail: list[float],
    expected_head: list[float],
) -> None:
    visual = _vectors(
        positions=np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        vectors=np.array([[2.0, -2.0, 1.0]], dtype=np.float64),
        colors=np.array([255, 0, 0, 255], dtype=np.uint8),
        scale=1.0,
        anchor=anchor,
    )
    tails, heads = visual.endpoint_values()
    np.testing.assert_array_equal(tails[0], expected_tail)
    np.testing.assert_array_equal(heads[0], expected_head)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        (
            "vectors",
            np.array([[1.0, 0.0]], dtype=np.float32),
            "same shape",
        ),
        (
            "vectors",
            np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            "nonzero",
        ),
        (
            "vectors",
            np.array([[np.nan, 0.0], [0.0, 1.0]], dtype=np.float32),
            "finite",
        ),
        ("widths_px", 0.0, "positive"),
        ("widths_px", np.inf, "finite"),
        ("scale", 0.0, "positive"),
        ("scale", np.inf, "finite"),
        ("anchor", "tail", "VectorAnchor"),
        ("end_cap", "triangle_out", "VectorCap"),
    ],
)
def test_vector_visual_rejects_invalid_fields(
    field: str, value: object, error: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _vectors(**{field: value})
