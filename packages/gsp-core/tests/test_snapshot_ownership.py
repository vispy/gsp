"""Scientific data cannot change underneath an accepted semantic snapshot."""

from dataclasses import replace

import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    CoordinateSpace,
    Panel,
    PointVisual,
    ScalarColorEncoding,
    ScalarColorSlot,
    TextVisual,
    Texture2D,
    VisualAttachment,
    VisualTransformBinding,
    full_target_panel_layout,
)


def test_scene_detaches_positions_colors_sizes_and_inline_transform() -> None:
    positions = np.asarray([[0, 0], [1, 1]], dtype=np.float32)
    colors = np.full((2, 4), 255, dtype=np.uint8)
    sizes = np.asarray([4, 8], dtype=np.float32)
    matrix = np.eye(3, dtype=np.float32)
    point = PointVisual(
        "visual:point",
        positions,
        colors,
        sizes,
        transform=VisualTransformBinding.inline_affine(matrix),
    )
    scene = Scene(
        "scene:owned",
        (Panel("panel:main"),),
        full_target_panel_layout("panel:main"),
        visuals=(point,),
        attachments=(VisualAttachment("visual:point", "panel:main"),),
    )
    positions[:] = 99
    colors[:] = 0
    sizes[:] = 100
    matrix[0, 2] = 42
    assert scene.visuals[0] is point
    np.testing.assert_array_equal(point.positions, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(point.colors, np.full((2, 4), 255))
    np.testing.assert_array_equal(point.sizes, [4, 8])
    np.testing.assert_array_equal(point.transform.inline.matrix, np.eye(3))
    for array in (point.positions, point.colors, point.sizes, point.transform.inline.matrix):
        with pytest.raises(ValueError):
            array.flat[0] = 1
        with pytest.raises(ValueError):
            array.flags.writeable = True


def test_readonly_view_does_not_borrow_mutable_owner() -> None:
    owner = np.asarray([[0, 1]], dtype=np.float32)
    view = owner.view()
    view.flags.writeable = False
    point = PointVisual("visual:point", view, np.full((1, 4), 255, dtype=np.uint8))
    owner[:] = 42
    np.testing.assert_array_equal(point.positions, [[0, 1]])
    assert not np.shares_memory(owner, point.positions)


def test_scalar_and_texture_payloads_are_owned_and_replacement_shares_safe_data() -> None:
    values = np.asarray([0, 1], dtype=np.float32)
    scalar = ScalarColorEncoding(ScalarColorSlot.COLOR, values, "scale:main")
    image = np.full((2, 2, 4), 255, dtype=np.uint8)
    texture = Texture2D("texture:main", image)
    values[:] = 10
    image[:] = 0
    np.testing.assert_array_equal(scalar.values, [0, 1])
    np.testing.assert_array_equal(texture.image, np.full((2, 2, 4), 255))
    assert replace(texture, id="texture:copy").image is texture.image
    assert replace(scalar, alpha=0.5).values is scalar.values


def test_text_sequence_is_detached_from_caller_list() -> None:
    labels = ["original"]
    visual = TextVisual(
        "visual:text", labels, np.asarray([[0, 0]], dtype=np.float32), CoordinateSpace.NDC
    )
    labels[0] = "changed"
    assert visual.texts == ("original",)
