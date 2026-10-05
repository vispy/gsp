"""Structural changes reject before adapters mutate retained resources."""

from dataclasses import replace

import numpy as np
import pytest

from gsp import Scene
from gsp.protocol import (
    CoordinateSpace,
    Panel,
    PointVisual,
    View2D,
    VisualAttachment,
    VisualTransformBinding,
    full_target_panel_layout,
)
from gsp.updates import prepare_point_update


def snapshot(*, hidden=False, transform=None):
    point = PointVisual(
        "visual:point",
        np.asarray([[0, 0]], dtype=np.float32),
        np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
        coordinate_space=CoordinateSpace.DATA,
        transform=transform,
    )
    return Scene(
        "scene:update",
        (Panel("panel:main"),),
        full_target_panel_layout("panel:main"),
        visuals=(point,),
        views2d=(View2D("view:main", "panel:main"),),
        attachments=(VisualAttachment(point.id, "panel:main", "view:main", visible=not hidden),),
    ), point


def test_replacement_owns_new_values_without_mutating_previous_scene():
    scene, point = snapshot()
    positions = np.asarray([[0.5, 0.25]], dtype=np.float32)
    changed = replace(point, positions=positions, sizes=12)
    updated = prepare_point_update(scene, changed)
    positions[:] = 42
    assert updated.id == scene.id
    assert updated.visuals == (changed,)
    assert updated.attachments is scene.attachments
    assert scene.visuals[0] is point
    np.testing.assert_array_equal(point.positions, [[0, 0]])
    np.testing.assert_array_equal(changed.positions, [[0.5, 0.25]])


@pytest.mark.parametrize("change", ["unknown", "count", "space", "transform", "hidden"])
def test_structural_change_rejects_without_mutating_snapshot(change):
    scene, point = snapshot(hidden=change == "hidden")
    options = {
        "unknown": {"id": "visual:unknown"},
        "count": {
            "positions": np.zeros((2, 2), dtype=np.float32),
            "colors": np.full((2, 4), 255, dtype=np.uint8),
        },
        "space": {"coordinate_space": CoordinateSpace.NDC},
        "transform": {"transform": VisualTransformBinding.inline_affine(np.eye(3))},
        "hidden": {},
    }
    with pytest.raises(ValueError):
        prepare_point_update(scene, replace(point, **options[change]))
    assert scene.visuals[0] is point


def test_equivalent_inline_transform_is_accepted_without_array_truth_comparison():
    scene, point = snapshot(transform=VisualTransformBinding.inline_affine(np.eye(3)))
    changed = replace(point, transform=VisualTransformBinding.inline_affine(np.eye(3)))
    assert prepare_point_update(scene, changed).visuals[0] is changed
