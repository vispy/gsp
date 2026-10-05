"""Validation for the bounded retained point-update extension.

This local session extension is deliberately smaller than a command server:
it replaces values on an existing point visual without changing scene topology.
"""

from dataclasses import replace

import numpy as np

from .scene import Scene
from .protocol import PointVisual, VisualTransformBinding

POINT_UPDATE_CAPABILITY = "scene.update.points.v1"


def _same_transform(a: VisualTransformBinding | None, b: VisualTransformBinding | None) -> bool:
    if a is b:
        return True
    if a is None or b is None or a.ref != b.ref:
        return False
    if a.inline is None or b.inline is None:
        return a.inline is b.inline
    return a.inline.kind is b.inline.kind and np.array_equal(a.inline.matrix, b.inline.matrix)


def prepare_point_update(scene: Scene, visual: PointVisual) -> Scene:
    """Validate a value replacement completely before touching a renderer."""
    if not isinstance(visual, PointVisual):
        raise TypeError("update_point requires a PointVisual")
    existing = next((item for item in scene.visuals if item.id == visual.id), None)
    if not isinstance(existing, PointVisual):
        raise ValueError(f"unknown retained point visual {visual.id!r}")
    attachment = scene.attachment_for_visual(visual.id)
    if not attachment.visible:
        raise ValueError("update_point does not support hidden attachments")
    if existing.positions.shape != visual.positions.shape:
        raise ValueError("update_point cannot change point count or dimension")
    if existing.coordinate_space is not visual.coordinate_space:
        raise ValueError("update_point cannot change coordinate space")
    if not _same_transform(existing.transform, visual.transform):
        raise ValueError("update_point cannot change transform binding")
    a, b = existing.color_encoding, visual.color_encoding
    if (a is None) != (b is None) or (
        a is not None
        and b is not None
        and (
            a.slot is not b.slot or a.domain is not b.domain or a.color_scale_id != b.color_scale_id
        )
    ):
        raise ValueError("update_point cannot change scalar color binding")
    return replace(
        scene, visuals=tuple(visual if item.id == visual.id else item for item in scene.visuals)
    )
