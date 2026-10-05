"""Prepare and apply bounded point updates to existing native collections."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib.collections

from gsp import Scene
from gsp.protocol import PointVisual, View2D
from gsp.updates import prepare_point_update

from .protocol_renderer import (
    MatplotlibProtocolRenderResult,
    _marker_area_values,
    _point_colors,
    _render_positions,
)


def update_point_collection(
    scene: Scene,
    result: MatplotlibProtocolRenderResult,
    visual: PointVisual,
    *,
    view: View2D | None = None,
) -> Scene:
    """Validate and lower all updated data before changing the retained artist."""
    updated = prepare_point_update(scene, visual)
    attachment = scene.attachment_for_visual(visual.id)
    if not attachment.visible:
        raise ValueError("point updates require a visible retained visual")
    axes = result.axes_for_panel(attachment.panel_id)
    artist = next(
        (
            collection
            for collection in axes.collections
            if isinstance(collection, matplotlib.collections.PathCollection)
            and collection.get_gid() == visual.id
        ),
        None,
    )
    if artist is None:
        raise ValueError(f"point visual {visual.id!r} has no retained native collection")
    if view is None:
        active_view = scene.primary_view_for_panel(attachment.panel_id)
        view = active_view if isinstance(active_view, View2D) else None
    offsets, _ = _render_positions(
        axes,
        visual,
        visual.positions,
        view if isinstance(view, View2D) else None,
        {item.id: item for item in scene.transforms},
    )
    colors = _point_colors(visual, color_scales={item.id: item for item in scene.color_scales})
    sizes = _marker_area_values(axes, visual.sizes, visual.positions.shape[0])
    # Matplotlib accepts RGBA arrays; its setter annotation only names sequences.
    rgba = cast(Sequence[tuple[float, float, float, float]], colors)
    artist.set_offsets(offsets)
    artist.set_facecolor(rgba)
    artist.set_edgecolor(rgba)
    artist.set_sizes(sizes)
    return updated
