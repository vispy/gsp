"""Public point queries in the actual native screen-space marker footprint."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib.collections
import numpy as np

from gsp import Scene
from gsp.protocol import (
    PointVisual,
    QueryCoordinateSpace,
    QueryRequest,
    QueryResult,
    QueryStatus,
    SCALAR_COLOR_QUERY_PAYLOAD_KIND,
    View2D,
    VisualFamily,
)

from .protocol_query import (
    _point_color_query_payload,
    _point_query_value,
    _transform_or_existing_payload,
)
from .protocol_renderer import MatplotlibProtocolRenderResult
from .transforms import transformed_positions


def native_point_query(
    scene: Scene,
    result: MatplotlibProtocolRenderResult,
    visual: PointVisual,
    view: View2D | None,
) -> Callable[[QueryRequest], QueryResult | None]:
    """Build a query evaluator sharing the existing native marker geometry."""
    attachment = scene.attachment_for_visual(visual.id)
    axes = result.axes_for_panel(attachment.panel_id)
    artist = next(
        collection
        for collection in axes.collections
        if isinstance(collection, matplotlib.collections.PathCollection)
        and collection.get_gid() == visual.id
    )
    transforms = {item.id: item for item in scene.transforms}
    scales = {item.id: item for item in scene.color_scales}

    def query(request: QueryRequest) -> QueryResult | None:
        target = result.layout_snapshot.render_target
        if request.coordinate_space is QueryCoordinateSpace.PANEL:
            x, y = request.coordinate
            display = np.array(
                [
                    x * result.figure.bbox.width / target.logical_width_px,
                    (target.logical_height_px - y)
                    * result.figure.bbox.height
                    / target.logical_height_px,
                ]
            )
        else:
            display = axes.transData.transform(request.coordinate)
        # Native PLOT clipping excludes markers outside the axes viewport.
        if not axes.bbox.contains(*display):
            return None
        offsets = artist.get_offset_transform().transform(artist.get_offsets())
        distances = np.linalg.norm(offsets - display, axis=1)
        areas = np.resize(artist.get_sizes(), len(offsets))
        widths = np.resize(artist.get_linewidth(), len(offsets))
        # Scatter scales its circle path by sqrt(area); the path diameter is one.
        # Include the stroke centered on that path boundary.
        radii = (np.sqrt(areas) + widths) * result.figure.dpi / 144.0
        eligible = np.flatnonzero((areas > 0) & (distances <= radii))
        if not len(eligible):
            return None
        index = int(eligible[np.argmin(distances[eligible])])
        positions = transformed_positions(visual.positions, visual.transform, transforms)
        point = tuple(float(value) for value in positions[index])
        source = tuple(float(value) for value in visual.positions[index, :2])
        rgba, payload = _point_color_query_payload(visual, index, color_scales=scales)
        kind, extension = _transform_or_existing_payload(
            visual,
            request,
            (point[0], point[1]),
            (source[0], source[1]),
            view,
            existing_kind=SCALAR_COLOR_QUERY_PAYLOAD_KIND if payload else None,
            existing_payload=payload,
        )
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.HIT,
            hit=True,
            panel_coordinate=request.coordinate,
            visual_id=visual.id,
            visual_family=VisualFamily.POINT,
            item_id=index,
            visual_coordinate=(point[0], point[1]),
            data_coordinate=(point[0], point[1]),
            displayed_rgba=rgba,
            value=_point_query_value(visual, index),
            extension_payload_kind=kind,
            extension_payload=extension,
        )

    return query
