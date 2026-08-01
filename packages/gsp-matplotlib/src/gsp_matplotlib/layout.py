"""Resolved layout snapshot extraction for Matplotlib reference figures."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math
from typing import cast

import matplotlib.axes
import matplotlib.backend_bases
import matplotlib.figure
import matplotlib.transforms
import numpy as np

from gsp.protocol import (
    AxisDimension,
    AxisGuide,
    LayoutDiagnostic,
    LayoutDiagnosticStatus,
    LayoutLayer,
    LogicalPixelRect,
    PixelOrigin,
    RenderTarget,
    ResolvedCanvas,
    ResolvedGuideBox,
    ResolvedLayoutSnapshot,
    ResolvedPanelLayout,
    View2D,
    View3D,
    quantize_logical_rect,
)
from gsp.protocol.guides import PanelTextGuide, PanelTextRole


@dataclass(frozen=True, slots=True)
class _LayoutMetrics:
    logical_width_px: float
    logical_height_px: float
    display_width_px: float
    display_height_px: float
    display_per_logical_x: float
    display_per_logical_y: float
    device_scale: float
    dpi: float


def resolve_matplotlib_layout_snapshot(
    figure: matplotlib.figure.Figure,
    axes: matplotlib.axes.Axes,
    *,
    snapshot_id: str,
    panel_id: str = "panel:default",
    view: View2D | None = None,
    view3d: View3D | None = None,
    axis_guides: Iterable[AxisGuide] = (),
    panel_text_guides: Iterable[PanelTextGuide] = (),
    device_scale: float = 1.0,
    panel_rect_px: LogicalPixelRect | None = None,
) -> ResolvedLayoutSnapshot:
    """Resolve a GSP layout snapshot from a drawn Matplotlib reference axes."""
    figure.canvas.draw()
    renderer = cast(
        matplotlib.backend_bases.RendererBase,
        getattr(figure.canvas, "get_renderer")(),
    )
    metrics = _resolve_layout_metrics(figure, device_scale=device_scale)
    render_target = RenderTarget(
        logical_width_px=metrics.logical_width_px,
        logical_height_px=metrics.logical_height_px,
        device_scale=metrics.device_scale,
        dpi=metrics.dpi,
        pixel_origin=PixelOrigin.TOP_LEFT,
        query_coordinate_space="plot",
    )
    panel_rect = panel_rect_px or LogicalPixelRect(
        0.0, 0.0, metrics.logical_width_px, metrics.logical_height_px
    )
    native_plot_rect = _rect_from_bbox(axes.get_window_extent(renderer), metrics)
    plot_rect = quantize_logical_rect(native_plot_rect, render_target)
    axis_guides_tuple = tuple(axis_guides)
    panel_text_guides_tuple = tuple(panel_text_guides)
    axis_label_boxes = _axis_label_boxes(axes, axis_guides_tuple, metrics, renderer)
    tick_label_boxes = _tick_label_boxes(axes, axis_guides_tuple, metrics, renderer)
    title_boxes = _title_boxes(axes, panel_text_guides_tuple, metrics, renderer)
    guide_boxes = axis_label_boxes + tick_label_boxes + title_boxes
    return ResolvedLayoutSnapshot(
        snapshot_id=snapshot_id,
        render_target=render_target,
        panels=(
            ResolvedPanelLayout(
                panel_id=panel_id,
                panel_rect_px=panel_rect,
                plot_rect_px=plot_rect,
                view_id=(
                    view.id if view is not None else view3d.id if view3d is not None else None
                ),
                data_to_screen_transform=_data_to_top_left_transform(
                    axes, metrics, native_plot_rect, plot_rect
                ),
                guide_boxes=guide_boxes,
                tick_label_boxes=tick_label_boxes,
                axis_label_boxes=axis_label_boxes,
                title_boxes=title_boxes,
                grid_clip_rect_px=plot_rect,
                z_layers=_layout_layers(axis_guides_tuple, panel_text_guides_tuple),
                diagnostics=(
                    LayoutDiagnostic(
                        code="matplotlib_native_layout_resolved",
                        status=LayoutDiagnosticStatus.RESOLVED,
                        message="Snapshot extracted from Matplotlib native artist layout after draw.",
                    ),
                ),
            ),
        ),
    )


def _axis_label_boxes(
    axes: matplotlib.axes.Axes,
    guides: tuple[AxisGuide, ...],
    metrics: _LayoutMetrics,
    renderer: matplotlib.backend_bases.RendererBase,
) -> tuple[ResolvedGuideBox, ...]:
    boxes: list[ResolvedGuideBox] = []
    for guide in guides:
        artist = axes.xaxis.label if guide.dimension == AxisDimension.X else axes.yaxis.label
        if not artist.get_visible() or not artist.get_text():
            continue
        boxes.append(
            ResolvedGuideBox(
                guide_id=guide.id,
                kind="axis_label",
                rect_px=_rect_from_bbox(artist.get_window_extent(renderer), metrics),
                role=f"{guide.dimension.value}_axis_label",
                layer="guides",
            )
        )
    return tuple(boxes)


def _tick_label_boxes(
    axes: matplotlib.axes.Axes,
    guides: tuple[AxisGuide, ...],
    metrics: _LayoutMetrics,
    renderer: matplotlib.backend_bases.RendererBase,
) -> tuple[ResolvedGuideBox, ...]:
    boxes: list[ResolvedGuideBox] = []
    for guide in guides:
        artists = (
            axes.get_xticklabels() if guide.dimension == AxisDimension.X else axes.get_yticklabels()
        )
        for artist in artists:
            if not artist.get_visible() or not artist.get_text():
                continue
            boxes.append(
                ResolvedGuideBox(
                    guide_id=guide.id,
                    kind="tick_label",
                    rect_px=_rect_from_bbox(artist.get_window_extent(renderer), metrics),
                    role=f"{guide.dimension.value}_tick_label",
                    layer="guides",
                )
            )
    return tuple(boxes)


def _title_boxes(
    axes: matplotlib.axes.Axes,
    guides: tuple[PanelTextGuide, ...],
    metrics: _LayoutMetrics,
    renderer: matplotlib.backend_bases.RendererBase,
) -> tuple[ResolvedGuideBox, ...]:
    boxes: list[ResolvedGuideBox] = []
    title_guides = tuple(guide for guide in guides if guide.role == PanelTextRole.TITLE)
    if not title_guides:
        return ()
    title_artist = axes.title
    if not title_artist.get_visible() or not title_artist.get_text():
        return ()
    for guide in title_guides:
        boxes.append(
            ResolvedGuideBox(
                guide_id=guide.id,
                kind="title",
                rect_px=_rect_from_bbox(title_artist.get_window_extent(renderer), metrics),
                role=guide.role.value,
                layer="guides",
            )
        )
    return tuple(boxes)


def _layout_layers(
    axis_guides: tuple[AxisGuide, ...], panel_text_guides: tuple[PanelTextGuide, ...]
) -> tuple[LayoutLayer, ...]:
    layers = [LayoutLayer(object_id=guide.id, layer="guides", z_order=0.0) for guide in axis_guides]
    layers.extend(
        LayoutLayer(object_id=guide.id, layer="guides", z_order=1.0) for guide in panel_text_guides
    )
    return tuple(layers)


def _data_to_top_left_transform(
    axes: matplotlib.axes.Axes,
    metrics: _LayoutMetrics,
    native_plot_rect: LogicalPixelRect,
    canonical_plot_rect: LogicalPixelRect,
) -> tuple[float, ...]:
    matrix = axes.transData.get_affine().get_matrix()
    display_to_logical_top_left = np.array(
        [
            [1.0 / metrics.display_per_logical_x, 0.0, 0.0],
            [
                0.0,
                -1.0 / metrics.display_per_logical_y,
                metrics.logical_height_px,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    top_left = display_to_logical_top_left @ matrix
    correction = np.array(
        [
            [
                canonical_plot_rect.width / native_plot_rect.width,
                0.0,
                canonical_plot_rect.x
                - native_plot_rect.x * canonical_plot_rect.width / native_plot_rect.width,
            ],
            [
                0.0,
                canonical_plot_rect.height / native_plot_rect.height,
                canonical_plot_rect.y
                - native_plot_rect.y * canonical_plot_rect.height / native_plot_rect.height,
            ],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    canonical = correction @ top_left
    return tuple(float(value) for value in canonical.reshape(-1))


def _rect_from_bbox(
    bbox: matplotlib.transforms.BboxBase, metrics: _LayoutMetrics
) -> LogicalPixelRect:
    return LogicalPixelRect(
        x=float(bbox.x0) / metrics.display_per_logical_x,
        y=(metrics.display_height_px - float(bbox.y1)) / metrics.display_per_logical_y,
        width=float(bbox.width) / metrics.display_per_logical_x,
        height=float(bbox.height) / metrics.display_per_logical_y,
    )


def _resolve_layout_metrics(
    figure: matplotlib.figure.Figure, *, device_scale: float
) -> _LayoutMetrics:
    resolved = getattr(figure, "_gsp_resolved_canvas", None)
    if isinstance(resolved, ResolvedCanvas):
        if resolved.device_scale_x != resolved.device_scale_y:
            raise ValueError(
                "Matplotlib layout requires equal resolved device_scale_x and device_scale_y values"
            )
        logical_width = float(resolved.canvas_width_px)
        logical_height = float(resolved.canvas_height_px)
        snapshot_device_scale = float(resolved.device_scale_x)
        dpi = float(resolved.output_dpi)
    else:
        width_in, height_in = figure.get_size_inches()
        dpi = float(getattr(figure, "_original_dpi", figure.dpi))
        logical_width = float(width_in) * dpi
        logical_height = float(height_in) * dpi
        snapshot_device_scale = float(device_scale)

    display_width = float(figure.bbox.width)
    display_height = float(figure.bbox.height)
    for name, value in (
        ("logical_width_px", logical_width),
        ("logical_height_px", logical_height),
        ("display_width_px", display_width),
        ("display_height_px", display_height),
        ("device_scale", snapshot_device_scale),
        ("dpi", dpi),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Matplotlib layout {name} must be finite and positive")

    display_per_logical_x = display_width / logical_width
    display_per_logical_y = display_height / logical_height
    if (
        not math.isfinite(display_per_logical_x)
        or display_per_logical_x <= 0.0
        or not math.isfinite(display_per_logical_y)
        or display_per_logical_y <= 0.0
    ):
        raise ValueError("Matplotlib display-to-logical layout factors must be finite and positive")
    return _LayoutMetrics(
        logical_width_px=logical_width,
        logical_height_px=logical_height,
        display_width_px=display_width,
        display_height_px=display_height,
        display_per_logical_x=display_per_logical_x,
        display_per_logical_y=display_per_logical_y,
        device_scale=snapshot_device_scale,
        dpi=dpi,
    )
