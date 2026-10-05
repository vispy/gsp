"""Native axis and colorbar realization."""

from __future__ import annotations

from typing import Any

import numpy as np
from gsp.protocol import (
    ColorbarGuide,
    ColorbarOrientation,
    ColorbarPlacement,
    ColorMapId,
)

from ._state import (
    _BUILTIN_COLORMAP_NAMES,
    DATOVIZ_REVIEW_PLOT_MARGINS,
    DVZ_COLORBAR_ORIENTATION_HORIZONTAL,
    DVZ_COLORBAR_ORIENTATION_VERTICAL,
    DVZ_COLORBAR_PLACEMENT_DETACHED,
    DVZ_HORIZONTAL_ANCHOR_RIGHT,
    DVZ_PLACEMENT_SPACE_PANEL,
    DVZ_SCENE_ANCHOR_PANEL_BOTTOM,
    DVZ_SCENE_ANCHOR_PANEL_LEFT,
    DVZ_SCENE_ANCHOR_PANEL_RIGHT,
    DVZ_SCENE_ANCHOR_PANEL_TOP,
    DVZ_VERTICAL_ANCHOR_CENTER,
    DatovizV04Unsupported,
)
from .native_api import (
    _ctypes_pointer_arg,
    _enum_value,
    _require_datoviz_success,
)


def _configure_axis_review_style(dvz: Any, axis: Any) -> None:
    style_factory = getattr(dvz, "dvz_axis_style", None)
    style_setter = getattr(dvz, "dvz_axis_set_style", None)
    if style_factory is None or style_setter is None:
        return
    style = style_factory()
    if hasattr(style, "spine_width"):
        style.spine_width = 1.75
    if hasattr(style, "major_tick_width"):
        style.major_tick_width = 1.5
    if hasattr(style, "minor_tick_width"):
        style.minor_tick_width = 1.0
    if hasattr(style, "grid_width"):
        style.grid_width = 1.0
    if hasattr(style, "major_tick_length"):
        style.major_tick_length = 7.0
    if hasattr(style, "minor_tick_length"):
        style.minor_tick_length = 4.0
    if hasattr(style, "tick_gap_px"):
        style.tick_gap_px = 5.0
    if hasattr(style, "label_gap_px"):
        style.label_gap_px = 9.0
    if hasattr(style, "tick_size_px"):
        style.tick_size_px = 15.0
    if hasattr(style, "label_size_px"):
        style.label_size_px = 17.0
    if hasattr(style, "plot_margin_left"):
        style.plot_margin_left = DATOVIZ_REVIEW_PLOT_MARGINS[0]
    if hasattr(style, "plot_margin_right"):
        style.plot_margin_right = DATOVIZ_REVIEW_PLOT_MARGINS[1]
    if hasattr(style, "plot_margin_bottom"):
        style.plot_margin_bottom = DATOVIZ_REVIEW_PLOT_MARGINS[2]
    if hasattr(style, "plot_margin_top"):
        style.plot_margin_top = DATOVIZ_REVIEW_PLOT_MARGINS[3]
    if hasattr(style, "show_spine"):
        style.show_spine = True
    if hasattr(style, "show_major_ticks"):
        style.show_major_ticks = True
    if hasattr(style, "show_minor_ticks"):
        style.show_minor_ticks = True
    _assign_style_color(style, "spine_color", (32, 32, 32, 255))
    _assign_style_color(style, "major_tick_color", (32, 32, 32, 255))
    _assign_style_color(style, "minor_tick_color", (90, 90, 90, 220))
    _assign_style_color(style, "grid_color", (150, 150, 150, 190))
    _require_datoviz_success(
        style_setter(axis, style),
        "Datoviz axis style configuration failed",
    )


def _configure_axis_review_plot_margins(dvz: Any, axis: Any) -> None:
    margin_setter = getattr(dvz, "dvz_axis_set_plot_margins", None)
    if margin_setter is None:
        return
    _require_datoviz_success(
        margin_setter(axis, *DATOVIZ_REVIEW_PLOT_MARGINS),
        "Datoviz axis plot margin configuration failed",
    )


def _assign_style_color(style: Any, field_name: str, color: tuple[int, int, int, int]) -> None:
    target = getattr(style, field_name, None)
    if target is None:
        return
    for index, channel in enumerate(color):
        target[index] = channel


def _builtin_colormap_value(dvz: Any, colormap_id: ColorMapId) -> int:
    name, fallback = _BUILTIN_COLORMAP_NAMES[colormap_id]
    return _enum_value(dvz, "DvzBuiltinColormap", name, fallback)


def _colorbar_orientation_value(dvz: Any, orientation: ColorbarOrientation) -> int:
    if orientation is ColorbarOrientation.VERTICAL:
        return _enum_value(
            dvz,
            "DvzColorbarOrientation",
            "DVZ_COLORBAR_ORIENTATION_VERTICAL",
            DVZ_COLORBAR_ORIENTATION_VERTICAL,
        )
    if orientation is ColorbarOrientation.HORIZONTAL:
        return _enum_value(
            dvz,
            "DvzColorbarOrientation",
            "DVZ_COLORBAR_ORIENTATION_HORIZONTAL",
            DVZ_COLORBAR_ORIENTATION_HORIZONTAL,
        )
    raise DatovizV04Unsupported(f"unsupported Datoviz colorbar orientation: {orientation.value}")


def _colorbar_anchor_value(dvz: Any, placement: ColorbarPlacement | None) -> int:
    if placement is ColorbarPlacement.RIGHT or placement is None:
        return _enum_value(
            dvz,
            "DvzSceneAnchor",
            "DVZ_SCENE_ANCHOR_PANEL_RIGHT",
            DVZ_SCENE_ANCHOR_PANEL_RIGHT,
        )
    if placement is ColorbarPlacement.LEFT:
        return _enum_value(
            dvz,
            "DvzSceneAnchor",
            "DVZ_SCENE_ANCHOR_PANEL_LEFT",
            DVZ_SCENE_ANCHOR_PANEL_LEFT,
        )
    if placement is ColorbarPlacement.BOTTOM:
        return _enum_value(
            dvz,
            "DvzSceneAnchor",
            "DVZ_SCENE_ANCHOR_PANEL_BOTTOM",
            DVZ_SCENE_ANCHOR_PANEL_BOTTOM,
        )
    if placement is ColorbarPlacement.TOP:
        return _enum_value(
            dvz,
            "DvzSceneAnchor",
            "DVZ_SCENE_ANCHOR_PANEL_TOP",
            DVZ_SCENE_ANCHOR_PANEL_TOP,
        )
    raise DatovizV04Unsupported(f"unsupported Datoviz colorbar placement: {placement.value}")


def _configure_colorbar_layout(
    dvz: Any,
    desc: Any,
    guide: ColorbarGuide,
    width: int,
    height: int,
    *,
    canvas_px_scale: float,
) -> None:
    """Use a bounded colorbar placement for visual QA captures."""
    ramp_width_px = float(guide.style.ramp_width_px * canvas_px_scale)
    tick_length_px = float(guide.style.tick_length_px * canvas_px_scale)
    label_gap_px = float(guide.style.label_gap_px * canvas_px_scale)
    min_length_px = float(guide.style.min_length_px * canvas_px_scale)
    if hasattr(desc, "orientation"):
        desc.orientation = _colorbar_orientation_value(dvz, guide.orientation)
    if hasattr(desc, "placement_mode"):
        desc.placement_mode = _enum_value(
            dvz,
            "DvzColorbarPlacementMode",
            "DVZ_COLORBAR_PLACEMENT_DETACHED",
            DVZ_COLORBAR_PLACEMENT_DETACHED,
        )
    if hasattr(desc, "ramp_width_px"):
        desc.ramp_width_px = ramp_width_px
    if hasattr(desc, "tick_length_px"):
        desc.tick_length_px = tick_length_px
    if hasattr(desc, "label_gap_px"):
        desc.label_gap_px = label_gap_px
    placement = getattr(desc, "placement", None)
    if placement is None:
        return
    if hasattr(placement, "space"):
        placement.space = _enum_value(
            dvz,
            "DvzPlacementSpace",
            "DVZ_PLACEMENT_SPACE_PANEL",
            DVZ_PLACEMENT_SPACE_PANEL,
        )
    if hasattr(placement, "horizontal_anchor"):
        placement.horizontal_anchor = _enum_value(
            dvz,
            "DvzHorizontalAnchor",
            "DVZ_HORIZONTAL_ANCHOR_RIGHT",
            DVZ_HORIZONTAL_ANCHOR_RIGHT,
        )
    if hasattr(placement, "vertical_anchor"):
        placement.vertical_anchor = _enum_value(
            dvz,
            "DvzVerticalAnchor",
            "DVZ_VERTICAL_ANCHOR_CENTER",
            DVZ_VERTICAL_ANCHOR_CENTER,
        )
    if hasattr(placement, "offset_x_px"):
        placement.offset_x_px = -max(48.0, float(width) * 0.060)
    if hasattr(placement, "offset_y_px"):
        placement.offset_y_px = 0.0
    if hasattr(placement, "width_px"):
        placement.width_px = ramp_width_px
    if hasattr(placement, "height_px"):
        placement.height_px = max(min_length_px, float(height) * guide.style.length_fraction)


def _configure_colorbar_format(dvz: Any, colorbar: Any) -> None:
    if not hasattr(dvz, "dvz_format_desc") or not hasattr(dvz, "dvz_colorbar_set_format"):
        return
    fmt = dvz.dvz_format_desc()
    if hasattr(fmt, "precision"):
        fmt.precision = 2
    if hasattr(fmt, "trim_trailing_zeros"):
        fmt.trim_trailing_zeros = True
    dvz.dvz_colorbar_set_format(colorbar, _ctypes_pointer_arg(fmt))


def _configure_colorbar_ticks(dvz: Any, colorbar: Any, guide: ColorbarGuide) -> None:
    if not guide.ticks:
        return
    setter = getattr(dvz, "dvz_colorbar_set_ticks", None)
    if setter is None:
        raise DatovizV04Unsupported(
            "Datoviz colorbar explicit ticks are unavailable: missing dvz_colorbar_set_ticks"
        )

    values = np.asarray(guide.ticks, dtype=np.float64)
    labels = list(guide.tick_labels) if guide.tick_labels else None
    _require_datoviz_success(
        setter(colorbar, values, labels),
        "Datoviz colorbar explicit tick configuration failed",
    )
