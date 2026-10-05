"""Retained text placement and style adaptation."""

from __future__ import annotations

import ctypes
from typing import Any

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    TextAnchorX,
    TextAnchorY,
)
from gsp.protocol.visuals import CoordinateSpace

from ._state import (
    DVZ_SCENE_ANCHOR_DATA,
    DVZ_SCENE_ANCHOR_PANEL_CENTER,
    DVZ_SCENE_ANCHOR_PANEL_TOP,
    DVZ_TEXT_PLACEMENT_DATA,
    DVZ_TEXT_PLACEMENT_SCREEN,
    DVZ_TEXT_RENDERER_MSDF_ATLAS,
    DatovizV04Unsupported,
    _CompatDvzTextPlacement,
)
from .layout import (
    _panel_pixel_size,
)
from .native_api import (
    _datoviz_shared_library_function,
    _require_datoviz_success,
)


def _text_placement_mode_value(dvz: Any, coordinate_space: CoordinateSpace) -> int:
    if coordinate_space == CoordinateSpace.NDC:
        name = "DVZ_TEXT_PLACEMENT_SCREEN"
        fallback = DVZ_TEXT_PLACEMENT_SCREEN
    elif coordinate_space == CoordinateSpace.DATA:
        name = "DVZ_TEXT_PLACEMENT_DATA"
        fallback = DVZ_TEXT_PLACEMENT_DATA
    else:
        raise DatovizV04Unsupported(
            f"Datoviz TextVisual coordinate space is unsupported: {coordinate_space.value}"
        )
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzTextPlacementMode", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return fallback


def _text_anchor_value(dvz: Any, coordinate_space: CoordinateSpace) -> int:
    if coordinate_space == CoordinateSpace.NDC:
        name = "DVZ_SCENE_ANCHOR_PANEL_CENTER"
        fallback = DVZ_SCENE_ANCHOR_PANEL_CENTER
    elif coordinate_space == CoordinateSpace.DATA:
        name = "DVZ_SCENE_ANCHOR_DATA"
        fallback = DVZ_SCENE_ANCHOR_DATA
    else:
        raise DatovizV04Unsupported(
            f"Datoviz TextVisual coordinate space is unsupported: {coordinate_space.value}"
        )
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzSceneAnchor", None)
    if enum_type is not None and hasattr(enum_type, name):
        return int(getattr(enum_type, name))
    return fallback


def _ndc_text_screen_position(
    position: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    width: int,
    height: int,
    panel_bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float]:
    panel_width, panel_height = _panel_pixel_size(width, height, panel_bounds)
    return (
        float(position[0]) * 0.5 * panel_width,
        -float(position[1]) * 0.5 * panel_height,
    )


def _text_renderer_value(dvz: Any) -> int:
    name = "DVZ_TEXT_RENDERER_MSDF_ATLAS"
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzTextRenderer", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return DVZ_TEXT_RENDERER_MSDF_ATLAS


def _text_placement(dvz: Any) -> Any:
    factory = getattr(dvz, "dvz_text_placement", None)
    if factory is not None:
        return factory()
    placement = _CompatDvzTextPlacement()
    placement.struct_size = ctypes.sizeof(_CompatDvzTextPlacement)
    placement.flags = 0
    placement.mode = DVZ_TEXT_PLACEMENT_SCREEN
    placement.anchor = DVZ_SCENE_ANCHOR_PANEL_TOP
    return placement


def _set_text_placement(dvz: Any, text: Any, placement: Any) -> None:
    if isinstance(placement, _CompatDvzTextPlacement):
        raw_setter = _datoviz_shared_library_function(dvz, "dvz_text_set_placement")
        if raw_setter is None:
            raise DatovizV04Unsupported(
                "Datoviz text placement fallback requires dvz_text_set_placement in libdatoviz"
            )
        raw_setter.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_CompatDvzTextPlacement),
        ]
        raw_setter.restype = None
        raw_setter(ctypes.cast(text, ctypes.c_void_p), ctypes.byref(placement))
        return
    _require_datoviz_success(
        dvz.dvz_text_set_placement(text, placement),
        "Datoviz text placement configuration failed",
    )


def _text_anchor_x_value(anchor: TextAnchorX) -> float:
    if anchor == TextAnchorX.LEFT:
        return 0.0
    if anchor == TextAnchorX.CENTER:
        return 0.5
    if anchor == TextAnchorX.RIGHT:
        return 1.0
    raise ValueError(f"unsupported text horizontal anchor: {anchor.value}")


def _text_anchor_y_value(anchor: TextAnchorY) -> float:
    if anchor == TextAnchorY.TOP:
        return 0.0
    if anchor in (TextAnchorY.CENTER, TextAnchorY.BASELINE):
        return 0.5
    if anchor == TextAnchorY.BOTTOM:
        return 1.0
    raise ValueError(f"unsupported text vertical anchor: {anchor.value}")


def _assign_rgba8(target: Any, color: npt.NDArray[np.uint8]) -> None:
    for index, channel in enumerate(color):
        target[index] = int(channel)
