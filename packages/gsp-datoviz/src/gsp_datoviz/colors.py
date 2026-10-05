"""RGBA conversion and scalar query payloads."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    ColorScale,
    MarkerVisual,
    PointVisual,
    QueryResult,
    ScalarColorQueryPayload,
)
from gsp.protocol.color_mapping import (
    map_scalar_value,
    map_scalar_values,
    resolve_color_scale,
)

from gsp_datoviz.query import (
    DATOVIZ_QUERY_PAYLOAD_KIND,
)

from ._state import (
    DatovizV04Unsupported,
    _ScalarVisualData,
)


def _rgba8(colors: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    if colors.dtype == np.dtype(np.uint8):
        return np.ascontiguousarray(colors)
    return np.ascontiguousarray(np.rint(np.asarray(colors) * 255.0).clip(0, 255).astype(np.uint8))


def _rgba8_image(image: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    if image.dtype != np.dtype(np.uint8):
        image = np.rint(np.asarray(image) * 255.0).clip(0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise DatovizV04Unsupported("Datoviz v0.4 slice only supports uint8 RGB/RGBA images")
    if image.shape[2] == 4:
        return np.ascontiguousarray(image)
    alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate([image, alpha], axis=2))


def _rgba8_broadcast(colors: npt.NDArray[Any], count: int) -> npt.NDArray[np.uint8]:
    converted = _rgba8(colors)
    if converted.shape == (4,):
        return np.ascontiguousarray(np.broadcast_to(converted, (count, 4)), dtype=np.uint8)
    return converted


def _rgba8_scalar_image(
    image: npt.NDArray[Any], clim: tuple[float, float] | None
) -> npt.NDArray[np.uint8]:
    values = np.asarray(image, dtype=np.float32)
    if clim is None:
        vmin = float(np.min(values))
        vmax = float(np.max(values))
        if vmin == vmax:
            vmax = vmin + 1.0
    else:
        vmin, vmax = clim
    normalized = ((values - vmin) / (vmax - vmin)).clip(0.0, 1.0)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    alpha = np.full(gray.shape, 255, dtype=np.uint8)
    return np.ascontiguousarray(np.stack([gray, gray, gray, alpha], axis=2))


def _point_colors(
    visual: PointVisual, *, color_scales: Mapping[str, ColorScale] | None
) -> npt.NDArray[np.uint8]:
    if visual.color_encoding is not None:
        scale = resolve_color_scale(color_scales, visual.color_encoding.color_scale_id)
        return _rgba8_scalar_values(
            visual.color_encoding.values,
            scale,
            alpha=float(visual.color_encoding.alpha),
        )
    if visual.colors is None:
        raise ValueError("PointVisual requires colors or color_encoding")
    return _rgba8(visual.colors)


def _marker_fill_colors(
    visual: MarkerVisual, *, color_scales: Mapping[str, ColorScale] | None
) -> npt.NDArray[np.uint8]:
    if visual.fill_color_encoding is not None:
        scale = resolve_color_scale(color_scales, visual.fill_color_encoding.color_scale_id)
        return _rgba8_scalar_values(
            visual.fill_color_encoding.values,
            scale,
            alpha=float(visual.fill_color_encoding.alpha),
        )
    if visual.fill_colors is None:
        raise ValueError("MarkerVisual requires fill_colors or fill_color_encoding")
    return _rgba8(visual.fill_colors)


def _rgba8_scalar_values(
    values: npt.ArrayLike, scale: ColorScale, *, alpha: float
) -> npt.NDArray[np.uint8]:
    mapped = map_scalar_values(values, scale, alpha=alpha)
    return np.ascontiguousarray(np.rint(mapped * 255.0).clip(0, 255).astype(np.uint8))


def _scalar_payload_for_query_result(
    metadata: _ScalarVisualData, result: QueryResult
) -> ScalarColorQueryPayload | None:
    item_id: int | None = None
    texel: tuple[int, int] | None = None
    if metadata.item_kind == "point":
        if result.item_id is None:
            return None
        if result.item_id < 0 or result.item_id >= metadata.values.shape[0]:
            return None
        source_value = float(metadata.values[result.item_id])
        item_id = result.item_id
    elif metadata.item_kind == "texel":
        texel = _resolved_scalar_texel(metadata.values, result)
        if texel is None:
            return None
        source_value = float(metadata.values[texel[0], texel[1]])
    else:
        return None

    mapped = map_scalar_value(source_value, metadata.color_scale, alpha=metadata.alpha)
    return ScalarColorQueryPayload(
        visual_id=metadata.visual_id,
        item_kind=metadata.item_kind,
        item_id=item_id,
        texel=texel,
        color_slot=metadata.color_slot,
        color_scale_id=metadata.color_scale.id,
        colormap_id=metadata.color_scale.colormap.id.value,
        source_value=mapped.source_value,
        normalized_value_raw=mapped.normalized_value_raw,
        normalized_value_clipped=mapped.normalized_value_clipped,
        range_class=mapped.range_class,
        lut_index=mapped.lut_index,
        displayed_rgba=mapped.displayed_rgba,
    )


def _resolved_scalar_texel(
    values: npt.NDArray[np.float64], result: QueryResult
) -> tuple[int, int] | None:
    if values.ndim != 2:
        return None
    height, width = values.shape
    if result.texel is not None:
        row, col = result.texel
        if 0 <= row < height and 0 <= col < width:
            return (row, col)
        flat_index = col if row == 0 else row * width + col
        if 0 <= flat_index < values.size:
            resolved_row, resolved_col = np.unravel_index(flat_index, values.shape)
            return (int(resolved_row), int(resolved_col))
    # The canonical GSP result intentionally does not turn Datoviz's flat texel_id into a
    # two-dimensional coordinate. Scalar-image decoration has the retained image shape, so it
    # can safely resolve that native id here without leaking an invented canonical texel.
    if result.extension_payload_kind == DATOVIZ_QUERY_PAYLOAD_KIND:
        flat_texel_id = getattr(result.extension_payload, "texel_id", None)
        if isinstance(flat_texel_id, int) and 0 <= flat_texel_id < values.size:
            resolved_row, resolved_col = np.unravel_index(flat_texel_id, values.shape)
            return (int(resolved_row), int(resolved_col))
    if result.item_id is not None and 0 <= result.item_id < values.size:
        resolved_row, resolved_col = np.unravel_index(result.item_id, values.shape)
        return (int(resolved_row), int(resolved_col))
    return None


def _rgba8_scalar(color: npt.NDArray[Any]) -> npt.NDArray[np.uint8]:
    rgba = _rgba8(np.asarray(color).reshape(1, 4))[0]
    return np.ascontiguousarray(rgba)
