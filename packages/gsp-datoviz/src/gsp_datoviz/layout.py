"""Panel allocation, native frame snapshots, and guide queries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from gsp.protocol import (
    GUIDE_QUERY_PAYLOAD_KIND,
    AxisDimension,
    GuideQueryPayload,
    LayoutAnchor,
    LayoutDiagnostic,
    LayoutDiagnosticStatus,
    LayoutLayer,
    LogicalPixelRect,
    PerspectiveProjection3D,
    PixelOrigin,
    QueryRequest,
    QueryResult,
    QueryStatus,
    RenderTarget,
    ResolvedCanvas,
    ResolvedGuideBox,
    ResolvedLayoutSnapshot,
    ResolvedPanelLayout,
    View2D,
    View3D,
    quantize_logical_rect,
)

from ._state import (
    DatovizV04Unavailable,
    DatovizV04Unsupported,
)
from .native_api import (
    _ctypes_pointer_arg,
    _datoviz_label,
    _is_null_handle,
    datoviz_v04_panel_frame_guide_query_ready,
    datoviz_v04_panel_frame_snapshot_diagnostics,
)


def _panel_pixel_size(
    width: int,
    height: int,
    panel_bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float]:
    if panel_bounds is None:
        return float(width), float(height)
    _x, _y, panel_width, panel_height = panel_bounds
    if 0.0 < panel_width <= 1.0 and 0.0 < panel_height <= 1.0:
        return float(width) * panel_width, float(height) * panel_height
    return panel_width, panel_height


def _datoviz_query_panel_bounds(
    width: int,
    height: int,
    panel_bounds: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float]:
    panel_width, panel_height = _panel_pixel_size(width, height, panel_bounds)
    return (0.0, panel_width, 0.0, panel_height)


def _resolve_datoviz_partial_layout_snapshot(
    dvz: Any,
    panel: Any,
    *,
    resolved_canvas: ResolvedCanvas,
    panel_id: str,
    view_id: str | None,
    snapshot_id_prefix: str,
) -> ResolvedLayoutSnapshot:
    diagnostics = datoviz_v04_panel_frame_snapshot_diagnostics(dvz)
    if diagnostics:
        raise DatovizV04Unavailable(
            "Datoviz panel frame snapshot binding is unavailable: " + "; ".join(diagnostics)
        )

    snapshot = dvz.dvz_panel_resolve_frame(panel)
    if _is_null_handle(snapshot):
        raise DatovizV04Unsupported("Datoviz panel frame snapshot resolution failed")
    try:
        info = dvz.DvzPanelFrameInfo()
        if not dvz.dvz_panel_frame_info(snapshot, _ctypes_pointer_arg(info)):
            raise DatovizV04Unsupported("Datoviz panel frame info copy failed")
        frame_snapshot_id = _native_uint_id(getattr(info, "snapshot_id", 0))
        if frame_snapshot_id == 0 and hasattr(dvz, "dvz_panel_frame_id"):
            frame_snapshot_id = _native_uint_id(dvz.dvz_panel_frame_id(snapshot))
        snapshot_id = f"{snapshot_id_prefix}:{frame_snapshot_id:x}"

        device_scale = _datoviz_frame_device_scale(info)
        render_target = RenderTarget(
            logical_width_px=_positive_or_fallback(
                getattr(info, "logical_width_px", 0),
                resolved_canvas.canvas_width_px,
            ),
            logical_height_px=_positive_or_fallback(
                getattr(info, "logical_height_px", 0),
                resolved_canvas.canvas_height_px,
            ),
            device_scale=device_scale,
            dpi=resolved_canvas.output_dpi,
        )
        _validate_datoviz_frame_units(info, render_target)

        guide_boxes, guide_diagnostics = _datoviz_frame_guide_boxes(
            dvz, snapshot, frame_snapshot_id
        )
        z_layers, contribution_diagnostics = _datoviz_frame_contribution_layers(
            dvz, snapshot, frame_snapshot_id
        )
        transform, transform_diagnostics = _datoviz_frame_data_to_screen_transform(info)
        grid_clip_rect = _optional_dvz_rect(getattr(info, "grid_clip_rect_px", None))

        guide_query_supported = datoviz_v04_panel_frame_guide_query_ready(dvz)
        guide_query_diagnostics = (
            (
                LayoutDiagnostic(
                    code="guide_query_native_verified",
                    status=LayoutDiagnosticStatus.RESOLVED,
                    message=(
                        "Datoviz panel frame guide hit/readback uses the same "
                        "snapshot id as guide layout records."
                    ),
                ),
                LayoutDiagnostic(
                    code="all_rendered_guides_native_verified",
                    status=LayoutDiagnosticStatus.RESOLVED,
                    message=(
                        "Datoviz panel frame contribution enumeration reports "
                        "guide contributions with snapshot ids."
                    ),
                ),
            )
            if guide_query_supported
            else (
                LayoutDiagnostic(
                    code="guide_query_missing",
                    status=LayoutDiagnosticStatus.MISSING,
                    message=("Datoviz guide hit/readback is not yet wired into GSP query results."),
                ),
                LayoutDiagnostic(
                    code="all_rendered_guides_unsupported",
                    status=LayoutDiagnosticStatus.UNSUPPORTED,
                    message=(
                        "Rendered contributions are diagnostic snapshot evidence "
                        "only until guide hit/readback is available."
                    ),
                ),
            )
        )

        snapshot_diagnostics = (
            LayoutDiagnostic(
                code="layout_snapshot_partial",
                status=LayoutDiagnosticStatus.RESOLVED,
                message=(
                    "Datoviz panel, plot, grid clip, guide layout, and contribution "
                    "records were mapped from the native panel frame snapshot."
                ),
            ),
            *guide_query_diagnostics,
            *guide_diagnostics,
            *contribution_diagnostics,
            *transform_diagnostics,
        )
        return ResolvedLayoutSnapshot(
            snapshot_id=snapshot_id,
            render_target=render_target,
            panels=(
                ResolvedPanelLayout(
                    panel_id=panel_id,
                    panel_rect_px=quantize_logical_rect(
                        _required_dvz_rect(info.panel_rect_px, "panel_rect_px"),
                        render_target,
                    ),
                    plot_rect_px=quantize_logical_rect(
                        _required_dvz_rect(info.plot_rect_px, "plot_rect_px"),
                        render_target,
                    ),
                    view_id=view_id,
                    data_to_screen_transform=transform,
                    guide_boxes=guide_boxes,
                    tick_label_boxes=tuple(box for box in guide_boxes if box.kind == "tick_label"),
                    axis_label_boxes=tuple(box for box in guide_boxes if box.kind == "axis_label"),
                    title_boxes=tuple(box for box in guide_boxes if box.kind == "title"),
                    legend_boxes=tuple(box for box in guide_boxes if box.kind == "legend"),
                    colorbar_boxes=tuple(box for box in guide_boxes if box.kind == "colorbar"),
                    grid_clip_rect_px=(
                        quantize_logical_rect(grid_clip_rect, render_target)
                        if grid_clip_rect is not None
                        else None
                    ),
                    z_layers=z_layers,
                    diagnostics=snapshot_diagnostics,
                ),
            ),
        )
    finally:
        frame_unref = getattr(dvz, "dvz_panel_frame_unref", None)
        if frame_unref is not None:
            frame_unref(snapshot)


def _query_datoviz_panel_frame_guides(
    dvz: Any,
    panel: Any,
    request: QueryRequest,
    *,
    all_rendered: bool,
) -> QueryResult:
    snapshot = dvz.dvz_panel_resolve_frame(panel)
    if _is_null_handle(snapshot):
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.FAILED,
            hit=False,
            panel_coordinate=request.coordinate,
            diagnostic="Datoviz panel frame snapshot resolution failed",
            layout_snapshot_id=request.layout_snapshot_id,
        )
    try:
        info = dvz.DvzPanelFrameInfo()
        if not dvz.dvz_panel_frame_info(snapshot, _ctypes_pointer_arg(info)):
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.FAILED,
                hit=False,
                panel_coordinate=request.coordinate,
                diagnostic="Datoviz panel frame info copy failed",
                layout_snapshot_id=request.layout_snapshot_id,
            )
        frame_snapshot_id = _native_uint_id(getattr(info, "snapshot_id", 0))
        if frame_snapshot_id == 0 and hasattr(dvz, "dvz_panel_frame_id"):
            frame_snapshot_id = _native_uint_id(dvz.dvz_panel_frame_id(snapshot))
        native_layout_snapshot_id = f"layout:datoviz:{frame_snapshot_id:x}"
        layout_snapshot_id = request.layout_snapshot_id or native_layout_snapshot_id
        if not _datoviz_layout_snapshot_id_matches(layout_snapshot_id, frame_snapshot_id):
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.STALE,
                hit=False,
                panel_coordinate=request.coordinate,
                diagnostic=(
                    "Datoviz guide query resolved a different layout snapshot id: "
                    f"{native_layout_snapshot_id}"
                ),
                layout_snapshot_id=native_layout_snapshot_id,
            )

        x, y = request.coordinate
        hit_record = dvz.DvzGuideHit()
        if not dvz.dvz_panel_frame_guide_hit(
            snapshot, float(x), float(y), _ctypes_pointer_arg(hit_record)
        ) or not bool(getattr(hit_record, "hit", False)):
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.MISS,
                hit=False,
                panel_coordinate=request.coordinate,
                layout_snapshot_id=layout_snapshot_id,
            )
        guide_id = _native_uint_id(getattr(hit_record, "guide_id", 0))
        if guide_id == 0:
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.FAILED,
                hit=False,
                panel_coordinate=request.coordinate,
                diagnostic="Datoviz guide hit omitted guide_id",
                layout_snapshot_id=layout_snapshot_id,
            )
        if frame_snapshot_id and _native_uint_id(getattr(hit_record, "snapshot_id", 0)) not in (
            0,
            frame_snapshot_id,
        ):
            return QueryResult(
                request_id=request.id,
                status=QueryStatus.FAILED,
                hit=False,
                panel_coordinate=request.coordinate,
                diagnostic="Datoviz guide hit snapshot id did not match frame snapshot id",
                layout_snapshot_id=layout_snapshot_id,
            )

        role = _datoviz_guide_role(dvz, int(getattr(hit_record, "role", 0)))
        label = _datoviz_label(getattr(hit_record, "label", b""))
        has_data_value = bool(getattr(hit_record, "has_data_value", False))
        data_value = float(getattr(hit_record, "data_value")) if has_data_value else None
        item_id = (
            int(getattr(hit_record, "item_index"))
            if bool(getattr(hit_record, "has_item_index", False))
            else None
        )
        payload = GuideQueryPayload(
            guide_id=_datoviz_protocol_id("guide", guide_id),
            role=role,
            axis_dimension=_datoviz_axis_dimension_for_role(role),
            tick_value=data_value,
            text_value=label,
        )
        diagnostic = "datoviz_all_rendered_guide_contribution_verified" if all_rendered else None
        return QueryResult(
            request_id=request.id,
            status=QueryStatus.HIT,
            hit=True,
            panel_coordinate=request.coordinate,
            visual_id=payload.guide_id,
            item_id=item_id,
            value=label if label is not None else data_value,
            extension_payload_kind=GUIDE_QUERY_PAYLOAD_KIND,
            extension_payload=payload,
            diagnostic=diagnostic,
            layout_snapshot_id=layout_snapshot_id,
        )
    finally:
        frame_unref = getattr(dvz, "dvz_panel_frame_unref", None)
        if frame_unref is not None:
            frame_unref(snapshot)


def _datoviz_frame_device_scale(info: Any) -> float:
    scale_x = float(getattr(info, "device_scale_x", 1.0))
    scale_y = float(getattr(info, "device_scale_y", scale_x))
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise DatovizV04Unsupported("Datoviz panel frame reported non-positive device scale")
    if not np.isclose(scale_x, scale_y, rtol=1e-6, atol=1e-6):
        raise DatovizV04Unsupported(
            "Datoviz panel frame reported asymmetric device_scale_x/device_scale_y; "
            "GSP RenderTarget currently requires one device_scale value"
        )
    return scale_x


def _validate_datoviz_frame_units(info: Any, render_target: RenderTarget) -> None:
    framebuffer_width = float(
        getattr(info, "framebuffer_width_px", render_target.framebuffer_width_px)
    )
    framebuffer_height = float(
        getattr(info, "framebuffer_height_px", render_target.framebuffer_height_px)
    )
    if not np.isclose(framebuffer_width, render_target.framebuffer_width_px, atol=1.0):
        raise DatovizV04Unsupported(
            "Datoviz panel frame framebuffer width disagrees with logical width/device scale"
        )
    if not np.isclose(framebuffer_height, render_target.framebuffer_height_px, atol=1.0):
        raise DatovizV04Unsupported(
            "Datoviz panel frame framebuffer height disagrees with logical height/device scale"
        )


def _datoviz_frame_guide_boxes(
    dvz: Any, snapshot: Any, frame_snapshot_id: int
) -> tuple[tuple[ResolvedGuideBox, ...], tuple[LayoutDiagnostic, ...]]:
    boxes: list[ResolvedGuideBox] = []
    diagnostics: list[LayoutDiagnostic] = []
    count = int(dvz.dvz_panel_frame_guide_count(snapshot))
    for index in range(count):
        record = dvz.DvzGuideLayout()
        if not dvz.dvz_panel_frame_guide_layout(snapshot, index, _ctypes_pointer_arg(record)):
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_guide_layout_copy_failed",
                    status=LayoutDiagnosticStatus.MISSING,
                    message=f"Datoviz did not copy guide layout record {index}.",
                )
            )
            continue
        if frame_snapshot_id and _native_uint_id(getattr(record, "snapshot_id", 0)) not in (
            0,
            frame_snapshot_id,
        ):
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_guide_snapshot_id_mismatch",
                    status=LayoutDiagnosticStatus.DEGRADED,
                    message=f"Guide layout record {index} did not match the frame snapshot id.",
                )
            )
        if not bool(getattr(record, "has_box", False)):
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_guide_box_missing",
                    status=LayoutDiagnosticStatus.MISSING,
                    message=f"Datoviz guide layout record {index} has no box.",
                )
            )
            continue
        guide_id = _native_uint_id(getattr(record, "guide_id", 0))
        if guide_id == 0:
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_guide_id_missing",
                    status=LayoutDiagnosticStatus.MISSING,
                    message=f"Datoviz guide layout record {index} has no guide id.",
                )
            )
            continue
        anchor = None
        if bool(getattr(record, "has_anchor", False)):
            anchor_values = getattr(record, "anchor_px", None)
            if anchor_values is not None:
                anchor = LayoutAnchor(x=float(anchor_values[0]), y=float(anchor_values[1]))
        role = _datoviz_guide_role(dvz, int(getattr(record, "role", 0)))
        boxes.append(
            ResolvedGuideBox(
                guide_id=_datoviz_protocol_id("guide", guide_id),
                kind=cast(Any, _datoviz_guide_box_kind(dvz, record)),
                rect_px=_required_dvz_rect(record.box_px, "guide.box_px"),
                anchor_px=anchor,
                role=role,
                diagnostics=(
                    LayoutDiagnostic(
                        code="datoviz_guide_layout_snapshot_first_slice",
                        status=LayoutDiagnosticStatus.ADAPTED,
                        message=(
                            "Datoviz reports retained logical-pixel guide boxes; "
                            "text boxes may be coarse and are not strict glyph extents."
                        ),
                    ),
                ),
            )
        )
    return tuple(boxes), tuple(diagnostics)


def _datoviz_frame_contribution_layers(
    dvz: Any, snapshot: Any, frame_snapshot_id: int
) -> tuple[tuple[LayoutLayer, ...], tuple[LayoutDiagnostic, ...]]:
    layers: list[LayoutLayer] = []
    diagnostics: list[LayoutDiagnostic] = []
    count = int(dvz.dvz_panel_frame_contribution_count(snapshot))
    for index in range(count):
        record = dvz.DvzRenderedContribution()
        if not dvz.dvz_panel_frame_contribution(snapshot, index, _ctypes_pointer_arg(record)):
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_contribution_copy_failed",
                    status=LayoutDiagnosticStatus.MISSING,
                    message=f"Datoviz did not copy rendered contribution record {index}.",
                )
            )
            continue
        if frame_snapshot_id and _native_uint_id(getattr(record, "snapshot_id", 0)) not in (
            0,
            frame_snapshot_id,
        ):
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_contribution_snapshot_id_mismatch",
                    status=LayoutDiagnosticStatus.DEGRADED,
                    message=f"Rendered contribution record {index} did not match the frame snapshot id.",
                )
            )
        contribution_id = _native_uint_id(getattr(record, "contribution_id", 0))
        if contribution_id == 0:
            diagnostics.append(
                LayoutDiagnostic(
                    code="datoviz_contribution_id_missing",
                    status=LayoutDiagnosticStatus.MISSING,
                    message=f"Datoviz rendered contribution record {index} has no contribution id.",
                )
            )
            continue
        layers.append(
            LayoutLayer(
                object_id=_datoviz_protocol_id("contribution", contribution_id),
                layer=_datoviz_contribution_layer(dvz, int(getattr(record, "kind", 0))),
                z_order=float(index),
            )
        )
    if count:
        diagnostics.append(
            LayoutDiagnostic(
                code="datoviz_rendered_contributions_reported",
                status=LayoutDiagnosticStatus.NATIVE,
                message=f"Datoviz reported {count} rendered contribution records.",
            )
        )
    return tuple(layers), tuple(diagnostics)


def _datoviz_frame_data_to_screen_transform(
    info: Any,
) -> tuple[tuple[float, ...], tuple[LayoutDiagnostic, ...]]:
    if not (
        bool(getattr(info, "has_valid_visible_x", False))
        and bool(getattr(info, "has_valid_visible_y", False))
    ):
        return (
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (
                LayoutDiagnostic(
                    code="datoviz_data_to_screen_transform_missing",
                    status=LayoutDiagnosticStatus.MISSING,
                    message="Datoviz did not report valid visible data ranges for a 2D transform.",
                ),
            ),
        )
    x0, x1 = (float(value) for value in getattr(info, "visible_data_x"))
    y0, y1 = (float(value) for value in getattr(info, "visible_data_y"))
    if np.isclose(x0, x1) or np.isclose(y0, y1):
        return (
            (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
            (
                LayoutDiagnostic(
                    code="datoviz_data_to_screen_transform_degenerate",
                    status=LayoutDiagnosticStatus.DEGRADED,
                    message="Datoviz reported a degenerate visible data range.",
                ),
            ),
        )
    plot = _required_dvz_rect(info.plot_rect_px, "plot_rect_px")
    sx = plot.width / (x1 - x0)
    sy = plot.height / (y1 - y0)
    tx = plot.x - sx * x0
    ty = plot.y - sy * y0
    return (sx, 0.0, tx, 0.0, sy, ty), ()


def _datoviz_guide_box_kind(dvz: Any, record: Any) -> str:
    role = int(getattr(record, "role", 0))
    if role in _datoviz_enum_values(
        dvz,
        "DvzGuideRole",
        {
            "DVZ_GUIDE_ROLE_AXIS_TICK_LABEL": 4,
            "DVZ_GUIDE_ROLE_COLORBAR_TICK_LABEL": 8,
        },
    ):
        return "tick_label"
    if role in _datoviz_enum_values(
        dvz,
        "DvzGuideRole",
        {
            "DVZ_GUIDE_ROLE_AXIS_LABEL": 5,
        },
    ):
        return "axis_label"
    if role in _datoviz_enum_values(
        dvz,
        "DvzGuideRole",
        {
            "DVZ_GUIDE_ROLE_AXIS_GRID": 3,
        },
    ):
        return "grid"
    if role in _datoviz_enum_values(
        dvz,
        "DvzGuideRole",
        {
            "DVZ_GUIDE_ROLE_COLORBAR": 6,
            "DVZ_GUIDE_ROLE_COLORBAR_RAMP": 7,
            "DVZ_GUIDE_ROLE_COLORBAR_TITLE": 9,
        },
    ):
        return "colorbar"
    if role in _datoviz_enum_values(
        dvz,
        "DvzGuideRole",
        {
            "DVZ_GUIDE_ROLE_LEGEND": 10,
            "DVZ_GUIDE_ROLE_LEGEND_ENTRY": 11,
            "DVZ_GUIDE_ROLE_LEGEND_TITLE": 12,
        },
    ):
        return "legend"
    if role in _datoviz_enum_values(
        dvz,
        "DvzGuideRole",
        {
            "DVZ_GUIDE_ROLE_GUIDE_LABEL": 15,
        },
    ):
        return "title"
    return "axis"


def _datoviz_guide_role(dvz: Any, role: int) -> str:
    names = {
        "DVZ_GUIDE_ROLE_X_AXIS": 1,
        "DVZ_GUIDE_ROLE_Y_AXIS": 2,
        "DVZ_GUIDE_ROLE_AXIS_GRID": 3,
        "DVZ_GUIDE_ROLE_AXIS_TICK_LABEL": 4,
        "DVZ_GUIDE_ROLE_AXIS_LABEL": 5,
        "DVZ_GUIDE_ROLE_COLORBAR": 6,
        "DVZ_GUIDE_ROLE_COLORBAR_RAMP": 7,
        "DVZ_GUIDE_ROLE_COLORBAR_TICK_LABEL": 8,
        "DVZ_GUIDE_ROLE_COLORBAR_TITLE": 9,
        "DVZ_GUIDE_ROLE_LEGEND": 10,
        "DVZ_GUIDE_ROLE_LEGEND_ENTRY": 11,
        "DVZ_GUIDE_ROLE_LEGEND_TITLE": 12,
        "DVZ_GUIDE_ROLE_GUIDE_LINE": 13,
        "DVZ_GUIDE_ROLE_GUIDE_SPAN": 14,
        "DVZ_GUIDE_ROLE_GUIDE_LABEL": 15,
    }
    for name, fallback in names.items():
        if role == _datoviz_enum_value(dvz, "DvzGuideRole", name, fallback):
            return name.removeprefix("DVZ_GUIDE_ROLE_").lower()
    return f"datoviz_role_{role}"


def _datoviz_axis_dimension_for_role(role: str) -> AxisDimension | None:
    if role.startswith("x_axis") or role == "axis_x":
        return AxisDimension.X
    if role.startswith("y_axis") or role == "axis_y":
        return AxisDimension.Y
    return None


def _datoviz_contribution_layer(dvz: Any, kind: int) -> str:
    if kind == _datoviz_enum_value(
        dvz,
        "DvzRenderedContributionKind",
        "DVZ_RENDERED_CONTRIBUTION_GUIDE",
        2,
    ):
        return "guide"
    if kind == _datoviz_enum_value(
        dvz,
        "DvzRenderedContributionKind",
        "DVZ_RENDERED_CONTRIBUTION_VISUAL",
        1,
    ):
        return "visual"
    return "datoviz"


def _datoviz_enum_values(dvz: Any, enum_type_name: str, values: Mapping[str, int]) -> set[int]:
    return {
        _datoviz_enum_value(dvz, enum_type_name, name, fallback)
        for name, fallback in values.items()
    }


def _datoviz_enum_value(dvz: Any, enum_type_name: str, name: str, fallback: int) -> int:
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, enum_type_name, None)
    if enum_type is not None and hasattr(enum_type, name):
        return int(getattr(enum_type, name))
    return fallback


def _required_dvz_rect(rect: Any, field_name: str) -> LogicalPixelRect:
    logical_rect = _optional_dvz_rect(rect)
    if logical_rect is None:
        raise DatovizV04Unsupported(f"Datoviz panel frame missing {field_name}")
    return logical_rect


def _optional_dvz_rect(rect: Any | None) -> LogicalPixelRect | None:
    if rect is None:
        return None
    return LogicalPixelRect(
        x=float(getattr(rect, "x")),
        y=float(getattr(rect, "y")),
        width=float(getattr(rect, "width")),
        height=float(getattr(rect, "height")),
    )


def _positive_or_fallback(value: object, fallback: float) -> float:
    number = float(cast(Any, value))
    return number if number > 0.0 else float(fallback)


def _native_uint_id(value: object) -> int:
    return int(cast(Any, value))


def _datoviz_protocol_id(prefix: str, value: int) -> str:
    return f"datoviz:{prefix}:{value:x}"


def _datoviz_layout_snapshot_id_matches(snapshot_id: str, native_id: int) -> bool:
    if native_id == 0:
        return True
    return snapshot_id.endswith(f":{native_id:x}")


def _datoviz_vec3_tuple(value: object) -> tuple[float, float, float] | None:
    try:
        raw = cast(Any, value)
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    except (TypeError, ValueError, IndexError):
        return None


def _datoviz_orthographic_bounds_tuple(
    state: object,
) -> tuple[float, float, float, float, float, float] | None:
    if not bool(getattr(state, "has_explicit_orthographic_bounds", False)):
        return None
    bounds = getattr(state, "orthographic_bounds", ())
    try:
        raw = cast(Any, bounds)
        return (
            float(raw[0]),
            float(raw[1]),
            float(raw[2]),
            float(raw[3]),
            float(raw[4]),
            float(raw[5]),
        )
    except (TypeError, ValueError, IndexError):
        return None


def _create_panel(dvz: Any, figure: Any, bounds: tuple[float, float, float, float] | None) -> Any:
    if bounds is None:
        return dvz.dvz_panel_full(figure)
    panel_factory = getattr(dvz, "dvz_panel", None)
    desc_factory = getattr(dvz, "dvz_panel_desc", None)
    if panel_factory is None or desc_factory is None:
        missing = tuple(
            name
            for name, value in (
                ("dvz_panel", panel_factory),
                ("dvz_panel_desc", desc_factory),
            )
            if value is None
        )
        raise DatovizV04Unavailable(
            "Datoviz resolved-layout viewport placement requires public " + " and ".join(missing)
        )

    x, y, width, height = bounds
    desc = cast(Any, desc_factory)()
    if desc is None or not all(hasattr(desc, name) for name in ("x", "y", "width", "height")):
        raise DatovizV04Unavailable(
            "Datoviz resolved-layout viewport placement requires a usable "
            "public dvz_panel_desc() descriptor"
        )
    desc.x = float(x)
    desc.y = float(y)
    desc.width = float(width)
    desc.height = float(height)
    panel = panel_factory(figure, _ctypes_pointer_arg(desc))
    if _is_null_handle(panel):
        raise DatovizV04Unavailable("Datoviz custom panel creation failed")
    return panel


def _preflight_consumed_layout_panel_api(dvz: Any) -> None:
    """Fail before creating native resources when custom panel placement is unavailable."""
    panel_factory = getattr(dvz, "dvz_panel", None)
    desc_factory = getattr(dvz, "dvz_panel_desc", None)
    missing = tuple(
        name
        for name, value in (
            ("dvz_panel_desc", desc_factory),
            ("dvz_panel", panel_factory),
        )
        if not callable(value)
    )
    if missing:
        raise DatovizV04Unavailable(
            "Datoviz resolved-layout viewport placement requires public " + " and ".join(missing)
        )
    desc = cast(Any, desc_factory)()
    if desc is None or not all(hasattr(desc, name) for name in ("x", "y", "width", "height")):
        raise DatovizV04Unavailable(
            "Datoviz resolved-layout viewport placement requires a usable "
            "public dvz_panel_desc() descriptor"
        )


def _validate_renderer_consumed_layout_view(
    snapshot: ResolvedLayoutSnapshot,
    *,
    view: View2D | None,
    view3d: View3D | None,
) -> None:
    active_view = view if view is not None else view3d
    if active_view is not None and snapshot.only_panel().view_id != active_view.id:
        raise ValueError("consumed layout view_id does not match the renderer view")
    if active_view is None and snapshot.only_panel().view_id is not None:
        raise ValueError("viewless renderer cannot consume a view-bound layout snapshot")


def _validate_consumed_perspective_aspect(
    snapshot: ResolvedLayoutSnapshot,
    view3d: View3D | None,
) -> None:
    """Qualify Datoviz's aspect-less retained perspective camera."""
    if view3d is None or not isinstance(view3d.projection, PerspectiveProjection3D):
        return
    authored_aspect = view3d.projection.aspect_ratio
    if authored_aspect is None:
        return
    plot = snapshot.only_panel().plot_rect_px
    plot_aspect = plot.width / plot.height
    if not math.isclose(authored_aspect, plot_aspect, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise DatovizV04Unsupported(
            "Datoviz consumed perspective layout requires omitted aspect_ratio "
            "or aspect_ratio equal to plot_rect_px width/height"
        )


def _consumed_layout_native_panel_bounds(
    snapshot: ResolvedLayoutSnapshot,
) -> tuple[float, float, float, float]:
    target = snapshot.render_target
    plot = snapshot.only_panel().plot_rect_px
    native_y = (
        plot.y / target.logical_height_px
        if target.pixel_origin is PixelOrigin.TOP_LEFT
        else 1.0 - (plot.y + plot.height) / target.logical_height_px
    )
    return (
        plot.x / target.logical_width_px,
        native_y,
        plot.width / target.logical_width_px,
        plot.height / target.logical_height_px,
    )


def _validate_resolved_canvas_matches_layout(
    canvas: ResolvedCanvas, snapshot: ResolvedLayoutSnapshot
) -> None:
    target = snapshot.render_target
    if (
        not math.isclose(canvas.canvas_width_px, target.logical_width_px)
        or not math.isclose(canvas.canvas_height_px, target.logical_height_px)
        or canvas.framebuffer_width != target.framebuffer_width_px
        or canvas.framebuffer_height != target.framebuffer_height_px
        or not math.isclose(canvas.device_scale_x, target.device_scale)
        or not math.isclose(canvas.device_scale_y, target.device_scale)
    ):
        raise ValueError(
            "Datoviz canvas resolution does not match the consumed layout render target"
        )
