"""Checked public Datoviz bindings and native resource operations."""

from __future__ import annotations

import ctypes
import math
import os
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from zlib import crc32

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    SCALAR_COLOR_QUERY_PAYLOAD_KIND,
    TRANSFORM_QUERY_PAYLOAD_KIND,
    CanvasMetricsSource,
    CanvasSize,
    CapabilitySnapshot,
    MarkerShape,
    OrthographicProjection3D,
    PathVisual,
    PerspectiveProjection3D,
    QueryCoordinateSpace,
    QueryHitPolicy,
    QueryPayload,
    QueryRequest,
    QueryResult,
    QueryScope,
    QueryStatus,
    ResolvedCanvas,
    StrokeCap,
    StrokeJoin,
    VectorCap,
    View3D,
    View3DDiagnosticCode,
)
from gsp.protocol.visuals import CoordinateSpace, ImageInterpolation

from gsp_datoviz.capabilities import (
    datoviz_v04_capability_snapshot,
)
from gsp_datoviz.capabilities import (
    datoviz_v04_view3d_state_readback_diagnostics as _state_readback_diagnostics,
)
from gsp_datoviz.latest_api_contract import (
    datoviz_current_api_contract_diagnostics,
    datoviz_text_api_diagnostics,
    datoviz_vector_api_diagnostics,
)
from gsp_datoviz.v04_import import bootstrap_datoviz_v04_source

from ._state import (
    _MARKER_SHAPE_FALLBACKS,
    _MARKER_SHAPE_NAMES,
    _OPTIONAL_UNVERIFIED_DVZ_MESH_FUNCTIONS,
    _OPTIONAL_UNVERIFIED_DVZ_TEXT_FUNCTIONS,
    _REQUIRED_DVZ_MARKER_FUNCTIONS,
    _REQUIRED_DVZ_MESH_FUNCTIONS,
    _REQUIRED_DVZ_PANEL_FRAME_GUIDE_QUERY_FUNCTIONS,
    _REQUIRED_DVZ_PANEL_FRAME_SNAPSHOT_FUNCTIONS,
    _REQUIRED_DVZ_PATH_FUNCTIONS,
    _REQUIRED_DVZ_SAMPLED_FIELD_FUNCTIONS,
    _REQUIRED_DVZ_SEGMENT_FUNCTIONS,
    _REQUIRED_DVZ_TEXTURE2D_MESH_FUNCTIONS,
    _REQUIRED_DVZ_VIEW3D_CAMERA_FUNCTIONS,
    _REQUIRED_DVZ_VIEW3D_RETAINED_DATA_FUNCTIONS,
    _STROKE_CAP_FALLBACKS,
    _STROKE_CAP_NAMES,
    _STROKE_JOIN_FALLBACKS,
    _STROKE_JOIN_NAMES,
    _VECTOR_CAP_NAMES,
    _VISUAL_ATTRIBUTE_ALIASES,
    DVZ_ALPHA_BLENDED,
    DVZ_CAMERA_ORTHOGRAPHIC,
    DVZ_CAMERA_PERSPECTIVE,
    DVZ_COLOR_PIPELINE_LEGACY_SRGB_BLEND,
    DVZ_COLOR_PIPELINE_LINEAR_SRGB,
    DVZ_CONTROLLER_APPLY,
    DVZ_CONTROLLER_FIXED,
    DVZ_IMAGE_SAMPLING_LINEAR,
    DVZ_IMAGE_SAMPLING_NEAREST,
    DVZ_SHAPE_ASPECT_FILLED,
    DVZ_SHAPE_ASPECT_OUTLINE,
    DatovizColorPipeline,
    DatovizV04Unavailable,
    DatovizV04Unsupported,
)
from .colors import (
    _rgba8,
    _rgba8_scalar,
)


def is_datoviz_v04_facade(module: ModuleType | Any) -> bool:
    """Return whether a module-like object exposes the required v0.4 facade."""
    return not datoviz_current_api_contract_diagnostics(module)


def datoviz_v04_sampled_field_ready(module: ModuleType | Any) -> bool:
    """Return whether a facade exposes the sampled-field image binding path."""
    return not datoviz_v04_sampled_field_diagnostics(module)


def datoviz_v04_sampled_field_diagnostics(module: ModuleType | Any) -> tuple[str, ...]:
    """Return missing sampled-field binding requirements."""
    return tuple(
        f"missing {name}"
        for name in _REQUIRED_DVZ_SAMPLED_FIELD_FUNCTIONS
        if not hasattr(module, name)
    )


def datoviz_v04_live_input_ready(module: ModuleType | Any) -> bool:
    """Return whether the Datoviz facade exposes the S035 live input substrate."""
    return not datoviz_v04_live_input_diagnostics(module)


def datoviz_v04_live_input_diagnostics(module: ModuleType | Any) -> tuple[str, ...]:
    """Return why Datoviz live pointer input is unavailable for GSP navigation."""
    diagnostics = [
        f"missing {name}"
        for name in (
            "dvz_view_input",
            "dvz_input_subscribe_event",
            "dvz_input_unsubscribe",
            "DvzPointerEvent",
            "DvzInputEvent",
            "DvzInputEventContent",
        )
        if not hasattr(module, name)
    ]
    pointer_event_type = getattr(module, "DvzPointerEvent", None)
    if pointer_event_type is not None:
        try:
            pointer_event = pointer_event_type()
        except TypeError:
            pointer_event = None
        if pointer_event is None or not hasattr(pointer_event, "window_size"):
            diagnostics.append("DvzPointerEvent missing window_size field")
    input_event_type = getattr(module, "DvzInputEvent", None)
    if input_event_type is not None:
        try:
            input_event = input_event_type()
        except TypeError:
            input_event = None
        if input_event is None or not hasattr(input_event, "content"):
            diagnostics.append("DvzInputEvent missing content union")
    input_event_content_type = getattr(module, "DvzInputEventContent", None)
    if input_event_content_type is not None and not hasattr(input_event_content_type, "pointer"):
        diagnostics.append("DvzInputEventContent missing pointer field")
    return tuple(diagnostics)


def datoviz_v04_mesh_diagnostics(module: ModuleType | Any) -> tuple[str, ...]:
    """Return why Datoviz MeshVisual rendering is disabled for this slice."""
    diagnostics = [
        f"missing {name}" for name in _REQUIRED_DVZ_MESH_FUNCTIONS if not hasattr(module, name)
    ]
    diagnostics.extend(
        f"unverified {name}"
        for name in _OPTIONAL_UNVERIFIED_DVZ_MESH_FUNCTIONS
        if not hasattr(module, name)
    )
    return tuple(diagnostics)


def datoviz_v04_mesh_ready(module: ModuleType | Any) -> bool:
    """Return whether this adapter slice may render MeshVisual through Datoviz."""
    return not datoviz_v04_mesh_diagnostics(module)


def _datoviz_texture2d_mesh_diagnostics(module: ModuleType | Any) -> tuple[str, ...]:
    """Return missing public APIs for the strict Texture2D mesh slice."""
    return tuple(
        f"missing {name}"
        for name in _REQUIRED_DVZ_TEXTURE2D_MESH_FUNCTIONS
        if not hasattr(module, name)
    )


def datoviz_v04_view3d_camera_diagnostics(module: ModuleType | Any) -> tuple[str, ...]:
    """Return why Datoviz View3D camera binding is unavailable."""
    return tuple(
        f"missing {name}"
        for name in _REQUIRED_DVZ_VIEW3D_CAMERA_FUNCTIONS
        if not callable(getattr(module, name, None))
    )


def datoviz_v04_view3d_retained_data_diagnostics(
    module: ModuleType | Any,
) -> tuple[str, ...]:
    """Return why retained DATA-space View3D visual attachment is unavailable."""
    return (
        *datoviz_v04_view3d_camera_diagnostics(module),
        *(
            f"missing {name}"
            for name in _REQUIRED_DVZ_VIEW3D_RETAINED_DATA_FUNCTIONS
            if not callable(getattr(module, name, None))
        ),
    )


def datoviz_v04_view3d_retained_data_ready(module: ModuleType | Any) -> bool:
    """Return whether Datoviz can keep 3D vertices in DATA space under View3D."""
    return not datoviz_v04_view3d_retained_data_diagnostics(module)


def datoviz_v04_view3d_state_readback_diagnostics(
    module: ModuleType | Any,
) -> tuple[str, ...]:
    """Return why native retained View3D state readback is unavailable."""
    return _state_readback_diagnostics(module)


def datoviz_v04_view3d_state_readback_ready(module: ModuleType | Any) -> bool:
    """Return whether native retained View3D state may be read through ctypes."""
    return not datoviz_v04_view3d_state_readback_diagnostics(module)


def datoviz_v04_panel_frame_snapshot_diagnostics(
    module: ModuleType | Any,
) -> tuple[str, ...]:
    """Return why Datoviz panel frame snapshot readback is unavailable."""
    missing = tuple(
        f"missing {name}"
        for name in _REQUIRED_DVZ_PANEL_FRAME_SNAPSHOT_FUNCTIONS
        if not hasattr(module, name)
    )
    return missing + _datoviz_incomplete_ctypes_records(
        module,
        ("DvzPanelFrameInfo", "DvzGuideLayout", "DvzRenderedContribution"),
    )


def datoviz_v04_panel_frame_snapshot_ready(module: ModuleType | Any) -> bool:
    """Return whether the facade exposes the bounded S043 snapshot readback path."""
    return not datoviz_v04_panel_frame_snapshot_diagnostics(module)


def datoviz_v04_panel_frame_guide_query_diagnostics(
    module: ModuleType | Any,
) -> tuple[str, ...]:
    """Return why Datoviz panel frame guide hit/readback is unavailable."""
    missing = tuple(
        f"missing {name}"
        for name in _REQUIRED_DVZ_PANEL_FRAME_GUIDE_QUERY_FUNCTIONS
        if not callable(getattr(module, name, None))
    )
    return missing + _datoviz_incomplete_ctypes_records(module, ("DvzGuideHit",))


def _datoviz_incomplete_ctypes_records(
    module: ModuleType | Any, names: tuple[str, ...]
) -> tuple[str, ...]:
    """Reject generated ctypes forward declarations with no usable layout."""
    diagnostics: list[str] = []
    for name in names:
        record_type = getattr(module, name, None)
        if not isinstance(record_type, type):
            continue
        try:
            is_ctypes_record = issubclass(record_type, ctypes.Structure)
        except TypeError:
            continue
        if is_ctypes_record and ctypes.sizeof(record_type) == 0:
            diagnostics.append(f"incomplete ctypes layout for {name}")
    return tuple(diagnostics)


def datoviz_v04_panel_frame_guide_query_ready(module: ModuleType | Any) -> bool:
    """Return whether the facade exposes Datoviz guide hit/readback APIs."""
    return datoviz_v04_panel_frame_snapshot_ready(
        module
    ) and not datoviz_v04_panel_frame_guide_query_diagnostics(module)


def datoviz_v04_view3d_live_navigation_diagnostics(
    module: ModuleType | Any,
) -> tuple[str, ...]:
    """Return why Datoviz View3D live navigation is unavailable."""
    diagnostics = [
        *datoviz_v04_live_input_diagnostics(module),
    ]
    if os.environ.get("GSP_DATOVIZ_ENABLE_EXPERIMENTAL_VIEW3D_NAV") != "1":
        diagnostics.append(
            "Datoviz View3D live navigation is experimental and failed manual "
            "review; set GSP_DATOVIZ_ENABLE_EXPERIMENTAL_VIEW3D_NAV=1 to opt in"
        )
    retained_diagnostics = datoviz_v04_view3d_retained_data_diagnostics(module)
    if retained_diagnostics:
        diagnostics.extend(retained_diagnostics)
        diagnostics.append(
            "Datoviz View3D live navigation requires the retained DATA-space "
            "View3D visual path from the current generated binding"
        )
    readback_diagnostics = datoviz_v04_view3d_state_readback_diagnostics(module)
    if readback_diagnostics:
        diagnostics.extend(readback_diagnostics)
        diagnostics.append("Datoviz View3D live navigation requires native View3D state readback")
    return tuple(diagnostics)


def datoviz_v04_text_diagnostics(module: ModuleType | Any) -> tuple[str, ...]:
    """Return why Datoviz TextVisual rendering is disabled for this slice."""
    diagnostics = list(datoviz_text_api_diagnostics(module))
    diagnostics.extend(
        f"unverified {name}"
        for name in _OPTIONAL_UNVERIFIED_DVZ_TEXT_FUNCTIONS
        if not hasattr(module, name)
    )
    return tuple(diagnostics)


def datoviz_v04_text_ready(module: ModuleType | Any) -> bool:
    """Return whether this adapter slice may render TextVisual through Datoviz."""
    return not datoviz_v04_text_diagnostics(module)


def import_datoviz_v04() -> ModuleType:
    """Import Datoviz and validate the C-shaped v0.4 facade."""
    bootstrap_datoviz_v04_source()
    try:
        import datoviz as dvz

        diagnostics = datoviz_current_api_contract_diagnostics(dvz)
        if diagnostics:
            raise DatovizV04Unavailable("; ".join(diagnostics))
    except ModuleNotFoundError as exc:
        raise DatovizV04Unavailable("Datoviz is not importable") from exc
    except (OSError, RuntimeError) as exc:
        raise DatovizV04Unavailable(f"Datoviz is not importable: {exc}") from exc
    return cast(ModuleType, dvz)


def capability_snapshot() -> CapabilitySnapshot:
    """Return the GSP capability surface for the current bounded adapter slice."""
    return datoviz_v04_capability_snapshot()


def _datoviz_color_pipeline_value(dvz: Any, color_pipeline: DatovizColorPipeline) -> int:
    """Return the Datoviz enum value for a GSP color-pipeline option."""
    if color_pipeline == "linear_srgb":
        return int(getattr(dvz, "DVZ_COLOR_PIPELINE_LINEAR_SRGB", DVZ_COLOR_PIPELINE_LINEAR_SRGB))
    if color_pipeline == "legacy_srgb_blend":
        return int(
            getattr(
                dvz,
                "DVZ_COLOR_PIPELINE_LEGACY_SRGB_BLEND",
                DVZ_COLOR_PIPELINE_LEGACY_SRGB_BLEND,
            )
        )
    raise ValueError(f"unsupported Datoviz color pipeline: {color_pipeline!r}")


def _set_figure_color_pipeline(dvz: Any, figure: Any, color_pipeline: DatovizColorPipeline) -> None:
    """Set the Datoviz figure color pipeline when the facade supports or requires it."""
    value = _datoviz_color_pipeline_value(dvz, color_pipeline)
    setter = getattr(dvz, "dvz_figure_set_color_pipeline", None)
    if setter is None:
        if color_pipeline == "linear_srgb":
            return
        raise DatovizV04Unavailable(
            "Datoviz legacy sRGB blend mode is unavailable: missing dvz_figure_set_color_pipeline"
        )
    setter(figure, value)


def _configure_datoviz_view3d_camera(dvz: Any, panel: Any, view3d: View3D) -> Any:
    diagnostics = datoviz_v04_view3d_camera_diagnostics(dvz)
    if diagnostics:
        raise DatovizV04Unavailable(
            "Datoviz View3D camera binding is unavailable: " + "; ".join(diagnostics)
        )

    desc = dvz.dvz_panel_view3d_desc()
    _fill_datoviz_camera_desc(dvz, _panel_view3d_camera_desc(desc), view3d)
    _require_datoviz_success(
        dvz.dvz_panel_set_view3d_desc(panel, _ctypes_pointer_arg(desc)),
        "Datoviz retained View3D descriptor setup failed",
    )
    camera = dvz.dvz_panel_camera(panel)
    if _is_null_handle(camera):
        raise DatovizV04Unsupported("Datoviz retained View3D panel camera is unavailable")
    _set_datoviz_camera_projection_state(dvz, camera, view3d)
    return camera


def _fill_datoviz_camera_desc(dvz: Any, desc: Any, view3d: View3D) -> None:
    for index, value in enumerate(view3d.camera.eye):
        desc.view.eye[index] = float(value)
    for index, value in enumerate(view3d.camera.target):
        desc.view.target[index] = float(value)
    for index, value in enumerate(view3d.camera.up):
        desc.view.up[index] = float(value)
    near, far = view3d.projection.near_far
    desc.projection.near_clip = float(near)
    desc.projection.far_clip = float(far)
    if isinstance(view3d.projection, PerspectiveProjection3D):
        desc.projection.type = _enum_value(
            dvz, "DvzCameraType", "DVZ_CAMERA_PERSPECTIVE", DVZ_CAMERA_PERSPECTIVE
        )
        desc.projection.fov_y = float(math.radians(view3d.projection.fov_y_degrees))
    else:
        desc.projection.type = _enum_value(
            dvz, "DvzCameraType", "DVZ_CAMERA_ORTHOGRAPHIC", DVZ_CAMERA_ORTHOGRAPHIC
        )
        desc.projection.ortho_height = float(
            abs(view3d.projection.ylim[1] - view3d.projection.ylim[0])
        )


def _set_datoviz_camera_projection_state(dvz: Any, camera: Any, view3d: View3D) -> None:
    if isinstance(view3d.projection, OrthographicProjection3D):
        _set_datoviz_camera_orthographic_bounds(dvz, camera, view3d)


def _set_datoviz_camera_orthographic_bounds(dvz: Any, camera: Any, view3d: View3D) -> None:
    if not isinstance(view3d.projection, OrthographicProjection3D):
        raise DatovizV04Unsupported("Datoviz orthographic bounds require OrthographicProjection3D")
    x0, x1 = view3d.projection.xlim
    y0, y1 = view3d.projection.ylim
    near, far = view3d.projection.near_far
    _require_datoviz_success(
        dvz.dvz_camera_set_orthographic_bounds(
            camera, float(x0), float(x1), float(y0), float(y1), float(near), float(far)
        ),
        "Datoviz View3D orthographic bounds setup failed",
    )


def _update_datoviz_view3d_camera(dvz: Any, panel: Any, view3d: View3D) -> Any:
    diagnostics = datoviz_v04_view3d_camera_diagnostics(dvz)
    if diagnostics:
        raise DatovizV04Unavailable(
            "Datoviz retained View3D descriptor binding is unavailable: " + "; ".join(diagnostics)
        )
    panel_desc = dvz.dvz_panel_view3d_desc()
    _fill_datoviz_camera_desc(dvz, _panel_view3d_camera_desc(panel_desc), view3d)
    _require_datoviz_success(
        dvz.dvz_panel_set_view3d_desc(panel, _ctypes_pointer_arg(panel_desc)),
        "Datoviz retained View3D panel camera descriptor update failed",
    )
    camera = dvz.dvz_panel_camera(panel)
    if _is_null_handle(camera):
        raise DatovizV04Unsupported("Datoviz retained View3D panel camera update failed")
    _set_datoviz_camera_projection_state(dvz, camera, view3d)
    return camera


def _panel_view3d_camera_desc(desc: Any) -> Any:
    """Return the camera-shaped payload inside a Datoviz View3D panel descriptor."""
    return getattr(desc, "camera", desc)


def _resolve_datoviz_canvas_size(dvz: Any, requested: CanvasSize) -> ResolvedCanvas:
    desc = _datoviz_view_size_desc(dvz, requested)
    resolver = getattr(dvz, "dvz_view_size_resolve", None)
    if desc is not None and resolver is not None:
        try:
            native = resolver(desc, _datoviz_view_kind_value(dvz, "glfw"))
        except (ctypes.ArgumentError, TypeError, ValueError):
            native = None
        if native is not None:
            return _resolved_canvas_from_datoviz(requested, native)

    scale = requested.requested_device_scale or 1.0
    output_dpi = requested.reference_dpi * scale
    return requested.resolve(
        output_dpi=output_dpi,
        device_scale=scale,
        metrics_source=CanvasMetricsSource.BACKEND_DEFAULT,
    )


def _datoviz_view_size_desc(dvz: Any, requested: CanvasSize) -> Any | None:
    factory_name = {
        "pixel_exact": "dvz_view_size_desc_framebuffer_px",
        "host_logical_px": "dvz_view_size_desc_host_logical_px",
        "reference_px": "dvz_view_size_desc_reference_px",
        "physical_mm": "dvz_view_size_desc_physical_mm",
    }[requested.policy.value]
    factory = getattr(dvz, factory_name, None)
    if factory is None:
        return None
    if requested.policy.value in {"reference_px", "physical_mm"}:
        desc = factory(requested.width, requested.height, requested.reference_dpi)
    else:
        desc = factory(requested.width, requested.height)
    if requested.requested_device_scale is not None and hasattr(desc, "requested_device_scale"):
        desc.requested_device_scale = float(requested.requested_device_scale)
    if requested.monitor_dpi_override is not None:
        _set_datoviz_monitor_dpi_override(desc, float(requested.monitor_dpi_override))
    if hasattr(desc, "strict_framebuffer_size"):
        desc.strict_framebuffer_size = bool(requested.strict_framebuffer_size)
    return desc


def _datoviz_view_kind_value(dvz: Any, name: str) -> int:
    if name == "glfw":
        return int(getattr(dvz, "DVZ_VIEW_GLFW", 1))
    return int(getattr(dvz, "DVZ_VIEW_OFFSCREEN", 2))


def _set_datoviz_monitor_dpi_override(desc: Any, dpi: float) -> None:
    if hasattr(desc, "monitor_dpi_x_override"):
        desc.monitor_dpi_x_override = dpi
    if hasattr(desc, "monitor_dpi_y_override"):
        desc.monitor_dpi_y_override = dpi
    if hasattr(desc, "monitor_dpi_override"):
        desc.monitor_dpi_override = dpi


def _resolved_canvas_from_datoviz(requested: CanvasSize, native: Any) -> ResolvedCanvas:
    fallback = requested.resolve()
    framebuffer_per_canvas_px_x = float(getattr(native, "framebuffer_per_canvas_px_x", 1.0))
    framebuffer_per_canvas_px_y = float(getattr(native, "framebuffer_per_canvas_px_y", 1.0))
    target_width_mm = _positive_native_float(native, "target_width_mm", fallback.target_width_mm)
    target_height_mm = _positive_native_float(native, "target_height_mm", fallback.target_height_mm)
    estimated_width_mm = _positive_native_float(native, "estimated_width_mm", target_width_mm)
    estimated_height_mm = _positive_native_float(native, "estimated_height_mm", target_height_mm)
    return ResolvedCanvas(
        requested_size=requested,
        canvas_width_px=float(getattr(native, "canvas_width_px")),
        canvas_height_px=float(getattr(native, "canvas_height_px")),
        host_logical_width=int(getattr(native, "host_logical_width")),
        host_logical_height=int(getattr(native, "host_logical_height")),
        framebuffer_width=int(getattr(native, "framebuffer_width")),
        framebuffer_height=int(getattr(native, "framebuffer_height")),
        device_scale_x=float(getattr(native, "device_scale_x", 1.0)),
        device_scale_y=float(getattr(native, "device_scale_y", 1.0)),
        canvas_to_host_scale_x=float(getattr(native, "canvas_to_host_scale_x", 1.0)),
        canvas_to_host_scale_y=float(getattr(native, "canvas_to_host_scale_y", 1.0)),
        framebuffer_per_canvas_px_x=framebuffer_per_canvas_px_x,
        framebuffer_per_canvas_px_y=framebuffer_per_canvas_px_y,
        target_width_mm=target_width_mm,
        target_height_mm=target_height_mm,
        estimated_width_mm=estimated_width_mm,
        estimated_height_mm=estimated_height_mm,
        output_dpi=float(requested.reference_dpi * framebuffer_per_canvas_px_x),
        metrics_source=CanvasMetricsSource.BACKEND_REPORTED,
        exactness=fallback.exactness,
        strict_framebuffer_size=bool(requested.strict_framebuffer_size),
    )


def _positive_native_float(native: Any, field_name: str, fallback: float) -> float:
    value = float(getattr(native, field_name, fallback))
    return value if value > 0.0 else fallback


def _set_datoviz_data_domain(panel_view: Any, field_name: str, limits: tuple[float, float]) -> None:
    domain = getattr(panel_view, field_name, None)
    if domain is None:
        raise DatovizV04Unsupported(f"Datoviz View2D descriptor is missing {field_name}")
    if not hasattr(domain, "min") or not hasattr(domain, "max"):
        raise DatovizV04Unsupported(f"Datoviz View2D descriptor {field_name} is not writable")
    domain.min = float(limits[0])
    domain.max = float(limits[1])


def _datoviz_view2d_descriptor_has_data_domains(panel_view: Any) -> bool:
    for field_name in ("data_x", "data_y"):
        domain = getattr(panel_view, field_name, None)
        if domain is None or not hasattr(domain, "min") or not hasattr(domain, "max"):
            return False
    return True


def _set_datoviz_panel_domains(
    dvz: Any,
    panel: Any,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> None:
    """Set ordered panel DATA domains through the current Datoviz v0.4 contract."""
    setter = getattr(dvz, "dvz_panel_set_domain", None)
    if setter is None:
        raise DatovizV04Unsupported(
            "Datoviz View2D data-domain setup failed: missing dvz_panel_set_domain"
        )
    dim_x = getattr(dvz, "DVZ_DIM_X", 0)
    dim_y = getattr(dvz, "DVZ_DIM_Y", 1)
    for dim, limits in ((dim_x, x_range), (dim_y, y_range)):
        _require_datoviz_success(
            setter(panel, dim, float(limits[0]), float(limits[1])),
            "Datoviz panel data-domain setup failed",
        )


def _set_axis_ticks(
    dvz: Any,
    axis: Any,
    values: tuple[float, ...],
    labels: tuple[str, ...] | None,
) -> None:
    tick_values = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    if labels is None:
        result = dvz.dvz_axis_set_ticks(axis, tick_values, None)
    else:
        encoded_labels = tuple(label.encode("utf-8") for label in labels)
        result = dvz.dvz_axis_set_ticks(axis, tick_values, encoded_labels)
    _require_datoviz_success(
        result,
        "Datoviz explicit axis tick configuration failed",
    )


def _is_null_handle(handle: Any) -> bool:
    """Return whether a Datoviz/ctypes handle is absent or a NULL pointer."""
    if handle is None:
        return True
    try:
        return not bool(handle)
    except TypeError:
        return False


def _visual_attach_desc(
    dvz: Any, *, coord_space: str, z_layer: int, controller_mode: str = "apply"
) -> Any:
    factory = getattr(dvz, "dvz_visual_attach_desc", None)
    if factory is not None:
        desc = factory()
    else:
        desc_type = getattr(dvz, "DvzVisualAttachDesc", None)
        if desc_type is None:
            raise DatovizV04Unavailable("Datoviz facade is missing DvzVisualAttachDesc")
        desc = desc_type()
        if hasattr(desc, "struct_size"):
            desc.struct_size = ctypes.sizeof(desc_type)
        if hasattr(desc, "flags"):
            desc.flags = 0

    _validate_visual_attach_desc_binding(desc)
    desc.z_layer = z_layer
    if controller_mode == "apply":
        desc.controller_mode = _controller_mode_value(
            dvz, "DVZ_CONTROLLER_APPLY", DVZ_CONTROLLER_APPLY
        )
    elif controller_mode == "fixed":
        desc.controller_mode = _controller_mode_value(
            dvz, "DVZ_CONTROLLER_FIXED", DVZ_CONTROLLER_FIXED
        )
    else:
        raise ValueError(f"unsupported Datoviz controller mode: {controller_mode}")
    if coord_space == "data":
        desc.coord_space = int(dvz.DVZ_VISUAL_COORD_DATA)
    elif coord_space == "view":
        desc.coord_space = int(dvz.DVZ_VISUAL_COORD_VIEW)
    else:
        raise ValueError(f"unsupported Datoviz coordinate space: {coord_space}")
    if hasattr(desc, "clip_rect"):
        desc.clip_rect = _enum_value(dvz, "DvzVisualClipRect", "DVZ_VISUAL_CLIP_AUTO", 0)
    if hasattr(desc, "viewport_rect"):
        desc.viewport_rect = _enum_value(
            dvz, "DvzVisualViewportRect", "DVZ_VISUAL_VIEWPORT_AUTO", 0
        )
    return desc


def _validate_visual_attach_desc_binding(desc: Any) -> None:
    missing = [name for name in ("clip_rect", "viewport_rect") if not hasattr(desc, name)]
    if missing:
        raise DatovizV04Unavailable(
            "Datoviz Python binding is stale: DvzVisualAttachDesc is missing "
            f"{', '.join(missing)}. Regenerate Datoviz bindings with `just ctypes`."
        )


def _add_visual_to_panel(dvz: Any, panel: Any, visual: Any, attach_desc: Any) -> None:
    _require_datoviz_success(
        dvz.dvz_panel_add_visual(panel, visual, attach_desc),
        "Datoviz visual panel attachment failed",
    )


def _datoviz_visual_coord_space(coordinate_space: CoordinateSpace) -> str:
    if coordinate_space is CoordinateSpace.NDC:
        return "view"
    if coordinate_space is CoordinateSpace.DATA:
        return "data"
    raise DatovizV04Unsupported(
        f"Datoviz visual coordinate space is unsupported: {coordinate_space.value}"
    )


def _controller_mode_value(dvz: Any, name: str, fallback: int) -> int:
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzControllerMode", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return fallback


def _retained_view3d_state_mismatch_diagnostics(
    view3d: View3D, snapshot: Mapping[str, object]
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if not bool(snapshot.get("enabled", False)):
        diagnostics.append(
            f"{View3DDiagnosticCode.VIEW3D_NAVIGATION_INVALID_RESULT.value}: "
            "Datoviz retained View3D state readback is disabled"
        )
    camera = view3d.camera
    expected_vectors = {
        "camera_eye": camera.eye,
        "camera_target": camera.target,
        "camera_up": camera.up,
    }
    for key, expected in expected_vectors.items():
        observed = snapshot.get(key)
        if observed is None or not np.allclose(
            np.asarray(observed, dtype=np.float64),
            np.asarray(expected, dtype=np.float64),
        ):
            diagnostics.append(
                f"{View3DDiagnosticCode.VIEW3D_NAVIGATION_INVALID_RESULT.value}: "
                f"Datoviz retained View3D {key} does not match canonical state"
            )
    observed_near_far = snapshot.get("near_far")
    if observed_near_far is None or not np.allclose(
        np.asarray(observed_near_far, dtype=np.float64),
        np.asarray(view3d.projection.near_far, dtype=np.float64),
    ):
        diagnostics.append(
            f"{View3DDiagnosticCode.VIEW3D_NAVIGATION_INVALID_RESULT.value}: "
            "Datoviz retained View3D near/far bounds do not match canonical state"
        )
    if isinstance(view3d.projection, OrthographicProjection3D):
        expected_bounds = (
            *view3d.projection.xlim,
            *view3d.projection.ylim,
            *view3d.projection.near_far,
        )
        observed_bounds = snapshot.get("orthographic_bounds")
        if observed_bounds is None or not np.allclose(
            np.asarray(observed_bounds, dtype=np.float64),
            np.asarray(expected_bounds, dtype=np.float64),
        ):
            diagnostics.append(
                f"{View3DDiagnosticCode.VIEW3D_NAVIGATION_INVALID_RESULT.value}: "
                "Datoviz retained View3D orthographic bounds do not match canonical state"
            )
    elif isinstance(view3d.projection, PerspectiveProjection3D):
        observed_fov_y = snapshot.get("fov_y_radians")
        expected_fov_y = math.radians(view3d.projection.fov_y_degrees)
        if not isinstance(observed_fov_y, (int, float)) or not np.isclose(
            float(observed_fov_y), expected_fov_y
        ):
            diagnostics.append(
                f"{View3DDiagnosticCode.VIEW3D_NAVIGATION_INVALID_RESULT.value}: "
                "Datoviz retained View3D perspective fov_y does not match canonical state"
            )
    return tuple(diagnostics)


def _datoviz_label(value: object) -> str | None:
    if isinstance(value, bytes):
        raw = value
    elif isinstance(value, bytearray):
        raw = bytes(value)
    elif isinstance(value, str):
        return value or None
    else:
        try:
            raw = bytes(cast(Any, value))
        except (TypeError, ValueError):
            return None
    text = raw.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return text or None


def _datoviz_shared_library_function(dvz: Any, name: str) -> Any | None:
    module_file = getattr(dvz, "__file__", None)
    if not isinstance(module_file, str):
        return None
    module_dir = Path(module_file).resolve().parent
    for filename in ("libdatoviz.so", "libdatoviz.dylib", "datoviz.dll"):
        library_path = module_dir / filename
        if not library_path.exists():
            continue
        try:
            library = ctypes.CDLL(str(library_path))
            return getattr(library, name)
        except (OSError, AttributeError):
            continue
    return None


def _set_data_view_payload(view: Any, pixels: npt.NDArray[np.uint8]) -> None:
    try:
        view.data = pixels
    except TypeError:
        view.data = pixels.ctypes.data


def _datoviz_call_succeeded(result: Any) -> bool:
    """Accept both old bool-returning mutators and pre-RC DvzResult returns."""
    if result is None:
        return True
    if isinstance(result, bool):
        return result
    value = getattr(result, "value", None)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value == 0
    if isinstance(result, int):
        return result == 0
    return bool(result)


def _require_datoviz_success(result: Any, message: str) -> None:
    if not _datoviz_call_succeeded(result):
        raise DatovizV04Unsupported(message)


def _set_visual_field(dvz: Any, visual: Any, slot_name: str, sampled_field: Any) -> bool:
    try:
        return _datoviz_call_succeeded(dvz.dvz_visual_set_field(visual, slot_name, sampled_field))
    except (ctypes.ArgumentError, TypeError):
        return _datoviz_call_succeeded(
            dvz.dvz_visual_set_field(visual, slot_name.encode("utf-8"), sampled_field)
        )


def _set_visual_field_sampling(dvz: Any, visual: Any, slot_name: str, sampling: Any) -> Any:
    """Call the slot-sampling setter across direct and ctypes facades."""
    sampling_arg = _ctypes_pointer_arg(sampling)
    try:
        return dvz.dvz_visual_set_field_sampling(visual, slot_name, sampling_arg)
    except (ctypes.ArgumentError, TypeError):
        return dvz.dvz_visual_set_field_sampling(visual, slot_name.encode("utf-8"), sampling_arg)


def _ctypes_pointer_arg(value: Any) -> Any:
    if isinstance(value, ctypes.Structure):
        return ctypes.byref(value)
    return value


def _enum_value(dvz: Any, enum_type_name: str, name: str, fallback: int) -> int:
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, enum_type_name, None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return fallback


def _image_sampling_value(dvz: Any, interpolation: ImageInterpolation) -> int:
    if interpolation == ImageInterpolation.NEAREST:
        name = "DVZ_IMAGE_SAMPLING_NEAREST"
        fallback = DVZ_IMAGE_SAMPLING_NEAREST
    elif interpolation == ImageInterpolation.LINEAR:
        name = "DVZ_IMAGE_SAMPLING_LINEAR"
        fallback = DVZ_IMAGE_SAMPLING_LINEAR
    else:
        raise DatovizV04Unsupported(f"unsupported Datoviz image interpolation: {interpolation}")

    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzImageSampling", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return fallback


def _set_image_sampling(dvz: Any, visual: Any, interpolation: ImageInterpolation) -> None:
    setter = getattr(dvz, "dvz_image_set_sampling", None)
    if setter is None:
        return
    _require_datoviz_success(
        setter(visual, _image_sampling_value(dvz, interpolation)),
        "Datoviz image sampling configuration failed",
    )


def _set_query_capabilities(dvz: Any, visual: Any, capabilities: int) -> None:
    setter = getattr(dvz, "dvz_visual_set_query_capabilities", None)
    if setter is None:
        return
    setter(visual, capabilities)


def _set_panel_background_color(dvz: Any, panel: Any, rgba: tuple[int, int, int, int]) -> None:
    setter = getattr(dvz, "dvz_panel_set_background_color", None)
    if setter is None:
        return
    color = _dvz_color(dvz, rgba)
    setter(panel, color)


def _dvz_color(dvz: Any, rgba: tuple[int, int, int, int]) -> Any:
    color_type = getattr(dvz, "DvzColor", None)
    if color_type is None:
        return rgba
    try:
        return color_type(*rgba)
    except TypeError:
        color = color_type()
        for channel, value in zip(("r", "g", "b", "a"), rgba, strict=True):
            setattr(color, channel, int(value))
        return color


def _set_visual_data(dvz: Any, visual: Any, attr_name: str, data: npt.NDArray[Any]) -> None:
    result = dvz.dvz_visual_set_data(visual, attr_name, data)
    if _datoviz_call_succeeded(result):
        return
    for alias in _VISUAL_ATTRIBUTE_ALIASES.get(attr_name, ()):
        result = dvz.dvz_visual_set_data(visual, alias, data)
        if _datoviz_call_succeeded(result):
            return
    raise DatovizV04Unsupported(f"Datoviz visual attribute {attr_name!r} upload failed")


def _set_visual_index_data(dvz: Any, visual: Any, indices: npt.NDArray[np.uint32]) -> None:
    _require_datoviz_success(
        dvz.dvz_visual_set_index_data(visual, indices, int(indices.shape[0])),
        "Datoviz visual index upload failed",
    )


def _set_filled_point_style(dvz: Any, visual: Any) -> None:
    """Force the documented filled/no-stroke point style when the facade exposes it."""
    style_factory = getattr(dvz, "dvz_point_style_desc", None)
    style_setter = getattr(dvz, "dvz_point_set_style", None)
    if style_factory is None or style_setter is None:
        return
    style = style_factory()
    if hasattr(style, "stroke_width"):
        style.stroke_width = 0.0
    if hasattr(style, "aspect"):
        style.aspect = int(getattr(dvz, "DVZ_SHAPE_ASPECT_FILLED", 0))
    _require_datoviz_success(
        style_setter(visual, style),
        "Datoviz point filled/no-stroke style configuration failed",
    )


def _set_alpha_mode_if_translucent(dvz: Any, visual: Any, colors: npt.NDArray[np.uint8]) -> None:
    if not np.any(colors[:, 3] < 255):
        return
    setter = getattr(dvz, "dvz_visual_set_alpha_mode", None)
    if setter is None:
        raise DatovizV04Unsupported("Datoviz translucent colors require dvz_visual_set_alpha_mode")
    _require_datoviz_success(
        setter(visual, _alpha_mode_value(dvz, "DVZ_ALPHA_BLENDED", DVZ_ALPHA_BLENDED)),
        "Datoviz alpha blending configuration failed",
    )


def _alpha_mode_value(dvz: Any, name: str, fallback: int) -> int:
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzAlphaMode", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return fallback


def _datoviz_marker_diagnostics(dvz: Any) -> tuple[str, ...]:
    return tuple(
        f"missing {name}" for name in _REQUIRED_DVZ_MARKER_FUNCTIONS if not hasattr(dvz, name)
    )


def _datoviz_segment_diagnostics(dvz: Any) -> tuple[str, ...]:
    return tuple(
        f"missing {name}" for name in _REQUIRED_DVZ_SEGMENT_FUNCTIONS if not hasattr(dvz, name)
    )


def _datoviz_path_diagnostics(dvz: Any) -> tuple[str, ...]:
    return tuple(
        f"missing {name}" for name in _REQUIRED_DVZ_PATH_FUNCTIONS if not hasattr(dvz, name)
    )


def _datoviz_colorbar_diagnostics(dvz: Any) -> tuple[str, ...]:
    required = (
        "dvz_scale_desc",
        "dvz_scale",
        "dvz_scale_set_domain",
        "dvz_scale_set_view_range",
        "dvz_scale_set_colormap",
        "dvz_colormap_builtin",
        "dvz_colorbar_desc",
        "dvz_colorbar",
        "dvz_colorbar_set_anchor",
        "dvz_colorbar_set_orientation",
        "dvz_colorbar_set_title",
    )
    return tuple(f"missing {name}" for name in required if not hasattr(dvz, name))


def _set_path_subpaths(dvz: Any, visual: Any, count: int, subpaths: npt.NDArray[np.uint32]) -> Any:
    try:
        return dvz.dvz_path_set_subpaths(visual, count, subpaths)
    except (ctypes.ArgumentError, TypeError):
        pointer = subpaths.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32))
        return dvz.dvz_path_set_subpaths(visual, count, pointer)


def _set_marker_style(
    dvz: Any, visual: Any, stroke_color: npt.NDArray[Any], stroke_width: float
) -> None:
    style = dvz.dvz_marker_style()
    _assign_rgba_field(style, "edge_color", _rgba8_scalar(stroke_color))
    if hasattr(style, "stroke_width_px"):
        style.stroke_width_px = float(stroke_width)
    elif hasattr(style, "stroke_width"):
        style.stroke_width = float(stroke_width)
    if hasattr(style, "aspect"):
        if stroke_width > 0.0 and _rgba8_scalar(stroke_color)[3] > 0:
            style.aspect = int(getattr(dvz, "DVZ_SHAPE_ASPECT_OUTLINE", DVZ_SHAPE_ASPECT_OUTLINE))
        else:
            style.aspect = int(getattr(dvz, "DVZ_SHAPE_ASPECT_FILLED", DVZ_SHAPE_ASPECT_FILLED))
    _require_datoviz_success(
        dvz.dvz_marker_set_style(visual, style),
        "Datoviz marker style configuration failed",
    )


def _assign_rgba_field(target: Any, field_name: str, rgba: npt.NDArray[np.uint8]) -> None:
    values = [int(value) for value in rgba]
    field = getattr(target, field_name, None)
    if field is not None:
        if all(hasattr(field, channel) for channel in ("r", "g", "b", "a")):
            field.r = values[0]
            field.g = values[1]
            field.b = values[2]
            field.a = values[3]
            return
        try:
            field[:] = values
            return
        except (TypeError, ValueError):
            pass
    color_type = getattr(type(target), field_name, None)
    if color_type is not None:
        try:
            setattr(target, field_name, color_type(*values))
            return
        except TypeError:
            pass
    setattr(target, field_name, values)


def _marker_shapes(dvz: Any, shapes: tuple[MarkerShape, ...]) -> npt.NDArray[np.uint32]:
    return np.ascontiguousarray(
        np.array([_marker_shape_value(dvz, shape) for shape in shapes], dtype=np.uint32)
    )


def _marker_shape_value(dvz: Any, shape: MarkerShape) -> int:
    name = _MARKER_SHAPE_NAMES[shape]
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzMarkerShape", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return _MARKER_SHAPE_FALLBACKS[shape]


def _stroke_cap_value(dvz: Any, cap: StrokeCap) -> int:
    name = _STROKE_CAP_NAMES[cap]
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzSegmentCap", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return _STROKE_CAP_FALLBACKS[cap]


def _preflight_datoviz_vector_api(dvz: Any) -> dict[VectorCap, int]:
    """Validate the public vector ABI before allocating a native visual."""
    diagnostics = datoviz_vector_api_diagnostics(dvz)
    if diagnostics:
        raise DatovizV04Unsupported(
            "Datoviz VectorVisual requires public vector ABI: " + ", ".join(diagnostics)
        )
    return {cap: int(getattr(dvz, name)) for cap, name in _VECTOR_CAP_NAMES.items()}


def _stroke_join_value(dvz: Any, join: StrokeJoin) -> int:
    name = _STROKE_JOIN_NAMES[join]
    value = getattr(dvz, name, None)
    if value is not None:
        return int(value)
    enum_type = getattr(dvz, "DvzPathJoin", None)
    if enum_type is not None:
        return int(getattr(enum_type, name))
    return _STROKE_JOIN_FALLBACKS[join]


def _expand_path_colors(visual: PathVisual) -> npt.NDArray[np.uint8]:
    colors = _rgba8(visual.colors)
    return np.ascontiguousarray(np.repeat(colors, visual.path_lengths, axis=0))


def _expand_path_widths(visual: PathVisual) -> npt.NDArray[np.float32]:
    widths = visual.width_values()
    return np.ascontiguousarray(np.repeat(widths, visual.path_lengths, axis=0))


def _configure_ndc_panel_view2d(dvz: Any, panel: Any) -> None:
    """Configure the panel so NDC data keeps equal X/Y screen scale when possible."""
    view_setter = getattr(dvz, "dvz_panel_set_view2d", None)
    if view_setter is None:
        return
    if hasattr(dvz, "dvz_panel_set_domain"):
        _set_datoviz_panel_domains(dvz, panel, (-1.0, 1.0), (-1.0, 1.0))
    try:
        view = _datoviz_panel_view2d_desc(dvz)
    except DatovizV04Unavailable:
        return
    if hasattr(view, "aspect"):
        view.aspect = int(getattr(dvz, "DVZ_PANEL_VIEW2D_ASPECT_EQUAL", 1))
    if hasattr(view, "padding"):
        view.padding = 0.0
    if _datoviz_view2d_descriptor_has_data_domains(view):
        _set_datoviz_data_domain(view, "data_x", (-1.0, 1.0))
        _set_datoviz_data_domain(view, "data_y", (-1.0, 1.0))
    _require_datoviz_success(
        view_setter(panel, view),
        "Datoviz NDC equal-aspect panel setup failed",
    )


def _datoviz_panel_view2d_desc(dvz: Any) -> Any:
    """Create a View2D descriptor across Datoviz pre-RC factory renames."""
    factory = getattr(dvz, "dvz_panel_view2d_desc", None)
    if factory is None:
        factory = getattr(dvz, "dvz_panel_view2d", None)
    if factory is None:
        raise DatovizV04Unavailable(
            "Datoviz View2D descriptor factory is unavailable: missing "
            "dvz_panel_view2d_desc/dvz_panel_view2d"
        )
    return factory()


def _query_frame_resolution_ready(dvz: Any) -> bool:
    if not hasattr(dvz, "dvz_app") or not hasattr(dvz, "dvz_view_offscreen"):
        return False
    return (
        hasattr(dvz, "dvz_view_render_once")
        or hasattr(dvz, "dvz_app_render_once")
        or hasattr(dvz, "dvz_app_run")
    )


def _panel_query(dvz: Any, panel: Any, x: float, y: float, request: Any) -> int:
    try:
        return int(dvz.dvz_panel_query_px(panel, x, y, ctypes.byref(request)))
    except TypeError:
        return int(dvz.dvz_panel_query_px(panel, x, y, request))


def _scene_poll_query(dvz: Any, scene: Any, out_result: Any) -> bool:
    try:
        return bool(dvz.dvz_scene_poll_query(scene, ctypes.byref(out_result)))
    except TypeError:
        return bool(dvz.dvz_scene_poll_query(scene, out_result))


def _datoviz_query_request_diagnostic(request: QueryRequest) -> str | None:
    if request.scope != QueryScope.DATA:
        if request.scope == QueryScope.GUIDES:
            return (
                "axis-guide-query-unsupported: Datoviz v0.4 query slice defers "
                "guide picking/query; guide rendering capability does not imply "
                "queryable ticks, labels, grids, or titles"
            )
        if request.scope == QueryScope.ALL_RENDERED:
            return (
                "all-rendered-guides-unsupported: Datoviz v0.4 query slice cannot "
                "merge data and guide contributions because guide query is deferred"
            )
        return f"Datoviz v0.4 query slice supports data scope only, got {request.scope.value!r}"
    if request.coordinate_space != QueryCoordinateSpace.PANEL:
        return f"Datoviz v0.4 query slice supports panel coordinates only, got {request.coordinate_space.value!r}"
    if request.hit_policy != QueryHitPolicy.FRONTMOST:
        return f"Datoviz v0.4 query slice supports frontmost hit policy only, got {request.hit_policy.value!r}"
    if SCALAR_COLOR_QUERY_PAYLOAD_KIND not in request.requested_extension_payload_kinds:
        unsupported_standard_payloads = tuple(
            payload for payload in request.requested_payload if payload != QueryPayload.IDENTITY
        )
        if unsupported_standard_payloads:
            return (
                "Datoviz v0.4 live data query currently supports identity payloads "
                f"only; unsupported requested payloads: {tuple(p.value for p in unsupported_standard_payloads)}"
            )
    unsupported_extension_payloads = tuple(
        kind
        for kind in request.requested_extension_payload_kinds
        if kind != SCALAR_COLOR_QUERY_PAYLOAD_KIND
    )
    if TRANSFORM_QUERY_PAYLOAD_KIND in unsupported_extension_payloads:
        return (
            "GSP_QUERY_INVERSE_UNSUPPORTED: Datoviz v0.4 query slice does not "
            "support gsp.transform-query@0.1 inverse payloads"
        )
    if unsupported_extension_payloads:
        return (
            "Datoviz v0.4 query slice does not support requested extension query "
            f"payloads: {unsupported_extension_payloads}"
        )
    return None


def _unsupported_query_result(request: QueryRequest, diagnostic: str | None) -> QueryResult | None:
    if diagnostic is None:
        return None
    return QueryResult(
        request_id=request.id,
        status=QueryStatus.UNSUPPORTED,
        hit=False,
        panel_coordinate=request.coordinate,
        diagnostic=diagnostic,
    )


def _datoviz_request_id(request_id: str) -> int:
    return crc32(request_id.encode("utf-8")) or 1
