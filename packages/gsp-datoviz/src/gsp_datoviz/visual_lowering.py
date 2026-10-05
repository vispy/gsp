"""Typed visual lowering into retained native resources."""

from __future__ import annotations

import ctypes
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    DepthMode,
    ImageVisual,
    MarkerVisual,
    MeshShading,
    MeshVisual,
    PathVisual,
    PixelVisual,
    PointVisual,
    PrimitiveVisual,
    ScalarColorSlot,
    SegmentVisual,
    SphereVisual,
    TextureFilter,
    TextVisual,
    VectorVisual,
    View2D,
    View3D,
    VisualFamily,
    VisualTransformBinding,
    project_view3d_data_point,
    validate_mesh_visual_texture2d_unlit,
)
from gsp.protocol.color_mapping import (
    resolve_color_scale,
)
from gsp.protocol.visuals import CoordinateSpace

from gsp_datoviz.latest_api_contract import (
    datoviz_primitive_api_diagnostics,
)

from ._state import (
    _PRIMITIVE_TOPOLOGY_NAMES,
    DVZ_ALPHA_BLENDED,
    DVZ_COLOR_ROLE_LINEAR_COLOR,
    DVZ_COLOR_ROLE_SRGB_COLOR,
    DVZ_FIELD_DIM_2D,
    DVZ_FIELD_FILTER_LINEAR,
    DVZ_FIELD_FILTER_NEAREST,
    DVZ_FIELD_FORMAT_RGBA8_UNORM,
    DVZ_FIELD_SEMANTIC_COLOR,
    DVZ_MATERIAL_MODEL_UNLIT,
    DVZ_QUERY_CAPABILITY_FACE,
    DVZ_QUERY_CAPABILITY_ITEM,
    DVZ_QUERY_CAPABILITY_PIXEL,
    DatovizV04Unavailable,
    DatovizV04Unsupported,
    _RetainedView2DPositionUpload,
    _RetainedView3DMeshAttachment,
    _RetainedView3DTextAttachment,
    _ScalarVisualData,
)
from .colors import (
    _marker_fill_colors,
    _point_colors,
    _rgba8,
    _rgba8_broadcast,
)
from .geometry import (
    _adapt_visual_positions,
    _datoviz_mesh3d_plot_ndc_positions,
    _datoviz_mesh_payload,
    _diameters_from_pixel_diameters,
    _image_positions,
    _image_texcoords,
    _mesh3d_face_depth_order,
    _positions_3d,
    _record_transform_adaptation,
    _rgba8_image_visual,
    _validate_datoviz_mesh3d_visual,
)
from .layout import (
    _panel_pixel_size,
)
from .native_api import (
    _add_visual_to_panel,
    _ctypes_pointer_arg,
    _datoviz_marker_diagnostics,
    _datoviz_path_diagnostics,
    _datoviz_segment_diagnostics,
    _datoviz_texture2d_mesh_diagnostics,
    _datoviz_visual_coord_space,
    _enum_value,
    _expand_path_colors,
    _expand_path_widths,
    _is_null_handle,
    _marker_shapes,
    _preflight_datoviz_vector_api,
    _require_datoviz_success,
    _set_alpha_mode_if_translucent,
    _set_data_view_payload,
    _set_filled_point_style,
    _set_image_sampling,
    _set_marker_style,
    _set_path_subpaths,
    _set_query_capabilities,
    _set_visual_data,
    _set_visual_field,
    _set_visual_field_sampling,
    _set_visual_index_data,
    _stroke_cap_value,
    _stroke_join_value,
    _visual_attach_desc,
    datoviz_v04_mesh_diagnostics,
    datoviz_v04_sampled_field_diagnostics,
    datoviz_v04_sampled_field_ready,
    datoviz_v04_text_diagnostics,
    datoviz_v04_view3d_retained_data_ready,
)
from .text import (
    _assign_rgba8,
    _ndc_text_screen_position,
    _set_text_placement,
    _text_anchor_value,
    _text_anchor_x_value,
    _text_anchor_y_value,
    _text_placement,
    _text_placement_mode_value,
    _text_renderer_value,
)

if TYPE_CHECKING:
    from .protocol_renderer import DatovizV04ProtocolRenderer


def add_point_visual(self: DatovizV04ProtocolRenderer, visual: PointVisual) -> Any:
    """Create and attach a Datoviz point visual."""
    positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    colors = _point_colors(visual, color_scales=self.color_scales)
    diameters = self._scale_canvas_px_array(
        _diameters_from_pixel_diameters(visual.sizes, positions.shape[0])
    )

    dvz_visual = self.dvz.dvz_point(self.scene, 0)
    _set_filled_point_style(self.dvz, dvz_visual)
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_data(self.dvz, dvz_visual, "diameter_px", diameters)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    self._retain_view2d_position_upload(
        visual.id,
        dvz_visual,
        "position",
        visual.positions,
        visual.transform,
        visual.coordinate_space,
    )
    if visual.color_encoding is not None:
        scale = resolve_color_scale(self.color_scales, visual.color_encoding.color_scale_id)
        self.scalar_visuals[visual.id] = _ScalarVisualData(
            visual_id=visual.id,
            visual_family=VisualFamily.POINT,
            item_kind="point",
            color_slot=ScalarColorSlot.COLOR,
            values=np.asarray(visual.color_encoding.values, dtype=np.float64),
            color_scale=scale,
            alpha=float(visual.color_encoding.alpha),
        )
    return dvz_visual


def add_pixel_visual(self: DatovizV04ProtocolRenderer, visual: PixelVisual) -> Any:
    """Create and attach a public Datoviz square-pixel visual."""
    is_3d = visual.positions.shape[1] == 3
    if is_3d:
        if visual.transform is not None:
            raise DatovizV04Unsupported(
                "Datoviz PixelVisual positions3d do not support a 2D transform"
            )
        if visual.coordinate_space is not CoordinateSpace.DATA or self.view3d is None:
            raise DatovizV04Unsupported(
                "Datoviz PixelVisual positions3d require DATA space and View3D"
            )
    elif visual.coordinate_space is CoordinateSpace.DATA and self.view is None:
        raise DatovizV04Unsupported("Datoviz PixelVisual DATA positions2d require View2D")
    positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    colors = _rgba8_broadcast(visual.colors, positions.shape[0])
    sizes = self._scale_canvas_px_array(visual.pixel_size_values())
    dvz_visual = self.dvz.dvz_pixel(self.scene, 0)
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_data(self.dvz, dvz_visual, "pixel_size_px", sizes)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    self._retain_view2d_position_upload(
        visual.id,
        dvz_visual,
        "position",
        visual.positions,
        visual.transform,
        visual.coordinate_space,
    )
    return dvz_visual


def add_sphere_visual(self: DatovizV04ProtocolRenderer, visual: SphereVisual) -> Any:
    """Create accurate public Datoviz raycast sphere impostors."""
    if self.view3d is None:
        raise DatovizV04Unsupported("Datoviz SphereVisual DATA positions3d require View3D")
    missing = tuple(
        name
        for name in ("dvz_sphere", "dvz_sphere_set_mode")
        if not callable(getattr(self.dvz, name, None))
    )
    if missing:
        raise DatovizV04Unsupported(
            "Datoviz SphereVisual requires public callable(s): " + ", ".join(missing)
        )
    if not hasattr(self.dvz, "DVZ_SPHERE_MODE_RAYCAST_IMPOSTOR"):
        raise DatovizV04Unsupported(
            "Datoviz SphereVisual requires DVZ_SPHERE_MODE_RAYCAST_IMPOSTOR"
        )
    positions = np.ascontiguousarray(visual.positions, dtype=np.float32)
    colors = _rgba8_broadcast(visual.colors, positions.shape[0])
    radii = visual.radius_values()
    dvz_visual = self.dvz.dvz_sphere(self.scene, 0)
    if _is_null_handle(dvz_visual):
        raise DatovizV04Unavailable("Datoviz dvz_sphere() failed")
    _require_datoviz_success(
        self.dvz.dvz_sphere_set_mode(dvz_visual, self.dvz.DVZ_SPHERE_MODE_RAYCAST_IMPOSTOR),
        "Datoviz accurate sphere raycast mode setup failed",
    )
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_data(self.dvz, dvz_visual, "radius", radii)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(CoordinateSpace.DATA),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    return dvz_visual


def add_vector_visual(self: DatovizV04ProtocolRenderer, visual: VectorVisual) -> Any:
    """Create a public Datoviz straight vector visual with visual-wide style."""
    is_3d = visual.positions.shape[1] == 3
    if is_3d:
        if visual.transform is not None:
            raise DatovizV04Unsupported(
                "Datoviz VectorVisual positions3d do not support a 2D transform"
            )
        if visual.coordinate_space is not CoordinateSpace.DATA or self.view3d is None:
            raise DatovizV04Unsupported(
                "Datoviz VectorVisual positions3d require DATA space and View3D"
            )
    elif visual.coordinate_space is CoordinateSpace.DATA and self.view is None:
        raise DatovizV04Unsupported("Datoviz VectorVisual DATA positions2d require View2D")
    cap_values = _preflight_datoviz_vector_api(self.dvz)
    source_tails, source_heads = visual.endpoint_values()
    positions = _adapt_visual_positions(
        visual.id,
        source_tails,
        visual.transform,
        visual.coordinate_space,
        self.view,
        self.transform_resources,
        cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
    )
    endpoints = _adapt_visual_positions(
        visual.id,
        source_heads,
        visual.transform,
        visual.coordinate_space,
        self.view,
        self.transform_resources,
        cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
    )
    vectors = _positions_3d(endpoints - positions)
    positions3 = _positions_3d(positions)
    colors = _rgba8_broadcast(visual.colors, positions3.shape[0])
    widths = self._scale_canvas_px_array(visual.width_values())

    dvz_visual = self.dvz.dvz_vector(self.scene, 0)
    if _is_null_handle(dvz_visual):
        raise DatovizV04Unavailable("Datoviz dvz_vector() failed")
    try:
        style = self.dvz.dvz_vector_style()
    except Exception as exc:
        raise DatovizV04Unavailable(
            f"Datoviz dvz_vector_style() failed: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(style, self.dvz.DvzVectorStyle):
        raise DatovizV04Unsupported(
            "Datoviz dvz_vector_style() returned an incompatible DvzVectorStyle"
        )
    style.scale = 1.0
    style.anchor = int(self.dvz.DVZ_VECTOR_ANCHOR_TAIL)
    style.start_cap = cap_values[visual.start_cap]
    style.end_cap = cap_values[visual.end_cap]
    _require_datoviz_success(
        self.dvz.dvz_vector_set_style(dvz_visual, ctypes.byref(style)),
        "Datoviz vector style setup failed",
    )
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions3)
    _set_visual_data(self.dvz, dvz_visual, "vector", vectors)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_data(self.dvz, dvz_visual, "stroke_width_px", widths)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    return dvz_visual


def add_primitive_visual(self: DatovizV04ProtocolRenderer, visual: PrimitiveVisual) -> Any:
    """Create a public Datoviz bounded primitive with optional public indices."""
    is_3d = visual.positions.shape[1] == 3
    if is_3d:
        if visual.transform is not None:
            raise DatovizV04Unsupported(
                "primitivevisual_transform_unsupported: Datoviz positions3d "
                "do not support a 2D transform"
            )
        if visual.coordinate_space is not CoordinateSpace.DATA or self.view3d is None:
            raise DatovizV04Unsupported(
                "primitivevisual_view3d_required: Datoviz positions3d require DATA space and View3D"
            )
    elif visual.coordinate_space is CoordinateSpace.DATA and self.view is None:
        raise DatovizV04Unsupported(
            "primitivevisual_view2d_required: Datoviz DATA positions2d require View2D"
        )
    diagnostics = datoviz_primitive_api_diagnostics(self.dvz, indexed=visual.indices is not None)
    if diagnostics:
        raise DatovizV04Unsupported(
            "primitivevisual_capability_unsupported: " + ", ".join(diagnostics)
        )
    topology_name = _PRIMITIVE_TOPOLOGY_NAMES[visual.topology]
    positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    colors = _rgba8_broadcast(visual.colors, positions.shape[0])
    dvz_visual = self.dvz.dvz_primitive(self.scene, int(getattr(self.dvz, topology_name)), 0)
    if _is_null_handle(dvz_visual):
        raise DatovizV04Unavailable("Datoviz dvz_primitive() failed")
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    indices = visual.index_values()
    if indices is not None:
        _set_visual_index_data(self.dvz, dvz_visual, indices)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    return dvz_visual


def update_point_visual(self: DatovizV04ProtocolRenderer, visual: PointVisual) -> None:
    """Update one retained point visual from semantic GSP state.

    This is an adapter operation used by bounded lifecycle/conformance probes. It keeps native
    handles private and deliberately rejects updates after close or for unknown visual IDs.
    """
    if self._closed:
        raise RuntimeError("Datoviz renderer is closed")
    native_visual = self.visuals.get(visual.id)
    if native_visual is None:
        raise ValueError(f"unknown retained point visual {visual.id!r}")

    positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    colors = _point_colors(visual, color_scales=self.color_scales)
    diameters = self._scale_canvas_px_array(
        _diameters_from_pixel_diameters(visual.sizes, positions.shape[0])
    )
    # Reset the previous blend state as colors can become opaque again.
    alpha_setter = getattr(self.dvz, "dvz_visual_set_alpha_mode", None)
    if alpha_setter is not None:
        from .native_api import _alpha_mode_value

        _require_datoviz_success(
            alpha_setter(
                native_visual,
                _alpha_mode_value(
                    self.dvz,
                    "DVZ_ALPHA_BLENDED" if np.any(colors[:, 3] < 255) else "DVZ_ALPHA_OPAQUE",
                    DVZ_ALPHA_BLENDED if np.any(colors[:, 3] < 255) else 0,
                ),
            ),
            "Datoviz point-update alpha configuration failed",
        )
    elif np.any(colors[:, 3] < 255):
        raise DatovizV04Unsupported("Datoviz translucent updates require an alpha-mode setter")
    _set_visual_data(self.dvz, native_visual, "position", positions)
    _set_visual_data(self.dvz, native_visual, "color", colors)
    _set_visual_data(self.dvz, native_visual, "diameter_px", diameters)

    if visual.color_encoding is not None:
        scale = resolve_color_scale(self.color_scales, visual.color_encoding.color_scale_id)
        self.scalar_visuals[visual.id] = _ScalarVisualData(
            visual_id=visual.id,
            visual_family=VisualFamily.POINT,
            item_kind="point",
            color_slot=ScalarColorSlot.COLOR,
            values=np.asarray(visual.color_encoding.values, dtype=np.float64),
            color_scale=scale,
            alpha=float(visual.color_encoding.alpha),
        )

    for upload in self.retained_view2d_position_uploads:
        if upload.visual_id == visual.id:
            upload.positions = visual.positions
            upload.transform = visual.transform
            upload.coordinate_space = visual.coordinate_space
            break


def add_marker_visual(self: DatovizV04ProtocolRenderer, visual: MarkerVisual) -> Any:
    """Create and attach a Datoviz marker visual."""
    marker_diagnostics = _datoviz_marker_diagnostics(self.dvz)
    if marker_diagnostics:
        raise DatovizV04Unsupported(
            f"Datoviz v0.4 marker facade is unavailable: {', '.join(marker_diagnostics)}"
        )

    positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    fill_colors = _marker_fill_colors(visual, color_scales=self.color_scales)
    diameters = self._scale_canvas_px_array(
        _diameters_from_pixel_diameters(visual.sizes, positions.shape[0])
    )
    shape_values = visual.shape_values()
    angles = np.ascontiguousarray(visual.angle_values())
    shapes = _marker_shapes(self.dvz, shape_values)

    dvz_visual = self.dvz.dvz_marker(self.scene, 0)
    if dvz_visual is None:
        raise DatovizV04Unsupported("Datoviz marker visual allocation failed")
    _set_marker_style(
        self.dvz,
        dvz_visual,
        visual.stroke_color,
        self._scale_canvas_px(visual.stroke_width),
    )
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, fill_colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", fill_colors)
    _set_visual_data(self.dvz, dvz_visual, "diameter_px", diameters)
    _set_visual_data(self.dvz, dvz_visual, "angle", angles)
    _set_visual_data(self.dvz, dvz_visual, "shape", shapes)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    self._retain_view2d_position_upload(
        visual.id,
        dvz_visual,
        "position",
        visual.positions,
        visual.transform,
        visual.coordinate_space,
    )
    if visual.fill_color_encoding is not None:
        scale = resolve_color_scale(self.color_scales, visual.fill_color_encoding.color_scale_id)
        self.scalar_visuals[visual.id] = _ScalarVisualData(
            visual_id=visual.id,
            visual_family="marker",
            item_kind="marker",
            color_slot=ScalarColorSlot.FILL,
            values=np.asarray(visual.fill_color_encoding.values, dtype=np.float64),
            color_scale=scale,
            alpha=float(visual.fill_color_encoding.alpha),
        )
    return dvz_visual


def add_segment_visual(self: DatovizV04ProtocolRenderer, visual: SegmentVisual) -> Any:
    """Create and attach a Datoviz segment visual."""
    segment_diagnostics = _datoviz_segment_diagnostics(self.dvz)
    if segment_diagnostics:
        raise DatovizV04Unsupported(
            f"Datoviz v0.4 segment facade is unavailable: {', '.join(segment_diagnostics)}"
        )

    start_positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.start_positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    end_positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.end_positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    colors = _rgba8(visual.colors)
    widths = self._scale_canvas_px_array(np.ascontiguousarray(visual.width_values()))

    dvz_visual = self.dvz.dvz_segment(self.scene, 0)
    if dvz_visual is None:
        raise DatovizV04Unsupported("Datoviz segment visual allocation failed")
    cap = _stroke_cap_value(self.dvz, visual.cap)
    _require_datoviz_success(
        self.dvz.dvz_segment_set_caps(dvz_visual, cap, cap),
        "Datoviz segment cap configuration failed",
    )
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position_start", start_positions)
    _set_visual_data(self.dvz, dvz_visual, "position_end", end_positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_data(self.dvz, dvz_visual, "stroke_width_px", widths)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    self._retain_view2d_position_upload(
        visual.id,
        dvz_visual,
        "position_start",
        visual.start_positions,
        visual.transform,
        visual.coordinate_space,
    )
    self._retain_view2d_position_upload(
        visual.id,
        dvz_visual,
        "position_end",
        visual.end_positions,
        visual.transform,
        visual.coordinate_space,
    )
    return dvz_visual


def add_path_visual(self: DatovizV04ProtocolRenderer, visual: PathVisual) -> Any:
    """Create and attach a Datoviz path visual."""
    path_diagnostics = _datoviz_path_diagnostics(self.dvz)
    if path_diagnostics:
        raise DatovizV04Unsupported(
            f"Datoviz v0.4 path facade is unavailable: {', '.join(path_diagnostics)}"
        )

    positions = _positions_3d(
        _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    colors = _expand_path_colors(visual)
    widths = self._scale_canvas_px_array(_expand_path_widths(visual))
    subpaths = np.ascontiguousarray(np.array(visual.path_lengths, dtype=np.uint32))

    dvz_visual = self.dvz.dvz_path(self.scene, 0)
    if dvz_visual is None:
        raise DatovizV04Unsupported("Datoviz path visual allocation failed")
    cap = _stroke_cap_value(self.dvz, visual.cap)
    _require_datoviz_success(
        self.dvz.dvz_path_set_caps(dvz_visual, cap, cap),
        "Datoviz path cap configuration failed",
    )
    _require_datoviz_success(
        self.dvz.dvz_path_set_join(
            dvz_visual,
            _stroke_join_value(self.dvz, visual.join),
            float(visual.miter_limit),
        ),
        "Datoviz path join configuration failed",
    )
    _require_datoviz_success(
        _set_path_subpaths(self.dvz, dvz_visual, len(visual.path_lengths), subpaths),
        "Datoviz path subpath configuration failed",
    )
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    _set_query_capabilities(self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM)
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_data(self.dvz, dvz_visual, "stroke_width_px", widths)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    self._retain_view2d_position_upload(
        visual.id,
        dvz_visual,
        "position",
        visual.positions,
        visual.transform,
        visual.coordinate_space,
    )
    return dvz_visual


def add_image_visual(self: DatovizV04ProtocolRenderer, visual: ImageVisual) -> Any:
    """Create and attach a Datoviz image visual in DATA or NDC coordinates."""
    pixels = _rgba8_image_visual(visual, color_scales=self.color_scales)
    positions = _image_positions(visual.extent)
    texcoords = _image_texcoords(visual.origin)
    height, width = pixels.shape[:2]

    dvz_visual = self.dvz.dvz_image(self.scene, 0)
    _set_image_sampling(self.dvz, dvz_visual, visual.interpolation)
    _set_query_capabilities(
        self.dvz, dvz_visual, DVZ_QUERY_CAPABILITY_ITEM | DVZ_QUERY_CAPABILITY_PIXEL
    )
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "texcoords", texcoords)
    if datoviz_v04_sampled_field_ready(self.dvz):
        sampled_field = self._create_rgba8_sampled_field(pixels, width, height)
        if not _set_visual_field(self.dvz, dvz_visual, "field", sampled_field):
            raise DatovizV04Unsupported("Datoviz sampled-field image binding failed")
        self.sampled_fields[visual.id] = sampled_field
    else:
        raise DatovizV04Unavailable(
            "Datoviz scalar/image field binding is unavailable: "
            + "; ".join(datoviz_v04_sampled_field_diagnostics(self.dvz))
        )
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=self._visual_coord_space(visual.coordinate_space),
            z_layer=0,
        ),
    )
    self.visuals[visual.id] = dvz_visual
    if visual.color_scale_id is not None:
        scale = resolve_color_scale(self.color_scales, visual.color_scale_id)
        self.scalar_visuals[visual.id] = _ScalarVisualData(
            visual_id=visual.id,
            visual_family=VisualFamily.IMAGE,
            item_kind="texel",
            color_slot=ScalarColorSlot.IMAGE,
            values=np.asarray(visual.image, dtype=np.float64),
            color_scale=scale,
        )
    return dvz_visual


def add_mesh_visual(self: DatovizV04ProtocolRenderer, visual: MeshVisual) -> Any:
    """Create and attach a bounded Datoviz triangle mesh visual."""
    texture2d_unlit = visual.canonical_shading() is MeshShading.TEXTURE2D_UNLIT
    if texture2d_unlit:
        diagnostics = _datoviz_texture2d_mesh_diagnostics(self.dvz)
        if diagnostics:
            raise DatovizV04Unsupported(
                "meshvisual_material_texture2d_unlit_unsupported: " + "; ".join(diagnostics)
            )
        validate_mesh_visual_texture2d_unlit(visual, texture_resources=self.texture_resources or {})
    is_3d_mesh = visual.positions.shape[1] == 3
    if is_3d_mesh:
        _validate_datoviz_mesh3d_visual(visual, self.view3d)
    diagnostics = datoviz_v04_mesh_diagnostics(self.dvz)
    if diagnostics:
        raise DatovizV04Unsupported(
            "Datoviz v0.4 MeshVisual support is unavailable: " + "; ".join(diagnostics)
        )

    face_order: npt.NDArray[np.int64] | None = None
    adapted_positions: npt.NDArray[np.float32] | npt.NDArray[np.float64]
    retained_view3d_data_path = (
        is_3d_mesh
        and visual.coordinate_space is CoordinateSpace.DATA
        and datoviz_v04_view3d_retained_data_ready(self.dvz)
    )
    if retained_view3d_data_path:
        adapted_positions = np.ascontiguousarray(np.asarray(visual.positions, dtype=np.float32))
    elif is_3d_mesh:
        panel_width, panel_height = _panel_pixel_size(self.width, self.height, self.panel_bounds)
        adapted_positions = _datoviz_mesh3d_plot_ndc_positions(
            visual,
            view3d=self.view3d,
            aspect_ratio=panel_width / panel_height,
        )
        if visual.depth_test is not DepthMode.DISABLED:
            face_order = _mesh3d_face_depth_order(adapted_positions[:, 2], visual.faces)
    else:
        adapted_positions = _adapt_visual_positions(
            visual.id,
            visual.positions,
            visual.transform,
            visual.coordinate_space,
            self.view,
            self.transform_resources,
            cpu_map_data_to_view=self._cpu_map_data_visuals_to_view,
        )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    positions, colors, indices = _datoviz_mesh_payload(
        visual,
        adapted_positions,
        view3d=self.view3d,
        face_order=face_order,
    )
    dvz_visual = self.dvz.dvz_mesh(self.scene, 0)
    if dvz_visual is None:
        raise DatovizV04Unsupported("Datoviz mesh visual allocation failed")
    _set_visual_data(self.dvz, dvz_visual, "position", positions)
    _set_visual_data(self.dvz, dvz_visual, "color", colors)
    _set_visual_index_data(self.dvz, dvz_visual, indices)
    if texture2d_unlit:
        self._configure_texture2d_unlit_mesh(dvz_visual, visual, positions)
    if is_3d_mesh:
        self.retained_view3d_update_stats.vertex_uploads += 1
        self.retained_view3d_update_stats.index_uploads += 1
        self.retained_view3d_update_stats.visual_rebuilds += 1
    native_depth_test = (
        visual.depth_test is not DepthMode.DISABLED and visual.depth_write is not DepthMode.DISABLED
        if retained_view3d_data_path
        else False
        if is_3d_mesh
        else visual.depth_test is DepthMode.ENABLED
    )
    _require_datoviz_success(
        self.dvz.dvz_visual_set_depth_test(dvz_visual, native_depth_test),
        "Datoviz mesh depth-test configuration failed",
    )
    _set_alpha_mode_if_translucent(self.dvz, dvz_visual, colors)
    query_capabilities = DVZ_QUERY_CAPABILITY_ITEM
    if is_3d_mesh and visual.coordinate_space is CoordinateSpace.DATA:
        query_capabilities |= getattr(
            self.dvz, "DVZ_QUERY_CAPABILITY_FACE", DVZ_QUERY_CAPABILITY_FACE
        )
    _set_query_capabilities(self.dvz, dvz_visual, query_capabilities)
    _add_visual_to_panel(
        self.dvz,
        self.panel,
        dvz_visual,
        _visual_attach_desc(
            self.dvz,
            coord_space=(
                "data"
                if retained_view3d_data_path
                else "view"
                if is_3d_mesh
                else self._visual_coord_space(visual.coordinate_space)
            ),
            z_layer=round(visual.order),
            controller_mode=(
                "apply" if retained_view3d_data_path else "fixed" if is_3d_mesh else "apply"
            ),
        ),
    )
    self.visuals[visual.id] = dvz_visual
    if retained_view3d_data_path:
        self.retained_view3d_meshes.append(
            _RetainedView3DMeshAttachment(
                visual_id=visual.id,
                native_visual=dvz_visual,
            )
        )
    return dvz_visual


def _configure_texture2d_unlit_mesh(
    self: DatovizV04ProtocolRenderer,
    dvz_visual: Any,
    visual: MeshVisual,
    positions: npt.NDArray[np.float32],
) -> None:
    """Bind strict RGBA8 nearest-or-linear unlit Texture2D mesh state."""
    if visual.texture2d_id is None or visual.uvs is None:
        raise DatovizV04Unsupported("texture2d_unlit mesh is missing texture or UV data")
    texture = (self.texture_resources or {}).get(visual.texture2d_id)
    if texture is None:
        raise DatovizV04Unsupported(
            f"texture2d_unknown_id: unknown Texture2D id {visual.texture2d_id!r}"
        )

    texcoords = np.ascontiguousarray(np.asarray(visual.uvs, dtype=np.float32))
    texcoords = texcoords.copy()
    texcoords[:, 1] = 1.0 - texcoords[:, 1]
    normals = np.zeros((positions.shape[0], 3), dtype=np.float32)
    normals[:, 2] = 1.0
    _set_visual_data(self.dvz, dvz_visual, "normal", normals)
    _set_visual_data(self.dvz, dvz_visual, "texcoords", texcoords)

    height, width = texture.image.shape[:2]
    sampled_field = self._create_rgba8_sampled_field(
        texture.image,
        width,
        height,
        color_role=DVZ_COLOR_ROLE_LINEAR_COLOR,
    )
    if not _set_visual_field(self.dvz, dvz_visual, "texture", sampled_field):
        raise DatovizV04Unsupported("Datoviz textured-mesh field binding failed")

    sampling = self.dvz.dvz_field_sampling_desc()
    if visual.texture_filter is TextureFilter.LINEAR:
        filter_member = "DVZ_FIELD_FILTER_LINEAR"
        filter_fallback = DVZ_FIELD_FILTER_LINEAR
    else:
        filter_member = "DVZ_FIELD_FILTER_NEAREST"
        filter_fallback = DVZ_FIELD_FILTER_NEAREST
    sampling.min_filter = _enum_value(self.dvz, "DvzFieldFilter", filter_member, filter_fallback)
    sampling.mag_filter = sampling.min_filter
    _require_datoviz_success(
        _set_visual_field_sampling(self.dvz, dvz_visual, "texture", sampling),
        "Datoviz textured-mesh field-slot sampling configuration failed",
    )

    material = self.dvz.dvz_material_desc()
    material.model = _enum_value(
        self.dvz,
        "DvzMaterialModel",
        "DVZ_MATERIAL_MODEL_UNLIT",
        DVZ_MATERIAL_MODEL_UNLIT,
    )
    _require_datoviz_success(
        self.dvz.dvz_visual_set_material(dvz_visual, _ctypes_pointer_arg(material)),
        "Datoviz textured-mesh unlit material configuration failed",
    )
    self.sampled_fields[visual.id] = sampled_field


def add_text_visual(self: DatovizV04ProtocolRenderer, visual: TextVisual) -> Any:
    """Create retained 2D text or projected screen-facing 3D billboards."""
    diagnostics = datoviz_v04_text_diagnostics(self.dvz)
    if diagnostics:
        raise DatovizV04Unsupported(
            "Datoviz v0.4 TextVisual support is unavailable: " + "; ".join(diagnostics)
        )

    is_billboard3d = visual.positions.shape[1] == 3
    if is_billboard3d:
        if visual.transform is not None:
            raise DatovizV04Unsupported(
                "Datoviz TextVisual billboard3d does not support a 2D transform"
            )
        if visual.coordinate_space is not CoordinateSpace.DATA or self.view3d is None:
            raise DatovizV04Unsupported(
                "Datoviz TextVisual positions3d require DATA space and View3D"
            )
        positions = _positions_3d(visual.positions)
    else:
        positions = _positions_3d(
            _adapt_visual_positions(
                visual.id,
                visual.positions,
                visual.transform,
                visual.coordinate_space,
                self.view,
                self.transform_resources,
            )
        )
    _record_transform_adaptation(self.transform_adaptations, visual.id, visual.transform)
    colors = _rgba8(visual.rgba_values())
    sizes = self._scale_canvas_px_array(visual.font_size_values())
    rotations = visual.rotation_values()
    anchor_x = visual.anchor_x_values()
    anchor_y = visual.anchor_y_values()
    placement_space = CoordinateSpace.NDC if is_billboard3d else visual.coordinate_space
    mode = _text_placement_mode_value(self.dvz, placement_space)
    texts: list[Any] = []

    for index, text_value in enumerate(visual.texts):
        text = self.dvz.dvz_text(self.panel, 0)
        if text is None:
            raise DatovizV04Unsupported("Datoviz text object allocation failed")

        style = self.dvz.dvz_text_style()
        style.size_px = float(sizes[index])
        style.renderer = _text_renderer_value(self.dvz)
        _assign_rgba8(style.color, colors[index])
        _require_datoviz_success(
            self.dvz.dvz_text_set_style(text, style),
            "Datoviz text style configuration failed",
        )

        placement = _text_placement(self.dvz)
        placement.mode = mode
        placement.anchor = _text_anchor_value(self.dvz, placement_space)
        if is_billboard3d:
            assert self.view3d is not None
            panel_width, panel_height = _panel_pixel_size(
                self.width, self.height, self.panel_bounds
            )
            projected = project_view3d_data_point(
                self.view3d,
                tuple(positions[index]),
                aspect_ratio=panel_width / panel_height,
            )
            position_x, position_y = _ndc_text_screen_position(
                np.asarray(projected),
                self.width,
                self.height,
                self.panel_bounds,
            )
        elif visual.coordinate_space is CoordinateSpace.NDC:
            position_x, position_y = _ndc_text_screen_position(
                positions[index],
                self.width,
                self.height,
                self.panel_bounds,
            )
        else:
            position_x = float(positions[index, 0])
            position_y = float(positions[index, 1])
        placement.position[0] = position_x
        placement.position[1] = position_y
        placement.position[2] = 0.0
        placement.text_anchor[0] = _text_anchor_x_value(anchor_x[index])
        placement.text_anchor[1] = _text_anchor_y_value(anchor_y[index])
        placement.has_text_anchor = True
        placement.angle = float(rotations[index])
        placement.depth_test = False
        _set_text_placement(self.dvz, text, placement)
        self.dvz.dvz_text_set_string(text, text_value.encode("utf-8"))
        texts.append(text)

    self.visuals[visual.id] = tuple(texts)
    if is_billboard3d:
        self.retained_view3d_texts.append(
            _RetainedView3DTextAttachment(visual=visual, native_texts=tuple(texts))
        )
    return tuple(texts)


def _update_billboard_text_placements(self: DatovizV04ProtocolRenderer, view3d: View3D) -> None:
    """Move retained overlay labels after a View3D camera/projection update."""
    panel_width, panel_height = _panel_pixel_size(self.width, self.height, self.panel_bounds)
    aspect_ratio = panel_width / panel_height
    for attachment in self.retained_view3d_texts:
        visual = attachment.visual
        anchor_x = visual.anchor_x_values()
        anchor_y = visual.anchor_y_values()
        rotations = visual.rotation_values()
        for index, text in enumerate(attachment.native_texts):
            projected = project_view3d_data_point(
                view3d,
                tuple(visual.positions[index]),
                aspect_ratio=aspect_ratio,
            )
            position_x, position_y = _ndc_text_screen_position(
                np.asarray(projected),
                self.width,
                self.height,
                self.panel_bounds,
            )
            placement = _text_placement(self.dvz)
            placement.mode = _text_placement_mode_value(self.dvz, CoordinateSpace.NDC)
            placement.anchor = _text_anchor_value(self.dvz, CoordinateSpace.NDC)
            placement.position[0] = position_x
            placement.position[1] = position_y
            placement.position[2] = 0.0
            placement.text_anchor[0] = _text_anchor_x_value(anchor_x[index])
            placement.text_anchor[1] = _text_anchor_y_value(anchor_y[index])
            placement.has_text_anchor = True
            placement.angle = float(rotations[index])
            placement.depth_test = False
            _set_text_placement(self.dvz, text, placement)


def _visual_coord_space(self: DatovizV04ProtocolRenderer, coordinate_space: CoordinateSpace) -> str:
    if coordinate_space is CoordinateSpace.DATA and self._cpu_map_data_visuals_to_view:
        return "view"
    return _datoviz_visual_coord_space(coordinate_space)


def _retain_view2d_position_upload(
    self: DatovizV04ProtocolRenderer,
    visual_id: str,
    native_visual: Any,
    attr_name: str,
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    transform: VisualTransformBinding | None,
    coordinate_space: CoordinateSpace,
) -> None:
    if not self._cpu_map_data_visuals_to_view or coordinate_space is not CoordinateSpace.DATA:
        return
    self.retained_view2d_position_uploads.append(
        _RetainedView2DPositionUpload(
            visual_id=visual_id,
            native_visual=native_visual,
            attr_name=attr_name,
            positions=np.ascontiguousarray(np.asarray(positions).copy()),
            transform=transform,
            coordinate_space=coordinate_space,
        )
    )


def _reupload_retained_view2d_positions(self: DatovizV04ProtocolRenderer, view: View2D) -> None:
    for upload in self.retained_view2d_position_uploads:
        positions = _positions_3d(
            _adapt_visual_positions(
                upload.visual_id,
                upload.positions,
                upload.transform,
                upload.coordinate_space,
                view,
                self.transform_resources,
                cpu_map_data_to_view=True,
            )
        )
        _set_visual_data(self.dvz, upload.native_visual, upload.attr_name, positions)


def _create_rgba8_sampled_field(
    self: DatovizV04ProtocolRenderer,
    pixels: npt.NDArray[np.uint8],
    width: int,
    height: int,
    *,
    color_role: int = DVZ_COLOR_ROLE_SRGB_COLOR,
) -> Any:
    """Create and upload a scene-owned RGBA8 sampled field."""
    desc = self.dvz.dvz_sampled_field_desc()
    desc.dim = DVZ_FIELD_DIM_2D
    desc.format = DVZ_FIELD_FORMAT_RGBA8_UNORM
    desc.semantic = DVZ_FIELD_SEMANTIC_COLOR
    desc.color_role = color_role
    desc.width = width
    desc.height = height
    desc.depth = 1

    sampled_field = self.dvz.dvz_sampled_field(self.scene, desc)
    if sampled_field is None:
        raise DatovizV04Unsupported("Datoviz sampled-field image allocation failed")

    view = self.dvz.dvz_field_data_view()
    _set_data_view_payload(view, pixels)
    view.bytes_per_row = width * 4
    view.rows_per_image = height
    _require_datoviz_success(
        self.dvz.dvz_sampled_field_set_data(sampled_field, view),
        "Datoviz sampled-field image upload failed",
    )
    return sampled_field
