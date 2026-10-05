"""Retained Datoviz renderer orchestration for typed GSP scenes."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    AffineTransform2DResource,
    CanvasSize,
    CapabilitySnapshot,
    ColorbarGuide,
    ColorScale,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    NavigationDiagnosticCode,
    PanelLayoutIntent,
    PathVisual,
    PixelOrigin,
    PixelVisual,
    PointVisual,
    PrimitiveVisual,
    QueryRequest,
    QueryResult,
    RenderTarget,
    ResolvedCanvas,
    ResolvedLayoutSnapshot,
    SegmentVisual,
    SphereVisual,
    Texture2D,
    TextVisual,
    VectorVisual,
    View2D,
    View3D,
    View3DDiagnosticCode,
    View3DMeshTrianglePickRequest,
    VisualTransformBinding,
    resolve_panel_layout_intent,
    resolve_view3d_projection_snapshot,
)
from gsp.protocol.color_mapping import (
    resolve_color_scale,
)
from gsp.protocol.view3d import (
    View3DNavigationAction,
    View3DNavigationResult,
    apply_view3d_navigation_action,
)
from gsp.protocol.visuals import CoordinateSpace

from gsp_datoviz.capabilities import (
    datoviz_v04_axis_provider_capability,
    datoviz_v04_capability_snapshot,
)

from . import capture_execution as _capture_execution
from . import panels as _panels
from . import query_execution as _query_execution
from . import visual_lowering as _visual_lowering
from ._state import (
    _BUILTIN_COLORMAP_NAMES as _BUILTIN_COLORMAP_NAMES,
)
from ._state import (
    _MARKER_SHAPE_FALLBACKS as _MARKER_SHAPE_FALLBACKS,
)
from ._state import (
    _MARKER_SHAPE_NAMES as _MARKER_SHAPE_NAMES,
)
from ._state import (
    _OPTIONAL_UNVERIFIED_DVZ_MESH_FUNCTIONS as _OPTIONAL_UNVERIFIED_DVZ_MESH_FUNCTIONS,
)
from ._state import (
    _OPTIONAL_UNVERIFIED_DVZ_TEXT_FUNCTIONS as _OPTIONAL_UNVERIFIED_DVZ_TEXT_FUNCTIONS,
)
from ._state import (
    _PRIMITIVE_TOPOLOGY_NAMES as _PRIMITIVE_TOPOLOGY_NAMES,
)
from ._state import (
    _REQUIRED_DVZ_MARKER_FUNCTIONS as _REQUIRED_DVZ_MARKER_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_MESH_FUNCTIONS as _REQUIRED_DVZ_MESH_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_PANEL_FRAME_GUIDE_QUERY_FUNCTIONS as _REQUIRED_DVZ_PANEL_FRAME_GUIDE_QUERY_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_PANEL_FRAME_SNAPSHOT_FUNCTIONS as _REQUIRED_DVZ_PANEL_FRAME_SNAPSHOT_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_PATH_FUNCTIONS as _REQUIRED_DVZ_PATH_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_SAMPLED_FIELD_FUNCTIONS as _REQUIRED_DVZ_SAMPLED_FIELD_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_SEGMENT_FUNCTIONS as _REQUIRED_DVZ_SEGMENT_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_TEXTURE2D_MESH_FUNCTIONS as _REQUIRED_DVZ_TEXTURE2D_MESH_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_V04_FUNCTIONS as _REQUIRED_DVZ_V04_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_VIEW3D_CAMERA_FUNCTIONS as _REQUIRED_DVZ_VIEW3D_CAMERA_FUNCTIONS,
)
from ._state import (
    _REQUIRED_DVZ_VIEW3D_RETAINED_DATA_FUNCTIONS as _REQUIRED_DVZ_VIEW3D_RETAINED_DATA_FUNCTIONS,
)
from ._state import (
    _STROKE_CAP_FALLBACKS as _STROKE_CAP_FALLBACKS,
)
from ._state import (
    _STROKE_CAP_NAMES as _STROKE_CAP_NAMES,
)
from ._state import (
    _STROKE_JOIN_FALLBACKS as _STROKE_JOIN_FALLBACKS,
)
from ._state import (
    _STROKE_JOIN_NAMES as _STROKE_JOIN_NAMES,
)
from ._state import (
    _VECTOR_CAP_NAMES as _VECTOR_CAP_NAMES,
)
from ._state import (
    _VISUAL_ATTRIBUTE_ALIASES as _VISUAL_ATTRIBUTE_ALIASES,
)
from ._state import (
    DATOVIZ_REVIEW_PLOT_MARGINS as DATOVIZ_REVIEW_PLOT_MARGINS,
)
from ._state import (
    DEFAULT_BACKGROUND_RGBA8 as DEFAULT_BACKGROUND_RGBA8,
)
from ._state import (
    DVZ_ALPHA_BLENDED as DVZ_ALPHA_BLENDED,
)
from ._state import (
    DVZ_BUILTIN_COLORMAP_CIVIDIS as DVZ_BUILTIN_COLORMAP_CIVIDIS,
)
from ._state import (
    DVZ_BUILTIN_COLORMAP_GRAY as DVZ_BUILTIN_COLORMAP_GRAY,
)
from ._state import (
    DVZ_BUILTIN_COLORMAP_INFERNO as DVZ_BUILTIN_COLORMAP_INFERNO,
)
from ._state import (
    DVZ_BUILTIN_COLORMAP_MAGMA as DVZ_BUILTIN_COLORMAP_MAGMA,
)
from ._state import (
    DVZ_BUILTIN_COLORMAP_PLASMA as DVZ_BUILTIN_COLORMAP_PLASMA,
)
from ._state import (
    DVZ_BUILTIN_COLORMAP_VIRIDIS as DVZ_BUILTIN_COLORMAP_VIRIDIS,
)
from ._state import (
    DVZ_CAMERA_ORTHOGRAPHIC as DVZ_CAMERA_ORTHOGRAPHIC,
)
from ._state import (
    DVZ_CAMERA_PERSPECTIVE as DVZ_CAMERA_PERSPECTIVE,
)
from ._state import (
    DVZ_COLOR_PIPELINE_LEGACY_SRGB_BLEND as DVZ_COLOR_PIPELINE_LEGACY_SRGB_BLEND,
)
from ._state import (
    DVZ_COLOR_PIPELINE_LINEAR_SRGB as DVZ_COLOR_PIPELINE_LINEAR_SRGB,
)
from ._state import (
    DVZ_COLOR_ROLE_LINEAR_COLOR as DVZ_COLOR_ROLE_LINEAR_COLOR,
)
from ._state import (
    DVZ_COLOR_ROLE_SRGB_COLOR as DVZ_COLOR_ROLE_SRGB_COLOR,
)
from ._state import (
    DVZ_COLORBAR_ORIENTATION_HORIZONTAL as DVZ_COLORBAR_ORIENTATION_HORIZONTAL,
)
from ._state import (
    DVZ_COLORBAR_ORIENTATION_VERTICAL as DVZ_COLORBAR_ORIENTATION_VERTICAL,
)
from ._state import (
    DVZ_COLORBAR_PLACEMENT_ATTACHED as DVZ_COLORBAR_PLACEMENT_ATTACHED,
)
from ._state import (
    DVZ_COLORBAR_PLACEMENT_DETACHED as DVZ_COLORBAR_PLACEMENT_DETACHED,
)
from ._state import (
    DVZ_CONTROLLER_APPLY as DVZ_CONTROLLER_APPLY,
)
from ._state import (
    DVZ_CONTROLLER_FIXED as DVZ_CONTROLLER_FIXED,
)
from ._state import (
    DVZ_FIELD_DIM_2D as DVZ_FIELD_DIM_2D,
)
from ._state import (
    DVZ_FIELD_FILTER_LINEAR as DVZ_FIELD_FILTER_LINEAR,
)
from ._state import (
    DVZ_FIELD_FILTER_NEAREST as DVZ_FIELD_FILTER_NEAREST,
)
from ._state import (
    DVZ_FIELD_FORMAT_RGBA8_UNORM as DVZ_FIELD_FORMAT_RGBA8_UNORM,
)
from ._state import (
    DVZ_FIELD_SEMANTIC_COLOR as DVZ_FIELD_SEMANTIC_COLOR,
)
from ._state import (
    DVZ_HORIZONTAL_ANCHOR_RIGHT as DVZ_HORIZONTAL_ANCHOR_RIGHT,
)
from ._state import (
    DVZ_IMAGE_SAMPLING_LINEAR as DVZ_IMAGE_SAMPLING_LINEAR,
)
from ._state import (
    DVZ_IMAGE_SAMPLING_NEAREST as DVZ_IMAGE_SAMPLING_NEAREST,
)
from ._state import (
    DVZ_INPUT_EVENT_POINTER as DVZ_INPUT_EVENT_POINTER,
)
from ._state import (
    DVZ_INPUT_EVENT_RESIZE as DVZ_INPUT_EVENT_RESIZE,
)
from ._state import (
    DVZ_INPUT_EVENT_SCALE as DVZ_INPUT_EVENT_SCALE,
)
from ._state import (
    DVZ_MATERIAL_MODEL_UNLIT as DVZ_MATERIAL_MODEL_UNLIT,
)
from ._state import (
    DVZ_PATH_JOIN_BEVEL as DVZ_PATH_JOIN_BEVEL,
)
from ._state import (
    DVZ_PATH_JOIN_MITER as DVZ_PATH_JOIN_MITER,
)
from ._state import (
    DVZ_PATH_JOIN_ROUND as DVZ_PATH_JOIN_ROUND,
)
from ._state import (
    DVZ_PLACEMENT_SPACE_PANEL as DVZ_PLACEMENT_SPACE_PANEL,
)
from ._state import (
    DVZ_POINTER_BUTTON_LEFT as DVZ_POINTER_BUTTON_LEFT,
)
from ._state import (
    DVZ_POINTER_BUTTON_RIGHT as DVZ_POINTER_BUTTON_RIGHT,
)
from ._state import (
    DVZ_POINTER_EVENT_DOUBLE_CLICK as DVZ_POINTER_EVENT_DOUBLE_CLICK,
)
from ._state import (
    DVZ_POINTER_EVENT_DRAG as DVZ_POINTER_EVENT_DRAG,
)
from ._state import (
    DVZ_POINTER_EVENT_DRAG_STOP as DVZ_POINTER_EVENT_DRAG_STOP,
)
from ._state import (
    DVZ_POINTER_EVENT_MOVE as DVZ_POINTER_EVENT_MOVE,
)
from ._state import (
    DVZ_POINTER_EVENT_PRESS as DVZ_POINTER_EVENT_PRESS,
)
from ._state import (
    DVZ_POINTER_EVENT_RELEASE as DVZ_POINTER_EVENT_RELEASE,
)
from ._state import (
    DVZ_POINTER_EVENT_WHEEL as DVZ_POINTER_EVENT_WHEEL,
)
from ._state import (
    DVZ_QUERY_CAPABILITY_FACE as DVZ_QUERY_CAPABILITY_FACE,
)
from ._state import (
    DVZ_QUERY_CAPABILITY_ITEM as DVZ_QUERY_CAPABILITY_ITEM,
)
from ._state import (
    DVZ_QUERY_CAPABILITY_PIXEL as DVZ_QUERY_CAPABILITY_PIXEL,
)
from ._state import (
    DVZ_QUERY_HIT_FRONTMOST as DVZ_QUERY_HIT_FRONTMOST,
)
from ._state import (
    DVZ_QUERY_PROFILE_UNSUPPORTED as DVZ_QUERY_PROFILE_UNSUPPORTED,
)
from ._state import (
    DVZ_SCALE_CONTINUOUS as DVZ_SCALE_CONTINUOUS,
)
from ._state import (
    DVZ_SCENE_ANCHOR_DATA as DVZ_SCENE_ANCHOR_DATA,
)
from ._state import (
    DVZ_SCENE_ANCHOR_PANEL_BOTTOM as DVZ_SCENE_ANCHOR_PANEL_BOTTOM,
)
from ._state import (
    DVZ_SCENE_ANCHOR_PANEL_CENTER as DVZ_SCENE_ANCHOR_PANEL_CENTER,
)
from ._state import (
    DVZ_SCENE_ANCHOR_PANEL_LEFT as DVZ_SCENE_ANCHOR_PANEL_LEFT,
)
from ._state import (
    DVZ_SCENE_ANCHOR_PANEL_RIGHT as DVZ_SCENE_ANCHOR_PANEL_RIGHT,
)
from ._state import (
    DVZ_SCENE_ANCHOR_PANEL_TOP as DVZ_SCENE_ANCHOR_PANEL_TOP,
)
from ._state import (
    DVZ_SCENE_TARGET_FACE as DVZ_SCENE_TARGET_FACE,
)
from ._state import (
    DVZ_SCENE_TARGET_ITEM as DVZ_SCENE_TARGET_ITEM,
)
from ._state import (
    DVZ_SCENE_TARGET_NONE as DVZ_SCENE_TARGET_NONE,
)
from ._state import (
    DVZ_SEGMENT_CAP_BUTT as DVZ_SEGMENT_CAP_BUTT,
)
from ._state import (
    DVZ_SEGMENT_CAP_ROUND as DVZ_SEGMENT_CAP_ROUND,
)
from ._state import (
    DVZ_SEGMENT_CAP_SQUARE as DVZ_SEGMENT_CAP_SQUARE,
)
from ._state import (
    DVZ_SHAPE_ASPECT_FILLED as DVZ_SHAPE_ASPECT_FILLED,
)
from ._state import (
    DVZ_SHAPE_ASPECT_OUTLINE as DVZ_SHAPE_ASPECT_OUTLINE,
)
from ._state import (
    DVZ_TEXT_PLACEMENT_DATA as DVZ_TEXT_PLACEMENT_DATA,
)
from ._state import (
    DVZ_TEXT_PLACEMENT_SCREEN as DVZ_TEXT_PLACEMENT_SCREEN,
)
from ._state import (
    DVZ_TEXT_RENDERER_MSDF_ATLAS as DVZ_TEXT_RENDERER_MSDF_ATLAS,
)
from ._state import (
    DVZ_VERTICAL_ANCHOR_CENTER as DVZ_VERTICAL_ANCHOR_CENTER,
)
from ._state import (
    DatovizColorPipeline as DatovizColorPipeline,
)
from ._state import (
    DatovizRetainedView3DUpdateStats as DatovizRetainedView3DUpdateStats,
)
from ._state import (
    DatovizV04Unavailable as DatovizV04Unavailable,
)
from ._state import (
    DatovizV04Unsupported as DatovizV04Unsupported,
)
from ._state import (
    NavigationAction as NavigationAction,
)
from ._state import (
    _CompatDvzTextPlacement as _CompatDvzTextPlacement,
)
from ._state import (
    _DatovizRetainedPanelState as _DatovizRetainedPanelState,
)
from ._state import (
    _DatovizView2DAxisState as _DatovizView2DAxisState,
)
from ._state import (
    _RetainedView2DPositionUpload as _RetainedView2DPositionUpload,
)
from ._state import (
    _RetainedView3DMeshAttachment as _RetainedView3DMeshAttachment,
)
from ._state import (
    _RetainedView3DTextAttachment as _RetainedView3DTextAttachment,
)
from ._state import (
    _ScalarVisualData as _ScalarVisualData,
)
from .capture import (
    _encode_rgba8_png as _encode_rgba8_png,
)
from .colors import (
    _marker_fill_colors as _marker_fill_colors,
)
from .colors import (
    _point_colors as _point_colors,
)
from .colors import (
    _resolved_scalar_texel as _resolved_scalar_texel,
)
from .colors import (
    _rgba8 as _rgba8,
)
from .colors import (
    _rgba8_broadcast as _rgba8_broadcast,
)
from .colors import (
    _rgba8_image as _rgba8_image,
)
from .colors import (
    _rgba8_scalar as _rgba8_scalar,
)
from .colors import (
    _rgba8_scalar_image as _rgba8_scalar_image,
)
from .colors import (
    _rgba8_scalar_values as _rgba8_scalar_values,
)
from .colors import (
    _scalar_payload_for_query_result as _scalar_payload_for_query_result,
)
from .geometry import (
    _adapt_visual_positions as _adapt_visual_positions,
)
from .geometry import (
    _cpu_adapt_affine_positions as _cpu_adapt_affine_positions,
)
from .geometry import (
    _datoviz_mesh3d_plot_ndc_positions as _datoviz_mesh3d_plot_ndc_positions,
)
from .geometry import (
    _datoviz_mesh_payload as _datoviz_mesh_payload,
)
from .geometry import (
    _diameters_from_pixel_diameters as _diameters_from_pixel_diameters,
)
from .geometry import (
    _image_positions as _image_positions,
)
from .geometry import (
    _image_texcoords as _image_texcoords,
)
from .geometry import (
    _map_view2d_data_positions_to_view as _map_view2d_data_positions_to_view,
)
from .geometry import (
    _mesh3d_face_depth_order as _mesh3d_face_depth_order,
)
from .geometry import (
    _positions_3d as _positions_3d,
)
from .geometry import (
    _record_transform_adaptation as _record_transform_adaptation,
)
from .geometry import (
    _resolve_datoviz_flat_lambert_facecolors as _resolve_datoviz_flat_lambert_facecolors,
)
from .geometry import (
    _rgba8_image_visual as _rgba8_image_visual,
)
from .geometry import (
    _transform_binding_matrix as _transform_binding_matrix,
)
from .geometry import (
    _validate_datoviz_mesh3d_visual as _validate_datoviz_mesh3d_visual,
)
from .guides import (
    _assign_style_color as _assign_style_color,
)
from .guides import (
    _builtin_colormap_value as _builtin_colormap_value,
)
from .guides import (
    _colorbar_anchor_value as _colorbar_anchor_value,
)
from .guides import (
    _colorbar_orientation_value as _colorbar_orientation_value,
)
from .guides import (
    _configure_axis_review_plot_margins as _configure_axis_review_plot_margins,
)
from .guides import (
    _configure_axis_review_style as _configure_axis_review_style,
)
from .guides import (
    _configure_colorbar_format as _configure_colorbar_format,
)
from .guides import (
    _configure_colorbar_layout as _configure_colorbar_layout,
)
from .guides import (
    _configure_colorbar_ticks as _configure_colorbar_ticks,
)
from .layout import (
    _consumed_layout_native_panel_bounds as _consumed_layout_native_panel_bounds,
)
from .layout import (
    _create_panel as _create_panel,
)
from .layout import (
    _datoviz_axis_dimension_for_role as _datoviz_axis_dimension_for_role,
)
from .layout import (
    _datoviz_contribution_layer as _datoviz_contribution_layer,
)
from .layout import (
    _datoviz_enum_value as _datoviz_enum_value,
)
from .layout import (
    _datoviz_enum_values as _datoviz_enum_values,
)
from .layout import (
    _datoviz_frame_contribution_layers as _datoviz_frame_contribution_layers,
)
from .layout import (
    _datoviz_frame_data_to_screen_transform as _datoviz_frame_data_to_screen_transform,
)
from .layout import (
    _datoviz_frame_device_scale as _datoviz_frame_device_scale,
)
from .layout import (
    _datoviz_frame_guide_boxes as _datoviz_frame_guide_boxes,
)
from .layout import (
    _datoviz_guide_box_kind as _datoviz_guide_box_kind,
)
from .layout import (
    _datoviz_guide_role as _datoviz_guide_role,
)
from .layout import (
    _datoviz_layout_snapshot_id_matches as _datoviz_layout_snapshot_id_matches,
)
from .layout import (
    _datoviz_orthographic_bounds_tuple as _datoviz_orthographic_bounds_tuple,
)
from .layout import (
    _datoviz_protocol_id as _datoviz_protocol_id,
)
from .layout import (
    _datoviz_query_panel_bounds as _datoviz_query_panel_bounds,
)
from .layout import (
    _datoviz_vec3_tuple as _datoviz_vec3_tuple,
)
from .layout import (
    _native_uint_id as _native_uint_id,
)
from .layout import (
    _optional_dvz_rect as _optional_dvz_rect,
)
from .layout import (
    _panel_pixel_size as _panel_pixel_size,
)
from .layout import (
    _positive_or_fallback as _positive_or_fallback,
)
from .layout import (
    _preflight_consumed_layout_panel_api as _preflight_consumed_layout_panel_api,
)
from .layout import (
    _query_datoviz_panel_frame_guides as _query_datoviz_panel_frame_guides,
)
from .layout import (
    _required_dvz_rect as _required_dvz_rect,
)
from .layout import (
    _resolve_datoviz_partial_layout_snapshot as _resolve_datoviz_partial_layout_snapshot,
)
from .layout import (
    _validate_consumed_perspective_aspect as _validate_consumed_perspective_aspect,
)
from .layout import (
    _validate_datoviz_frame_units as _validate_datoviz_frame_units,
)
from .layout import (
    _validate_renderer_consumed_layout_view as _validate_renderer_consumed_layout_view,
)
from .layout import (
    _validate_resolved_canvas_matches_layout as _validate_resolved_canvas_matches_layout,
)
from .native_api import (
    _add_visual_to_panel as _add_visual_to_panel,
)
from .native_api import (
    _alpha_mode_value as _alpha_mode_value,
)
from .native_api import (
    _assign_rgba_field as _assign_rgba_field,
)
from .native_api import (
    _configure_datoviz_view3d_camera as _configure_datoviz_view3d_camera,
)
from .native_api import (
    _configure_ndc_panel_view2d as _configure_ndc_panel_view2d,
)
from .native_api import (
    _controller_mode_value as _controller_mode_value,
)
from .native_api import (
    _ctypes_pointer_arg as _ctypes_pointer_arg,
)
from .native_api import (
    _datoviz_call_succeeded as _datoviz_call_succeeded,
)
from .native_api import (
    _datoviz_color_pipeline_value as _datoviz_color_pipeline_value,
)
from .native_api import (
    _datoviz_colorbar_diagnostics as _datoviz_colorbar_diagnostics,
)
from .native_api import (
    _datoviz_incomplete_ctypes_records as _datoviz_incomplete_ctypes_records,
)
from .native_api import (
    _datoviz_label as _datoviz_label,
)
from .native_api import (
    _datoviz_marker_diagnostics as _datoviz_marker_diagnostics,
)
from .native_api import (
    _datoviz_panel_view2d_desc as _datoviz_panel_view2d_desc,
)
from .native_api import (
    _datoviz_path_diagnostics as _datoviz_path_diagnostics,
)
from .native_api import (
    _datoviz_query_request_diagnostic as _datoviz_query_request_diagnostic,
)
from .native_api import (
    _datoviz_request_id as _datoviz_request_id,
)
from .native_api import (
    _datoviz_segment_diagnostics as _datoviz_segment_diagnostics,
)
from .native_api import (
    _datoviz_shared_library_function as _datoviz_shared_library_function,
)
from .native_api import (
    _datoviz_texture2d_mesh_diagnostics as _datoviz_texture2d_mesh_diagnostics,
)
from .native_api import (
    _datoviz_view2d_descriptor_has_data_domains as _datoviz_view2d_descriptor_has_data_domains,
)
from .native_api import (
    _datoviz_view_kind_value as _datoviz_view_kind_value,
)
from .native_api import (
    _datoviz_view_size_desc as _datoviz_view_size_desc,
)
from .native_api import (
    _datoviz_visual_coord_space as _datoviz_visual_coord_space,
)
from .native_api import (
    _dvz_color as _dvz_color,
)
from .native_api import (
    _enum_value as _enum_value,
)
from .native_api import (
    _expand_path_colors as _expand_path_colors,
)
from .native_api import (
    _expand_path_widths as _expand_path_widths,
)
from .native_api import (
    _fill_datoviz_camera_desc as _fill_datoviz_camera_desc,
)
from .native_api import (
    _image_sampling_value as _image_sampling_value,
)
from .native_api import (
    _is_null_handle as _is_null_handle,
)
from .native_api import (
    _marker_shape_value as _marker_shape_value,
)
from .native_api import (
    _marker_shapes as _marker_shapes,
)
from .native_api import (
    _panel_query as _panel_query,
)
from .native_api import (
    _panel_view3d_camera_desc as _panel_view3d_camera_desc,
)
from .native_api import (
    _positive_native_float as _positive_native_float,
)
from .native_api import (
    _preflight_datoviz_vector_api as _preflight_datoviz_vector_api,
)
from .native_api import (
    _query_frame_resolution_ready as _query_frame_resolution_ready,
)
from .native_api import (
    _require_datoviz_success as _require_datoviz_success,
)
from .native_api import (
    _resolve_datoviz_canvas_size as _resolve_datoviz_canvas_size,
)
from .native_api import (
    _resolved_canvas_from_datoviz as _resolved_canvas_from_datoviz,
)
from .native_api import (
    _retained_view3d_state_mismatch_diagnostics as _retained_view3d_state_mismatch_diagnostics,
)
from .native_api import (
    _scene_poll_query as _scene_poll_query,
)
from .native_api import (
    _set_alpha_mode_if_translucent as _set_alpha_mode_if_translucent,
)
from .native_api import (
    _set_axis_ticks as _set_axis_ticks,
)
from .native_api import (
    _set_data_view_payload as _set_data_view_payload,
)
from .native_api import (
    _set_datoviz_camera_orthographic_bounds as _set_datoviz_camera_orthographic_bounds,
)
from .native_api import (
    _set_datoviz_camera_projection_state as _set_datoviz_camera_projection_state,
)
from .native_api import (
    _set_datoviz_data_domain as _set_datoviz_data_domain,
)
from .native_api import (
    _set_datoviz_monitor_dpi_override as _set_datoviz_monitor_dpi_override,
)
from .native_api import (
    _set_datoviz_panel_domains as _set_datoviz_panel_domains,
)
from .native_api import (
    _set_figure_color_pipeline as _set_figure_color_pipeline,
)
from .native_api import (
    _set_filled_point_style as _set_filled_point_style,
)
from .native_api import (
    _set_image_sampling as _set_image_sampling,
)
from .native_api import (
    _set_marker_style as _set_marker_style,
)
from .native_api import (
    _set_panel_background_color as _set_panel_background_color,
)
from .native_api import (
    _set_path_subpaths as _set_path_subpaths,
)
from .native_api import (
    _set_query_capabilities as _set_query_capabilities,
)
from .native_api import (
    _set_visual_data as _set_visual_data,
)
from .native_api import (
    _set_visual_field as _set_visual_field,
)
from .native_api import (
    _set_visual_field_sampling as _set_visual_field_sampling,
)
from .native_api import (
    _set_visual_index_data as _set_visual_index_data,
)
from .native_api import (
    _stroke_cap_value as _stroke_cap_value,
)
from .native_api import (
    _stroke_join_value as _stroke_join_value,
)
from .native_api import (
    _unsupported_query_result as _unsupported_query_result,
)
from .native_api import (
    _update_datoviz_view3d_camera as _update_datoviz_view3d_camera,
)
from .native_api import (
    _validate_visual_attach_desc_binding as _validate_visual_attach_desc_binding,
)
from .native_api import (
    _visual_attach_desc as _visual_attach_desc,
)
from .native_api import (
    capability_snapshot as capability_snapshot,
)
from .native_api import (
    datoviz_v04_live_input_diagnostics as datoviz_v04_live_input_diagnostics,
)
from .native_api import (
    datoviz_v04_live_input_ready as datoviz_v04_live_input_ready,
)
from .native_api import (
    datoviz_v04_mesh_diagnostics as datoviz_v04_mesh_diagnostics,
)
from .native_api import (
    datoviz_v04_mesh_ready as datoviz_v04_mesh_ready,
)
from .native_api import (
    datoviz_v04_panel_frame_guide_query_diagnostics as datoviz_v04_panel_frame_guide_query_diagnostics,
)
from .native_api import (
    datoviz_v04_panel_frame_guide_query_ready as datoviz_v04_panel_frame_guide_query_ready,
)
from .native_api import (
    datoviz_v04_panel_frame_snapshot_diagnostics as datoviz_v04_panel_frame_snapshot_diagnostics,
)
from .native_api import (
    datoviz_v04_panel_frame_snapshot_ready as datoviz_v04_panel_frame_snapshot_ready,
)
from .native_api import (
    datoviz_v04_sampled_field_diagnostics as datoviz_v04_sampled_field_diagnostics,
)
from .native_api import (
    datoviz_v04_sampled_field_ready as datoviz_v04_sampled_field_ready,
)
from .native_api import (
    datoviz_v04_text_diagnostics as datoviz_v04_text_diagnostics,
)
from .native_api import (
    datoviz_v04_text_ready as datoviz_v04_text_ready,
)
from .native_api import (
    datoviz_v04_view3d_camera_diagnostics as datoviz_v04_view3d_camera_diagnostics,
)
from .native_api import (
    datoviz_v04_view3d_live_navigation_diagnostics as datoviz_v04_view3d_live_navigation_diagnostics,
)
from .native_api import (
    datoviz_v04_view3d_retained_data_diagnostics as datoviz_v04_view3d_retained_data_diagnostics,
)
from .native_api import (
    datoviz_v04_view3d_retained_data_ready as datoviz_v04_view3d_retained_data_ready,
)
from .native_api import (
    datoviz_v04_view3d_state_readback_diagnostics as datoviz_v04_view3d_state_readback_diagnostics,
)
from .native_api import (
    datoviz_v04_view3d_state_readback_ready as datoviz_v04_view3d_state_readback_ready,
)
from .native_api import (
    import_datoviz_v04 as import_datoviz_v04,
)
from .native_api import (
    is_datoviz_v04_facade as is_datoviz_v04_facade,
)
from .navigation import (
    _apply_view2d_navigation_action as _apply_view2d_navigation_action,
)
from .navigation import (
    _datoviz_pointer_y_to_gsp_logical_px as _datoviz_pointer_y_to_gsp_logical_px,
)
from .navigation import (
    _datoviz_resize_logical_size as _datoviz_resize_logical_size,
)
from .navigation import (
    _DatovizLiveView2DNavigation as _DatovizLiveView2DNavigation,
)
from .navigation import (
    _DatovizLiveView3DNavigation as _DatovizLiveView3DNavigation,
)
from .navigation import (
    _navigation_pointer_event_from_datoviz as _navigation_pointer_event_from_datoviz,
)
from .navigation import (
    _reject_navigation_action as _reject_navigation_action,
)
from .text import (
    _assign_rgba8 as _assign_rgba8,
)
from .text import (
    _ndc_text_screen_position as _ndc_text_screen_position,
)
from .text import (
    _set_text_placement as _set_text_placement,
)
from .text import (
    _text_anchor_value as _text_anchor_value,
)
from .text import (
    _text_anchor_x_value as _text_anchor_x_value,
)
from .text import (
    _text_anchor_y_value as _text_anchor_y_value,
)
from .text import (
    _text_placement as _text_placement,
)
from .text import (
    _text_placement_mode_value as _text_placement_mode_value,
)
from .text import (
    _text_renderer_value as _text_renderer_value,
)


@dataclass
class DatovizV04ProtocolRenderer:
    """Minimal point/image renderer using Datoviz v0.4 top-level functions."""

    dvz: Any = None
    color_scales: Mapping[str, ColorScale] | None = None
    texture_resources: Mapping[str, Texture2D] | None = None
    width: int = 800
    height: int = 600
    canvas_size: CanvasSize | None = None
    background_rgba8: tuple[int, int, int, int] = DEFAULT_BACKGROUND_RGBA8
    color_pipeline: DatovizColorPipeline = "legacy_srgb_blend"
    view: View2D | None = None
    view3d: View3D | None = None
    transform_resources: Mapping[str, AffineTransform2DResource] | None = None
    panel_bounds: tuple[float, float, float, float] | None = None
    panel_id: str = "panel:default"
    panel_layout: PanelLayoutIntent | None = None
    consumed_layout_snapshot: ResolvedLayoutSnapshot | None = None
    scene: Any = field(init=False)
    figure: Any = field(init=False)
    panel: Any = field(init=False)
    app: Any | None = field(default=None, init=False)
    offscreen_view: Any | None = field(default=None, init=False)
    live_view: Any | None = field(default=None, init=False)
    native_panzoom: Any | None = field(default=None, init=False)
    native_arcball: Any | None = field(default=None, init=False)
    live_navigation: "_DatovizLiveView2DNavigation | None" = field(default=None, init=False)
    live_view3d_navigation: "_DatovizLiveView3DNavigation | None" = field(default=None, init=False)
    live_navigations: dict[str, "_DatovizLiveView2DNavigation"] = field(
        default_factory=dict, init=False
    )
    live_view3d_navigations: dict[str, "_DatovizLiveView3DNavigation"] = field(
        default_factory=dict, init=False
    )
    visuals: dict[str, Any] = field(default_factory=dict, init=False)
    sampled_fields: dict[str, Any] = field(default_factory=dict, init=False)
    native_scales: dict[str, Any] = field(default_factory=dict, init=False)
    native_colormaps: dict[str, Any] = field(default_factory=dict, init=False)
    colorbars: dict[str, Any] = field(default_factory=dict, init=False)
    resolved_canvas: ResolvedCanvas = field(init=False)
    scalar_visuals: dict[str, _ScalarVisualData] = field(default_factory=dict, init=False)
    retained_view2d_position_uploads: list[_RetainedView2DPositionUpload] = field(
        default_factory=list, init=False
    )
    retained_view3d_meshes: list[_RetainedView3DMeshAttachment] = field(
        default_factory=list, init=False
    )
    retained_view3d_texts: list[_RetainedView3DTextAttachment] = field(
        default_factory=list, init=False
    )
    retained_view3d_update_stats: DatovizRetainedView3DUpdateStats = field(
        default_factory=DatovizRetainedView3DUpdateStats, init=False
    )
    view2d_axis_state: _DatovizView2DAxisState | None = field(default=None, init=False)
    native_view3d_camera: Any | None = field(default=None, init=False)
    transform_adaptations: dict[str, tuple[str, ...]] = field(default_factory=dict, init=False)
    last_view2d_carrier_diagnostics: dict[str, object] = field(default_factory=dict, init=False)
    _panel_states: dict[str, _DatovizRetainedPanelState] = field(default_factory=dict, init=False)
    _authoritative_scene_layout_snapshot: ResolvedLayoutSnapshot | None = field(
        default=None, init=False
    )
    _cpu_map_data_visuals_to_view: bool = field(default=False, init=False)
    _closed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.view is not None and self.view3d is not None:
            raise ValueError("Datoviz renderer accepts either view or view3d, not both")
        if self.dvz is None:
            self.dvz = import_datoviz_v04()
        elif not is_datoviz_v04_facade(self.dvz):
            missing = [name for name in _REQUIRED_DVZ_V04_FUNCTIONS if not hasattr(self.dvz, name)]
            raise DatovizV04Unavailable(f"Datoviz facade is missing v0.4 functions: {missing}")
        if self.consumed_layout_snapshot is not None:
            _preflight_consumed_layout_panel_api(self.dvz)
            _validate_renderer_consumed_layout_view(
                self.consumed_layout_snapshot,
                view=self.view,
                view3d=self.view3d,
            )
            _validate_consumed_perspective_aspect(self.consumed_layout_snapshot, self.view3d)
            expected_bounds = _consumed_layout_native_panel_bounds(self.consumed_layout_snapshot)
            if self.panel_bounds is None:
                self.panel_bounds = expected_bounds
            elif not all(
                math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12)
                for actual, expected in zip(self.panel_bounds, expected_bounds, strict=True)
            ):
                raise ValueError("panel_bounds conflicts with the consumed layout plot rectangle")
            if self.canvas_size is None:
                target = self.consumed_layout_snapshot.render_target
                self.canvas_size = CanvasSize.reference_px(
                    target.logical_width_px,
                    target.logical_height_px,
                    reference_dpi=target.dpi or 96.0,
                ).with_requested_device_scale(target.device_scale)

        requested_size = self.canvas_size or CanvasSize.pixel_exact(self.width, self.height)
        self.resolved_canvas = _resolve_datoviz_canvas_size(self.dvz, requested_size)
        if self.consumed_layout_snapshot is not None:
            _validate_resolved_canvas_matches_layout(
                self.resolved_canvas, self.consumed_layout_snapshot
            )
        elif self.panel_layout is not None:
            target = RenderTarget(
                logical_width_px=self.resolved_canvas.canvas_width_px,
                logical_height_px=self.resolved_canvas.canvas_height_px,
                device_scale=self.resolved_canvas.framebuffer_per_canvas_px,
                dpi=self.resolved_canvas.output_dpi,
                pixel_origin=PixelOrigin.TOP_LEFT,
                query_coordinate_space="plot",
            )
            resolved_panels = resolve_panel_layout_intent(self.panel_layout, target)
            matching_panels = tuple(
                panel for panel in resolved_panels if panel.panel_id == self.panel_id
            )
            if len(matching_panels) != 1:
                raise ValueError("panel_layout must resolve the rendered panel exactly once")
            panel = matching_panels[0].panel_rect_px
            self.panel_bounds = (
                panel.x / target.logical_width_px,
                panel.y / target.logical_height_px,
                panel.width / target.logical_width_px,
                panel.height / target.logical_height_px,
            )
        self.width = self.resolved_canvas.framebuffer_width
        self.height = self.resolved_canvas.framebuffer_height

        self.scene = self.dvz.dvz_scene()
        self.figure = self.dvz.dvz_figure(self.scene, self.width, self.height, 0)
        _set_figure_color_pipeline(self.dvz, self.figure, self.color_pipeline)
        self.panel = _create_panel(self.dvz, self.figure, self.panel_bounds)
        _set_panel_background_color(self.dvz, self.panel, self.background_rgba8)
        _configure_ndc_panel_view2d(self.dvz, self.panel)
        if self.view is not None:
            self.apply_datoviz_data_view2d(self.view)
        if self.view3d is not None:
            self.native_view3d_camera = _configure_datoviz_view3d_camera(
                self.dvz, self.panel, self.view3d
            )
            self.retained_view3d_update_stats.view_projection_uniform_updates += 1
        self._store_active_panel_state()

    def _store_active_panel_state(self) -> None:
        return _panels._store_active_panel_state(self)

    def activate_panel(self, panel_id: str) -> None:
        """Select one retained panel as the target of subsequent adapter operations."""
        return _panels.activate_panel(self, panel_id)

    def add_retained_panel(
        self,
        *,
        panel_id: str,
        view: View2D | None,
        view3d: View3D | None,
        panel_layout: PanelLayoutIntent,
        consumed_layout_snapshot: ResolvedLayoutSnapshot | None = None,
    ) -> Any:
        """Create another typed panel in this renderer's retained figure."""
        return _panels.add_retained_panel(
            self,
            panel_id=panel_id,
            view=view,
            view3d=view3d,
            panel_layout=panel_layout,
            consumed_layout_snapshot=consumed_layout_snapshot,
        )

    def set_authoritative_scene_layout_snapshot(
        self, snapshot: ResolvedLayoutSnapshot | None
    ) -> None:
        return _panels.set_authoritative_scene_layout_snapshot(self, snapshot)

    def capabilities(self) -> CapabilitySnapshot:
        """Return the capability snapshot for this adapter slice."""
        return datoviz_v04_capability_snapshot(self.dvz)

    def resolve_partial_layout_snapshot(
        self, *, snapshot_id_prefix: str = "layout:datoviz"
    ) -> ResolvedLayoutSnapshot:
        """Map Datoviz-reported panel frame fields into a partial GSP snapshot.

        The adapter only copies geometry and identities reported by Datoviz. Missing guide/query
        semantics remain explicit diagnostics and are intentionally not synthesized here.
        """
        return _panels.resolve_partial_layout_snapshot(self, snapshot_id_prefix=snapshot_id_prefix)

    def authoritative_layout_snapshot(self) -> ResolvedLayoutSnapshot | None:
        """Return the consumed GSP layout retained independently of the native panel."""
        return _panels.authoritative_layout_snapshot(self)

    def _canvas_px_scale(self) -> float:
        """Convert GSP canvas pixels to Datoviz logical screen pixels.

        Datoviz ``*_px`` visual attributes are authored in logical pixels and
        its runtime applies the view device scale during frame emission.
        Scaling by framebuffer-per-canvas here would therefore apply HiDPI
        twice. Use the canvas-to-host conversion instead.
        """
        return 0.5 * (
            self.resolved_canvas.canvas_to_host_scale_x
            + self.resolved_canvas.canvas_to_host_scale_y
        )

    def _scale_canvas_px_array(self, values: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        return np.ascontiguousarray(
            values.astype(np.float32, copy=False) * np.float32(self._canvas_px_scale())
        )

    def _scale_canvas_px(self, value: float) -> float:
        return float(value * self._canvas_px_scale())

    def close(self) -> None:
        """Destroy the scene when the facade exposes a destroy helper."""
        if self._closed:
            return
        for navigation2d in self.live_navigations.values():
            navigation2d.close()
        for navigation3d in self.live_view3d_navigations.values():
            navigation3d.close()
        self.live_navigations.clear()
        self.live_view3d_navigations.clear()
        self.live_navigation = None
        self.live_view3d_navigation = None
        destroy_app = getattr(self.dvz, "dvz_app_destroy", None)
        if destroy_app is not None and self.app is not None:
            destroy_app(self.app)
        destroy = getattr(self.dvz, "dvz_scene_destroy", None)
        if destroy is not None:
            destroy(self.scene)
        self._closed = True

    def __enter__(self) -> "DatovizV04ProtocolRenderer":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def add_point_visual(self, visual: PointVisual) -> Any:
        """Create and attach a Datoviz point visual."""
        return _visual_lowering.add_point_visual(self, visual)

    def add_pixel_visual(self, visual: PixelVisual) -> Any:
        """Create and attach a public Datoviz square-pixel visual."""
        return _visual_lowering.add_pixel_visual(self, visual)

    def add_sphere_visual(self, visual: SphereVisual) -> Any:
        """Create accurate public Datoviz raycast sphere impostors."""
        return _visual_lowering.add_sphere_visual(self, visual)

    def add_vector_visual(self, visual: VectorVisual) -> Any:
        """Create a public Datoviz straight vector visual with visual-wide style."""
        return _visual_lowering.add_vector_visual(self, visual)

    def add_primitive_visual(self, visual: PrimitiveVisual) -> Any:
        """Create a public Datoviz bounded primitive with optional public indices."""
        return _visual_lowering.add_primitive_visual(self, visual)

    def update_point_visual(self, visual: PointVisual) -> None:
        """Update one retained point visual from semantic GSP state.

        This is an adapter operation used by bounded lifecycle/conformance probes. It keeps native
        handles private and deliberately rejects updates after close or for unknown visual IDs.
        """
        return _visual_lowering.update_point_visual(self, visual)

    def add_marker_visual(self, visual: MarkerVisual) -> Any:
        """Create and attach a Datoviz marker visual."""
        return _visual_lowering.add_marker_visual(self, visual)

    def add_segment_visual(self, visual: SegmentVisual) -> Any:
        """Create and attach a Datoviz segment visual."""
        return _visual_lowering.add_segment_visual(self, visual)

    def add_path_visual(self, visual: PathVisual) -> Any:
        """Create and attach a Datoviz path visual."""
        return _visual_lowering.add_path_visual(self, visual)

    def add_image_visual(self, visual: ImageVisual) -> Any:
        """Create and attach a Datoviz image visual in DATA or NDC coordinates."""
        return _visual_lowering.add_image_visual(self, visual)

    def add_mesh_visual(self, visual: MeshVisual) -> Any:
        """Create and attach a bounded Datoviz triangle mesh visual."""
        return _visual_lowering.add_mesh_visual(self, visual)

    def _configure_texture2d_unlit_mesh(
        self,
        dvz_visual: Any,
        visual: MeshVisual,
        positions: npt.NDArray[np.float32],
    ) -> None:
        """Bind strict RGBA8 nearest-or-linear unlit Texture2D mesh state."""
        return _visual_lowering._configure_texture2d_unlit_mesh(self, dvz_visual, visual, positions)

    def add_text_visual(self, visual: TextVisual) -> Any:
        """Create retained 2D text or projected screen-facing 3D billboards."""
        return _visual_lowering.add_text_visual(self, visual)

    def _update_billboard_text_placements(self, view3d: View3D) -> None:
        """Move retained overlay labels after a View3D camera/projection update."""
        return _visual_lowering._update_billboard_text_placements(self, view3d)

    def _visual_coord_space(self, coordinate_space: CoordinateSpace) -> str:
        return _visual_lowering._visual_coord_space(self, coordinate_space)

    def _retain_view2d_position_upload(
        self,
        visual_id: str,
        native_visual: Any,
        attr_name: str,
        positions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
        transform: VisualTransformBinding | None,
        coordinate_space: CoordinateSpace,
    ) -> None:
        return _visual_lowering._retain_view2d_position_upload(
            self, visual_id, native_visual, attr_name, positions, transform, coordinate_space
        )

    def _reupload_retained_view2d_positions(self, view: View2D) -> None:
        return _visual_lowering._reupload_retained_view2d_positions(self, view)

    def add_colorbar_guide(self, guide: ColorbarGuide) -> Any:
        """Create a native Datoviz colorbar bound to the guide's color scale."""
        scale = resolve_color_scale(self.color_scales, guide.color_scale_id)
        diagnostics = _datoviz_colorbar_diagnostics(self.dvz)
        if diagnostics:
            raise DatovizV04Unsupported(
                "colorbar_render_unsupported: Datoviz v0.4 ColorbarGuide "
                "facade is unavailable: " + "; ".join(diagnostics)
            )

        native_scale = self._create_native_colorbar_scale(scale, guide)
        desc = self.dvz.dvz_colorbar_desc()
        _configure_colorbar_layout(
            self.dvz,
            desc,
            guide,
            self.resolved_canvas.host_logical_width,
            self.resolved_canvas.host_logical_height,
            canvas_px_scale=self._canvas_px_scale(),
        )
        if hasattr(desc, "anchor"):
            desc.anchor = _colorbar_anchor_value(self.dvz, guide.placement)
        if hasattr(desc, "title"):
            desc.title = guide.label.encode("utf-8") if guide.label else None

        colorbar = self.dvz.dvz_colorbar(self.panel, native_scale, _ctypes_pointer_arg(desc))
        if colorbar is None:
            raise DatovizV04Unsupported("Datoviz colorbar allocation failed")
        self.dvz.dvz_colorbar_set_orientation(
            colorbar, _colorbar_orientation_value(self.dvz, guide.orientation)
        )
        _require_datoviz_success(
            self.dvz.dvz_colorbar_set_anchor(
                colorbar, _colorbar_anchor_value(self.dvz, guide.placement)
            ),
            "Datoviz colorbar anchor configuration failed",
        )
        _configure_colorbar_format(self.dvz, colorbar)
        _configure_colorbar_ticks(self.dvz, colorbar, guide)
        if guide.label:
            self.dvz.dvz_colorbar_set_title(colorbar, guide.label.encode("utf-8"))
        self.colorbars[guide.id] = colorbar
        return colorbar

    def _create_native_colorbar_scale(self, scale: ColorScale, guide: ColorbarGuide) -> Any:
        if scale.id in self.native_scales:
            return self.native_scales[scale.id]

        desc = self.dvz.dvz_scale_desc()
        if hasattr(desc, "kind"):
            desc.kind = _enum_value(
                self.dvz, "DvzScaleKind", "DVZ_SCALE_CONTINUOUS", DVZ_SCALE_CONTINUOUS
            )
        if hasattr(desc, "label"):
            desc.label = guide.label.encode("utf-8") if guide.label else None
        native_scale = self.dvz.dvz_scale(self.scene, _ctypes_pointer_arg(desc))
        if native_scale is None:
            raise DatovizV04Unsupported("Datoviz scale allocation failed")
        self.dvz.dvz_scale_set_domain(native_scale, scale.normalize.vmin, scale.normalize.vmax)
        self.dvz.dvz_scale_set_view_range(native_scale, scale.normalize.vmin, scale.normalize.vmax)
        colormap = self.dvz.dvz_colormap_builtin(
            self.scene, _builtin_colormap_value(self.dvz, scale.colormap.id)
        )
        if colormap is None:
            raise DatovizV04Unsupported("Datoviz colormap allocation failed")
        self.dvz.dvz_scale_set_colormap(native_scale, colormap)
        self.native_colormaps[scale.id] = colormap
        self.native_scales[scale.id] = native_scale
        return native_scale

    def show(self, *, frame_count: int = 0) -> None:
        """Open an interactive Datoviz window and run the app.

        A ``frame_count`` of 0 follows Datoviz convention and runs until the user
        closes the window. Tests may set ``GSP_TEST=True`` to avoid blocking.
        """
        if os.environ.get("GSP_TEST") == "True":
            return
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        self._ensure_live_view()
        app_run = getattr(self.dvz, "dvz_app_run", None)
        if app_run is None:
            raise DatovizV04Unavailable(
                "Datoviz interactive app run is unavailable: missing dvz_app_run"
            )
        app_run(self.app, frame_count)

    def enable_native_panzoom(self) -> Any:
        """Enable Datoviz v0.4 native pan/zoom on the renderer's live panel."""
        view_panzoom = getattr(self.dvz, "dvz_view_panzoom", None)
        if view_panzoom is None:
            raise DatovizV04Unavailable(
                "Datoviz native panzoom is unavailable: missing dvz_view_panzoom"
            )
        live_view = self._ensure_live_view()
        desc = None
        desc_factory = getattr(self.dvz, "dvz_panzoom_desc", None)
        if desc_factory is not None:
            desc = desc_factory()
            if hasattr(desc, "width"):
                desc.width = float(self.resolved_canvas.host_logical_width)
            if hasattr(desc, "height"):
                desc.height = float(self.resolved_canvas.host_logical_height)
        self.native_panzoom = view_panzoom(
            live_view,
            self.panel,
            _ctypes_pointer_arg(desc) if desc is not None else None,
        )
        if _is_null_handle(self.native_panzoom):
            raise DatovizV04Unavailable("Datoviz native panzoom creation failed")
        return self.native_panzoom

    def enable_native_view3d_arcball(self) -> Any:
        """Enable Datoviz v0.4 native arcball on the renderer's live 3D panel."""
        view_arcball = getattr(self.dvz, "dvz_view_arcball", None)
        if view_arcball is None:
            raise DatovizV04Unavailable(
                "Datoviz native arcball is unavailable: missing dvz_view_arcball"
            )
        live_view = self._ensure_live_view()
        desc = None
        desc_factory = getattr(self.dvz, "dvz_arcball_desc", None)
        if desc_factory is not None:
            desc = desc_factory()
            if hasattr(desc, "width"):
                desc.width = float(self.resolved_canvas.host_logical_width)
            if hasattr(desc, "height"):
                desc.height = float(self.resolved_canvas.host_logical_height)
        self.native_arcball = view_arcball(
            live_view,
            self.panel,
            _ctypes_pointer_arg(desc) if desc is not None else None,
        )
        if _is_null_handle(self.native_arcball):
            raise DatovizV04Unavailable("Datoviz native arcball creation failed")
        return self.native_arcball

    def enable_gsp_view2d_navigation(
        self,
        view: View2D | None = None,
        *,
        controller_id: str = "nav:datoviz-live",
        layout_snapshot_id: str | None = None,
    ) -> "_DatovizLiveView2DNavigation":
        """Enable Datoviz pointer input as canonical S035 View2D navigation."""
        diagnostics = datoviz_v04_live_input_diagnostics(self.dvz)
        if diagnostics:
            raise DatovizV04Unavailable(
                "Datoviz live input binding is unavailable: " + "; ".join(diagnostics)
            )
        target_view = view or self.view
        if target_view is None:
            raise DatovizV04Unavailable("Datoviz GSP navigation requires an initial View2D")
        self.activate_panel(target_view.panel_id)
        live_view = self._ensure_live_view()
        router = self.dvz.dvz_view_input(live_view)
        if _is_null_handle(router):
            raise DatovizV04Unavailable("Datoviz live input router is unavailable")
        previous = self.live_navigations.get(target_view.panel_id)
        if previous is not None:
            previous.close()
        effective_layout_snapshot_id = (
            self.consumed_layout_snapshot.snapshot_id
            if self.consumed_layout_snapshot is not None
            else layout_snapshot_id or "layout:datoviz-live"
        )
        self.live_navigation = _DatovizLiveView2DNavigation(
            renderer=self,
            router=router,
            live_view=live_view,
            view=target_view,
            controller_id=controller_id,
            layout_snapshot_id=effective_layout_snapshot_id,
        )
        self.live_navigation.subscription_id = self.dvz.dvz_input_subscribe_event(
            router, self.live_navigation.handle_input_event, None
        )
        self.live_navigations[target_view.panel_id] = self.live_navigation
        return self.live_navigation

    def enable_gsp_view3d_navigation(
        self,
        view3d: View3D | None = None,
        *,
        controller_id: str = "nav:datoviz-live-3d",
        layout_snapshot_id: str | None = None,
    ) -> "_DatovizLiveView3DNavigation":
        """Enable Datoviz pointer input as canonical S037 View3D navigation."""
        target_view3d = view3d or self.view3d
        if target_view3d is None:
            raise DatovizV04Unavailable("Datoviz GSP View3D navigation requires an initial View3D")
        self.activate_panel(target_view3d.panel_id)
        diagnostics = datoviz_v04_view3d_live_navigation_diagnostics(self.dvz)
        if diagnostics:
            raise DatovizV04Unavailable("; ".join(diagnostics))
        live_view = self._ensure_live_view()
        router = self.dvz.dvz_view_input(live_view)
        if _is_null_handle(router):
            raise DatovizV04Unavailable("Datoviz live input router is unavailable")
        previous = self.live_view3d_navigations.get(target_view3d.panel_id)
        if previous is not None:
            previous.close()
        effective_layout_snapshot_id = (
            self.consumed_layout_snapshot.snapshot_id
            if self.consumed_layout_snapshot is not None
            else layout_snapshot_id or "layout:datoviz-live-3d"
        )
        self.live_view3d_navigation = _DatovizLiveView3DNavigation(
            renderer=self,
            router=router,
            live_view=live_view,
            view3d=target_view3d,
            controller_id=controller_id,
            layout_snapshot_id=effective_layout_snapshot_id,
        )
        self.live_view3d_navigation.subscription_id = self.dvz.dvz_input_subscribe_event(
            router, self.live_view3d_navigation.handle_input_event, None
        )
        self.live_view3d_navigations[target_view3d.panel_id] = self.live_view3d_navigation
        return self.live_view3d_navigation

    def _create_rgba8_sampled_field(
        self,
        pixels: npt.NDArray[np.uint8],
        width: int,
        height: int,
        *,
        color_role: int = DVZ_COLOR_ROLE_SRGB_COLOR,
    ) -> Any:
        """Create and upload a scene-owned RGBA8 sampled field."""
        return _visual_lowering._create_rgba8_sampled_field(
            self, pixels, width, height, color_role=color_role
        )

    def capture_png_bytes(self) -> bytes:
        """Render one offscreen frame and return PNG screenshot/export bytes."""
        return _capture_execution.capture_png_bytes(self)

    def _ensure_offscreen_view(self) -> Any:
        """Create the lazy offscreen app/view pair used by PNG capture."""
        return _capture_execution._ensure_offscreen_view(self)

    def _ensure_live_view(self) -> Any:
        """Create the lazy interactive app/view pair used by show()."""
        return _capture_execution._ensure_live_view(self)

    def _ensure_app(self, purpose: str) -> Any:
        """Create the lazy Datoviz app shared by live and offscreen views."""
        return _capture_execution._ensure_app(self, purpose)

    def _render_offscreen_frame(self) -> None:
        return _capture_execution._render_offscreen_frame(self)

    def query_panel(
        self, request: QueryRequest, *, native_target: int = DVZ_SCENE_TARGET_ITEM
    ) -> QueryResult:
        """Queue and poll one Datoviz panel query for data-scope panel coordinates."""
        return _query_execution.query_panel(self, request, native_target=native_target)

    def _semantic_visual_id(self, native_visual_id: int) -> str | None:
        """Resolve one native Datoviz ID back to the GSP scene visual identity."""
        return _query_execution._semantic_visual_id(self, native_visual_id)

    def _native_query_coordinate(
        self, coordinate: tuple[float, float]
    ) -> tuple[float, float] | None:
        return _query_execution._native_query_coordinate(self, coordinate)

    def _stale_consumed_layout_result(self, request: QueryRequest) -> QueryResult | None:
        return _query_execution._stale_consumed_layout_result(self, request)

    def _query_panel_guides(self, request: QueryRequest) -> QueryResult:
        return _query_execution._query_panel_guides(self, request)

    def query_view3d_ray_context(
        self, request: QueryRequest, *, layout_snapshot_id: str
    ) -> QueryResult:
        """Return a canonical View3D ray-context payload for the current Datoviz panel."""
        return _query_execution.query_view3d_ray_context(
            self, request, layout_snapshot_id=layout_snapshot_id
        )

    def _query_plot_bounds(self) -> tuple[float, float, float, float]:
        return _query_execution._query_plot_bounds(self)

    def query_view3d_mesh_triangle_pick(
        self,
        request: View3DMeshTrianglePickRequest,
        *,
        layout_snapshot_id: str,
    ) -> QueryResult:
        """Use Datoviz public FACE queries for the bounded single-mesh S044 subset."""
        return _query_execution.query_view3d_mesh_triangle_pick(
            self, request, layout_snapshot_id=layout_snapshot_id
        )

    def _decorate_scalar_query_result(
        self, result: QueryResult, request: QueryRequest
    ) -> QueryResult:
        """Attach exact S026 scalar payloads from retained protocol scene data."""
        return _query_execution._decorate_scalar_query_result(self, result, request)

    def _scalar_metadata_for_query_result(self, result: QueryResult) -> _ScalarVisualData | None:
        return _query_execution._scalar_metadata_for_query_result(self, result)

    def configure_view2d_axes(
        self,
        view: View2D,
        *,
        x_label: str | None = None,
        y_label: str | None = None,
        grid: bool = False,
        backend_auto_ticks: bool = True,
        x_tick_values: tuple[float, ...] = (),
        x_tick_labels: tuple[str, ...] | None = None,
        y_tick_values: tuple[float, ...] = (),
        y_tick_labels: tuple[str, ...] | None = None,
    ) -> None:
        """Configure Datoviz v0.4-dev native panel domains and panel-owned axes.

        This is a capability-gated proof. It uses only the local v0.4-dev C ABI names
        verified in ``include/datoviz/scene.h`` and exposed by the supplied Python facade.
        """
        provider = datoviz_v04_axis_provider_capability(self.dvz)
        if provider.provider_status == "unsupported":
            diagnostic = (
                provider.diagnostics[0]
                if provider.diagnostics
                else "Datoviz native axis provider is unavailable"
            )
            raise DatovizV04Unavailable(diagnostic)
        has_explicit_ticks = bool(x_tick_values or y_tick_values)
        if not backend_auto_ticks and not has_explicit_ticks:
            raise DatovizV04Unsupported(
                "Datoviz native axis provider cannot realize explicit GSP ticks in this slice"
            )
        dim_x = getattr(self.dvz, "DVZ_DIM_X", 0)
        dim_y = getattr(self.dvz, "DVZ_DIM_Y", 1)
        self.view = view
        self._cpu_map_data_visuals_to_view = False
        self.apply_datoviz_data_view2d(view)

        x_axis = self.dvz.dvz_panel_axis(self.panel, dim_x)
        y_axis = self.dvz.dvz_panel_axis(self.panel, dim_y)

        _configure_axis_review_style(self.dvz, x_axis)
        _configure_axis_review_style(self.dvz, y_axis)
        _configure_axis_review_plot_margins(self.dvz, x_axis)
        _configure_axis_review_plot_margins(self.dvz, y_axis)

        self.view2d_axis_state = _DatovizView2DAxisState(
            x_axis=x_axis,
            y_axis=y_axis,
            x_label=x_label,
            y_label=y_label,
            grid=grid,
            backend_auto_ticks=backend_auto_ticks,
            x_tick_values=x_tick_values,
            x_tick_labels=x_tick_labels,
            y_tick_values=y_tick_values,
            y_tick_labels=y_tick_labels,
        )
        self.update_view2d_axes(view)

    def apply_datoviz_data_view2d(self, view: View2D) -> Any:
        """Apply GSP ordered data ranges and Datoviz View2D policy."""
        has_panel_domain = hasattr(self.dvz, "dvz_panel_set_domain")
        if has_panel_domain:
            _set_datoviz_panel_domains(self.dvz, self.panel, view.x_range, view.y_range)
        panel_view = _datoviz_panel_view2d_desc(self.dvz)
        descriptor_has_data_domains = _datoviz_view2d_descriptor_has_data_domains(panel_view)
        if descriptor_has_data_domains:
            _set_datoviz_data_domain(panel_view, "data_x", view.x_range)
            _set_datoviz_data_domain(panel_view, "data_y", view.y_range)
        elif not has_panel_domain:
            raise DatovizV04Unsupported(
                "Datoviz View2D data-domain setup failed: missing ordered domain carrier"
            )
        if hasattr(panel_view, "padding"):
            panel_view.padding = 0.0
        _require_datoviz_success(
            self.dvz.dvz_panel_set_view2d(self.panel, panel_view),
            "Datoviz View2D data-domain setup failed",
        )
        carrier = (
            "DvzPanelView2D.data_x/data_y"
            if descriptor_has_data_domains
            else "dvz_panel_set_domain+DvzPanelView2D policy"
        )
        self.last_view2d_carrier_diagnostics = {
            "datoviz_view2d_carrier": carrier,
            "ordered_ranges_preserved": True,
            "reversed_x": view.x_range[0] > view.x_range[1],
            "reversed_y": view.y_range[0] > view.y_range[1],
            "legacy_panel_domain_sync": has_panel_domain,
            "datoviz_visible_domain_readback": (
                "available" if hasattr(self.dvz, "dvz_panel_visible_domain") else "missing"
            ),
            "datoviz_transform_point": (
                "available" if hasattr(self.dvz, "dvz_panel_transform_point") else "missing"
            ),
        }
        return panel_view

    def apply_retained_view2d_navigation(self, view: View2D) -> Any:
        """Apply an accepted S035 navigation View2D as a retained panel update."""
        self.view = view
        panel_view = self.apply_datoviz_data_view2d(view)
        self.update_view2d_axes(view)
        return panel_view

    def apply_retained_view3d_navigation(
        self, view3d: View3D, *, layout_snapshot_id: str = "layout:datoviz"
    ) -> dict[str, object]:
        """Apply an accepted S037 View3D state as retained camera/projection updates."""
        diagnostics = datoviz_v04_view3d_retained_data_diagnostics(self.dvz)
        if diagnostics:
            raise DatovizV04Unavailable(
                "Datoviz retained View3D DATA-space visual path is unavailable: "
                + "; ".join(diagnostics)
            )
        readback_diagnostics = datoviz_v04_view3d_state_readback_diagnostics(self.dvz)
        if readback_diagnostics:
            raise DatovizV04Unavailable(
                "Datoviz retained View3D navigation state readback is unavailable: "
                + "; ".join(readback_diagnostics)
            )
        self.native_view3d_camera = _update_datoviz_view3d_camera(self.dvz, self.panel, view3d)
        self._update_billboard_text_placements(view3d)
        self.view3d = view3d
        self.retained_view3d_update_stats.view_projection_uniform_updates += 1
        return self.resolve_retained_view3d_state_snapshot(layout_snapshot_id=layout_snapshot_id)

    def apply_gsp_view3d_navigation_action(
        self,
        action: View3DNavigationAction,
        *,
        layout_snapshot_id: str = "layout:datoviz-live-3d",
    ) -> View3DNavigationResult:
        """Replay one canonical S037 View3D navigation action into retained Datoviz state."""
        if self.consumed_layout_snapshot is not None and (
            layout_snapshot_id != self.consumed_layout_snapshot.snapshot_id
            or action.base_layout_snapshot_id
            not in (None, self.consumed_layout_snapshot.snapshot_id)
        ):
            return View3DNavigationResult(
                accepted=False,
                view_id=action.view_id,
                action_kind=action.kind,
                old_revision=action.base_view_revision,
                diagnostics=(
                    f"{NavigationDiagnosticCode.NAVIGATION_STALE_LAYOUT.value}: "
                    "navigation references a stale consumed layout snapshot",
                ),
                layout_snapshot_id=self.consumed_layout_snapshot.snapshot_id,
            )
        if self.view3d is None:
            return View3DNavigationResult(
                accepted=False,
                view_id=action.view_id,
                action_kind=action.kind,
                old_revision=action.base_view_revision,
                diagnostics=(
                    f"{View3DDiagnosticCode.VIEW3D_NAVIGATION_UNSUPPORTED.value}: "
                    "Datoviz GSP View3D navigation requires a renderer View3D",
                ),
                layout_snapshot_id=action.base_layout_snapshot_id,
            )
        result = apply_view3d_navigation_action(
            self.view3d, action, layout_snapshot_id=layout_snapshot_id
        )
        if not result.accepted or result.view is None:
            return result
        snapshot = self.apply_retained_view3d_navigation(
            result.view, layout_snapshot_id=layout_snapshot_id
        )
        state_diagnostics = _retained_view3d_state_mismatch_diagnostics(result.view, snapshot)
        if state_diagnostics:
            return View3DNavigationResult(
                accepted=False,
                view_id=result.view_id,
                action_kind=result.action_kind,
                old_revision=result.old_revision,
                diagnostics=state_diagnostics,
                layout_snapshot_id=result.layout_snapshot_id,
            )
        return result

    def resolve_retained_view3d_state_snapshot(
        self, *, layout_snapshot_id: str = "layout:datoviz"
    ) -> dict[str, object]:
        """Read back retained Datoviz View3D state and canonical GSP snapshot identity."""
        diagnostics = datoviz_v04_view3d_state_readback_diagnostics(self.dvz)
        if diagnostics:
            raise DatovizV04Unavailable(
                "Datoviz retained View3D state readback is unavailable: " + "; ".join(diagnostics)
            )
        if self.view3d is None:
            raise DatovizV04Unavailable(
                "Datoviz retained View3D state readback requires a renderer View3D"
            )
        state = self.dvz.DvzPanelView3DState()
        if not self.dvz.dvz_panel_view3d_state(self.panel, _ctypes_pointer_arg(state)):
            raise DatovizV04Unsupported("Datoviz retained View3D state copy failed")
        self.retained_view3d_update_stats.snapshot_resolves += 1
        projection_snapshot = resolve_view3d_projection_snapshot(
            self.view3d,
            layout_snapshot=self.consumed_layout_snapshot,
            layout_snapshot_id=layout_snapshot_id,
        )
        view = getattr(state, "view", None)
        projection = getattr(state, "projection", None)
        return {
            "layout_snapshot_id": layout_snapshot_id,
            "view_projection_snapshot_id": (projection_snapshot.view_projection_snapshot_id),
            "native_view_id": int(getattr(state, "view_id", 0)),
            "native_revision": int(getattr(state, "revision", 0)),
            "enabled": bool(getattr(state, "enabled", False)),
            "camera_eye": _datoviz_vec3_tuple(getattr(view, "eye", ())),
            "camera_target": _datoviz_vec3_tuple(getattr(view, "target", ())),
            "camera_up": _datoviz_vec3_tuple(getattr(view, "up", ())),
            "projection_type": int(getattr(projection, "type", 0)),
            "near_far": (
                float(getattr(projection, "near_clip", 0.0)),
                float(getattr(projection, "far_clip", 0.0)),
            ),
            "fov_y_radians": float(getattr(projection, "fov_y", 0.0)),
            "orthographic_bounds": _datoviz_orthographic_bounds_tuple(state),
        }

    def apply_adapted_view2d_navigation_cpu_remap(self, view: View2D) -> Any:
        """Apply View2D navigation for the adapted CPU-remapped DATA placement."""
        self.view = view
        panel_view = self.apply_datoviz_data_view2d(view)
        self.update_view2d_axes(view)
        self._reupload_retained_view2d_positions(view)
        return panel_view

    def update_view2d_axes(self, view: View2D) -> None:
        """Refresh native Datoviz axes from the committed canonical View2D."""
        del view
        state = self.view2d_axis_state
        if state is None:
            return
        tick_policy = self.dvz.dvz_axis_tick_policy()
        if state.backend_auto_ticks and hasattr(self.dvz, "dvz_axis_clear_ticks"):
            self.dvz.dvz_axis_clear_ticks(state.x_axis)
            self.dvz.dvz_axis_clear_ticks(state.y_axis)
        self.dvz.dvz_axis_set_tick_policy(state.x_axis, tick_policy)
        self.dvz.dvz_axis_set_tick_policy(state.y_axis, tick_policy)
        if state.x_tick_values and hasattr(self.dvz, "dvz_axis_set_ticks"):
            _set_axis_ticks(self.dvz, state.x_axis, state.x_tick_values, state.x_tick_labels)
        if state.y_tick_values and hasattr(self.dvz, "dvz_axis_set_ticks"):
            _set_axis_ticks(self.dvz, state.y_axis, state.y_tick_values, state.y_tick_labels)
        if hasattr(self.dvz, "dvz_axis_set_grid"):
            self.dvz.dvz_axis_set_grid(state.x_axis, state.grid)
            self.dvz.dvz_axis_set_grid(state.y_axis, state.grid)
        if state.x_label is not None:
            self.dvz.dvz_axis_set_label(state.x_axis, state.x_label.encode("utf-8"))
        if state.y_label is not None:
            self.dvz.dvz_axis_set_label(state.y_axis, state.y_label.encode("utf-8"))
