"""Retained Datoviz state records and native constants."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    ColorMapId,
    ColorScale,
    MarkerShape,
    PanByAction,
    PrimitiveTopology,
    ResetViewAction,
    ResolvedLayoutSnapshot,
    ScalarColorSlot,
    SetViewAction,
    StrokeCap,
    StrokeJoin,
    TextVisual,
    VectorCap,
    View2D,
    View3D,
    VisualFamily,
    VisualTransformBinding,
    ZoomAboutAction,
)
from gsp.protocol.visuals import CoordinateSpace

from gsp_datoviz.latest_api_contract import (
    REQUIRED_DATOVIZ_V04_DEV_SYMBOLS,
)

_REQUIRED_DVZ_V04_FUNCTIONS = REQUIRED_DATOVIZ_V04_DEV_SYMBOLS


_REQUIRED_DVZ_SAMPLED_FIELD_FUNCTIONS = (
    "dvz_sampled_field_desc",
    "dvz_field_data_view",
    "dvz_sampled_field",
    "dvz_sampled_field_set_data",
    "dvz_visual_set_field",
)


_REQUIRED_DVZ_MARKER_FUNCTIONS = (
    "dvz_marker",
    "dvz_marker_style",
    "dvz_marker_set_style",
)


_REQUIRED_DVZ_SEGMENT_FUNCTIONS = (
    "dvz_segment",
    "dvz_segment_set_caps",
)


_REQUIRED_DVZ_PATH_FUNCTIONS = (
    "dvz_path",
    "dvz_path_set_subpaths",
    "dvz_path_set_caps",
    "dvz_path_set_join",
)


_REQUIRED_DVZ_MESH_FUNCTIONS = (
    "dvz_mesh",
    "dvz_visual_set_data",
    "dvz_visual_set_index_data",
    "dvz_visual_set_depth_test",
)


_PRIMITIVE_TOPOLOGY_NAMES = {
    PrimitiveTopology.POINT_LIST: "DVZ_PRIMITIVE_TOPOLOGY_POINT_LIST",
    PrimitiveTopology.LINE_LIST: "DVZ_PRIMITIVE_TOPOLOGY_LINE_LIST",
    PrimitiveTopology.LINE_STRIP: "DVZ_PRIMITIVE_TOPOLOGY_LINE_STRIP",
    PrimitiveTopology.TRIANGLE_LIST: "DVZ_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST",
    PrimitiveTopology.TRIANGLE_STRIP: "DVZ_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP",
}


_REQUIRED_DVZ_TEXTURE2D_MESH_FUNCTIONS = (
    "dvz_field_sampling_desc",
    "dvz_visual_set_field_sampling",
    "dvz_material_desc",
    "dvz_visual_set_material",
    *_REQUIRED_DVZ_SAMPLED_FIELD_FUNCTIONS,
)


_OPTIONAL_UNVERIFIED_DVZ_MESH_FUNCTIONS: tuple[str, ...] = ()


_REQUIRED_DVZ_VIEW3D_CAMERA_FUNCTIONS = (
    "DvzCameraDesc",
    "DvzCameraView",
    "DvzCameraProjection",
    "dvz_camera_desc",
    "dvz_camera_set_orthographic_bounds",
    "dvz_panel_view3d_desc",
    "dvz_panel_set_view3d_desc",
    "dvz_panel_camera",
)


_REQUIRED_DVZ_VIEW3D_RETAINED_DATA_FUNCTIONS = (
    "DvzPanelView3DDesc",
    "dvz_panel_view3d_desc",
    "dvz_panel_set_view3d_desc",
    "dvz_panel_camera",
    "dvz_camera_set_view",
)


_REQUIRED_DVZ_PANEL_FRAME_SNAPSHOT_FUNCTIONS = (
    "DvzPanelFrameInfo",
    "DvzGuideLayout",
    "DvzRenderedContribution",
    "dvz_panel_resolve_frame",
    "dvz_panel_frame_info",
    "dvz_panel_frame_guide_count",
    "dvz_panel_frame_guide_layout",
    "dvz_panel_frame_contribution_count",
    "dvz_panel_frame_contribution",
)


_REQUIRED_DVZ_PANEL_FRAME_GUIDE_QUERY_FUNCTIONS = (
    "DvzGuideHit",
    "dvz_panel_frame_guide_hit",
)


_OPTIONAL_UNVERIFIED_DVZ_TEXT_FUNCTIONS: tuple[str, ...] = ()


DVZ_POINTER_EVENT_RELEASE = 0


DVZ_POINTER_EVENT_PRESS = 1


DVZ_POINTER_EVENT_MOVE = 2


DVZ_POINTER_EVENT_DOUBLE_CLICK = 5


DVZ_POINTER_EVENT_DRAG = 11


DVZ_POINTER_EVENT_DRAG_STOP = 12


DVZ_POINTER_EVENT_WHEEL = 20


DVZ_POINTER_BUTTON_LEFT = 1


DVZ_POINTER_BUTTON_RIGHT = 3


DVZ_INPUT_EVENT_POINTER = 1


DVZ_INPUT_EVENT_RESIZE = 3


DVZ_INPUT_EVENT_SCALE = 4


DVZ_FIELD_DIM_2D = 0


DVZ_FIELD_FORMAT_RGBA8_UNORM = 22


DVZ_FIELD_SEMANTIC_COLOR = 4


DVZ_COLOR_ROLE_SRGB_COLOR = 1


DVZ_COLOR_ROLE_LINEAR_COLOR = 2


DVZ_FIELD_FILTER_LINEAR = 0


DVZ_FIELD_FILTER_NEAREST = 1


DVZ_MATERIAL_MODEL_UNLIT = 0


DVZ_SCENE_TARGET_NONE = 0


DVZ_SCENE_TARGET_ITEM = 2


DVZ_SCENE_TARGET_FACE = 4


DVZ_QUERY_HIT_FRONTMOST = 0


DVZ_QUERY_PROFILE_UNSUPPORTED = 0


DVZ_QUERY_CAPABILITY_ITEM = 0x02


DVZ_QUERY_CAPABILITY_FACE = 0x08


DVZ_QUERY_CAPABILITY_PIXEL = 0x10


DVZ_CAMERA_PERSPECTIVE = 0


DVZ_CAMERA_ORTHOGRAPHIC = 1


DVZ_CONTROLLER_APPLY = 0


DVZ_CONTROLLER_FIXED = 1


DVZ_SHAPE_ASPECT_FILLED = 0


DVZ_SHAPE_ASPECT_OUTLINE = 2


DVZ_ALPHA_BLENDED = 1


DVZ_IMAGE_SAMPLING_LINEAR = 0


DVZ_IMAGE_SAMPLING_NEAREST = 1


DVZ_COLOR_PIPELINE_LINEAR_SRGB = 0


DVZ_COLOR_PIPELINE_LEGACY_SRGB_BLEND = 1


DVZ_SCALE_CONTINUOUS = 0


DVZ_BUILTIN_COLORMAP_VIRIDIS = 1


DVZ_BUILTIN_COLORMAP_MAGMA = 2


DVZ_BUILTIN_COLORMAP_PLASMA = 3


DVZ_BUILTIN_COLORMAP_INFERNO = 4


DVZ_BUILTIN_COLORMAP_CIVIDIS = 5


DVZ_BUILTIN_COLORMAP_GRAY = 7


DVZ_SEGMENT_CAP_ROUND = 1


DVZ_SEGMENT_CAP_SQUARE = 4


DVZ_SEGMENT_CAP_BUTT = 5


DVZ_PATH_JOIN_MITER = 0


DVZ_PATH_JOIN_ROUND = 1


DVZ_PATH_JOIN_BEVEL = 2


DVZ_TEXT_PLACEMENT_SCREEN = 0


DVZ_TEXT_PLACEMENT_DATA = 1


DVZ_COLORBAR_ORIENTATION_VERTICAL = 0


DVZ_COLORBAR_ORIENTATION_HORIZONTAL = 1


DVZ_COLORBAR_PLACEMENT_ATTACHED = 0


DVZ_COLORBAR_PLACEMENT_DETACHED = 1


DVZ_PLACEMENT_SPACE_PANEL = 0


DVZ_HORIZONTAL_ANCHOR_RIGHT = 2


DVZ_VERTICAL_ANCHOR_CENTER = 1


DVZ_SCENE_ANCHOR_PANEL_TOP = 2


DVZ_SCENE_ANCHOR_PANEL_LEFT = 4


DVZ_SCENE_ANCHOR_PANEL_CENTER = 5


DVZ_SCENE_ANCHOR_PANEL_RIGHT = 6


DVZ_SCENE_ANCHOR_PANEL_BOTTOM = 8


DVZ_SCENE_ANCHOR_DATA = 10


DVZ_TEXT_RENDERER_MSDF_ATLAS = 3


DEFAULT_BACKGROUND_RGBA8 = (255, 255, 255, 255)


DATOVIZ_REVIEW_PLOT_MARGINS = (0.0, 0.0, 0.0, 0.08)


DatovizColorPipeline = Literal["linear_srgb", "legacy_srgb_blend"]


class _CompatDvzTextPlacement(ctypes.Structure):
    """Header-matched fallback for builds that omit the Python placement factory."""

    _fields_ = (
        ("struct_size", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("mode", ctypes.c_int),
        ("anchor", ctypes.c_int),
        ("position", ctypes.c_double * 3),
        ("offset", ctypes.c_float * 2),
        ("text_anchor", ctypes.c_float * 2),
        ("has_text_anchor", ctypes.c_bool),
        ("angle", ctypes.c_float),
        ("depth_test", ctypes.c_bool),
    )


_MARKER_SHAPE_FALLBACKS = {
    MarkerShape.DISC: 0,
    MarkerShape.SQUARE: 1,
    MarkerShape.TRIANGLE: 2,
    MarkerShape.DIAMOND: 3,
    MarkerShape.CROSS: 4,
}


_MARKER_SHAPE_NAMES = {
    MarkerShape.DISC: "DVZ_MARKER_SHAPE_DISC",
    MarkerShape.SQUARE: "DVZ_MARKER_SHAPE_SQUARE",
    MarkerShape.TRIANGLE: "DVZ_MARKER_SHAPE_TRIANGLE",
    MarkerShape.DIAMOND: "DVZ_MARKER_SHAPE_DIAMOND",
    MarkerShape.CROSS: "DVZ_MARKER_SHAPE_CROSS",
}


_STROKE_CAP_FALLBACKS = {
    StrokeCap.BUTT: DVZ_SEGMENT_CAP_BUTT,
    StrokeCap.ROUND: DVZ_SEGMENT_CAP_ROUND,
    StrokeCap.SQUARE: DVZ_SEGMENT_CAP_SQUARE,
}


_STROKE_CAP_NAMES = {
    StrokeCap.BUTT: "DVZ_SEGMENT_CAP_BUTT",
    StrokeCap.ROUND: "DVZ_SEGMENT_CAP_ROUND",
    StrokeCap.SQUARE: "DVZ_SEGMENT_CAP_SQUARE",
}


_VECTOR_CAP_NAMES = {
    VectorCap.NONE: "DVZ_SEGMENT_CAP_NONE",
    VectorCap.BUTT: "DVZ_SEGMENT_CAP_BUTT",
    VectorCap.ROUND: "DVZ_SEGMENT_CAP_ROUND",
    VectorCap.TRIANGLE_IN: "DVZ_SEGMENT_CAP_TRIANGLE_IN",
    VectorCap.TRIANGLE_OUT: "DVZ_SEGMENT_CAP_TRIANGLE_OUT",
    VectorCap.SQUARE: "DVZ_SEGMENT_CAP_SQUARE",
}


_STROKE_JOIN_FALLBACKS = {
    StrokeJoin.MITER: DVZ_PATH_JOIN_MITER,
    StrokeJoin.ROUND: DVZ_PATH_JOIN_ROUND,
    StrokeJoin.BEVEL: DVZ_PATH_JOIN_BEVEL,
}


_STROKE_JOIN_NAMES = {
    StrokeJoin.MITER: "DVZ_PATH_JOIN_MITER",
    StrokeJoin.ROUND: "DVZ_PATH_JOIN_ROUND",
    StrokeJoin.BEVEL: "DVZ_PATH_JOIN_BEVEL",
}


_VISUAL_ATTRIBUTE_ALIASES = {
    "diameter_px": ("diameter",),
    "stroke_width_px": ("stroke_width",),
}


_BUILTIN_COLORMAP_NAMES = {
    ColorMapId.GRAY: ("DVZ_BUILTIN_COLORMAP_GRAY", DVZ_BUILTIN_COLORMAP_GRAY),
    ColorMapId.VIRIDIS: (
        "DVZ_BUILTIN_COLORMAP_VIRIDIS",
        DVZ_BUILTIN_COLORMAP_VIRIDIS,
    ),
    ColorMapId.MAGMA: ("DVZ_BUILTIN_COLORMAP_MAGMA", DVZ_BUILTIN_COLORMAP_MAGMA),
    ColorMapId.PLASMA: ("DVZ_BUILTIN_COLORMAP_PLASMA", DVZ_BUILTIN_COLORMAP_PLASMA),
    ColorMapId.INFERNO: (
        "DVZ_BUILTIN_COLORMAP_INFERNO",
        DVZ_BUILTIN_COLORMAP_INFERNO,
    ),
    ColorMapId.CIVIDIS: (
        "DVZ_BUILTIN_COLORMAP_CIVIDIS",
        DVZ_BUILTIN_COLORMAP_CIVIDIS,
    ),
}


class DatovizV04Unavailable(RuntimeError):
    """Raised when the imported Datoviz facade is not the expected v0.4 shape."""


class DatovizV04Unsupported(ValueError):
    """Raised when a GSP v0.1 visual asks for semantics this slice does not support."""


@dataclass
class _ScalarVisualData:
    """Scene-owned scalar color data retained for semantic query payloads."""

    visual_id: str
    visual_family: VisualFamily | str
    item_kind: str
    color_slot: ScalarColorSlot
    values: npt.NDArray[np.float64]
    color_scale: ColorScale
    alpha: float = 1.0


@dataclass
class _RetainedView2DPositionUpload:
    """Source positions for CPU-mapped DATA visuals that must follow View2D changes."""

    visual_id: str
    native_visual: Any
    attr_name: str
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64]
    transform: VisualTransformBinding | None
    coordinate_space: CoordinateSpace


@dataclass
class DatovizRetainedView3DUpdateStats:
    """Counters proving retained View3D updates avoid unchanged visual buffer writes."""

    vertex_uploads: int = 0
    index_uploads: int = 0
    visual_rebuilds: int = 0
    view_projection_uniform_updates: int = 0
    snapshot_resolves: int = 0


@dataclass
class _RetainedView3DMeshAttachment:
    """Native DATA-space 3D mesh attached to a retained Datoviz View3D."""

    visual_id: str
    native_visual: Any


@dataclass
class _RetainedView3DTextAttachment:
    """Source billboard anchors and retained Datoviz text handles."""

    visual: TextVisual
    native_texts: tuple[Any, ...]


@dataclass(frozen=True)
class _DatovizView2DAxisState:
    """Native Datoviz axis handles and semantic configuration for one View2D."""

    x_axis: Any
    y_axis: Any
    x_label: str | None
    y_label: str | None
    grid: bool
    backend_auto_ticks: bool
    x_tick_values: tuple[float, ...]
    x_tick_labels: tuple[str, ...] | None
    y_tick_values: tuple[float, ...]
    y_tick_labels: tuple[str, ...] | None


@dataclass
class _DatovizRetainedPanelState:
    """All panel-local state retained inside one Datoviz scene/figure."""

    panel: Any
    view: View2D | None
    view3d: View3D | None
    panel_bounds: tuple[float, float, float, float] | None
    consumed_layout_snapshot: ResolvedLayoutSnapshot | None
    retained_view2d_position_uploads: list[_RetainedView2DPositionUpload]
    retained_view3d_meshes: list[_RetainedView3DMeshAttachment]
    retained_view3d_texts: list[_RetainedView3DTextAttachment]
    retained_view3d_update_stats: DatovizRetainedView3DUpdateStats
    view2d_axis_state: _DatovizView2DAxisState | None
    native_view3d_camera: Any | None


NavigationAction = PanByAction | ZoomAboutAction | SetViewAction | ResetViewAction
