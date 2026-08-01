"""Resolved layout protocol models for GSP guide geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Final, Literal, TypeAlias

from .ids import validate_id


class ConformanceTier(str, Enum):
    """Tiered conformance target for guide/layout review."""

    SEMANTIC_STRICT = "semantic_strict"
    LAYOUT_STRICT = "layout_strict"
    RASTER_TOLERANT = "raster_tolerant"
    PIXEL_PARITY = "pixel_parity"


class PixelOrigin(str, Enum):
    """Pixel origin for logical screen coordinates."""

    TOP_LEFT = "top-left"
    BOTTOM_LEFT = "bottom-left"


class LayoutResolveStatus(str, Enum):
    """Outcome of resolving layout for a render target."""

    RESOLVED = "resolved"
    ADAPTED = "adapted"
    UNSUPPORTED = "unsupported"
    FAILED = "failed"


class LayoutDiagnosticStatus(str, Enum):
    """Standard statuses for layout and guide adaptation diagnostics."""

    NATIVE = "native"
    RESOLVED = "resolved"
    ADAPTED = "adapted"
    DEGRADED = "degraded"
    UNSUPPORTED = "unsupported"
    MISSING = "missing"
    BACKEND_DEFAULT_USED = "backend_default_used"
    FONT_SUBSTITUTED = "font_substituted"
    LAYOUT_SNAPSHOT_NOT_USED = "layout_snapshot_not_used"
    QUERY_SEMANTICS_MISSING = "query_semantics_missing"
    GRID_CLIP_NOT_ENFORCED = "grid_clip_not_enforced"


class LogicalCoordinateRegion(str, Enum):
    """Resolved panel region containing one absolute logical coordinate."""

    OUTSIDE_PANEL = "outside-panel"
    PANEL_GUIDE_LANE = "panel-guide-lane"
    DATA_PLOT = "data-plot"


EXPLICIT_PANEL_LAYOUT_V1_KIND: Final[Literal["layout.panel.explicit_rects.v1"]] = (
    "layout.panel.explicit_rects.v1"
)


@dataclass(frozen=True, slots=True)
class NormalizedRenderTargetRect:
    """Outer-panel allocation intent in normalized top-left target coordinates."""

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        for index, value in enumerate(values):
            if not isinstance(value, (int, float)):
                raise TypeError(f"normalized rectangle value {index} must be a number")
            _validate_finite(f"normalized rectangle value {index}", value)
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("normalized rectangle width and height must be positive")
        if self.left < 0.0 or self.top < 0.0:
            raise ValueError("normalized rectangle origin must be non-negative")
        if self.left + self.width > 1.0 or self.top + self.height > 1.0:
            raise ValueError("normalized rectangle must be contained by the render target")


@dataclass(frozen=True, slots=True)
class PanelPlacement:
    """Allocate one scene panel inside the logical render target."""

    panel_id: str
    allocation_rect: NormalizedRenderTargetRect

    def __post_init__(self) -> None:
        validate_id(self.panel_id)
        if not isinstance(self.allocation_rect, NormalizedRenderTargetRect):
            raise TypeError("allocation_rect must be a NormalizedRenderTargetRect")


@dataclass(frozen=True, slots=True)
class ExplicitPanelLayoutV1:
    """Closed v1 explicit, non-overlapping panel-allocation intent."""

    placements: tuple[PanelPlacement, ...]
    kind: Literal["layout.panel.explicit_rects.v1"] = EXPLICIT_PANEL_LAYOUT_V1_KIND

    def __post_init__(self) -> None:
        if self.kind != EXPLICIT_PANEL_LAYOUT_V1_KIND:
            raise ValueError(f"kind must be {EXPLICIT_PANEL_LAYOUT_V1_KIND!r}")
        panel_ids = [placement.panel_id for placement in self.placements]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("panel layout contains duplicate panel_id placements")
        for index, left in enumerate(self.placements):
            for right in self.placements[index + 1 :]:
                if _normalized_rects_overlap(left.allocation_rect, right.allocation_rect):
                    raise ValueError(
                        f"panel allocations overlap: {left.panel_id!r} and {right.panel_id!r}"
                    )


PanelLayoutIntent: TypeAlias = ExplicitPanelLayoutV1


def full_target_panel_layout(panel_id: str) -> ExplicitPanelLayoutV1:
    """Build explicit full-target intent for a single-panel producer convenience."""
    return ExplicitPanelLayoutV1(
        placements=(
            PanelPlacement(
                panel_id=panel_id,
                allocation_rect=NormalizedRenderTargetRect(0.0, 0.0, 1.0, 1.0),
            ),
        )
    )


GuideBoxKind = Literal[
    "axis",
    "axis_label",
    "tick_label",
    "title",
    "legend",
    "colorbar",
    "grid",
    "panel_text",
]


@dataclass(frozen=True, slots=True)
class LogicalPixelRect:
    """Rectangle in logical pixels."""

    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        _validate_finite("x", self.x)
        _validate_finite("y", self.y)
        _validate_finite("width", self.width)
        _validate_finite("height", self.height)
        if self.width < 0.0 or self.height < 0.0:
            raise ValueError("logical pixel rectangle width/height must be non-negative")


@dataclass(frozen=True, slots=True)
class LayoutAnchor:
    """Anchor point in logical pixels."""

    x: float
    y: float

    def __post_init__(self) -> None:
        _validate_finite("x", self.x)
        _validate_finite("y", self.y)


@dataclass(frozen=True, slots=True)
class RenderTarget:
    """Logical render target used to resolve layout."""

    logical_width_px: float
    logical_height_px: float
    device_scale: float = 1.0
    dpi: float | None = None
    pixel_origin: PixelOrigin = PixelOrigin.TOP_LEFT
    query_coordinate_space: str = "plot"

    def __post_init__(self) -> None:
        _validate_positive("logical_width_px", self.logical_width_px)
        _validate_positive("logical_height_px", self.logical_height_px)
        _validate_positive("device_scale", self.device_scale)
        if self.dpi is not None:
            _validate_positive("dpi", self.dpi)
        if isinstance(self.pixel_origin, str):
            try:
                object.__setattr__(self, "pixel_origin", PixelOrigin(self.pixel_origin))
            except ValueError as exc:
                raise ValueError(
                    f"pixel_origin must be one of {[origin.value for origin in PixelOrigin]}"
                ) from exc
        elif not isinstance(self.pixel_origin, PixelOrigin):
            raise TypeError("pixel_origin must be a PixelOrigin or its string value")
        if not self.query_coordinate_space:
            raise ValueError("query_coordinate_space must not be empty")

    @property
    def framebuffer_width_px(self) -> int:
        """Physical framebuffer width implied by logical size and device scale."""
        return int(round(self.logical_width_px * self.device_scale))

    @property
    def framebuffer_height_px(self) -> int:
        """Physical framebuffer height implied by logical size and device scale."""
        return int(round(self.logical_height_px * self.device_scale))


@dataclass(frozen=True, slots=True)
class LayoutDiagnostic:
    """Diagnostic for layout support, adaptation, or fallback behavior."""

    code: str
    status: LayoutDiagnosticStatus
    message: str | None = None

    def __post_init__(self) -> None:
        if not self.code:
            raise ValueError("layout diagnostic code must not be empty")
        if self.message is not None and not self.message:
            raise ValueError("layout diagnostic message must not be empty")


@dataclass(frozen=True, slots=True)
class ResolvedGuideBox:
    """Resolved box for one guide contribution."""

    guide_id: str
    kind: GuideBoxKind
    rect_px: LogicalPixelRect
    anchor_px: LayoutAnchor | None = None
    role: str | None = None
    layer: str | None = None
    diagnostics: tuple[LayoutDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.guide_id)
        if not self.kind:
            raise ValueError("guide box kind must not be empty")
        if self.role is not None and not self.role:
            raise ValueError("guide box role must not be empty")
        if self.layer is not None and not self.layer:
            raise ValueError("guide box layer must not be empty")


@dataclass(frozen=True, slots=True)
class LayoutLayer:
    """Layer assignment for a resolved scene object."""

    object_id: str
    layer: str
    z_order: float = 0.0

    def __post_init__(self) -> None:
        validate_id(self.object_id)
        if not self.layer:
            raise ValueError("layout layer name must not be empty")
        _validate_finite("z_order", self.z_order)


@dataclass(frozen=True, slots=True)
class ResolvedPanelLayout:
    """Resolved layout and guide geometry for one scene panel."""

    panel_id: str
    panel_rect_px: LogicalPixelRect
    plot_rect_px: LogicalPixelRect
    view_id: str | None = None
    data_to_screen_transform: tuple[float, ...] = (
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    guide_boxes: tuple[ResolvedGuideBox, ...] = ()
    guide_anchors: tuple[ResolvedGuideBox, ...] = ()
    tick_label_boxes: tuple[ResolvedGuideBox, ...] = ()
    axis_label_boxes: tuple[ResolvedGuideBox, ...] = ()
    title_boxes: tuple[ResolvedGuideBox, ...] = ()
    legend_boxes: tuple[ResolvedGuideBox, ...] = ()
    colorbar_boxes: tuple[ResolvedGuideBox, ...] = ()
    grid_clip_rect_px: LogicalPixelRect | None = None
    z_layers: tuple[LayoutLayer, ...] = ()
    diagnostics: tuple[LayoutDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.panel_id)
        if self.view_id is not None:
            validate_id(self.view_id)
        if self.panel_rect_px.width <= 0.0 or self.panel_rect_px.height <= 0.0:
            raise ValueError("panel_rect_px must have positive width and height")
        if self.plot_rect_px.width <= 0.0 or self.plot_rect_px.height <= 0.0:
            raise ValueError("plot_rect_px must have positive width and height")
        if not _rect_contains_rect(self.panel_rect_px, self.plot_rect_px):
            raise ValueError("plot_rect_px must be contained by panel_rect_px")
        if len(self.data_to_screen_transform) not in (6, 9):
            raise ValueError("data_to_screen_transform must contain 6 or 9 finite values")
        for value in self.data_to_screen_transform:
            _validate_finite("data_to_screen_transform", value)


@dataclass(frozen=True, slots=True)
class ResolvedLayoutSnapshot:
    """Derived per-panel layout state for one scene and logical render target."""

    snapshot_id: str
    render_target: RenderTarget
    panels: tuple[ResolvedPanelLayout, ...]

    def __post_init__(self) -> None:
        validate_id(self.snapshot_id)
        panel_ids = [panel.panel_id for panel in self.panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("resolved layout contains duplicate panel_id entries")
        for panel in self.panels:
            if not isinstance(panel, ResolvedPanelLayout):
                raise TypeError("panels must contain ResolvedPanelLayout values")
            _validate_rect_inside_render_target(
                "panel_rect_px", panel.panel_rect_px, self.render_target
            )
            _validate_rect_inside_render_target(
                "plot_rect_px", panel.plot_rect_px, self.render_target
            )

    def panel(self, panel_id: str) -> ResolvedPanelLayout:
        """Return one resolved panel by exact scene-scoped identifier."""
        validate_id(panel_id)
        matches = [panel for panel in self.panels if panel.panel_id == panel_id]
        if not matches:
            raise ValueError(f"resolved layout has no panel {panel_id!r}")
        return matches[0]

    def only_panel(self) -> ResolvedPanelLayout:
        """Return the sole panel or reject an ambiguous multi-panel shortcut."""
        if len(self.panels) != 1:
            raise ValueError("operation requires exactly one resolved panel")
        return self.panels[0]


def quantize_normalized_edge(coordinate: float, target_extent_px: int) -> int:
    """Resolve one normalized edge with the protocol-defined P038 quantizer."""
    _validate_finite("coordinate", coordinate)
    if coordinate < 0.0 or coordinate > 1.0:
        raise ValueError("coordinate must be in the closed interval [0, 1]")
    if not isinstance(target_extent_px, int):
        raise TypeError("target_extent_px must be an int")
    if target_extent_px <= 0:
        raise ValueError("target_extent_px must be positive")
    return min(
        target_extent_px,
        max(0, math.floor(coordinate * target_extent_px + 0.5)),
    )


def quantize_logical_rect(rect: LogicalPixelRect, render_target: RenderTarget) -> LogicalPixelRect:
    """Quantize logical rectangle edges to canonical integer pixel boundaries."""
    width = _integral_target_extent("logical_width_px", render_target.logical_width_px)
    height = _integral_target_extent("logical_height_px", render_target.logical_height_px)
    left = min(width, max(0, math.floor(rect.x + 0.5)))
    right = min(width, max(0, math.floor(rect.x + rect.width + 0.5)))
    top = min(height, max(0, math.floor(rect.y + 0.5)))
    bottom = min(height, max(0, math.floor(rect.y + rect.height + 0.5)))
    return LogicalPixelRect(left, top, right - left, bottom - top)


def resolve_panel_layout_intent(
    layout: PanelLayoutIntent, render_target: RenderTarget
) -> tuple[ResolvedPanelLayout, ...]:
    """Resolve all explicit outer-panel allocations using the canonical quantizer."""
    if not isinstance(layout, ExplicitPanelLayoutV1):
        raise TypeError("layout must be an ExplicitPanelLayoutV1")
    if not isinstance(render_target, RenderTarget):
        raise TypeError("render_target must be a RenderTarget")
    width = _integral_target_extent("logical_width_px", render_target.logical_width_px)
    height = _integral_target_extent("logical_height_px", render_target.logical_height_px)
    resolved: list[ResolvedPanelLayout] = []
    for placement in layout.placements:
        rect = placement.allocation_rect
        left = quantize_normalized_edge(rect.left, width)
        right = quantize_normalized_edge(rect.left + rect.width, width)
        top = quantize_normalized_edge(rect.top, height)
        bottom = quantize_normalized_edge(rect.top + rect.height, height)
        panel_rect = LogicalPixelRect(left, top, right - left, bottom - top)
        if panel_rect.width <= 0 or panel_rect.height <= 0:
            raise ValueError(f"panel {placement.panel_id!r} resolves to zero logical-pixel area")
        resolved.append(
            ResolvedPanelLayout(
                panel_id=placement.panel_id,
                panel_rect_px=panel_rect,
                plot_rect_px=panel_rect,
            )
        )
    return tuple(resolved)


@dataclass(frozen=True, slots=True)
class LayoutResolveRequest:
    """Request to resolve layout for a scene and render target."""

    request_id: str
    scene_id: str
    render_target: RenderTarget
    requested_tier: ConformanceTier = ConformanceTier.LAYOUT_STRICT

    def __post_init__(self) -> None:
        validate_id(self.request_id)
        validate_id(self.scene_id)


@dataclass(frozen=True, slots=True)
class LayoutResolveResult:
    """Result of a layout resolve/get operation."""

    request_id: str
    status: LayoutResolveStatus
    snapshot: ResolvedLayoutSnapshot | None = None
    diagnostics: tuple[LayoutDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        validate_id(self.request_id)
        if self.status == LayoutResolveStatus.RESOLVED and self.snapshot is None:
            raise ValueError("resolved layout results require a snapshot")
        if self.status != LayoutResolveStatus.RESOLVED and not self.diagnostics:
            raise ValueError("non-resolved layout results require diagnostics")


def logical_px_to_points(logical_px: float, dpi: float) -> float:
    """Convert logical pixels to typographic points for a DPI-bound backend."""
    _validate_finite("logical_px", logical_px)
    _validate_positive("dpi", dpi)
    return logical_px * 72.0 / dpi


def plot_logical_px_to_plot_ndc(
    snapshot: ResolvedLayoutSnapshot, coordinate_px: tuple[float, float]
) -> tuple[float, float]:
    """Map a render-target logical coordinate through the resolved data viewport."""
    _validate_layout_snapshot(snapshot)
    x_px, y_px = _validate_logical_coordinate(coordinate_px)
    plot = snapshot.only_panel().plot_rect_px
    _validate_nonempty_plot_rect(plot)
    if not _rect_contains_coordinate(plot, x_px, y_px):
        raise ValueError("coordinate_px must be inside the closed plot_rect_px")
    x_ndc = -1.0 + 2.0 * (x_px - plot.x) / plot.width
    y_fraction = (y_px - plot.y) / plot.height
    y_ndc = (
        1.0 - 2.0 * y_fraction
        if snapshot.render_target.pixel_origin is PixelOrigin.TOP_LEFT
        else -1.0 + 2.0 * y_fraction
    )
    return (x_ndc, y_ndc)


def plot_ndc_to_plot_logical_px(
    snapshot: ResolvedLayoutSnapshot, plot_ndc: tuple[float, float]
) -> tuple[float, float]:
    """Map plot NDC through the resolved data viewport into logical coordinates."""
    _validate_layout_snapshot(snapshot)
    x_ndc, y_ndc = _validate_logical_coordinate(plot_ndc)
    plot = snapshot.only_panel().plot_rect_px
    _validate_nonempty_plot_rect(plot)
    x_px = plot.x + (x_ndc + 1.0) * 0.5 * plot.width
    y_fraction = (
        (1.0 - y_ndc) * 0.5
        if snapshot.render_target.pixel_origin is PixelOrigin.TOP_LEFT
        else (y_ndc + 1.0) * 0.5
    )
    return (x_px, plot.y + y_fraction * plot.height)


def resolved_plot_aspect_ratio(snapshot: ResolvedLayoutSnapshot) -> float:
    """Return the positive width/height aspect of the resolved data viewport."""
    _validate_layout_snapshot(snapshot)
    plot = snapshot.only_panel().plot_rect_px
    _validate_nonempty_plot_rect(plot)
    return plot.width / plot.height


def logical_coordinate_in_data_viewport(
    snapshot: ResolvedLayoutSnapshot, coordinate_px: tuple[float, float]
) -> bool:
    """Return whether a logical coordinate lies in the closed plot rectangle."""
    return classify_logical_coordinate(snapshot, coordinate_px) is LogicalCoordinateRegion.DATA_PLOT


def classify_logical_coordinate(
    snapshot: ResolvedLayoutSnapshot, coordinate_px: tuple[float, float]
) -> LogicalCoordinateRegion:
    """Classify an absolute render-target logical coordinate for query routing."""
    _validate_layout_snapshot(snapshot)
    x_px, y_px = _validate_logical_coordinate(coordinate_px)
    panel = snapshot.only_panel()
    if not _rect_contains_coordinate(panel.panel_rect_px, x_px, y_px):
        return LogicalCoordinateRegion.OUTSIDE_PANEL
    plot = panel.plot_rect_px
    if plot.width > 0.0 and plot.height > 0.0 and _rect_contains_coordinate(plot, x_px, y_px):
        return LogicalCoordinateRegion.DATA_PLOT
    return LogicalCoordinateRegion.PANEL_GUIDE_LANE


def _validate_layout_snapshot(snapshot: ResolvedLayoutSnapshot) -> None:
    if not isinstance(snapshot, ResolvedLayoutSnapshot):
        raise TypeError("snapshot must be a ResolvedLayoutSnapshot")


def _validate_logical_coordinate(
    coordinate: tuple[float, float],
) -> tuple[float, float]:
    if len(coordinate) != 2:
        raise ValueError("coordinate must contain two values")
    x, y = coordinate
    _validate_finite("coordinate[0]", x)
    _validate_finite("coordinate[1]", y)
    return (x, y)


def _validate_nonempty_plot_rect(rect: LogicalPixelRect) -> None:
    if rect.width <= 0.0 or rect.height <= 0.0:
        raise ValueError("plot_rect_px must have positive width and height")


def _validate_rect_inside_render_target(
    field_name: str, rect: LogicalPixelRect, render_target: RenderTarget
) -> None:
    if rect.x < 0.0 or rect.y < 0.0:
        raise ValueError(f"{field_name} origin must be non-negative")
    if (
        rect.x + rect.width > render_target.logical_width_px
        or rect.y + rect.height > render_target.logical_height_px
    ):
        raise ValueError(f"{field_name} must be inside the render target")


def _rect_contains_rect(outer: LogicalPixelRect, inner: LogicalPixelRect) -> bool:
    return (
        inner.x >= outer.x
        and inner.y >= outer.y
        and inner.x + inner.width <= outer.x + outer.width
        and inner.y + inner.height <= outer.y + outer.height
    )


def _rect_contains_coordinate(rect: LogicalPixelRect, x: float, y: float) -> bool:
    return rect.x <= x <= rect.x + rect.width and rect.y <= y <= rect.y + rect.height


def _validate_finite(field_name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite")


def _validate_positive(field_name: str, value: float) -> None:
    _validate_finite(field_name, value)
    if value <= 0.0:
        raise ValueError(f"{field_name} must be positive")


def _normalized_rects_overlap(
    left: NormalizedRenderTargetRect, right: NormalizedRenderTargetRect
) -> bool:
    return (
        left.left < right.left + right.width
        and right.left < left.left + left.width
        and left.top < right.top + right.height
        and right.top < left.top + left.height
    )


def _integral_target_extent(field_name: str, value: float) -> int:
    if not float(value).is_integer():
        raise ValueError(f"{field_name} must be an integral logical-pixel extent")
    return int(value)
