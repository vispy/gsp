"""Panel and View2D protocol models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .ids import validate_id
from .layout import LogicalPixelRect, ResolvedLayoutSnapshot
from .transforms import TransformDiagnosticCode, ViewKind, validate_view2d_limits


class AspectPolicy(str, Enum):
    """2D view aspect policy."""

    AUTO = "auto"
    EQUAL = "equal"


class VisualCoordinateSpace(str, Enum):
    """Coordinate space used by a visual attachment."""

    DATA = "data"
    VIEW = "view"
    PANEL = "panel"


class ClipScope(str, Enum):
    """Rectangular raster-clipping boundary for one visual attachment."""

    PLOT = "plot"
    PANEL = "panel"
    RENDER_TARGET = "render_target"


@dataclass(frozen=True, slots=True)
class Panel:
    """Scene-scoped semantic panel identity."""

    id: str

    def __post_init__(self) -> None:
        validate_id(self.id)


@dataclass(frozen=True, slots=True)
class View2D:
    """Semantic 2D Cartesian view attached to a panel."""

    id: str
    panel_id: str
    x_range: tuple[float, float] = (-1.0, 1.0)
    y_range: tuple[float, float] = (-1.0, 1.0)
    aspect_policy: AspectPolicy = AspectPolicy.AUTO
    kind: ViewKind = ViewKind.VIEW2D_LINEAR

    def __post_init__(self) -> None:
        validate_id(self.id)
        validate_id(self.panel_id)
        if self.kind is not ViewKind.VIEW2D_LINEAR:
            raise ValueError("only VIEW2D_LINEAR views are accepted in S027")
        if self.aspect_policy is not AspectPolicy.AUTO:
            raise ValueError(
                f"{TransformDiagnosticCode.VIEW2D_ASPECT_UNSUPPORTED.value}: "
                "equal/fixed aspect layout is deferred"
            )
        validate_view2d_limits("x_range", self.x_range)
        validate_view2d_limits("y_range", self.y_range)

    @property
    def xlim(self) -> tuple[float, float]:
        """S027 alias for the x data limits."""
        return self.x_range

    @property
    def ylim(self) -> tuple[float, float]:
        """S027 alias for the y data limits."""
        return self.y_range


@dataclass(frozen=True, slots=True)
class VisualAttachment:
    """Attach a data visual to a panel/view without making axes part of the visual stream."""

    visual_id: str
    panel_id: str
    view_id: str
    coordinate_space: VisualCoordinateSpace = VisualCoordinateSpace.DATA
    z_order: int = 0
    clip_scope: ClipScope = ClipScope.PLOT

    def __post_init__(self) -> None:
        validate_id(self.visual_id)
        validate_id(self.panel_id)
        validate_id(self.view_id)
        if isinstance(self.clip_scope, str):
            try:
                object.__setattr__(self, "clip_scope", ClipScope(self.clip_scope))
            except ValueError as exc:
                raise ValueError(
                    f"clip_scope must be one of {[scope.value for scope in ClipScope]}"
                ) from exc
        elif not isinstance(self.clip_scope, ClipScope):
            raise TypeError("clip_scope must be a ClipScope or its string value")


def resolved_attachment_clip_rect(
    attachment: VisualAttachment, snapshot: ResolvedLayoutSnapshot
) -> LogicalPixelRect:
    """Return the exact logical-pixel scissor selected by one attachment."""
    if not isinstance(attachment, VisualAttachment):
        raise TypeError("attachment must be a VisualAttachment")
    if not isinstance(snapshot, ResolvedLayoutSnapshot):
        raise TypeError("snapshot must be a ResolvedLayoutSnapshot")
    panel = snapshot.panel(attachment.panel_id)
    if attachment.clip_scope is ClipScope.PLOT:
        return panel.plot_rect_px
    if attachment.clip_scope is ClipScope.PANEL:
        return panel.panel_rect_px
    return LogicalPixelRect(
        0,
        0,
        snapshot.render_target.logical_width_px,
        snapshot.render_target.logical_height_px,
    )
