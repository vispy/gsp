"""Panel and View2D protocol models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .ids import validate_id
from .layout import LogicalPixelRect, RenderTarget
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


@dataclass(frozen=True, slots=True)
class Panel:
    """Semantic plot panel with a stable protocol identity."""

    id: str
    figure_id: str
    viewport_rect: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        validate_id(self.id)
        validate_id(self.figure_id)
        if len(self.viewport_rect) != 4:
            raise ValueError("panel viewport_rect must contain four values")
        x, y, width, height = self.viewport_rect
        for index, value in enumerate(self.viewport_rect):
            if not isinstance(value, (int, float)):
                raise TypeError(f"panel viewport_rect[{index}] must be a number")
            if not math.isfinite(value):
                raise ValueError(f"panel viewport_rect[{index}] must be finite")
        if width <= 0 or height <= 0:
            raise ValueError("panel viewport width and height must be positive")
        if x < 0 or y < 0:
            raise ValueError("panel viewport origin must be non-negative")
        if x + width > 1.0 or y + height > 1.0:
            raise ValueError("panel viewport_rect must be contained by the normalized render target")


def resolve_panel_viewport_rect(panel: Panel, render_target: RenderTarget) -> LogicalPixelRect:
    """Resolve normalized outer-panel allocation intent into logical pixels."""
    if not isinstance(panel, Panel):
        raise TypeError("panel must be a Panel")
    if not isinstance(render_target, RenderTarget):
        raise TypeError("render_target must be a RenderTarget")
    x, y, width, height = panel.viewport_rect
    return LogicalPixelRect(
        x=x * render_target.logical_width_px,
        y=y * render_target.logical_height_px,
        width=width * render_target.logical_width_px,
        height=height * render_target.logical_height_px,
    )


@dataclass(frozen=True, slots=True)
class View2D:
    """Semantic 2D Cartesian view attached to a panel."""

    id: str
    panel_id: str
    x_range: tuple[float, float] = (-1.0, 1.0)
    y_range: tuple[float, float] = (-1.0, 1.0)
    aspect_policy: AspectPolicy = AspectPolicy.AUTO
    kind: ViewKind = ViewKind.VIEW2D_LINEAR
    clip: bool = True

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
        if not isinstance(self.clip, bool):
            raise TypeError("clip must be a bool")
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

    def __post_init__(self) -> None:
        validate_id(self.visual_id)
        validate_id(self.panel_id)
        validate_id(self.view_id)
