"""Immutable backend-neutral scene snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import (
    AffineTransform2DResource,
    AxisGuide,
    CanvasSize,
    ColorScale,
    ColorbarGuide,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    Panel,
    PanelTextGuide,
    PathVisual,
    PointVisual,
    SegmentVisual,
    TextVisual,
    Texture2D,
    View2D,
    View3D,
    VisualAttachment,
)

SceneVisual = PointVisual | MarkerVisual | SegmentVisual | PathVisual | ImageVisual | TextVisual | MeshVisual


@dataclass(frozen=True, slots=True)
class Scene:
    """One logically immutable semantic scene ready for capability planning."""

    id: str
    visuals: tuple[SceneVisual, ...] = ()
    panels: tuple[Panel, ...] = ()
    view2d: View2D | None = None
    view3d: View3D | None = None
    attachments: tuple[VisualAttachment, ...] = ()
    axis_guides: tuple[AxisGuide, ...] = ()
    panel_text_guides: tuple[PanelTextGuide, ...] = ()
    color_scales: tuple[ColorScale, ...] = ()
    colorbar_guides: tuple[ColorbarGuide, ...] = ()
    textures: tuple[Texture2D, ...] = ()
    transforms: tuple[AffineTransform2DResource, ...] = ()
    canvas_size: CanvasSize | None = None

