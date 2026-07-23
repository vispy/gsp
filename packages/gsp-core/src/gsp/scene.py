"""Immutable backend-neutral scene snapshot."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import (
    AffineTransform2DResource,
    AxisGuide,
    CanvasSize,
    ColorScale,
    ColorbarGuide,
    CoordinateSpace,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    Panel,
    PanelTextGuide,
    PathVisual,
    PixelVisual,
    PointVisual,
    PrimitiveVisual,
    SegmentVisual,
    SphereVisual,
    TextVisual,
    Texture2D,
    View2D,
    View3D,
    VisualAttachment,
    VectorVisual,
)

SceneVisual = (
    PointVisual
    | PixelVisual
    | SphereVisual
    | VectorVisual
    | PrimitiveVisual
    | MarkerVisual
    | SegmentVisual
    | PathVisual
    | ImageVisual
    | TextVisual
    | MeshVisual
)


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

    def __post_init__(self) -> None:
        """Reject an ambiguous scene while preserving viewless NDC scenes."""
        if self.view2d is not None and self.view3d is not None:
            raise ValueError("Scene cannot define both view2d and view3d")
        for visual in self.visuals:
            if isinstance(visual, SphereVisual):
                if self.view3d is None:
                    raise ValueError(
                        "SphereVisual DATA positions3d require Scene.view3d"
                    )
                continue
            if not isinstance(visual, (PixelVisual, VectorVisual, PrimitiveVisual)):
                continue
            if visual.positions.shape[1] == 3:
                if visual.coordinate_space is not CoordinateSpace.DATA:
                    raise ValueError(
                        f"{type(visual).__name__} positions3d require CoordinateSpace.DATA"
                    )
                if self.view3d is None:
                    raise ValueError(
                        f"{type(visual).__name__} DATA positions3d require Scene.view3d"
                    )
            elif (
                visual.coordinate_space is CoordinateSpace.DATA
                and self.view2d is None
            ):
                raise ValueError(
                    f"{type(visual).__name__} DATA positions2d require Scene.view2d"
                )
