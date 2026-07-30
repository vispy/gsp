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
    ExplicitPanelLayoutV1,
    ImageVisual,
    MarkerVisual,
    MeshVisual,
    Panel,
    PanelLayoutIntent,
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
from .protocol.ids import validate_id

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
    panels: tuple[Panel, ...]
    panel_layout: PanelLayoutIntent
    visuals: tuple[SceneVisual, ...] = ()
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
        validate_id(self.id)
        if not self.panels:
            raise ValueError("Scene requires at least one panel")
        if not isinstance(self.panel_layout, ExplicitPanelLayoutV1):
            raise TypeError("Scene.panel_layout must be an ExplicitPanelLayoutV1")
        panel_ids = [panel.id for panel in self.panels]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError("Scene panels must have unique ids")
        placement_ids = [placement.panel_id for placement in self.panel_layout.placements]
        if set(placement_ids) != set(panel_ids):
            missing = sorted(set(panel_ids) - set(placement_ids))
            unknown = sorted(set(placement_ids) - set(panel_ids))
            raise ValueError(
                "Scene.panel_layout must place every panel exactly once; "
                f"missing={missing}, unknown={unknown}"
            )
        if self.view2d is not None and self.view3d is not None:
            raise ValueError("Scene cannot define both view2d and view3d")
        for view in (self.view2d, self.view3d):
            if view is not None and view.panel_id not in panel_ids:
                raise ValueError(f"view references unknown panel_id {view.panel_id!r}")
        visual_ids = {visual.id for visual in self.visuals}
        view_ids = {view.id for view in (self.view2d, self.view3d) if view is not None}
        for attachment in self.attachments:
            if attachment.visual_id not in visual_ids:
                raise ValueError(
                    f"attachment references unknown visual_id {attachment.visual_id!r}"
                )
            if attachment.panel_id not in panel_ids:
                raise ValueError(f"attachment references unknown panel_id {attachment.panel_id!r}")
            if attachment.view_id not in view_ids:
                raise ValueError(f"attachment references unknown view_id {attachment.view_id!r}")
        for visual in self.visuals:
            if isinstance(visual, SphereVisual):
                if self.view3d is None:
                    raise ValueError("SphereVisual DATA positions3d require Scene.view3d")
                continue
            if not isinstance(visual, (PixelVisual, VectorVisual, PrimitiveVisual, TextVisual)):
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
                if isinstance(visual, TextVisual) and visual.transform is not None:
                    raise ValueError(
                        "TextVisual billboard3d does not support a 2D visual transform"
                    )
            elif visual.coordinate_space is CoordinateSpace.DATA and self.view2d is None:
                raise ValueError(f"{type(visual).__name__} DATA positions2d require Scene.view2d")
