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
    views2d: tuple[View2D, ...] = ()
    views3d: tuple[View3D, ...] = ()
    attachments: tuple[VisualAttachment, ...] = ()
    axis_guides: tuple[AxisGuide, ...] = ()
    panel_text_guides: tuple[PanelTextGuide, ...] = ()
    color_scales: tuple[ColorScale, ...] = ()
    colorbar_guides: tuple[ColorbarGuide, ...] = ()
    textures: tuple[Texture2D, ...] = ()
    transforms: tuple[AffineTransform2DResource, ...] = ()
    canvas_size: CanvasSize | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous panel, view, visual, attachment, and guide relationships."""
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
        if any(not isinstance(view, View2D) for view in self.views2d):
            raise TypeError("Scene.views2d must contain only View2D values")
        if any(not isinstance(view, View3D) for view in self.views3d):
            raise TypeError("Scene.views3d must contain only View3D values")
        views: tuple[View2D | View3D, ...] = (*self.views2d, *self.views3d)
        view_ids = [view.id for view in views]
        if len(view_ids) != len(set(view_ids)):
            raise ValueError("Scene views must have unique ids across views2d and views3d")
        view_panel_ids = [view.panel_id for view in views]
        if len(view_panel_ids) != len(set(view_panel_ids)):
            raise ValueError("a panel may have at most one primary data view")
        for view in views:
            if view.panel_id not in panel_ids:
                raise ValueError(f"view references unknown panel_id {view.panel_id!r}")

        visual_ids = [visual.id for visual in self.visuals]
        if len(visual_ids) != len(set(visual_ids)):
            raise ValueError("Scene visuals must have unique ids")
        attachments_by_visual: dict[str, VisualAttachment] = {}
        for attachment in self.attachments:
            if attachment.visual_id not in set(visual_ids):
                raise ValueError(
                    f"attachment references unknown visual_id {attachment.visual_id!r}"
                )
            if attachment.visual_id in attachments_by_visual:
                raise ValueError(
                    f"visual {attachment.visual_id!r} must have exactly one attachment"
                )
            attachments_by_visual[attachment.visual_id] = attachment
            if attachment.panel_id not in panel_ids:
                raise ValueError(f"attachment references unknown panel_id {attachment.panel_id!r}")
            if attachment.view_id is not None and attachment.view_id not in set(view_ids):
                raise ValueError(f"attachment references unknown view_id {attachment.view_id!r}")
        missing_attachments = sorted(set(visual_ids) - set(attachments_by_visual))
        if missing_attachments:
            raise ValueError(
                f"every visual requires exactly one attachment; missing={missing_attachments}"
            )

        for visual in self.visuals:
            attachment = attachments_by_visual[visual.id]
            dimension = _visual_dimension(visual)
            if (
                dimension == 3
                and isinstance(visual, (PixelVisual, VectorVisual, PrimitiveVisual, TextVisual))
                and visual.coordinate_space is not CoordinateSpace.DATA
            ):
                raise ValueError(
                    f"{type(visual).__name__} positions3d require CoordinateSpace.DATA"
                )
            if visual.coordinate_space is CoordinateSpace.NDC:
                if attachment.view_id is not None:
                    raise ValueError(f"NDC visual {visual.id!r} requires a viewless attachment")
                continue
            if attachment.view_id is None:
                raise ValueError(f"DATA visual {visual.id!r} requires an attachment view_id")
            view = self.view(attachment.view_id)
            if view.panel_id != attachment.panel_id:
                raise ValueError(
                    f"attachment panel_id {attachment.panel_id!r} does not match "
                    f"view panel_id {view.panel_id!r}"
                )
            expected_type = View3D if dimension == 3 else View2D
            if not isinstance(view, expected_type):
                raise ValueError(
                    f"{type(visual).__name__} DATA positions{dimension}d require "
                    f"{expected_type.__name__}"
                )
            if dimension == 3 and isinstance(visual, TextVisual) and visual.transform is not None:
                raise ValueError("TextVisual billboard3d does not support a 2D visual transform")

        for axis_guide in self.axis_guides:
            try:
                view = self.view(axis_guide.view_id)
            except KeyError as exc:
                raise ValueError(
                    f"axis guide references unknown view_id {axis_guide.view_id!r}"
                ) from exc
            if not isinstance(view, View2D):
                raise ValueError("AxisGuide requires a View2D")
        for text_guide in self.panel_text_guides:
            if text_guide.panel_id not in panel_ids:
                raise ValueError(
                    f"panel text guide references unknown panel_id {text_guide.panel_id!r}"
                )
        for colorbar_guide in self.colorbar_guides:
            if colorbar_guide.panel_id not in panel_ids:
                raise ValueError(
                    f"colorbar guide references unknown panel_id {colorbar_guide.panel_id!r}"
                )
            for visual_id in colorbar_guide.linked_visual_ids:
                if visual_id not in set(visual_ids):
                    raise ValueError(f"colorbar guide references unknown visual_id {visual_id!r}")

    def view(self, view_id: str) -> View2D | View3D:
        """Return one view by semantic identity."""
        validate_id(view_id)
        views: tuple[View2D | View3D, ...] = (*self.views2d, *self.views3d)
        for view in views:
            if view.id == view_id:
                return view
        raise KeyError(view_id)

    def primary_view_for_panel(self, panel_id: str) -> View2D | View3D | None:
        """Return the panel's sole primary data view, if present."""
        validate_id(panel_id)
        views: tuple[View2D | View3D, ...] = (*self.views2d, *self.views3d)
        for view in views:
            if view.panel_id == panel_id:
                return view
        return None

    def attachment_for_visual(self, visual_id: str) -> VisualAttachment:
        """Return the sole attachment for one visual identity."""
        validate_id(visual_id)
        for attachment in self.attachments:
            if attachment.visual_id == visual_id:
                return attachment
        raise KeyError(visual_id)

    def visuals_for_panel(
        self, panel_id: str, *, include_hidden: bool = False
    ) -> tuple[SceneVisual, ...]:
        """Return panel visuals in stable attachment order."""
        validate_id(panel_id)
        by_id = {visual.id: visual for visual in self.visuals}
        indexed = [
            (index, attachment)
            for index, attachment in enumerate(self.attachments)
            if attachment.panel_id == panel_id and (include_hidden or attachment.visible)
        ]
        indexed.sort(key=lambda item: (item[1].z_order, item[0]))
        return tuple(by_id[attachment.visual_id] for _, attachment in indexed)


def _visual_dimension(visual: SceneVisual) -> int:
    if isinstance(visual, ImageVisual):
        return 2
    if isinstance(visual, SegmentVisual):
        return int(visual.start_positions.shape[1])
    return int(visual.positions.shape[1])
