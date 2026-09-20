"""Construction helpers for pre-P038 internal fixture migration."""

from typing import Any

from gsp.protocol import (
    CoordinateSpace,
    LogicalPixelRect,
    Panel,
    RenderTarget,
    ResolvedLayoutSnapshot,
    ResolvedPanelLayout,
    View2D,
    View3D,
    VisualAttachment,
    full_target_panel_layout,
)
from gsp.scene import Scene


def resolved_single_panel_fixture(
    *,
    snapshot_id: str,
    render_target: RenderTarget,
    panel_rect_px: LogicalPixelRect,
    plot_rect_px: LogicalPixelRect,
    panel_id: str = "panel:main",
    **panel_fields: Any,
) -> ResolvedLayoutSnapshot:
    """Build one canonical per-panel snapshot from concise fixture inputs."""
    return ResolvedLayoutSnapshot(
        snapshot_id=snapshot_id,
        render_target=render_target,
        panels=(
            ResolvedPanelLayout(
                panel_id=panel_id,
                panel_rect_px=panel_rect_px,
                plot_rect_px=plot_rect_px,
                **panel_fields,
            ),
        ),
    )


def single_panel_scene(**scene_fields: Any) -> Scene:
    """Build a canonical explicit-layout scene for one-panel test fixtures."""
    view2d = scene_fields.pop("view2d", None)
    view3d = scene_fields.pop("view3d", None)
    if view2d is not None and view3d is not None:
        raise ValueError("single-panel fixture accepts only one primary view")
    view = view2d or view3d
    panel_id = view.panel_id if isinstance(view, (View2D, View3D)) else "panel:main"
    panels = scene_fields.pop("panels", (Panel(id=panel_id),))
    panel_layout = scene_fields.pop("panel_layout", full_target_panel_layout(panels[0].id))
    visuals = tuple(scene_fields.get("visuals", ()))
    if "attachments" not in scene_fields:
        scene_fields["attachments"] = tuple(
            VisualAttachment(
                visual_id=visual.id,
                panel_id=panel_id,
                view_id=(
                    view.id
                    if visual.coordinate_space is CoordinateSpace.DATA and view is not None
                    else None
                ),
            )
            for visual in visuals
        )
    return Scene(
        panels=panels,
        panel_layout=panel_layout,
        views2d=(view2d,) if isinstance(view2d, View2D) else (),
        views3d=(view3d,) if isinstance(view3d, View3D) else (),
        **scene_fields,
    )
