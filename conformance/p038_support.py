"""Construction helpers for pre-P038 internal fixture migration."""

from typing import Any

from gsp.protocol import (
    LogicalPixelRect,
    Panel,
    RenderTarget,
    ResolvedLayoutSnapshot,
    ResolvedPanelLayout,
    View2D,
    View3D,
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
    view = scene_fields.get("view2d") or scene_fields.get("view3d")
    panel_id = view.panel_id if isinstance(view, (View2D, View3D)) else "panel:main"
    panels = scene_fields.pop("panels", (Panel(id=panel_id),))
    panel_layout = scene_fields.pop("panel_layout", full_target_panel_layout(panels[0].id))
    return Scene(
        panels=panels,
        panel_layout=panel_layout,
        **scene_fields,
    )
