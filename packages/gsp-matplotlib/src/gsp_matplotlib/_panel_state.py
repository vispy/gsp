"""Internal panel-result composition and consumed-layout validation."""

from __future__ import annotations

from gsp import Scene
from gsp.protocol import CanvasSize, ResolvedLayoutSnapshot, resolve_panel_layout_intent

from .protocol_renderer import MatplotlibProtocolRenderResult


def _scene_panel_ids(scene: Scene) -> frozenset[str]:
    return frozenset(panel.id for panel in scene.panels)


def _matplotlib_panel_bounds(
    result: MatplotlibProtocolRenderResult,
    panel_id: str,
) -> tuple[float, float, float, float]:
    rect = result.layout_snapshot.panel(panel_id).plot_rect_px
    return (rect.x, rect.x + rect.width, rect.y, rect.y + rect.height)


def _snapshot_for_panel(snapshot: ResolvedLayoutSnapshot, panel_id: str) -> ResolvedLayoutSnapshot:
    """Project one panel out of a scene-wide snapshot for singular helpers."""
    return ResolvedLayoutSnapshot(
        snapshot_id=snapshot.snapshot_id,
        render_target=snapshot.render_target,
        panels=(snapshot.panel(panel_id),),
    )


def _combine_panel_results(
    panel_results: list[MatplotlibProtocolRenderResult], *, consumed: bool
) -> MatplotlibProtocolRenderResult:
    if not panel_results:
        raise ValueError("Matplotlib rendering requires at least one panel")
    first = panel_results[0]
    render_target = first.layout_snapshot.render_target
    if any(result.layout_snapshot.render_target != render_target for result in panel_results[1:]):
        raise ValueError("Matplotlib panel passes resolved inconsistent render targets")
    snapshot_ids = {result.layout_snapshot.snapshot_id for result in panel_results}
    snapshot_id = next(iter(snapshot_ids)) if len(snapshot_ids) == 1 else "layout:matplotlib"
    snapshot = ResolvedLayoutSnapshot(
        snapshot_id=snapshot_id,
        render_target=render_target,
        panels=tuple(result.layout_snapshot.only_panel() for result in panel_results),
    )
    panel_axes = tuple(
        (result.layout_snapshot.only_panel().panel_id, result.axes) for result in panel_results
    )
    panel_view_snapshot_ids = tuple(
        (result.layout_snapshot.only_panel().panel_id, result.view_snapshot_id)
        for result in panel_results
    )
    panel_view3d_projection_snapshots = tuple(
        (
            result.layout_snapshot.only_panel().panel_id,
            result.view3d_projection_snapshot,
        )
        for result in panel_results
    )
    return MatplotlibProtocolRenderResult(
        figure=first.figure,
        axes=first.axes,
        layout_snapshot=snapshot,
        resolved_canvas=first.resolved_canvas,
        view_snapshot_id=first.view_snapshot_id if len(panel_results) == 1 else None,
        view3d_projection_snapshot=(
            first.view3d_projection_snapshot if len(panel_results) == 1 else None
        ),
        layout_was_consumed=consumed,
        panel_axes=panel_axes,
        panel_view_snapshot_ids=panel_view_snapshot_ids,
        panel_view3d_projection_snapshots=panel_view3d_projection_snapshots,
    )


def _replace_resolved_panel(
    current: ResolvedLayoutSnapshot, replacement: ResolvedLayoutSnapshot
) -> ResolvedLayoutSnapshot:
    panel = replacement.only_panel()
    return ResolvedLayoutSnapshot(
        snapshot_id=replacement.snapshot_id,
        render_target=current.render_target,
        panels=tuple(
            panel if existing.panel_id == panel.panel_id else existing
            for existing in current.panels
        ),
    )


def _set_panel_view_snapshot_id(
    result: MatplotlibProtocolRenderResult,
    panel_id: str,
    snapshot_id: str | None,
) -> None:
    updated = tuple(
        (candidate, snapshot_id if candidate == panel_id else existing)
        for candidate, existing in result.panel_view_snapshot_ids
    )
    object.__setattr__(result, "panel_view_snapshot_ids", updated)
    if len(updated) == 1:
        object.__setattr__(result, "view_snapshot_id", snapshot_id)


def _validate_consumed_layout_scene(
    scene: Scene, layout_snapshot: ResolvedLayoutSnapshot | None
) -> None:
    if layout_snapshot is None:
        return
    if not isinstance(layout_snapshot, ResolvedLayoutSnapshot):
        raise TypeError("layout_snapshot must be a ResolvedLayoutSnapshot")
    _validate_scene_canvas_target(scene.canvas_size, layout_snapshot)
    scene_panel_ids = tuple(panel.id for panel in scene.panels)
    snapshot_panel_ids = tuple(panel.panel_id for panel in layout_snapshot.panels)
    if set(scene_panel_ids) != set(snapshot_panel_ids):
        raise ValueError("layout_snapshot panels do not match the consumed scene panels")
    for panel in scene.panels:
        active_view = scene.primary_view_for_panel(panel.id)
        resolved_panel = layout_snapshot.panel(panel.id)
        if active_view is not None:
            if resolved_panel.view_id != active_view.id:
                raise ValueError("layout_snapshot view_id does not match the active scene view")
        elif resolved_panel.view_id is not None:
            raise ValueError("viewless scene panel cannot consume a view-bound layout_snapshot")
    expected = resolve_panel_layout_intent(scene.panel_layout, layout_snapshot.render_target)
    expected_rects = {panel.panel_id: panel.panel_rect_px for panel in expected}
    if any(
        expected_rects.get(panel.id) != layout_snapshot.panel(panel.id).panel_rect_px
        for panel in scene.panels
    ):
        raise ValueError("layout_snapshot panel_rect_px does not match the scene panel allocation")
    if scene.axis_guides or scene.colorbar_guides:
        raise ValueError(
            "consumed Matplotlib layout does not prove native axis/colorbar guide geometry"
        )
    for panel in scene.panels:
        panel_guides = tuple(
            guide for guide in scene.panel_text_guides if guide.panel_id == panel.id
        )
        title_guides = tuple(guide for guide in panel_guides if guide.role.value == "title")
        if len(title_guides) != len(panel_guides) or len(title_guides) > 1:
            raise ValueError("consumed Matplotlib layout supports at most one title per panel")
        title_box_ids = {box.guide_id for box in layout_snapshot.panel(panel.id).title_boxes}
        if title_guides and title_guides[0].id not in title_box_ids:
            raise ValueError("supplied title guide requires matching resolved title geometry")


def _validate_scene_canvas_target(
    canvas_size: CanvasSize | None, snapshot: ResolvedLayoutSnapshot
) -> None:
    if canvas_size is None:
        return
    target = snapshot.render_target
    resolved = canvas_size.resolve(
        output_dpi=canvas_size.reference_dpi * target.device_scale,
        device_scale=target.device_scale,
    )
    if (
        resolved.canvas_width_px != target.logical_width_px
        or resolved.canvas_height_px != target.logical_height_px
        or resolved.framebuffer_width != target.framebuffer_width_px
        or resolved.framebuffer_height != target.framebuffer_height_px
        or resolved.device_scale_x != target.device_scale
        or resolved.device_scale_y != target.device_scale
    ):
        raise ValueError(
            "scene canvas policy does not resolve to the consumed layout render target"
        )
