import pytest

from gsp.protocol import (
    Camera3D,
    LogicalCoordinateRegion,
    LogicalPixelRect,
    PerspectiveAspectRatioSource,
    PerspectiveProjection3D,
    PixelOrigin,
    Panel,
    RenderTarget,
    ResolvedLayoutSnapshot,
    View3D,
    View2D,
    View3DNavigationAction,
    View3DNavigationActionKind,
    Zoom3DPayload,
    apply_view3d_navigation_action,
    classify_logical_coordinate,
    logical_coordinate_in_data_viewport,
    panel_ndc_to_plot_logical_px,
    pan_view2d,
    plot_logical_px_to_panel_ndc,
    project_view3d_data_point,
    resolve_view3d_projection_snapshot,
    resolve_panel_viewport_rect,
    unproject_view3d_panel_ndc_point,
    zoom_view3d,
    zoom_view2d_about,
)


def _layout(
    snapshot_id: str,
    *,
    panel: LogicalPixelRect,
    plot: LogicalPixelRect,
) -> ResolvedLayoutSnapshot:
    return ResolvedLayoutSnapshot(
        snapshot_id=snapshot_id,
        view_id="view:perspective",
        render_target=RenderTarget(800.0, 600.0),
        panel_rect_px=panel,
        plot_rect_px=plot,
    )


def _view(*, authored_aspect: float | None = None) -> View3D:
    return View3D(
        id="view:perspective",
        panel_id="panel:main",
        camera=Camera3D(
            eye=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
        ),
        projection=PerspectiveProjection3D(
            fov_y_degrees=60.0,
            near_far=(0.1, 100.0),
            aspect_ratio=authored_aspect,
        ),
    )


def test_normalized_panel_intent_and_view2d_origin_are_executable() -> None:
    panel = Panel(
        id="panel:main",
        figure_id="figure:main",
        viewport_rect=(0.125, 0.1, 0.75, 0.8),
    )
    view = View2D(
        id="view:main",
        panel_id=panel.id,
        x_range=(0.0, 100.0),
        y_range=(0.0, 100.0),
    )
    top_left = ResolvedLayoutSnapshot(
        snapshot_id="layout:top-left",
        view_id=view.id,
        render_target=RenderTarget(800.0, 600.0, pixel_origin=PixelOrigin.TOP_LEFT),
        panel_rect_px=resolve_panel_viewport_rect(panel, RenderTarget(800.0, 600.0)),
        plot_rect_px=LogicalPixelRect(140.0, 100.0, 520.0, 400.0),
    )
    bottom_left = ResolvedLayoutSnapshot(
        snapshot_id="layout:bottom-left",
        view_id=view.id,
        render_target=RenderTarget(800.0, 600.0, pixel_origin=PixelOrigin.BOTTOM_LEFT),
        panel_rect_px=top_left.panel_rect_px,
        plot_rect_px=top_left.plot_rect_px,
    )

    assert top_left.panel_rect_px == LogicalPixelRect(100.0, 60.0, 600.0, 480.0)
    assert zoom_view2d_about(
        view,
        top_left.plot_rect_px,
        (400.0, 100.0),
        1.0,
        2.0,
        layout_snapshot=top_left,
    ).y_range == pytest.approx((50.0, 100.0))
    assert zoom_view2d_about(
        view,
        bottom_left.plot_rect_px,
        (400.0, 100.0),
        1.0,
        2.0,
        layout_snapshot=bottom_left,
    ).y_range == pytest.approx((0.0, 50.0))
    assert pan_view2d(
        view,
        top_left.plot_rect_px,
        0.0,
        40.0,
        layout_snapshot=top_left,
    ).y_range == pytest.approx((10.0, 110.0))
    assert pan_view2d(
        view,
        bottom_left.plot_rect_px,
        0.0,
        40.0,
        layout_snapshot=bottom_left,
    ).y_range == pytest.approx((-10.0, 90.0))


@pytest.mark.parametrize(
    ("panel", "plot"),
    [
        (
            LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
            LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        ),
        (
            LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
            LogicalPixelRect(90.0, 80.0, 620.0, 462.0),
        ),
        (
            LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
            LogicalPixelRect(140.0, 120.0, 520.0, 390.0),
        ),
    ],
)
def test_plot_viewport_round_trips_and_drives_perspective_projection(
    panel: LogicalPixelRect, plot: LogicalPixelRect
) -> None:
    layout = _layout("layout:strict", panel=panel, plot=plot)
    view = _view()
    projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=layout
    )
    point = (0.75, -0.4, 0.0)

    ndc = project_view3d_data_point(
        view, point, aspect_ratio=projection.aspect_ratio
    )
    logical = panel_ndc_to_plot_logical_px(layout, ndc[:2])

    assert plot_logical_px_to_panel_ndc(layout, logical) == pytest.approx(ndc[:2])
    assert unproject_view3d_panel_ndc_point(
        view, ndc, aspect_ratio=projection.aspect_ratio
    ) == pytest.approx(point)
    assert projection.aspect_ratio == pytest.approx(plot.width / plot.height)
    assert (
        projection.aspect_ratio_source
        is PerspectiveAspectRatioSource.RESOLVED_LAYOUT
    )


def test_title_lane_is_outside_data_viewport_but_inside_outer_panel() -> None:
    layout = _layout(
        "layout:title",
        panel=LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
        plot=LogicalPixelRect(140.0, 120.0, 520.0, 390.0),
    )
    title_coordinate = (400.0, 80.0)

    assert 100.0 <= title_coordinate[0] <= 700.0
    assert 50.0 <= title_coordinate[1] <= 550.0
    assert not logical_coordinate_in_data_viewport(layout, title_coordinate)
    assert (
        classify_logical_coordinate(layout, title_coordinate)
        is LogicalCoordinateRegion.PANEL_GUIDE_LANE
    )
    with pytest.raises(ValueError, match="closed plot_rect_px"):
        plot_logical_px_to_panel_ndc(layout, title_coordinate)


def test_effective_plot_viewport_and_layout_identity_change_projection_identity() -> None:
    view = _view()
    full = _layout(
        "layout:shared",
        panel=LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        plot=LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
    )
    inset = _layout(
        "layout:shared",
        panel=LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        plot=LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
    )
    stale_id = _layout(
        "layout:stale",
        panel=full.panel_rect_px,
        plot=full.plot_rect_px,
    )

    full_projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=full
    )
    repeated_projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=full
    )
    inset_projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=inset
    )
    stale_projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=stale_id
    )

    assert full_projection.view_projection_snapshot_id != (
        inset_projection.view_projection_snapshot_id
    )
    assert full_projection.view_projection_snapshot_id == (
        repeated_projection.view_projection_snapshot_id
    )
    assert full_projection.view_projection_snapshot_id != (
        stale_projection.view_projection_snapshot_id
    )


def test_authored_aspect_wins_and_legacy_id_only_path_is_diagnosed() -> None:
    layout = _layout(
        "layout:strict",
        panel=LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        plot=LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
    )
    explicit = resolve_view3d_projection_snapshot(
        _view(authored_aspect=2.0), layout_snapshot=layout
    )
    legacy = resolve_view3d_projection_snapshot(
        _view(), layout_snapshot_id="layout:legacy"
    )

    assert explicit.aspect_ratio == pytest.approx(2.0)
    assert explicit.aspect_ratio_source is PerspectiveAspectRatioSource.EXPLICIT
    assert explicit.diagnostics == ()
    assert legacy.aspect_ratio == pytest.approx(1.0)
    assert (
        legacy.aspect_ratio_source
        is PerspectiveAspectRatioSource.COMPATIBILITY_DEFAULT
    )
    assert "layout_geometry_missing" in legacy.diagnostics[0]


def test_layout_resolved_navigation_refreshes_identity_and_rejects_stale_layout() -> None:
    view = _view()
    layout = _layout(
        "layout:current",
        panel=LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        plot=LogicalPixelRect(100.0, 80.0, 600.0, 450.0),
    )
    current = resolve_view3d_projection_snapshot(view, layout_snapshot=layout)
    action = View3DNavigationAction(
        kind=View3DNavigationActionKind.ZOOM,
        view_id=view.id,
        base_view_revision=view.revision,
        base_view_projection_snapshot_id=current.view_projection_snapshot_id,
        base_layout_snapshot_id=layout.snapshot_id,
        payload=Zoom3DPayload(scale=2.0),
    )

    result = apply_view3d_navigation_action(view, action, layout_snapshot=layout)

    assert result.accepted
    assert result.view is not None
    updated = resolve_view3d_projection_snapshot(
        result.view, layout_snapshot=layout
    )
    assert result.view_projection_snapshot_id == updated.view_projection_snapshot_id

    stale_layout = _layout(
        "layout:changed",
        panel=layout.panel_rect_px,
        plot=LogicalPixelRect(120.0, 80.0, 560.0, 450.0),
    )
    rejected = apply_view3d_navigation_action(
        view, action, layout_snapshot=stale_layout
    )
    assert not rejected.accepted
    assert "snapshot_mismatch" in rejected.diagnostics[0]


def test_guide_lane_cannot_construct_ray_or_mesh_pick_ndc() -> None:
    layout = _layout(
        "layout:guide-lane",
        panel=LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        plot=LogicalPixelRect(100.0, 100.0, 600.0, 450.0),
    )
    coordinate = (400.0, 50.0)

    assert (
        classify_logical_coordinate(layout, coordinate)
        is LogicalCoordinateRegion.PANEL_GUIDE_LANE
    )
    with pytest.raises(ValueError, match="closed plot_rect_px"):
        plot_logical_px_to_panel_ndc(layout, coordinate)
    with pytest.raises(ValueError, match="closed data plot"):
        zoom_view3d(
            _view(),
            Zoom3DPayload(scale=2.0, anchor_panel_ndc_xy=(0.0, 1.01)),
        )
