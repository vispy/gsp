import pytest

from gsp.protocol import (
    Camera3D,
    LogicalPixelRect,
    PerspectiveAspectRatioSource,
    PerspectiveProjection3D,
    RenderTarget,
    ResolvedLayoutSnapshot,
    View3D,
    logical_coordinate_in_data_viewport,
    panel_ndc_to_plot_logical_px,
    plot_logical_px_to_panel_ndc,
    project_view3d_data_point,
    resolve_view3d_projection_snapshot,
    unproject_view3d_panel_ndc_point,
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
    assert plot_logical_px_to_panel_ndc(layout, title_coordinate)[1] > 1.0


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
    inset_projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=inset
    )
    stale_projection = resolve_view3d_projection_snapshot(
        view, layout_snapshot=stale_id
    )

    assert full_projection.view_projection_snapshot_id != (
        inset_projection.view_projection_snapshot_id
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
