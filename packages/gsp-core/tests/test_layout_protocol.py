import pytest

from gsp.protocol import (
    LogicalCoordinateRegion,
    LogicalPixelRect,
    PixelOrigin,
    RenderTarget,
    ResolvedLayoutSnapshot,
    classify_logical_coordinate,
    logical_coordinate_in_data_viewport,
    panel_ndc_to_plot_logical_px,
    plot_logical_px_to_panel_ndc,
    resolved_plot_aspect_ratio,
)


def _snapshot(
    *,
    panel: LogicalPixelRect = LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
    plot: LogicalPixelRect = LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
    origin: PixelOrigin = PixelOrigin.TOP_LEFT,
) -> ResolvedLayoutSnapshot:
    return ResolvedLayoutSnapshot(
        snapshot_id="layout:test",
        view_id="view:test",
        render_target=RenderTarget(
            logical_width_px=800.0,
            logical_height_px=600.0,
            pixel_origin=origin,
        ),
        panel_rect_px=panel,
        plot_rect_px=plot,
    )


def test_render_target_coerces_valid_pixel_origin_string() -> None:
    target = RenderTarget(800.0, 600.0, pixel_origin="bottom-left")  # type: ignore[arg-type]

    assert target.pixel_origin is PixelOrigin.BOTTOM_LEFT


@pytest.mark.parametrize("invalid", ("leftish", 7, object()))
def test_render_target_rejects_invalid_pixel_origin(invalid: object) -> None:
    error = ValueError if isinstance(invalid, str) else TypeError
    with pytest.raises(error, match="pixel_origin"):
        RenderTarget(800.0, 600.0, pixel_origin=invalid)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("panel", "plot"),
    [
        (
            LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
            LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
        ),
        (
            LogicalPixelRect(0.0, 0.0, 800.0, 600.0),
            LogicalPixelRect(80.0, 70.0, 640.0, 480.0),
        ),
        (
            LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
            LogicalPixelRect(140.0, 110.0, 520.0, 400.0),
        ),
    ],
)
def test_plot_logical_pixel_panel_ndc_round_trip(
    panel: LogicalPixelRect, plot: LogicalPixelRect
) -> None:
    snapshot = _snapshot(panel=panel, plot=plot)
    center = (plot.x + 0.37 * plot.width, plot.y + 0.61 * plot.height)

    ndc = plot_logical_px_to_panel_ndc(snapshot, center)

    assert panel_ndc_to_plot_logical_px(snapshot, ndc) == pytest.approx(center)
    assert panel_ndc_to_plot_logical_px(snapshot, (-1.0, 1.0)) == pytest.approx(
        (plot.x, plot.y)
    )
    assert panel_ndc_to_plot_logical_px(snapshot, (1.0, -1.0)) == pytest.approx(
        (plot.x + plot.width, plot.y + plot.height)
    )


def test_bottom_left_pixel_origin_round_trip_preserves_panel_ndc_orientation() -> None:
    snapshot = _snapshot(origin=PixelOrigin.BOTTOM_LEFT)

    assert panel_ndc_to_plot_logical_px(snapshot, (-1.0, -1.0)) == pytest.approx(
        (0.0, 0.0)
    )
    assert plot_logical_px_to_panel_ndc(snapshot, (800.0, 600.0)) == pytest.approx(
        (1.0, 1.0)
    )


@pytest.mark.parametrize("origin", list(PixelOrigin))
def test_exact_plot_corners_center_and_closed_edges(origin: PixelOrigin) -> None:
    snapshot = _snapshot(
        plot=LogicalPixelRect(100.0, 75.0, 600.0, 450.0),
        origin=origin,
    )
    top_y_ndc = 1.0 if origin is PixelOrigin.TOP_LEFT else -1.0
    bottom_y_ndc = -top_y_ndc

    assert panel_ndc_to_plot_logical_px(
        snapshot, (-1.0, top_y_ndc)
    ) == pytest.approx((100.0, 75.0))
    assert panel_ndc_to_plot_logical_px(
        snapshot, (1.0, bottom_y_ndc)
    ) == pytest.approx((700.0, 525.0))
    assert panel_ndc_to_plot_logical_px(snapshot, (0.0, 0.0)) == pytest.approx(
        (400.0, 300.0)
    )
    assert logical_coordinate_in_data_viewport(snapshot, (100.0, 75.0))
    assert logical_coordinate_in_data_viewport(snapshot, (700.0, 525.0))


def test_resolved_plot_aspect_and_three_way_coordinate_classification() -> None:
    snapshot = _snapshot(
        panel=LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
        plot=LogicalPixelRect(140.0, 110.0, 520.0, 400.0),
    )

    assert resolved_plot_aspect_ratio(snapshot) == pytest.approx(1.3)
    assert logical_coordinate_in_data_viewport(snapshot, (140.0, 110.0))
    assert logical_coordinate_in_data_viewport(snapshot, (660.0, 510.0))
    assert not logical_coordinate_in_data_viewport(snapshot, (120.0, 80.0))
    assert not logical_coordinate_in_data_viewport(snapshot, (680.0, 530.0))
    assert (
        classify_logical_coordinate(snapshot, (140.0, 110.0))
        is LogicalCoordinateRegion.DATA_PLOT
    )
    assert (
        classify_logical_coordinate(snapshot, (120.0, 80.0))
        is LogicalCoordinateRegion.PANEL_GUIDE_LANE
    )
    assert (
        classify_logical_coordinate(snapshot, (90.0, 80.0))
        is LogicalCoordinateRegion.OUTSIDE_PANEL
    )


@pytest.mark.parametrize(
    ("panel", "plot", "message"),
    [
        (
            LogicalPixelRect(0.0, 0.0, 0.0, 600.0),
            LogicalPixelRect(0.0, 0.0, 0.0, 600.0),
            "panel_rect_px must have positive",
        ),
        (
            LogicalPixelRect(-1.0, 0.0, 100.0, 100.0),
            LogicalPixelRect(0.0, 0.0, 50.0, 50.0),
            "panel_rect_px origin",
        ),
        (
            LogicalPixelRect(0.0, 0.0, 801.0, 600.0),
            LogicalPixelRect(0.0, 0.0, 50.0, 50.0),
            "panel_rect_px must be inside",
        ),
        (
            LogicalPixelRect(100.0, 100.0, 400.0, 300.0),
            LogicalPixelRect(50.0, 120.0, 200.0, 200.0),
            "plot_rect_px must be contained",
        ),
    ],
)
def test_resolved_layout_rejects_invalid_panel_and_plot_geometry(
    panel: LogicalPixelRect, plot: LogicalPixelRect, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _snapshot(panel=panel, plot=plot)


def test_plot_helpers_reject_empty_data_viewport() -> None:
    snapshot = _snapshot(
        plot=LogicalPixelRect(100.0, 100.0, 0.0, 200.0)
    )

    with pytest.raises(ValueError, match="positive width and height"):
        resolved_plot_aspect_ratio(snapshot)
    with pytest.raises(ValueError, match="positive width and height"):
        plot_logical_px_to_panel_ndc(snapshot, (100.0, 100.0))
    assert (
        classify_logical_coordinate(snapshot, (100.0, 100.0))
        is LogicalCoordinateRegion.PANEL_GUIDE_LANE
    )


def test_plot_conversion_rejects_guide_lane_instead_of_extrapolating() -> None:
    snapshot = _snapshot(
        panel=LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
        plot=LogicalPixelRect(140.0, 110.0, 520.0, 400.0),
    )

    with pytest.raises(ValueError, match="inside the closed plot_rect_px"):
        plot_logical_px_to_panel_ndc(snapshot, (120.0, 80.0))
