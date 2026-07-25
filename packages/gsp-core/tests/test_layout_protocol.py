import pytest

from gsp.protocol import (
    LogicalPixelRect,
    PixelOrigin,
    RenderTarget,
    ResolvedLayoutSnapshot,
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


def test_resolved_plot_aspect_and_outside_plot_classification() -> None:
    snapshot = _snapshot(
        panel=LogicalPixelRect(100.0, 50.0, 600.0, 500.0),
        plot=LogicalPixelRect(140.0, 110.0, 520.0, 400.0),
    )

    assert resolved_plot_aspect_ratio(snapshot) == pytest.approx(1.3)
    assert logical_coordinate_in_data_viewport(snapshot, (140.0, 110.0))
    assert logical_coordinate_in_data_viewport(snapshot, (660.0, 510.0))
    assert not logical_coordinate_in_data_viewport(snapshot, (120.0, 80.0))
    assert not logical_coordinate_in_data_viewport(snapshot, (680.0, 530.0))


@pytest.mark.parametrize(
    ("panel", "plot", "message"),
    [
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
