from __future__ import annotations

import pytest

from gsp.protocol import LogicalPixelRect, PanByAction, View2D, View2DNavigationController
from gsp_datoviz.protocol_renderer import _apply_view2d_navigation_action as apply_datoviz
from gsp_matplotlib.navigation import apply_view2d_navigation_action as apply_matplotlib


def test_providers_share_canonical_pan_ranges_and_revision_transition() -> None:
    view = View2D(
        id="view:main",
        panel_id="panel:main",
        x_range=(-2.0, 2.0),
        y_range=(-1.0, 3.0),
    )
    controller = View2DNavigationController(
        id="nav:main",
        panel_id=view.panel_id,
        view_id=view.id,
        current_view2d_revision="view-rev:1",
        home_view=view,
    )
    rect = LogicalPixelRect(x=0.0, y=0.0, width=800.0, height=400.0)
    action = PanByAction(
        controller_id=controller.id,
        view2d_revision=controller.current_view2d_revision,
        dx_px=80.0,
        dy_px=-40.0,
        layout_snapshot_id="layout:main",
    )
    kwargs = {
        "next_view2d_revision": "view-rev:2",
        "view_snapshot_id": "view-snapshot:2",
        "expected_layout_snapshot_id": "layout:main",
    }

    mpl = apply_matplotlib(controller, view, rect, action, **kwargs)
    dvz = apply_datoviz(controller, view, rect, action, **kwargs)

    assert mpl.accepted and dvz.accepted
    assert mpl.new_view2d_revision == dvz.new_view2d_revision == "view-rev:2"
    assert mpl.view is not None and dvz.view is not None
    assert mpl.view.x_range == pytest.approx(dvz.view.x_range)
    assert mpl.view.y_range == pytest.approx(dvz.view.y_range)
    assert mpl.view.x_range == pytest.approx((-2.4, 1.6))
    assert mpl.view.y_range == pytest.approx((-0.6, 3.4))
