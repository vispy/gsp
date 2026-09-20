"""Scene-level validation for panel-scoped GSP views and attachments."""

import pytest
import numpy as np

from gsp import Scene
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    ExplicitPanelLayoutV1,
    NormalizedRenderTargetRect,
    OrthographicProjection3D,
    Panel,
    PanelPlacement,
    PointVisual,
    TextVisual,
    View2D,
    View3D,
    VisualAttachment,
    VisualTransformBinding,
    full_target_panel_layout,
)


def _scene(**kwargs: object) -> Scene:
    views2d = kwargs.get("views2d", ())
    views3d = kwargs.get("views3d", ())
    views = (*views2d, *views3d)  # type: ignore[misc]
    panel_id = views[0].panel_id if views else "panel:main"
    panels = (Panel(id=panel_id),)
    visuals = kwargs.get("visuals", ())
    if "attachments" not in kwargs:
        kwargs["attachments"] = tuple(
            VisualAttachment(
                visual_id=visual.id,
                panel_id=panel_id,
                view_id=(
                    next(
                        (
                            view.id
                            for view in views
                            if isinstance(
                                view, View3D if visual.positions.shape[1] == 3 else View2D
                            )
                        ),
                        None,
                    )
                    if visual.coordinate_space is CoordinateSpace.DATA
                    else None
                ),
            )
            for visual in visuals  # type: ignore[union-attr]
        )
    return Scene(
        panels=panels,
        panel_layout=full_target_panel_layout(panel_id),
        **kwargs,  # type: ignore[arg-type]
    )


def _view2d() -> View2D:
    return View2D(id="view:2d", panel_id="panel:main")


def _view3d() -> View3D:
    return View3D(
        id="view:3d",
        panel_id="panel:main",
        camera=Camera3D(
            eye=(3.0, 3.0, 3.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 1.0),
        ),
        projection=OrthographicProjection3D(),
    )


def test_scene_rejects_both_active_views() -> None:
    with pytest.raises(ValueError, match="at most one primary data view"):
        _scene(id="scene:invalid", views2d=(_view2d(),), views3d=(_view3d(),))


def test_scene_accepts_viewless_ndc_or_one_active_view() -> None:
    assert _scene(id="scene:viewless").views2d == ()
    assert _scene(id="scene:2d", views2d=(_view2d(),)).views2d == (_view2d(),)
    assert _scene(id="scene:3d", views3d=(_view3d(),)).views3d == (_view3d(),)


def test_scene_accepts_disjoint_panel_views_and_routes_ordered_visuals() -> None:
    panels = (Panel("panel:left"), Panel("panel:right"))
    left_view = View2D("view:left", "panel:left")
    right_view = View2D("view:right", "panel:right")
    left = PointVisual(
        "visual:left",
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        colors=np.asarray([[255, 255, 255, 255]], dtype=np.uint8),
        coordinate_space=CoordinateSpace.DATA,
    )
    overlay = PointVisual(
        "visual:overlay",
        np.asarray([[0.5, 0.5]], dtype=np.float32),
        colors=np.asarray([[255, 255, 255, 255]], dtype=np.uint8),
        coordinate_space=CoordinateSpace.NDC,
    )
    right = PointVisual(
        "visual:right",
        np.asarray([[1.0, 1.0]], dtype=np.float32),
        colors=np.asarray([[255, 255, 255, 255]], dtype=np.uint8),
        coordinate_space=CoordinateSpace.DATA,
    )
    scene = Scene(
        id="scene:multi",
        panels=panels,
        panel_layout=ExplicitPanelLayoutV1(
            placements=(
                PanelPlacement("panel:left", NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
                PanelPlacement("panel:right", NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
            )
        ),
        visuals=(left, overlay, right),
        views2d=(left_view, right_view),
        attachments=(
            VisualAttachment("visual:left", "panel:left", "view:left", z_order=2),
            VisualAttachment("visual:overlay", "panel:left", None, z_order=1),
            VisualAttachment("visual:right", "panel:right", "view:right", visible=False),
        ),
    )

    assert scene.primary_view_for_panel("panel:left") is left_view
    assert scene.view("view:right") is right_view
    assert scene.attachment_for_visual("visual:overlay").view_id is None
    assert scene.visuals_for_panel("panel:left") == (overlay, left)
    assert scene.visuals_for_panel("panel:right") == ()
    assert scene.visuals_for_panel("panel:right", include_hidden=True) == (right,)


def test_scene_rejects_ambiguous_or_cross_panel_attachment_relationships() -> None:
    point = PointVisual(
        "visual:data",
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        colors=np.asarray([[255, 255, 255, 255]], dtype=np.uint8),
        coordinate_space=CoordinateSpace.DATA,
    )
    with pytest.raises(ValueError, match="exactly one attachment"):
        _scene(
            id="scene:duplicate-attachment",
            visuals=(point,),
            views2d=(_view2d(),),
            attachments=(
                VisualAttachment("visual:data", "panel:main", "view:2d"),
                VisualAttachment("visual:data", "panel:main", "view:2d"),
            ),
        )
    with pytest.raises(ValueError, match="viewless attachment"):
        overlay = PointVisual(
            "visual:ndc",
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            colors=np.asarray([[255, 255, 255, 255]], dtype=np.uint8),
            coordinate_space=CoordinateSpace.NDC,
        )
        _scene(
            id="scene:ndc-view",
            visuals=(overlay,),
            views2d=(_view2d(),),
            attachments=(VisualAttachment("visual:ndc", "panel:main", "view:2d"),),
        )

    other_view = View2D("view:other", "panel:other")
    with pytest.raises(ValueError, match="does not match view panel_id"):
        Scene(
            id="scene:cross-panel",
            panels=(Panel("panel:main"), Panel("panel:other")),
            panel_layout=ExplicitPanelLayoutV1(
                placements=(
                    PanelPlacement("panel:main", NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
                    PanelPlacement("panel:other", NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
                )
            ),
            visuals=(point,),
            views2d=(other_view,),
            attachments=(VisualAttachment("visual:data", "panel:main", "view:other"),),
        )


def test_scene_accepts_text_billboard3d_only_with_data_view3d_contract() -> None:
    billboard = TextVisual(
        id="visual:billboard",
        texts=("origin",),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
    )

    assert _scene(id="scene:billboard", visuals=(billboard,), views3d=(_view3d(),)).visuals == (
        billboard,
    )
    with pytest.raises(ValueError, match="requires an attachment view_id"):
        _scene(id="scene:no-view", visuals=(billboard,))

    ndc3 = TextVisual(
        id="visual:ndc3",
        texts=("invalid",),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.NDC,
    )
    with pytest.raises(ValueError, match="positions3d require CoordinateSpace.DATA"):
        _scene(id="scene:ndc3", visuals=(ndc3,), views3d=(_view3d(),))

    transformed = TextVisual(
        id="visual:transformed3",
        texts=("invalid",),
        positions=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        coordinate_space=CoordinateSpace.DATA,
        transform=VisualTransformBinding.from_ref("transform:invalid"),
    )
    with pytest.raises(ValueError, match="does not support a 2D visual transform"):
        _scene(id="scene:transform3", visuals=(transformed,), views3d=(_view3d(),))
