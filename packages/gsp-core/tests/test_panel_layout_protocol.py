from dataclasses import fields

import pytest

from gsp.protocol import (
    ClipScope,
    EXPLICIT_PANEL_LAYOUT_V1_KIND,
    ExplicitPanelLayoutV1,
    LogicalPixelRect,
    NormalizedRenderTargetRect,
    Panel,
    PanelPlacement,
    RenderTarget,
    ResolvedLayoutSnapshot,
    ResolvedPanelLayout,
    VisualAttachment,
    quantize_normalized_edge,
    resolve_panel_layout_intent,
    resolved_attachment_clip_rect,
)


def test_panel_is_identity_only() -> None:
    assert tuple(field.name for field in fields(Panel)) == ("id",)
    assert Panel(id="panel:main").id == "panel:main"
    with pytest.raises(TypeError):
        Panel(id="panel:main", figure_id="figure:main")  # type: ignore[call-arg]


def test_explicit_layout_resolves_with_canonical_quantizer() -> None:
    layout = ExplicitPanelLayoutV1(
        placements=(
            PanelPlacement(
                panel_id="panel:left",
                allocation_rect=NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0),
            ),
            PanelPlacement(
                panel_id="panel:right",
                allocation_rect=NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0),
            ),
        )
    )
    assert layout.kind == EXPLICIT_PANEL_LAYOUT_V1_KIND
    resolved = resolve_panel_layout_intent(layout, RenderTarget(801, 601))
    assert resolved[0].panel_rect_px == LogicalPixelRect(0, 0, 401, 601)
    assert resolved[1].panel_rect_px == LogicalPixelRect(401, 0, 400, 601)


@pytest.mark.parametrize(
    ("coordinate", "extent", "expected"),
    [(0.0, 5, 0), (0.1, 5, 1), (0.5, 5, 3), (0.9, 5, 5), (1.0, 5, 5)],
)
def test_quantize_normalized_edge_golden(coordinate: float, extent: int, expected: int) -> None:
    assert quantize_normalized_edge(coordinate, extent) == expected


def test_touching_is_allowed_but_overlap_is_rejected() -> None:
    touching = (
        PanelPlacement("panel:a", NormalizedRenderTargetRect(0.0, 0.0, 0.5, 1.0)),
        PanelPlacement("panel:b", NormalizedRenderTargetRect(0.5, 0.0, 0.5, 1.0)),
    )
    ExplicitPanelLayoutV1(touching)
    with pytest.raises(ValueError, match="overlap"):
        ExplicitPanelLayoutV1(
            (
                touching[0],
                PanelPlacement("panel:b", NormalizedRenderTargetRect(0.49, 0.0, 0.51, 1.0)),
            )
        )


def test_zero_resolved_area_fails_closed() -> None:
    layout = ExplicitPanelLayoutV1(
        (
            PanelPlacement(
                "panel:tiny",
                NormalizedRenderTargetRect(0.0, 0.0, 0.001, 1.0),
            ),
        )
    )
    with pytest.raises(ValueError, match="zero logical-pixel area"):
        resolve_panel_layout_intent(layout, RenderTarget(10, 10))


@pytest.mark.parametrize("scope", list(ClipScope))
def test_attachment_clip_scope_is_explicit_and_closed(scope: ClipScope) -> None:
    attachment = VisualAttachment(
        visual_id="visual:main",
        panel_id="panel:main",
        view_id="view:main",
        clip_scope=scope,
    )
    assert attachment.clip_scope is scope


def test_attachment_clip_scope_defaults_to_plot_and_rejects_unknown() -> None:
    attachment = VisualAttachment(
        visual_id="visual:main",
        panel_id="panel:main",
        view_id="view:main",
    )
    assert attachment.clip_scope is ClipScope.PLOT
    with pytest.raises(ValueError, match="clip_scope"):
        VisualAttachment(
            visual_id="visual:main",
            panel_id="panel:main",
            view_id="view:main",
            clip_scope="viewport",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("scope", "expected"),
    [
        (ClipScope.PLOT, LogicalPixelRect(30, 40, 100, 80)),
        (ClipScope.PANEL, LogicalPixelRect(10, 20, 160, 120)),
        (ClipScope.RENDER_TARGET, LogicalPixelRect(0, 0, 200, 150)),
    ],
)
def test_resolved_attachment_clip_rect_selects_exact_scope(
    scope: ClipScope, expected: LogicalPixelRect
) -> None:
    snapshot = ResolvedLayoutSnapshot(
        snapshot_id="layout:clip",
        render_target=RenderTarget(200, 150),
        panels=(
            ResolvedPanelLayout(
                panel_id="panel:main",
                panel_rect_px=LogicalPixelRect(10, 20, 160, 120),
                plot_rect_px=LogicalPixelRect(30, 40, 100, 80),
            ),
        ),
    )
    attachment = VisualAttachment(
        visual_id="visual:main",
        panel_id="panel:main",
        view_id="view:main",
        clip_scope=scope,
    )
    assert resolved_attachment_clip_rect(attachment, snapshot) == expected
