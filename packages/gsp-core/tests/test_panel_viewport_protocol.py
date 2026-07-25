import math

import pytest

from gsp.protocol import LogicalPixelRect, Panel, RenderTarget, resolve_panel_viewport_rect


def test_panel_viewport_intent_resolves_deterministically_to_outer_panel() -> None:
    panel = Panel(
        id="panel:inset",
        figure_id="figure:main",
        viewport_rect=(0.125, 0.25, 0.75, 0.5),
    )
    target = RenderTarget(logical_width_px=800.0, logical_height_px=600.0)

    assert resolve_panel_viewport_rect(panel, target) == LogicalPixelRect(
        100.0, 150.0, 600.0, 300.0
    )


@pytest.mark.parametrize(
    "viewport_rect",
    (
        (0.0, 0.0, 1.0, 1.0),
        (0.25, 0.5, 0.75, 0.5),
    ),
)
def test_panel_viewport_accepts_exact_right_and_bottom_edges(
    viewport_rect: tuple[float, float, float, float],
) -> None:
    assert Panel(
        id="panel:edge",
        figure_id="figure:main",
        viewport_rect=viewport_rect,
    ).viewport_rect == viewport_rect


@pytest.mark.parametrize("index", range(4))
@pytest.mark.parametrize("invalid", (math.nan, math.inf, -math.inf))
def test_panel_viewport_rejects_every_nonfinite_slot(index: int, invalid: float) -> None:
    values = [0.0, 0.0, 0.5, 0.5]
    values[index] = invalid

    with pytest.raises(ValueError, match=rf"viewport_rect\[{index}\] must be finite"):
        Panel(
            id="panel:invalid",
            figure_id="figure:main",
            viewport_rect=tuple(values),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("viewport_rect", "message"),
    (
        ((-0.1, 0.0, 0.5, 0.5), "origin must be non-negative"),
        ((0.0, -0.1, 0.5, 0.5), "origin must be non-negative"),
        ((0.0, 0.0, -0.1, 0.5), "width and height must be positive"),
        ((0.0, 0.0, 0.5, -0.1), "width and height must be positive"),
        ((0.0, 0.0, 0.0, 0.5), "width and height must be positive"),
        ((0.0, 0.0, 0.5, 0.0), "width and height must be positive"),
        ((0.75, 0.0, 0.5, 0.5), "contained by the normalized render target"),
        ((0.0, 0.75, 0.5, 0.5), "contained by the normalized render target"),
    ),
)
def test_panel_viewport_rejects_invalid_extent_or_overflow(
    viewport_rect: tuple[float, float, float, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Panel(
            id="panel:invalid",
            figure_id="figure:main",
            viewport_rect=viewport_rect,
        )


def test_resolve_panel_viewport_rect_validates_protocol_types() -> None:
    panel = Panel(id="panel:main", figure_id="figure:main")
    target = RenderTarget(800.0, 600.0)

    with pytest.raises(TypeError, match="panel must be a Panel"):
        resolve_panel_viewport_rect(object(), target)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="render_target must be a RenderTarget"):
        resolve_panel_viewport_rect(panel, object())  # type: ignore[arg-type]
