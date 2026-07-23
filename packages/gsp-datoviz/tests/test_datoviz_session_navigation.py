from __future__ import annotations

from typing import Any

import numpy as np

import gsp
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    MeshVisual,
    OrthographicProjection3D,
    View2D,
    View3D,
)
from gsp_datoviz.session import DatovizSession


class _FakeRenderer:
    def __init__(self, view: View2D) -> None:
        self.view = view
        self.enable_calls: list[View2D | None] = []
        self.show_calls: list[int] = []
        self.closed = False

    def enable_gsp_view2d_navigation(self, view: View2D | None = None) -> object:
        self.enable_calls.append(view)
        return object()

    def show(self, *, frame_count: int) -> None:
        self.show_calls.append(frame_count)

    def close(self) -> None:
        self.closed = True


def _session(renderer: _FakeRenderer) -> DatovizSession:
    session = object.__new__(DatovizSession)
    session.request = None  # type: ignore[assignment]
    session._dvz = object()
    session.capabilities = None  # type: ignore[assignment]
    session._diagnostics = []
    session._renderers = []
    session._renderer_scenes = {}
    session._interactive_view2d_renderers = set()
    session._closed = False
    session._build_renderer = lambda scene: renderer  # type: ignore[method-assign]
    return session


def _scene() -> gsp.Scene:
    return gsp.Scene(
        id="scene:live-view2d",
        view2d=View2D(
            id="view:main",
            panel_id="panel:main",
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        ),
    )


def _mesh3d_scene() -> gsp.Scene:
    return gsp.Scene(
        id="scene:mesh3d",
        visuals=(
            MeshVisual(
                id="visual:mesh3d",
                positions=np.asarray(
                    [[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
                color=np.asarray([70, 130, 220, 255], dtype=np.uint8),
                coordinate_space=CoordinateSpace.DATA,
            ),
        ),
        view3d=View3D(
            id="view:main",
            panel_id="panel:main",
            camera=Camera3D(
                eye=(3.0, 3.0, 3.0),
                target=(0.0, 0.0, 0.0),
                up=(0.0, 0.0, 1.0),
            ),
            projection=OrthographicProjection3D(),
        ),
    )


def test_interactive_2d_display_enables_canonical_navigation_exactly_once() -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    session = _session(renderer)

    assert session.display(scene, block=False) is renderer  # type: ignore[comparison-overlap]
    session.run()

    assert renderer.enable_calls == [scene.view2d]
    assert renderer.show_calls == [0]

    session.close()
    assert renderer.closed


def test_run_after_noninteractive_render_enables_navigation_once() -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    session = _session(renderer)

    session.render(scene)
    assert renderer.enable_calls == []

    session.run()

    assert renderer.enable_calls == [scene.view2d]


def test_offscreen_render_does_not_enable_interactive_navigation(tmp_path: Any) -> None:
    scene = _scene()
    assert scene.view2d is not None
    renderer = _FakeRenderer(scene.view2d)
    renderer.capture_png_bytes = lambda: b"png"  # type: ignore[attr-defined]
    session = _session(renderer)

    session.render(scene, target=tmp_path / "frame.png")

    assert renderer.enable_calls == []


def test_offscreen_render_accepts_static_view3d_mesh_scene(tmp_path: Any) -> None:
    scene = _mesh3d_scene()
    assert scene.view3d is not None
    renderer = _FakeRenderer(scene.view3d)  # type: ignore[arg-type]
    renderer.capture_png_bytes = lambda: b"png"  # type: ignore[attr-defined]
    session = _session(renderer)
    target = tmp_path / "mesh3d.png"

    session.render(scene, target=target)

    assert target.read_bytes() == b"png"
    assert renderer.enable_calls == []
