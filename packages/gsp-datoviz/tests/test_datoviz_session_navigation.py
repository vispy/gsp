from __future__ import annotations

from typing import Any

import numpy as np

import gsp
from gsp.protocol import (
    Camera3D,
    CoordinateSpace,
    MeshVisual,
    OrthographicProjection3D,
    PointVisual,
    TextVisual,
    VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY,
    View2D,
    View3D,
)
from gsp_datoviz.session import DatovizSession
import gsp_datoviz.session as session_module


class _FakeRenderer:
    def __init__(self, view: View2D | View3D) -> None:
        self.view = view
        self.enable_calls: list[View2D | None] = []
        self.enable_view3d_calls: list[View3D | None] = []
        self.show_calls: list[int] = []
        self.closed = False

    def enable_gsp_view2d_navigation(self, view: View2D | None = None) -> object:
        self.enable_calls.append(view)
        return object()

    def enable_gsp_view3d_navigation(self, view: View3D | None = None) -> object:
        self.enable_view3d_calls.append(view)
        return object()

    def show(self, *, frame_count: int) -> None:
        self.show_calls.append(frame_count)

    def close(self) -> None:
        self.closed = True


class _FakeCapabilities:
    def __init__(self, *, live_view3d: bool = False) -> None:
        self.live_view3d = live_view3d

    def supports_view3d_capability(self, capability: str) -> bool:
        return (
            capability == VIEW3D_NAVIGATION_ORBIT_PAN_ZOOM_CAPABILITY
            and self.live_view3d
        )


def _session(
    renderer: _FakeRenderer, *, live_view3d: bool = False
) -> DatovizSession:
    session = object.__new__(DatovizSession)
    session.request = None  # type: ignore[assignment]
    session._dvz = object()  # type: ignore[assignment]
    session.capabilities = _FakeCapabilities(  # type: ignore[assignment]
        live_view3d=live_view3d
    )
    session._diagnostics = []
    session._renderers = []
    session._renderer_scenes = {}
    session._interactive_view2d_renderers = set()
    session._interactive_view3d_renderers = set()
    session._closed = False
    session._build_renderer = lambda scene: renderer  # type: ignore[assignment,method-assign,return-value]
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
    renderer = _FakeRenderer(scene.view3d)
    renderer.capture_png_bytes = lambda: b"png"  # type: ignore[attr-defined]
    session = _session(renderer, live_view3d=True)
    target = tmp_path / "mesh3d.png"

    session.render(scene, target=target)

    assert target.read_bytes() == b"png"
    assert renderer.enable_calls == []
    assert renderer.enable_view3d_calls == []


def test_interactive_view3d_is_enabled_only_from_advertised_session_capability() -> None:
    scene = _mesh3d_scene()
    assert scene.view3d is not None

    static_renderer = _FakeRenderer(scene.view3d)
    static_session = _session(static_renderer)
    static_session.display(scene, block=False)
    static_session.run()
    assert static_renderer.enable_view3d_calls == []

    live_renderer = _FakeRenderer(scene.view3d)
    live_session = _session(live_renderer, live_view3d=True)
    live_session.display(scene, block=False)
    live_session.run()

    assert live_renderer.enable_view3d_calls == [scene.view3d]
    assert live_renderer.show_calls == [0]


def test_scene_emits_geometry_then_stably_z_ordered_overlay_text(
    monkeypatch: Any,
) -> None:
    calls: list[tuple[str, str]] = []

    class RecordingRenderer:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def add_point_visual(self, visual: PointVisual) -> None:
            calls.append(("point", visual.id))

        def add_text_visual(self, visual: TextVisual) -> None:
            calls.append(("text", visual.id))

        def close(self) -> None:
            pass

    monkeypatch.setattr(
        session_module, "DatovizV04ProtocolRenderer", RecordingRenderer
    )
    session = object.__new__(DatovizSession)
    session._dvz = object()  # type: ignore[assignment]
    scene = gsp.Scene(
        id="scene:text-order",
        visuals=(
            TextVisual(
                id="text:z5",
                texts=("z5",),
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                coordinate_space=CoordinateSpace.NDC,
                z_order=5,
            ),
            PointVisual(
                id="point:geometry",
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                colors=np.array([[255, 255, 255, 255]], dtype=np.uint8),
                coordinate_space=CoordinateSpace.NDC,
            ),
            TextVisual(
                id="text:z1-first",
                texts=("first",),
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                coordinate_space=CoordinateSpace.NDC,
                z_order=1,
            ),
            TextVisual(
                id="text:z1-second",
                texts=("second",),
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                coordinate_space=CoordinateSpace.NDC,
                z_order=1,
            ),
        ),
    )

    session._build_renderer(scene)

    assert calls == [
        ("point", "point:geometry"),
        ("text", "text:z1-first"),
        ("text", "text:z1-second"),
        ("text", "text:z5"),
    ]
