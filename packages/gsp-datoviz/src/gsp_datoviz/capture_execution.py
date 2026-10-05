"""Native view ownership and offscreen capture execution."""

from __future__ import annotations

import ctypes
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gsp_datoviz.capabilities import (
    datoviz_v04_capture_diagnostics,
    datoviz_v04_capture_ready,
)

from ._state import (
    DatovizV04Unavailable,
    DatovizV04Unsupported,
)
from .capture import (
    _encode_rgba8_png,
)
from .native_api import (
    _is_null_handle,
    _require_datoviz_success,
)

if TYPE_CHECKING:
    from .protocol_renderer import DatovizV04ProtocolRenderer


def capture_png_bytes(self: DatovizV04ProtocolRenderer) -> bytes:
    """Render one offscreen frame and return PNG screenshot/export bytes."""
    if not datoviz_v04_capture_ready(self.dvz):
        diagnostics = ", ".join(datoviz_v04_capture_diagnostics(self.dvz))
        raise DatovizV04Unavailable(f"Datoviz offscreen PNG capture is unavailable: {diagnostics}")

    view = self._ensure_offscreen_view()
    self._render_offscreen_frame()

    if all(hasattr(self.dvz, name) for name in ("dvz_view_canvas", "dvz_canvas_capture_rgba_into")):
        width = self.resolved_canvas.framebuffer_width
        height = self.resolved_canvas.framebuffer_height
        byte_count = width * height * 4
        rgba = (ctypes.c_uint8 * byte_count)()
        canvas = self.dvz.dvz_view_canvas(view)
        if _is_null_handle(canvas):
            raise DatovizV04Unavailable("Datoviz offscreen canvas lookup failed")
        _require_datoviz_success(
            self.dvz.dvz_canvas_capture_rgba_into(
                canvas,
                width,
                height,
                rgba,
                byte_count,
            ),
            "Datoviz offscreen RGBA capture failed",
        )
        return _encode_rgba8_png(width, height, bytes(rgba))

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as file:
            path = Path(file.name)
        _require_datoviz_success(
            self.dvz.dvz_view_capture_png(view, str(path).encode()),
            "Datoviz offscreen PNG capture failed",
        )
        return path.read_bytes()
    finally:
        if path is not None:
            path.unlink(missing_ok=True)


def _ensure_offscreen_view(self: DatovizV04ProtocolRenderer) -> Any:
    """Create the lazy offscreen app/view pair used by PNG capture."""
    if self.offscreen_view is not None:
        return self.offscreen_view

    self._ensure_app("offscreen")
    self.offscreen_view = self.dvz.dvz_view_offscreen(
        self.app,
        self.figure,
        self.resolved_canvas.framebuffer_width,
        self.resolved_canvas.framebuffer_height,
    )
    if _is_null_handle(self.offscreen_view):
        raise DatovizV04Unavailable("Datoviz offscreen view creation failed")
    return self.offscreen_view


def _ensure_live_view(self: DatovizV04ProtocolRenderer) -> Any:
    """Create the lazy interactive app/view pair used by show()."""
    if self.live_view is not None:
        return self.live_view

    self._ensure_app("interactive")
    view_window = getattr(self.dvz, "dvz_view_window", None)
    if view_window is not None:
        self.live_view = view_window(
            self.app,
            self.figure,
            self.resolved_canvas.host_logical_width,
            self.resolved_canvas.host_logical_height,
            b"GSP Datoviz review",
        )
        if _is_null_handle(self.live_view):
            raise DatovizV04Unavailable("Datoviz interactive window view creation failed")
        return self.live_view

    view_glfw = getattr(self.dvz, "dvz_view_glfw", None)
    if view_glfw is not None:
        self.live_view = view_glfw(
            self.app,
            self.figure,
            self.resolved_canvas.host_logical_width,
            self.resolved_canvas.host_logical_height,
            b"GSP Datoviz review",
        )
        if _is_null_handle(self.live_view):
            raise DatovizV04Unavailable("Datoviz interactive GLFW view creation failed")
        return self.live_view

    view = getattr(self.dvz, "dvz_view", None)
    if view is None:
        raise DatovizV04Unavailable("Datoviz interactive view is unavailable: missing dvz_view")
    self.live_view = view(self.app, self.figure, None)
    if _is_null_handle(self.live_view):
        raise DatovizV04Unavailable("Datoviz interactive view creation failed")
    return self.live_view


def _ensure_app(self: DatovizV04ProtocolRenderer, purpose: str) -> Any:
    """Create the lazy Datoviz app shared by live and offscreen views."""
    if self.app is not None:
        return self.app
    app = getattr(self.dvz, "dvz_app", None)
    if app is None:
        raise DatovizV04Unavailable(
            f"Datoviz {purpose} app creation is unavailable: missing dvz_app"
        )
    self.app = app(self.scene)
    if _is_null_handle(self.app):
        raise DatovizV04Unavailable(f"Datoviz {purpose} app creation failed")
    return self.app


def _render_offscreen_frame(self: DatovizV04ProtocolRenderer) -> None:
    view_render_once = getattr(self.dvz, "dvz_view_render_once", None)
    if view_render_once is not None:
        result = view_render_once(self.offscreen_view)
    else:
        app_render_once = getattr(self.dvz, "dvz_app_render_once", None)
        if app_render_once is not None:
            result = app_render_once(self.app)
        else:
            result = self.dvz.dvz_app_run(self.app, 1)
    frame_ready = getattr(self.dvz, "DVZ_CANVAS_FRAME_READY", 0)
    if result not in (0, None, frame_ready):
        raise DatovizV04Unsupported("Datoviz offscreen frame render failed")
