"""Native input translation and retained navigation bindings."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
from gsp.protocol import (
    LogicalPixelRect,
    NavigationDiagnosticCode,
    NavigationPointerEvent,
    NavigationPointerEventKind,
    NavigationResult,
    OrthographicProjection3D,
    PanByAction,
    PixelOrigin,
    ResetViewAction,
    ResolvedLayoutSnapshot,
    SetViewAction,
    View2D,
    View2DNavigationController,
    View2DNavigationInputAdapter,
    View3D,
    ZoomAboutAction,
    pan_view2d,
    resolve_view3d_projection_snapshot,
    zoom_view2d_about,
)
from gsp.protocol.view3d import (
    Orbit3DPayload,
    Pan3DPayload,
    ResetView3DPayload,
    View3DNavigationAction,
    View3DNavigationActionKind,
    Zoom3DPayload,
)

from ._state import (
    DVZ_INPUT_EVENT_POINTER,
    DVZ_INPUT_EVENT_RESIZE,
    DVZ_INPUT_EVENT_SCALE,
    DVZ_POINTER_BUTTON_LEFT,
    DVZ_POINTER_BUTTON_RIGHT,
    DVZ_POINTER_EVENT_DOUBLE_CLICK,
    DVZ_POINTER_EVENT_DRAG,
    DVZ_POINTER_EVENT_DRAG_STOP,
    DVZ_POINTER_EVENT_MOVE,
    DVZ_POINTER_EVENT_PRESS,
    DVZ_POINTER_EVENT_RELEASE,
    DVZ_POINTER_EVENT_WHEEL,
    NavigationAction,
)
from .native_api import (
    _enum_value,
)

if TYPE_CHECKING:
    from .protocol_renderer import DatovizV04ProtocolRenderer


class _DatovizLiveView2DNavigation:
    """Translate Datoviz pointer callbacks into S035 semantic navigation."""

    def __init__(
        self,
        *,
        renderer: DatovizV04ProtocolRenderer,
        router: Any,
        live_view: Any,
        view: View2D,
        controller_id: str,
        layout_snapshot_id: str,
    ) -> None:
        self.renderer = renderer
        self.router = router
        self.live_view = live_view
        self.view = view
        self.revision_index = 1
        self.layout_snapshot_id = layout_snapshot_id
        self.controller = View2DNavigationController(
            id=controller_id,
            panel_id=view.panel_id,
            view_id=view.id,
            current_view2d_revision="view-rev:datoviz-live-1",
            home_view=view,
        )
        consumed_layout = cast(
            ResolvedLayoutSnapshot | None,
            getattr(renderer, "consumed_layout_snapshot", None),
        )
        self._panel_rect = self._initial_panel_rect()
        if consumed_layout is not None:
            self.adapter = View2DNavigationInputAdapter(
                controller_id=self.controller.id,
                view2d_revision=self.controller.current_view2d_revision,
                layout_snapshot=consumed_layout,
            )
        else:
            self.adapter = View2DNavigationInputAdapter(
                controller_id=self.controller.id,
                view2d_revision=self.controller.current_view2d_revision,
                panel_rect=self.panel_rect,
                layout_snapshot_id=layout_snapshot_id,
            )
        self.subscription_id: Any = None
        self._closed = False

    @property
    def panel_rect(self) -> LogicalPixelRect:
        """Return the live Datoviz panel rectangle in host logical pixels."""
        return self._panel_rect

    def _initial_panel_rect(self) -> LogicalPixelRect:
        """Return the initial live Datoviz panel rectangle in host logical pixels."""
        consumed_layout = cast(
            ResolvedLayoutSnapshot | None,
            getattr(self.renderer, "consumed_layout_snapshot", None),
        )
        if consumed_layout is not None:
            return consumed_layout.only_panel().plot_rect_px
        return LogicalPixelRect(
            x=0.0,
            y=0.0,
            width=float(self.renderer.resolved_canvas.host_logical_width),
            height=float(self.renderer.resolved_canvas.host_logical_height),
        )

    def close(self) -> None:
        """Unsubscribe from Datoviz pointer callbacks."""
        if self._closed:
            return
        unsubscribe = getattr(self.renderer.dvz, "dvz_input_unsubscribe", None)
        if unsubscribe is not None and self.subscription_id is not None:
            unsubscribe(self.router, self.subscription_id)
        else:
            unsubscribe_event = getattr(self.renderer.dvz, "dvz_input_unsubscribe_event", None)
            if unsubscribe_event is not None:
                unsubscribe_event(self.router, self.handle_input_event, None)
        self._closed = True

    def handle_input_event(self, _router: Any, event_ptr: Any, _user_data: Any) -> None:
        """Handle one routed Datoviz input callback."""
        activate_panel = getattr(self.renderer, "activate_panel", None)
        if activate_panel is not None:
            activate_panel(self.view.panel_id)
        input_event = getattr(event_ptr, "contents", event_ptr)
        input_event_type = int(getattr(input_event, "type"))
        if input_event_type == _enum_value(
            self.renderer.dvz,
            "DvzInputEventType",
            "DVZ_INPUT_EVENT_POINTER",
            DVZ_INPUT_EVENT_POINTER,
        ):
            pointer_event = getattr(input_event.content, "pointer")
            self.handle_pointer_event(_router, pointer_event, _user_data)
            return
        if input_event_type == _enum_value(
            self.renderer.dvz,
            "DvzInputEventType",
            "DVZ_INPUT_EVENT_RESIZE",
            DVZ_INPUT_EVENT_RESIZE,
        ):
            self.handle_resize_event(getattr(input_event.content, "resize"))
            return
        if input_event_type == _enum_value(
            self.renderer.dvz,
            "DvzInputEventType",
            "DVZ_INPUT_EVENT_SCALE",
            DVZ_INPUT_EVENT_SCALE,
        ):
            self._refresh_view_after_viewport_change()

    def handle_pointer_event(self, _router: Any, event_ptr: Any, _user_data: Any) -> None:
        """Handle one raw Datoviz pointer callback."""
        event = getattr(event_ptr, "contents", event_ptr)
        consumed_layout = cast(
            ResolvedLayoutSnapshot | None,
            getattr(self.renderer, "consumed_layout_snapshot", None),
        )
        pointer_event = _navigation_pointer_event_from_datoviz(
            self.renderer.dvz,
            event,
            pixel_origin=(
                consumed_layout.render_target.pixel_origin if consumed_layout is not None else None
            ),
        )
        if pointer_event is None:
            return
        self._apply_event(pointer_event)

    def handle_resize_event(self, event: Any) -> None:
        """Update the live logical panel size from one Datoviz resize event."""
        if getattr(self.renderer, "consumed_layout_snapshot", None) is not None:
            request_frame = getattr(self.renderer.dvz, "dvz_view_request_frame", None)
            if request_frame is not None:
                request_frame(self.live_view)
            return
        width, height = _datoviz_resize_logical_size(event)
        if width <= 0.0 or height <= 0.0:
            return
        self._panel_rect = LogicalPixelRect(x=0.0, y=0.0, width=width, height=height)
        self.adapter.set_panel_rect(self._panel_rect)
        self._refresh_view_after_viewport_change()

    def _apply_event(self, event: NavigationPointerEvent) -> None:
        consumed_layout = cast(
            ResolvedLayoutSnapshot | None,
            getattr(self.renderer, "consumed_layout_snapshot", None),
        )
        if consumed_layout is None:
            self.adapter.set_panel_rect(self.panel_rect)
        action = self.adapter.handle_pointer_event(event)
        if action is None:
            return
        result = _apply_view2d_navigation_action(
            self.controller,
            self.view,
            self.adapter.panel_rect,
            action,
            next_view2d_revision=self._next_revision(),
            view_snapshot_id=f"view-snapshot:datoviz-live-{self.revision_index}",
            expected_layout_snapshot_id=self.layout_snapshot_id,
            layout_snapshot=consumed_layout,
        )
        if not result.accepted or result.view is None or result.new_view2d_revision is None:
            return
        self.view = result.view
        self.controller = replace(
            self.controller, current_view2d_revision=result.new_view2d_revision
        )
        self.adapter.accept_navigation_result(result)
        self.renderer.apply_retained_view2d_navigation(result.view)
        request_frame = getattr(self.renderer.dvz, "dvz_view_request_frame", None)
        if request_frame is not None:
            request_frame(self.live_view)

    def _refresh_view_after_viewport_change(self) -> None:
        self.renderer.update_view2d_axes(self.view)
        request_frame = getattr(self.renderer.dvz, "dvz_view_request_frame", None)
        if request_frame is not None:
            request_frame(self.live_view)

    def _next_revision(self) -> str:
        self.revision_index += 1
        return f"view-rev:datoviz-live-{self.revision_index}"


class _DatovizLiveView3DNavigation:
    """Translate Datoviz pointer callbacks into S037 semantic View3D navigation."""

    def __init__(
        self,
        *,
        renderer: DatovizV04ProtocolRenderer,
        router: Any,
        live_view: Any,
        view3d: View3D,
        controller_id: str,
        layout_snapshot_id: str,
    ) -> None:
        self.renderer = renderer
        self.router = router
        self.live_view = live_view
        self.view3d = view3d
        self.home_view3d = view3d
        self.controller_id = controller_id
        self.layout_snapshot_id = layout_snapshot_id
        self._panel_rect = self._initial_panel_rect()
        self._drag_last_px: tuple[float, float] | None = None
        self._drag_mode: Literal["orbit", "pan"] | None = None
        self.subscription_id: Any = None
        self._closed = False

    @property
    def panel_rect(self) -> LogicalPixelRect:
        """Return the live Datoviz panel rectangle in host logical pixels."""
        return self._panel_rect

    def _initial_panel_rect(self) -> LogicalPixelRect:
        """Return the initial live Datoviz panel rectangle in host logical pixels."""
        consumed_layout = cast(
            ResolvedLayoutSnapshot | None,
            getattr(self.renderer, "consumed_layout_snapshot", None),
        )
        if consumed_layout is not None:
            return consumed_layout.only_panel().plot_rect_px
        return LogicalPixelRect(
            x=0.0,
            y=0.0,
            width=float(self.renderer.resolved_canvas.host_logical_width),
            height=float(self.renderer.resolved_canvas.host_logical_height),
        )

    def close(self) -> None:
        """Unsubscribe from Datoviz pointer callbacks."""
        if self._closed:
            return
        unsubscribe = getattr(self.renderer.dvz, "dvz_input_unsubscribe", None)
        if unsubscribe is not None and self.subscription_id is not None:
            unsubscribe(self.router, self.subscription_id)
        else:
            unsubscribe_event = getattr(self.renderer.dvz, "dvz_input_unsubscribe_event", None)
            if unsubscribe_event is not None:
                unsubscribe_event(self.router, self.handle_input_event, None)
        self._closed = True

    def handle_input_event(self, _router: Any, event_ptr: Any, _user_data: Any) -> None:
        """Handle one routed Datoviz input callback."""
        activate_panel = getattr(self.renderer, "activate_panel", None)
        if activate_panel is not None:
            activate_panel(self.view3d.panel_id)
        input_event = getattr(event_ptr, "contents", event_ptr)
        input_event_type = int(getattr(input_event, "type"))
        if input_event_type == _enum_value(
            self.renderer.dvz,
            "DvzInputEventType",
            "DVZ_INPUT_EVENT_POINTER",
            DVZ_INPUT_EVENT_POINTER,
        ):
            pointer_event = getattr(input_event.content, "pointer")
            self.handle_pointer_event(_router, pointer_event, _user_data)
            return
        if input_event_type == _enum_value(
            self.renderer.dvz,
            "DvzInputEventType",
            "DVZ_INPUT_EVENT_RESIZE",
            DVZ_INPUT_EVENT_RESIZE,
        ):
            self.handle_resize_event(getattr(input_event.content, "resize"))
            return
        if input_event_type == _enum_value(
            self.renderer.dvz,
            "DvzInputEventType",
            "DVZ_INPUT_EVENT_SCALE",
            DVZ_INPUT_EVENT_SCALE,
        ):
            self._request_frame()

    def handle_pointer_event(self, _router: Any, event_ptr: Any, _user_data: Any) -> None:
        """Handle one raw Datoviz pointer callback."""
        event = getattr(event_ptr, "contents", event_ptr)
        consumed_layout = getattr(self.renderer, "consumed_layout_snapshot", None)
        pointer_event = _navigation_pointer_event_from_datoviz(
            self.renderer.dvz,
            event,
            pixel_origin=(
                consumed_layout.render_target.pixel_origin if consumed_layout is not None else None
            ),
        )
        if pointer_event is None:
            return
        self._apply_event(pointer_event)

    def handle_resize_event(self, event: Any) -> None:
        """Update the live logical panel size from one Datoviz resize event."""
        if getattr(self.renderer, "consumed_layout_snapshot", None) is not None:
            self._request_frame()
            return
        width, height = _datoviz_resize_logical_size(event)
        if width <= 0.0 or height <= 0.0:
            return
        self._panel_rect = LogicalPixelRect(x=0.0, y=0.0, width=width, height=height)
        self._request_frame()

    def _apply_event(self, event: NavigationPointerEvent) -> None:
        inside_panel = (
            self.panel_rect.x <= event.x_px <= self.panel_rect.x + self.panel_rect.width
            and self.panel_rect.y <= event.y_px <= self.panel_rect.y + self.panel_rect.height
        )
        if event.kind is NavigationPointerEventKind.BUTTON_PRESS:
            if not inside_panel:
                return
            if event.left_button:
                self._drag_mode = "orbit"
            elif event.right_button:
                self._drag_mode = "pan"
            else:
                return
            self._drag_last_px = (event.x_px, event.y_px)
            return
        if event.kind is NavigationPointerEventKind.BUTTON_RELEASE:
            self._drag_last_px = None
            self._drag_mode = None
            return
        if event.kind is NavigationPointerEventKind.MOUSE_MOVE:
            self._apply_drag_event(event)
            return
        if event.kind is NavigationPointerEventKind.WHEEL and event.scroll_steps != 0.0:
            if not inside_panel:
                return
            self._apply_payload(
                View3DNavigationActionKind.ZOOM,
                Zoom3DPayload(scale=1.1**event.scroll_steps),
            )
            return
        if event.kind is NavigationPointerEventKind.DOUBLE_CLICK:
            if not inside_panel:
                return
            self._apply_payload(
                View3DNavigationActionKind.RESET,
                ResetView3DPayload(
                    camera=self.home_view3d.camera,
                    projection=self.home_view3d.projection,
                ),
            )

    def _apply_drag_event(self, event: NavigationPointerEvent) -> None:
        if self._drag_last_px is None or self._drag_mode is None:
            return
        last_x, last_y = self._drag_last_px
        self._drag_last_px = (event.x_px, event.y_px)
        dx_px = event.x_px - last_x
        dy_px = event.y_px - last_y
        if dx_px == 0.0 and dy_px == 0.0:
            return
        if self._drag_mode == "orbit":
            self._apply_payload(
                View3DNavigationActionKind.ORBIT,
                self._orbit_payload_from_pixels(dx_px, dy_px),
            )
        else:
            self._apply_payload(
                View3DNavigationActionKind.PAN,
                self._pan_payload_from_pixels(dx_px, dy_px),
            )

    def _apply_payload(
        self,
        kind: View3DNavigationActionKind,
        payload: Orbit3DPayload | Pan3DPayload | Zoom3DPayload | ResetView3DPayload,
    ) -> None:
        snapshot = resolve_view3d_projection_snapshot(
            self.view3d,
            layout_snapshot=getattr(self.renderer, "consumed_layout_snapshot", None),
            layout_snapshot_id=self.layout_snapshot_id,
        )
        action = View3DNavigationAction(
            kind=kind,
            view_id=self.view3d.id,
            base_view_revision=self.view3d.revision,
            base_view_projection_snapshot_id=snapshot.view_projection_snapshot_id,
            payload=payload,
            base_layout_snapshot_id=self.layout_snapshot_id,
        )
        result = self.renderer.apply_gsp_view3d_navigation_action(
            action, layout_snapshot_id=self.layout_snapshot_id
        )
        if not result.accepted or result.view is None:
            return
        self.view3d = result.view
        self._request_frame()

    def _orbit_payload_from_pixels(self, dx_px: float, dy_px: float) -> Orbit3DPayload:
        rect = self.panel_rect
        return Orbit3DPayload(
            delta_yaw_radians=-dx_px / rect.width * np.pi,
            delta_pitch_radians=-dy_px / rect.height * np.pi,
        )

    def _pan_payload_from_pixels(self, dx_px: float, dy_px: float) -> Pan3DPayload:
        rect = self.panel_rect
        consumed_layout = getattr(self.renderer, "consumed_layout_snapshot", None)
        projection = self.view3d.projection
        if isinstance(projection, OrthographicProjection3D):
            x_span = projection.xlim[1] - projection.xlim[0]
            y_span = projection.ylim[1] - projection.ylim[0]
        else:
            basis = self.view3d.camera.basis()
            target_distance = max(
                sum(
                    (target - eye) * forward
                    for target, eye, forward in zip(
                        self.view3d.camera.target,
                        self.view3d.camera.eye,
                        basis.forward,
                        strict=True,
                    )
                ),
                projection.near_far[0],
            )
            aspect_ratio = projection.aspect_ratio or (rect.width / rect.height)
            half_height = target_distance * math.tan(math.radians(projection.fov_y_degrees) * 0.5)
            x_span = 2.0 * half_height * aspect_ratio
            y_span = 2.0 * half_height
        return Pan3DPayload(
            delta_view_right=-dx_px / rect.width * x_span,
            delta_view_up=(
                dy_px / rect.height * y_span
                if (
                    consumed_layout is not None
                    and consumed_layout.render_target.pixel_origin is PixelOrigin.TOP_LEFT
                )
                else -dy_px / rect.height * y_span
            ),
        )

    def _request_frame(self) -> None:
        request_frame = getattr(self.renderer.dvz, "dvz_view_request_frame", None)
        if request_frame is not None:
            request_frame(self.live_view)


def _navigation_pointer_event_from_datoviz(
    dvz: Any,
    event: Any,
    *,
    pixel_origin: PixelOrigin | None = None,
) -> NavigationPointerEvent | None:
    event_type = int(getattr(event, "type"))
    x_px = float(event.pos[0])
    y_px = _datoviz_pointer_y_to_gsp_logical_px(event, pixel_origin=pixel_origin)
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_PRESS",
        DVZ_POINTER_EVENT_PRESS,
    ):
        left_button = int(getattr(event, "button", 0)) == _enum_value(
            dvz,
            "DvzPointerButton",
            "DVZ_POINTER_BUTTON_LEFT",
            DVZ_POINTER_BUTTON_LEFT,
        )
        right_button = int(getattr(event, "button", 0)) == _enum_value(
            dvz,
            "DvzPointerButton",
            "DVZ_POINTER_BUTTON_RIGHT",
            DVZ_POINTER_BUTTON_RIGHT,
        )
        return NavigationPointerEvent(
            NavigationPointerEventKind.BUTTON_PRESS,
            x_px,
            y_px,
            left_button=left_button,
            right_button=right_button,
        )
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_RELEASE",
        DVZ_POINTER_EVENT_RELEASE,
    ):
        return NavigationPointerEvent(NavigationPointerEventKind.BUTTON_RELEASE, x_px, y_px)
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_MOVE",
        DVZ_POINTER_EVENT_MOVE,
    ):
        return None
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_DRAG",
        DVZ_POINTER_EVENT_DRAG,
    ):
        return NavigationPointerEvent(NavigationPointerEventKind.MOUSE_MOVE, x_px, y_px)
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_DRAG_STOP",
        DVZ_POINTER_EVENT_DRAG_STOP,
    ):
        return NavigationPointerEvent(NavigationPointerEventKind.BUTTON_RELEASE, x_px, y_px)
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_WHEEL",
        DVZ_POINTER_EVENT_WHEEL,
    ):
        return NavigationPointerEvent(
            NavigationPointerEventKind.WHEEL,
            x_px,
            y_px,
            scroll_steps=float(event.content.w.dir[1]),
        )
    if event_type == _enum_value(
        dvz,
        "DvzPointerEventType",
        "DVZ_POINTER_EVENT_DOUBLE_CLICK",
        DVZ_POINTER_EVENT_DOUBLE_CLICK,
    ):
        return NavigationPointerEvent(
            NavigationPointerEventKind.DOUBLE_CLICK,
            x_px,
            y_px,
        )
    return None


def _datoviz_pointer_y_to_gsp_logical_px(
    event: Any, *, pixel_origin: PixelOrigin | None = None
) -> float:
    if pixel_origin is PixelOrigin.TOP_LEFT:
        return float(event.pos[1])
    window_size = getattr(event, "window_size", None)
    if window_size is not None:
        height = float(window_size[1])
        if height > 0.0:
            return height - float(event.pos[1])
    return float(event.pos[1])


def _datoviz_resize_logical_size(event: Any) -> tuple[float, float]:
    width = float(getattr(event, "window_width", 0))
    height = float(getattr(event, "window_height", 0))
    if width > 0.0 and height > 0.0:
        return width, height

    framebuffer_width = float(getattr(event, "framebuffer_width", 0))
    framebuffer_height = float(getattr(event, "framebuffer_height", 0))
    scale_x = float(getattr(event, "content_scale_x", 1.0))
    scale_y = float(getattr(event, "content_scale_y", 1.0))
    if framebuffer_width > 0.0 and framebuffer_height > 0.0 and scale_x > 0.0 and scale_y > 0.0:
        return framebuffer_width / scale_x, framebuffer_height / scale_y
    return 0.0, 0.0


def _apply_view2d_navigation_action(
    controller: View2DNavigationController,
    current_view: View2D,
    panel_rect: LogicalPixelRect,
    action: NavigationAction,
    *,
    next_view2d_revision: str,
    view_snapshot_id: str | None,
    expected_layout_snapshot_id: str,
    layout_snapshot: ResolvedLayoutSnapshot | None = None,
) -> NavigationResult:
    if action.controller_id != controller.id:
        return _reject_navigation_action(
            controller,
            action,
            NavigationDiagnosticCode.NAVIGATION_UNSUPPORTED,
            "navigation action targets a different controller",
        )
    if action.view2d_revision != controller.current_view2d_revision:
        return _reject_navigation_action(
            controller,
            action,
            NavigationDiagnosticCode.NAVIGATION_STALE_VIEW,
            "navigation action references a stale View2D revision",
        )
    if action.layout_snapshot_id not in (None, expected_layout_snapshot_id):
        return _reject_navigation_action(
            controller,
            action,
            NavigationDiagnosticCode.NAVIGATION_STALE_LAYOUT,
            "navigation action references a stale layout snapshot",
        )
    if isinstance(action, PanByAction):
        next_view = pan_view2d(
            current_view,
            panel_rect,
            action.dx_px,
            action.dy_px,
            layout_snapshot=layout_snapshot,
        )
    elif isinstance(action, ZoomAboutAction):
        next_view = zoom_view2d_about(
            current_view,
            panel_rect,
            action.anchor_px,
            action.factor_x,
            action.factor_y,
            layout_snapshot=layout_snapshot,
        )
    elif isinstance(action, SetViewAction):
        next_view = action.view
    elif isinstance(action, ResetViewAction) and controller.home_view is not None:
        next_view = controller.home_view
    else:
        return _reject_navigation_action(
            controller,
            action,
            NavigationDiagnosticCode.NAVIGATION_UNSUPPORTED,
            "unsupported navigation action",
        )
    return NavigationResult(
        accepted=True,
        controller_id=controller.id,
        old_view2d_revision=controller.current_view2d_revision,
        new_view2d_revision=next_view2d_revision,
        view=next_view,
        view_snapshot_id=view_snapshot_id,
        layout_snapshot_id=action.layout_snapshot_id or expected_layout_snapshot_id,
    )


def _reject_navigation_action(
    controller: View2DNavigationController,
    action: NavigationAction,
    code: NavigationDiagnosticCode,
    message: str,
) -> NavigationResult:
    return NavigationResult(
        accepted=False,
        controller_id=controller.id,
        old_view2d_revision=action.view2d_revision,
        diagnostics=(f"{code.value}: {message}",),
        layout_snapshot_id=action.layout_snapshot_id,
    )
