"""Side-effect-free Matplotlib backend provider entry point."""

from __future__ import annotations

from importlib.util import find_spec

from gsp.backends import (
    BackendDescriptor,
    BackendInfo,
    BackendProvider,
    BackendSession,
    PLUGIN_API_VERSION,
    SessionRequest,
)

_CAPABILITIES = frozenset(
    {
        "output.file",
        "visual.points",
        "visual.markers",
        "visual.segments",
        "visual.paths",
        "visual.pixels",
        "visual.image",
        "visual.text",
        "visual.mesh",
        "visual.sphere",
        "visual.vector",
        "visual.primitive",
        "guides.axes",
        "guides.colorbar",
        "query.panel",
    }
)


class MatplotlibProvider:
    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            name="matplotlib",
            plugin_api_version=PLUGIN_API_VERSION,
            protocol_versions=("0.2",),
            declared_capabilities=_CAPABILITIES,
        )

    def probe(self) -> BackendInfo:
        available = find_spec("matplotlib") is not None
        descriptor = self.describe()
        return BackendInfo(
            name=descriptor.name,
            installed=True,
            available=available,
            descriptor=descriptor,
            capabilities=_CAPABILITIES if available else frozenset(),
            diagnostics=() if available else ("matplotlib is not importable",),
        )

    def open_session(self, request: SessionRequest) -> BackendSession:
        from .session import MatplotlibSession

        return MatplotlibSession(request=request)


def get_provider() -> BackendProvider:
    return MatplotlibProvider()
