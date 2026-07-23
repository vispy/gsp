"""Lazy Datoviz backend provider entry point."""

from __future__ import annotations

from gsp.backends import (
    BackendDescriptor,
    BackendInfo,
    BackendProvider,
    BackendSession,
    PLUGIN_API_VERSION,
    SessionRequest,
)

_DECLARED_CAPABILITIES = frozenset(
    {
        "display.interactive",
        "output.file",
        "visual.image",
        "visual.markers",
        "visual.mesh",
        "visual.paths",
        "visual.pixels",
        "visual.points",
        "visual.segments",
        "visual.sphere",
        "visual.vector",
        "visual.primitive",
        "visual.text",
        "texture.rgba8",
        "texture.filter.nearest",
        "texture.filter.linear",
    }
)


class DatovizProvider:
    def describe(self) -> BackendDescriptor:
        return BackendDescriptor(
            name="datoviz",
            plugin_api_version=PLUGIN_API_VERSION,
            protocol_versions=("0.2",),
            declared_capabilities=_DECLARED_CAPABILITIES,
        )

    def probe(self) -> BackendInfo:
        try:
            from .protocol_renderer import import_datoviz_v04

            dvz = import_datoviz_v04()
        except Exception as exc:
            return BackendInfo(
                name="datoviz",
                installed=True,
                available=False,
                descriptor=self.describe(),
                diagnostics=(f"{type(exc).__name__}: {exc}",),
            )

        from .capabilities import datoviz_v04_capability_snapshot
        from .latest_api_contract import datoviz_current_api_contract_diagnostics

        diagnostics = datoviz_current_api_contract_diagnostics(dvz)
        if diagnostics:
            return BackendInfo(
                name="datoviz",
                installed=True,
                available=False,
                descriptor=self.describe(),
                diagnostics=diagnostics,
            )
        snapshot = datoviz_v04_capability_snapshot(dvz)
        capabilities = set(_DECLARED_CAPABILITIES)
        capabilities.update(snapshot.transform_capabilities)
        capabilities.update(snapshot.navigation_capabilities)
        capabilities.update(snapshot.view3d_capabilities)
        if not snapshot.supports_visual("sphere"):
            capabilities.discard("visual.sphere")
        if not snapshot.supports_visual("vector"):
            capabilities.discard("visual.vector")
        if not snapshot.supports_visual("primitive"):
            capabilities.discard("visual.primitive")
        if not snapshot.supports_visual("text"):
            capabilities.discard("visual.text")
        return BackendInfo(
            name="datoviz",
            installed=True,
            available=True,
            descriptor=self.describe(),
            capabilities=frozenset(capabilities),
            diagnostics=(),
        )

    def open_session(self, request: SessionRequest) -> BackendSession:
        from .session import DatovizSession

        return DatovizSession(request=request)


def get_provider() -> BackendProvider:
    return DatovizProvider()
