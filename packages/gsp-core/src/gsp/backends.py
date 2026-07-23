"""Lazy backend discovery and session selection."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from importlib import metadata
from typing import Any, Protocol, runtime_checkable

from .protocol import PROTOCOL_VERSION, CapabilitySnapshot, QueryRequest, QueryResult

BACKEND_ENTRY_POINT_GROUP = "gsp.backends"
PLUGIN_API_VERSION = 1


class BackendError(RuntimeError):
    """Base backend discovery or selection error."""


class BackendSelectionRequired(BackendError):
    """Raised when no explicit backend policy was provided."""


class BackendUnavailable(BackendError):
    """Raised when a requested backend is missing or cannot run."""


class BackendCompatibilityError(BackendError):
    """Raised when plugin or protocol versions do not match."""


class DuplicateBackendError(BackendError):
    """Raised when multiple installed distributions claim one backend name."""


class BackendCapabilityError(BackendError):
    """Raised when a selected backend lacks required semantic capabilities."""


@dataclass(frozen=True, slots=True)
class BackendDescriptor:
    name: str
    plugin_api_version: int
    protocol_versions: tuple[str, ...]
    declared_capabilities: frozenset[str]


@dataclass(frozen=True, slots=True)
class BackendInfo:
    name: str
    installed: bool
    available: bool | None = None
    descriptor: BackendDescriptor | None = None
    capabilities: frozenset[str] = frozenset()
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SessionRequest:
    require: frozenset[str] = frozenset()
    adaptation: frozenset[str] = frozenset()


@runtime_checkable
class BackendSession(Protocol):
    backend_name: str
    capabilities: CapabilitySnapshot

    @property
    def diagnostics(self) -> tuple[str, ...]: ...
    def render(self, scene: Any, **kwargs: Any) -> Any: ...
    def display(self, scene: Any, **kwargs: Any) -> Any: ...
    def query(
        self, request: QueryRequest, *, scene_id: str | None = None
    ) -> QueryResult: ...
    def run(self) -> None: ...
    def close(self) -> None: ...
    def __enter__(self) -> "BackendSession": ...
    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None: ...


@runtime_checkable
class BackendProvider(Protocol):
    def describe(self) -> BackendDescriptor: ...
    def probe(self) -> BackendInfo: ...
    def open_session(self, request: SessionRequest) -> BackendSession: ...


def _entry_points() -> tuple[metadata.EntryPoint, ...]:
    return tuple(metadata.entry_points(group=BACKEND_ENTRY_POINT_GROUP))


def _entry_points_by_name() -> dict[str, metadata.EntryPoint]:
    result: dict[str, metadata.EntryPoint] = {}
    for entry_point in _entry_points():
        if entry_point.name in result:
            raise DuplicateBackendError(
                f"multiple installed plugins claim backend {entry_point.name!r}"
            )
        result[entry_point.name] = entry_point
    return result


def _load_provider(entry_point: metadata.EntryPoint) -> BackendProvider:
    factory = entry_point.load()
    provider = factory()
    if not isinstance(provider, BackendProvider):
        raise BackendCompatibilityError(
            f"backend {entry_point.name!r} does not implement BackendProvider"
        )
    descriptor = provider.describe()
    if descriptor.name != entry_point.name:
        raise BackendCompatibilityError(
            f"backend entry point {entry_point.name!r} describes {descriptor.name!r}"
        )
    if descriptor.plugin_api_version != PLUGIN_API_VERSION:
        raise BackendCompatibilityError(
            f"backend {descriptor.name!r} uses plugin API {descriptor.plugin_api_version}; "
            f"expected {PLUGIN_API_VERSION}"
        )
    if PROTOCOL_VERSION not in descriptor.protocol_versions:
        raise BackendCompatibilityError(
            f"backend {descriptor.name!r} does not support GSP {PROTOCOL_VERSION}"
        )
    return provider


def discover_backends(*, probe: bool = False) -> tuple[BackendInfo, ...]:
    """Discover installed plugins, optionally loading each lightweight provider."""
    entries = _entry_points_by_name()
    if not probe:
        return tuple(BackendInfo(name=name, installed=True) for name in sorted(entries))
    infos: list[BackendInfo] = []
    for name in sorted(entries):
        provider = _load_provider(entries[name])
        info = provider.probe()
        if info.name != name or not info.installed:
            raise BackendCompatibilityError(f"backend {name!r} returned inconsistent probe data")
        infos.append(info)
    return tuple(infos)


def open_session(
    backend: str | None = None,
    *,
    require: Collection[str] = (),
    prefer: tuple[str, ...] = (),
    adaptation: Collection[str] = (),
) -> BackendSession:
    """Open a backend session using an explicit name or ordered caller policy."""
    if backend is not None and prefer:
        raise ValueError("backend and prefer are mutually exclusive")
    if backend is None and not prefer:
        raise BackendSelectionRequired("provide backend= or a non-empty prefer= policy")
    candidates = (backend,) if backend is not None else prefer
    entries = _entry_points_by_name()
    required = frozenset(require)
    failures: list[str] = []
    for candidate in candidates:
        assert candidate is not None
        entry_point = entries.get(candidate)
        if entry_point is None:
            failures.append(f"backend {candidate!r} is not installed")
            if backend is not None:
                break
            continue
        provider = _load_provider(entry_point)
        info = provider.probe()
        if not info.available:
            failures.append(
                f"backend {candidate!r} is unavailable: " + "; ".join(info.diagnostics)
            )
            if backend is not None:
                break
            continue
        missing = required - info.capabilities
        if missing:
            failures.append(
                f"backend {candidate!r} lacks required capabilities: {sorted(missing)!r}"
            )
            if backend is not None:
                raise BackendCapabilityError(failures[-1])
            continue
        return provider.open_session(
            SessionRequest(require=required, adaptation=frozenset(adaptation))
        )
    raise BackendUnavailable("; ".join(failures))
