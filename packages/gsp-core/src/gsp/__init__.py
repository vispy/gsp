"""Backend-independent Graphics Server Protocol."""

from . import protocol
from .backends import (
    BackendCapabilityError,
    BackendCompatibilityError,
    BackendDescriptor,
    BackendError,
    BackendInfo,
    BackendSelectionRequired,
    BackendUnavailable,
    DuplicateBackendError,
    discover_backends,
    open_session,
)
from .scene import Scene

__version__ = "0.2.0a1"

__all__ = [
    "__version__",
    "BackendCapabilityError",
    "BackendCompatibilityError",
    "BackendDescriptor",
    "BackendError",
    "BackendInfo",
    "BackendSelectionRequired",
    "BackendUnavailable",
    "DuplicateBackendError",
    "Scene",
    "discover_backends",
    "open_session",
    "protocol",
]

