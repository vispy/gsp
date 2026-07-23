from __future__ import annotations

import pytest

import gsp.backends as backends


class FakeEntryPoint:
    def __init__(self, name: str, factory):
        self.name = name
        self._factory = factory

    def load(self):
        return self._factory


class FakeSession:
    backend_name = "fake"
    capabilities = None
    diagnostics = ()

    def render(self, scene, **kwargs): return scene
    def display(self, scene, **kwargs): return scene
    def query(self, request, *, scene_id=None): return request
    def run(self): return None
    def close(self): return None
    def __enter__(self): return self
    def __exit__(self, exc_type, exc, traceback): self.close()


class FakeProvider:
    def describe(self):
        return backends.BackendDescriptor("fake", 1, ("0.2",), frozenset({"visual.points"}))

    def probe(self):
        return backends.BackendInfo(
            name="fake",
            installed=True,
            available=True,
            descriptor=self.describe(),
            capabilities=self.describe().declared_capabilities,
        )

    def open_session(self, request):
        return FakeSession()


def test_metadata_discovery_does_not_load_provider(monkeypatch):
    loaded = False

    def factory():
        nonlocal loaded
        loaded = True
        return FakeProvider()

    monkeypatch.setattr(backends, "_entry_points", lambda: (FakeEntryPoint("fake", factory),))
    assert backends.discover_backends() == (backends.BackendInfo(name="fake", installed=True),)
    assert loaded is False


def test_explicit_selection_and_capability_check(monkeypatch):
    monkeypatch.setattr(
        backends, "_entry_points", lambda: (FakeEntryPoint("fake", FakeProvider),)
    )
    session = backends.open_session("fake", require={"visual.points"})
    assert session.backend_name == "fake"
    with pytest.raises(backends.BackendCapabilityError):
        backends.open_session("fake", require={"visual.mesh"})


def test_selection_requires_policy_and_rejects_duplicates(monkeypatch):
    with pytest.raises(backends.BackendSelectionRequired):
        backends.open_session()
    monkeypatch.setattr(
        backends,
        "_entry_points",
        lambda: (FakeEntryPoint("fake", FakeProvider), FakeEntryPoint("fake", FakeProvider)),
    )
    with pytest.raises(backends.DuplicateBackendError):
        backends.discover_backends()
