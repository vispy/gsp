from __future__ import annotations

from types import SimpleNamespace

import pytest

import gsp_datoviz.capabilities as capability_module
import gsp_datoviz.latest_api_contract as contract_module
import gsp_datoviz.protocol_renderer as renderer_module
from gsp_datoviz.plugin import DatovizProvider


@pytest.mark.parametrize("vector_ready", [False, True])
def test_probe_removes_vector_capability_when_runtime_probe_rejects_it(
    monkeypatch: pytest.MonkeyPatch, vector_ready: bool
) -> None:
    monkeypatch.setattr(renderer_module, "import_datoviz_v04", lambda: object())
    monkeypatch.setattr(
        contract_module, "datoviz_current_api_contract_diagnostics", lambda _dvz: ()
    )
    monkeypatch.setattr(
        capability_module,
        "datoviz_v04_capability_snapshot",
        lambda _dvz: SimpleNamespace(
            transform_capabilities=(),
            navigation_capabilities=(),
            view3d_capabilities=(),
            supports_visual=lambda family: (
                vector_ready if family == "vector" else True
            ),
        ),
    )

    info = DatovizProvider().probe()

    assert info.available
    assert ("visual.vector" in info.capabilities) is vector_ready


@pytest.mark.parametrize("primitive_ready", [False, True])
def test_probe_removes_primitive_capability_when_runtime_probe_rejects_it(
    monkeypatch: pytest.MonkeyPatch, primitive_ready: bool
) -> None:
    monkeypatch.setattr(renderer_module, "import_datoviz_v04", lambda: object())
    monkeypatch.setattr(
        contract_module, "datoviz_current_api_contract_diagnostics", lambda _dvz: ()
    )
    monkeypatch.setattr(
        capability_module,
        "datoviz_v04_capability_snapshot",
        lambda _dvz: SimpleNamespace(
            transform_capabilities=(),
            navigation_capabilities=(),
            view3d_capabilities=(),
            supports_visual=lambda family: (
                primitive_ready if family == "primitive" else True
            ),
        ),
    )

    provider = DatovizProvider()
    assert "visual.primitive" in provider.describe().declared_capabilities
    info = provider.probe()

    assert info.available
    assert ("visual.primitive" in info.capabilities) is primitive_ready


@pytest.mark.parametrize("text_ready", [False, True])
def test_probe_removes_text_capability_when_runtime_probe_rejects_it(
    monkeypatch: pytest.MonkeyPatch, text_ready: bool
) -> None:
    monkeypatch.setattr(renderer_module, "import_datoviz_v04", lambda: object())
    monkeypatch.setattr(
        contract_module, "datoviz_current_api_contract_diagnostics", lambda _dvz: ()
    )
    monkeypatch.setattr(
        capability_module,
        "datoviz_v04_capability_snapshot",
        lambda _dvz: SimpleNamespace(
            transform_capabilities=(),
            navigation_capabilities=(),
            view3d_capabilities=(),
            supports_visual=lambda family: text_ready if family == "text" else True,
        ),
    )

    info = DatovizProvider().probe()

    assert info.available
    assert ("visual.text" in info.capabilities) is text_ready
