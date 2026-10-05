"""Bounded public single-mesh session picking retains native diagnostics."""

from dataclasses import replace

import numpy as np
import pytest

from conformance.p038_support import single_panel_scene
from gsp import MeshPickSession
from gsp.backends import SessionRequest
from gsp.protocol import (
    CoordinateSpace,
    MeshVisual,
    QueryStatus,
    View3DMeshTrianglePickPayload,
    View3DMeshTrianglePickRequest,
)
from gsp_datoviz.capabilities import (
    SINGLE_MESH_PICK_CAPABILITY,
    datoviz_single_mesh_pick_ready,
    datoviz_v04_capability_snapshot,
)
from gsp_datoviz.session import DatovizSession
from test_datoviz_v04_protocol_renderer import (
    DVZ_QUERY_STATUS_HIT,
    DVZ_QUERY_STATUS_MISS,
    FakeDatovizV04WithRetainedView3D,
    FakeDvzQueryResult,
    _canonical_view3d_for_datoviz_query,
)


class FaceResult(FakeDvzQueryResult):
    def __init__(self, **kwargs):
        super().__init__(face_id=0, freshness_serial=0, **kwargs)


class FaceDatoviz(FakeDatovizV04WithRetainedView3D):
    DvzQueryResult = FaceResult
    DVZ_SCENE_TARGET_FACE = 4
    DVZ_QUERY_CAPABILITY_FACE = 0x08

    def __init__(self):
        super().__init__()
        self.hit = True
        self.freshness_serial = 17

    def dvz_query_request(self):
        return type("FakeRequest", (), {})()

    def dvz_panel_query_px(self, panel, x, y, request):
        self.calls.append(("panel_query_px", panel, x, y, request))
        return 0

    def dvz_scene_poll_query(self, scene, out_result):
        result = FakeDvzQueryResult(
            status=DVZ_QUERY_STATUS_HIT if self.hit else DVZ_QUERY_STATUS_MISS,
            hit=self.hit,
            visual_id=789,
            visual_family=8,
            resolved_target=4,
            face_id=0,
            freshness_serial=self.freshness_serial,
        )
        for name, value in vars(result).items():
            setattr(out_result, name, value)
        return True


def _mesh(visual_id="visual:mesh"):
    return MeshVisual(
        id=visual_id,
        positions=np.asarray([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32),
        faces=np.asarray([[0, 1, 2]], dtype=np.uint32),
        coordinate_space=CoordinateSpace.DATA,
        color=np.asarray([255, 255, 255, 255], dtype=np.uint8),
    )


@pytest.fixture
def retained(monkeypatch):
    native = FaceDatoviz()
    monkeypatch.setattr("gsp_datoviz.session.import_datoviz_v04", lambda: native)
    session = DatovizSession(request=SessionRequest())
    view = _canonical_view3d_for_datoviz_query()
    scene = single_panel_scene(id="scene:mesh", visuals=(_mesh(),), view3d=view)
    renderer = session.render(scene)
    request = View3DMeshTrianglePickRequest(view.id, (50.0, 50.0), panel_id=view.panel_id)
    yield session, scene, renderer, request, native
    session.close()


def test_public_mesh_pick_delegates_native_face_identity_and_miss(retained):
    session, scene, renderer, request, native = retained
    assert isinstance(session, MeshPickSession)
    assert session.capabilities.supports_extension(SINGLE_MESH_PICK_CAPABILITY)
    assert not session.capabilities.supports_extension("query.view3d.mesh_triangle_pick.v1")
    result = session.pick_mesh(request)
    assert result.status is QueryStatus.HIT
    assert result.visual_id == scene.visuals[0].id
    payload = result.extension_payload
    assert isinstance(payload, View3DMeshTrianglePickPayload)
    assert payload.primitive_index == 0
    assert payload.pick_scene_snapshot_id == "pick-scene:datoviz-17"
    assert any(call[0] == "panel_query_px" and call[4].target == 4 for call in native.calls)
    assert session._scene_renderers[scene.id][1] is renderer
    native.hit = False
    assert session.pick_mesh(request, scene_id=scene.id).status is QueryStatus.MISS


@pytest.mark.parametrize(
    "changes,code",
    [
        ({"expected_view_revision": 999}, "pick.stale.view_revision"),
        ({"expected_pick_scene_snapshot_id": "pick-scene:old"}, "pick.stale.pick_scene_snapshot"),
    ],
)
def test_public_mesh_pick_preserves_renderer_freshness_diagnostics(retained, changes, code):
    session, _, _, request, _ = retained
    result = session.pick_mesh(replace(request, **changes))
    assert result.status is QueryStatus.STALE
    assert result.diagnostic == code
    assert result.extension_payload.diagnostics[0].code.value == code


def test_public_mesh_pick_validates_panel_view_scene_and_session(retained):
    session, _, _, request, _ = retained
    assert (
        session.pick_mesh(replace(request, panel_id="panel:missing")).status is QueryStatus.INVALID
    )
    assert session.pick_mesh(replace(request, view_id="view:missing")).status is QueryStatus.INVALID
    assert session.pick_mesh(replace(request, panel_id=None)).hit
    with pytest.raises(TypeError, match="View3DMeshTrianglePickRequest"):
        session.pick_mesh(object())
    with pytest.raises(RuntimeError, match="has not been rendered"):
        session.pick_mesh(request, scene_id="scene:missing")
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.pick_mesh(request)


def test_public_mesh_pick_rejects_additional_visuals_without_native_query(retained):
    session, scene, _, request, native = retained
    session.render(
        replace(
            scene,
            id="scene:occluded",
            visuals=(scene.visuals[0], _mesh("visual:other")),
            attachments=scene.attachments
            + (replace(scene.attachments[0], visual_id="visual:other"),),
        )
    )
    count = len(native.calls)
    result = session.pick_mesh(request)
    assert result.status is QueryStatus.UNSUPPORTED
    assert result.diagnostic == "pick.unsupported.scene_occluder"
    assert not any(call[0] == "panel_query_px" for call in native.calls[count:])


def test_single_mesh_capability_requires_public_face_result_field():
    ready = FaceDatoviz()
    assert datoviz_single_mesh_pick_ready(ready)
    ready.DvzQueryResult = FakeDvzQueryResult
    assert not datoviz_single_mesh_pick_ready(ready)
    assert not datoviz_v04_capability_snapshot(ready).supports_extension(
        SINGLE_MESH_PICK_CAPABILITY
    )


@pytest.mark.parametrize("serial", [0, -1])
def test_mesh_miss_without_proven_native_freshness_is_unsupported(retained, serial):
    session, _, _, request, native = retained
    native.hit = False
    native.freshness_serial = serial
    result = session.pick_mesh(request)
    assert result.status is QueryStatus.UNSUPPORTED
    assert result.diagnostic == "pick.unsupported.native_state_only"


def test_mesh_miss_retains_freshness_and_checks_expected_scene(retained):
    session, _, _, request, native = retained
    native.hit = False
    result = session.pick_mesh(request)
    assert result.status is QueryStatus.MISS
    assert not result.hits
    assert result.visual_id is None
    assert result.extension_payload.pick_scene_snapshot_id == "pick-scene:datoviz-17"
    stale = session.pick_mesh(replace(request, expected_pick_scene_snapshot_id="pick-scene:old"))
    assert stale.status is QueryStatus.STALE
    assert stale.diagnostic == "pick.stale.pick_scene_snapshot"


def test_decoder_retains_miss_freshness_without_hit_payload_fields():
    from gsp_datoviz.query import decode_dvz_query_result

    metadata = {}
    raw = FakeDvzQueryResult(status=DVZ_QUERY_STATUS_MISS, hit=False, freshness_serial=42)
    decoded = decode_dvz_query_result(raw, native_metadata=metadata)
    assert metadata == {"freshness_serial": 42}
    assert decoded.status is QueryStatus.MISS
    assert not decoded.hits
    assert decoded.visual_id is None
    assert decoded.extension_payload is None


def test_public_mesh_pick_requires_scene_render_before_query(monkeypatch):
    monkeypatch.setattr("gsp_datoviz.session.import_datoviz_v04", FaceDatoviz)
    with DatovizSession(request=SessionRequest()) as session:
        with pytest.raises(RuntimeError, match="requires a rendered scene"):
            session.pick_mesh(View3DMeshTrianglePickRequest("view:main", (50.0, 50.0)))


def test_public_mesh_pick_rejects_non_view3d_panel(retained):
    from gsp.protocol import View2D

    session, _, _, request, native = retained
    scene = single_panel_scene(
        id="scene:2d",
        view2d=View2D("view:2d", "panel:main"),
    )
    session.render(scene)
    count = len(native.calls)
    answer = session.pick_mesh(replace(request, view_id="view:2d"))
    assert answer.status is QueryStatus.UNSUPPORTED
    assert answer.diagnostic == "pick.unsupported.backend"
    assert not any(call[0] == "panel_query_px" for call in native.calls[count:])


def test_mesh_pick_rejects_hidden_visual(retained):
    session, scene, _, request, native = retained
    hidden = replace(scene, attachments=(replace(scene.attachments[0], visible=False),))
    session.render(hidden)
    count = len(native.calls)
    answer = session.pick_mesh(request)
    assert answer.status is QueryStatus.UNSUPPORTED
    assert answer.diagnostic == "pick.unsupported.scene_occluder"
    assert not any(call[0] == "panel_query_px" for call in native.calls[count:])


def test_mesh_pick_rejects_visual_in_another_panel(retained):
    from gsp.protocol import (
        ExplicitPanelLayoutV1,
        NormalizedRenderTargetRect,
        Panel,
        PanelPlacement,
        VisualAttachment,
    )

    session, scene, _, request, native = retained
    other_view = replace(scene.views3d[0], id="view:other", panel_id="panel:other")
    other = _mesh("visual:other")
    multiple = replace(
        scene,
        panels=scene.panels + (Panel("panel:other"),),
        panel_layout=ExplicitPanelLayoutV1(
            (
                PanelPlacement("panel:main", NormalizedRenderTargetRect(0, 0, 0.5, 1)),
                PanelPlacement("panel:other", NormalizedRenderTargetRect(0.5, 0, 0.5, 1)),
            )
        ),
        views3d=scene.views3d + (other_view,),
        visuals=scene.visuals + (other,),
        attachments=scene.attachments
        + (VisualAttachment(other.id, other_view.panel_id, other_view.id),),
    )
    session.render(multiple)
    count = len(native.calls)
    answer = session.pick_mesh(request)
    assert answer.status is QueryStatus.UNSUPPORTED
    assert answer.diagnostic == "pick.unsupported.scene_occluder"
    assert not any(call[0] == "panel_query_px" for call in native.calls[count:])
