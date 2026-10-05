"""Native resource identity and semantic state survive bounded point updates."""

from dataclasses import replace

import numpy as np
import pytest

from gsp import Scene, PointUpdateSession
from gsp.backends import SessionRequest
from gsp.protocol import (
    CoordinateSpace,
    Panel,
    PointVisual,
    QueryCoordinateSpace,
    QueryPayload,
    QueryRequest,
    QueryStatus,
    View2D,
    VisualAttachment,
    full_target_panel_layout,
)
from gsp_datoviz.session import DatovizSession
from test_datoviz_v04_protocol_renderer import FakeDatovizV04
from test_datoviz_v04_protocol_renderer import (
    FakeDatovizV04WithRuntimeQuery,
    FakeDvzQueryResult,
)


@pytest.fixture
def retained(monkeypatch):
    dvz = FakeDatovizV04()
    monkeypatch.setattr("gsp_datoviz.session.import_datoviz_v04", lambda: dvz)
    session = DatovizSession(request=SessionRequest())
    point = PointVisual(
        "visual:point",
        np.asarray([[0, 0]], dtype=np.float32),
        np.asarray([[255, 0, 0, 255]], dtype=np.uint8),
        coordinate_space=CoordinateSpace.DATA,
    )
    scene = Scene(
        "scene:main",
        (Panel("panel:main"),),
        full_target_panel_layout("panel:main"),
        visuals=(point,),
        views2d=(View2D("view:main", "panel:main"),),
        attachments=(VisualAttachment(point.id, "panel:main", "view:main"),),
    )
    renderer = session.render(scene)
    yield session, scene, point, renderer, dvz
    session.close()


def test_point_update_keeps_native_scene_figure_visual_and_revisions(retained):
    session, scene, point, renderer, dvz = retained
    assert isinstance(session, PointUpdateSession)
    native = renderer.visuals[point.id]
    calls = len(dvz.calls)
    changed = replace(
        point,
        positions=np.asarray([[0.5, 0.25]], dtype=np.float32),
        colors=np.asarray([[0, 0, 255, 255]], dtype=np.uint8),
        sizes=12,
    )
    assert session.scene_revision() == 0
    assert session.update_point(changed) == 1
    assert session._scene_renderers[scene.id][1] is renderer
    assert renderer.visuals[point.id] is native
    assert session._scene_renderers[scene.id][0].visuals[0] is changed
    assert not any(call[0] in ("point", "scene", "figure") for call in dvz.calls[calls:])
    uploaded = [
        call[3] for call in dvz.calls[calls:] if call[0] == "set_data" and call[2] == "position"
    ]
    np.testing.assert_array_equal(uploaded[0], [[0.5, 0.25, 0]])
    session.render(scene)
    assert session.scene_revision(scene.id) == 2


def test_rejected_update_and_target_failure_do_not_publish_new_state(retained, tmp_path):
    session, scene, point, renderer, dvz = retained
    calls = len(dvz.calls)
    changed = replace(
        point,
        positions=np.zeros((2, 2), dtype=np.float32),
        colors=np.full((2, 4), 255, dtype=np.uint8),
    )
    with pytest.raises(ValueError, match="count or dimension"):
        session.update_point(changed)
    assert session.scene_revision() == 0
    assert len(dvz.calls) == calls
    with pytest.raises(ValueError, match="unknown retained"):
        session.update_point(replace(point, id="visual:unknown"))
    with pytest.raises(RuntimeError, match="has not been rendered"):
        session.update_point(point, scene_id="scene:unknown")
    # Capture fails before state publication and the prior successful scene stays live.
    with pytest.raises(Exception):
        session.render(scene, target=tmp_path / "missing" / "output.png")
    assert session._scene_renderers[scene.id][1] is renderer
    assert session.scene_revision() == 0
    assert len(session._renderers) == 1


def test_native_update_failure_closes_session(retained, monkeypatch):
    session, scene, point, renderer, _ = retained

    def fail(_visual):
        raise RuntimeError("native upload failed")

    monkeypatch.setattr(renderer, "update_point_visual", fail)
    with pytest.raises(RuntimeError, match="native upload failed"):
        session.update_point(point)
    assert renderer._closed
    with pytest.raises(RuntimeError, match="closed"):
        session.scene_revision(scene.id)


def test_update_after_close_rejects(retained):
    session, _, point, _, _ = retained
    session.close()
    with pytest.raises(RuntimeError, match="closed"):
        session.update_point(point)


def test_native_identity_is_mapped_in_top_level_and_individual_hits(monkeypatch):
    dvz = FakeDatovizV04WithRuntimeQuery(
        FakeDvzQueryResult(status=1, hit=True, visual_id=123, visual_family=1, item_id=0)
    )
    monkeypatch.setattr("gsp_datoviz.session.import_datoviz_v04", lambda: dvz)
    with DatovizSession(request=SessionRequest()) as session:
        point = PointVisual(
            "visual:point", np.zeros((1, 2), dtype=np.float32), np.full((1, 4), 255, np.uint8)
        )
        scene = Scene(
            "scene:query",
            (Panel("panel:main"),),
            full_target_panel_layout("panel:main"),
            visuals=(point,),
            attachments=(VisualAttachment(point.id, "panel:main"),),
        )
        session.render(scene)
        result = session.query(
            QueryRequest(
                "query:identity",
                "panel:main",
                (100, 100),
                coordinate_space=QueryCoordinateSpace.PANEL,
                requested_payload=(QueryPayload.IDENTITY,),
            )
        )
        assert result.status is QueryStatus.HIT
        assert result.visual_id == point.id
        assert result.hits[0].visual_id == point.id


def test_update_resets_blended_alpha_to_opaque(retained, monkeypatch):
    session, _, point, renderer, dvz = retained
    modes = []
    monkeypatch.setattr(
        dvz,
        "dvz_visual_set_alpha_mode",
        lambda visual, mode: modes.append(mode) or 0,
        raising=False,
    )
    session.update_point(replace(point, colors=np.asarray([[0, 0, 255, 128]], np.uint8)))
    session.update_point(point)
    assert modes == [1, 0]
