"""Focused unit coverage for the Datoviz query-result adapter boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from gsp.protocol import QueryStatus, VisualFamily
from gsp_datoviz.query import (
    DATOVIZ_QUERY_PAYLOAD_KIND,
    DVZ_QUERY_STATUS_HIT,
    DVZ_SCENE_VISUAL_FAMILY_GLYPH,
    DVZ_SCENE_VISUAL_FAMILY_IMAGE,
    DVZ_SCENE_VISUAL_FAMILY_LABELS,
    DVZ_SCENE_VISUAL_FAMILY_MARKER,
    DVZ_SCENE_VISUAL_FAMILY_MESH,
    DVZ_SCENE_VISUAL_FAMILY_PATH,
    DVZ_SCENE_VISUAL_FAMILY_PIXEL,
    DVZ_SCENE_VISUAL_FAMILY_POINT,
    DVZ_SCENE_VISUAL_FAMILY_PRIMITIVE,
    DVZ_SCENE_VISUAL_FAMILY_SEGMENT,
    DVZ_SCENE_VISUAL_FAMILY_SPHERE,
    DVZ_SCENE_VISUAL_FAMILY_SPLAT,
    DVZ_SCENE_VISUAL_FAMILY_TEXT,
    DVZ_SCENE_VISUAL_FAMILY_VECTOR,
    DVZ_SCENE_VISUAL_FAMILY_VOLUME,
    DVZ_QUERY_VALUE_CATEGORY,
    DVZ_QUERY_VALUE_TEXT,
    DVZ_QUERY_VALUE_VEC3,
    decode_dvz_query_result,
)


def _result(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "request_id": 17,
        "status": DVZ_QUERY_STATUS_HIT,
        "hit": True,
        "panel_position": (12.0, 34.0),
        "visual_id": 23,
        "visual_family": DVZ_SCENE_VISUAL_FAMILY_POINT,
        "item_id": 5,
        "texel_id": 99,
        "has_visual_position": True,
        "visual_position": (0.25, 0.5, 0.0),
        "has_data_position": True,
        "data_position": (1.0, 2.0, 3.0),
        "has_display_rgba": True,
        "display_rgba": (0.1, 0.2, 0.3, 1.0),
        "value_kind": 0,
        "vector": (0.0, 0.0, 0.0, 0.0),
        "scalar": 0.0,
        "category_id": 0,
        "label": b"",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    ("native_family", "expected"),
    [
        (DVZ_SCENE_VISUAL_FAMILY_POINT, VisualFamily.POINT),
        (DVZ_SCENE_VISUAL_FAMILY_PIXEL, "pixel"),
        (DVZ_SCENE_VISUAL_FAMILY_MARKER, "marker"),
        (DVZ_SCENE_VISUAL_FAMILY_SEGMENT, "segment"),
        (DVZ_SCENE_VISUAL_FAMILY_VECTOR, "vector"),
        (DVZ_SCENE_VISUAL_FAMILY_PATH, "path"),
        (DVZ_SCENE_VISUAL_FAMILY_IMAGE, VisualFamily.IMAGE),
        (DVZ_SCENE_VISUAL_FAMILY_MESH, VisualFamily.MESH),
        (DVZ_SCENE_VISUAL_FAMILY_VOLUME, "volume"),
        (DVZ_SCENE_VISUAL_FAMILY_PRIMITIVE, "primitive"),
        (DVZ_SCENE_VISUAL_FAMILY_SPHERE, "sphere"),
        (DVZ_SCENE_VISUAL_FAMILY_GLYPH, "glyph"),
        (DVZ_SCENE_VISUAL_FAMILY_TEXT, VisualFamily.TEXT),
        (DVZ_SCENE_VISUAL_FAMILY_LABELS, "labels"),
        (DVZ_SCENE_VISUAL_FAMILY_SPLAT, "splat"),
    ],
)
def test_decode_maps_all_datoviz_visual_families(native_family: int, expected: object) -> None:
    result = decode_dvz_query_result(_result(visual_family=native_family))

    assert result.status is QueryStatus.HIT
    assert result.visual_family == expected
    assert result.item_id == 5


def test_decode_unknown_visual_family_remains_unmapped() -> None:
    result = decode_dvz_query_result(_result(visual_family=999))

    assert result.visual_family is None
    assert result.item_id is None


def test_decode_does_not_invent_two_dimensional_texel_coordinates() -> None:
    result = decode_dvz_query_result(
        _result(visual_family=DVZ_SCENE_VISUAL_FAMILY_IMAGE, texel_id=99)
    )

    assert result.texel is None


def test_decode_retains_zero_flat_texel_in_native_payload() -> None:
    result = decode_dvz_query_result(
        _result(visual_family=DVZ_SCENE_VISUAL_FAMILY_IMAGE, texel_id=0)
    )

    assert result.extension_payload_kind == DATOVIZ_QUERY_PAYLOAD_KIND
    assert result.extension_payload.texel_id == 0


def test_decode_retains_bounded_native_metadata() -> None:
    result = decode_dvz_query_result(
        _result(
            payload_version=3,
            resolved_target=7,
            resolved_id=41,
            group_id=42,
            auxiliary_id=43,
            instance_id=44,
            face_id=45,
            primitive_id=46,
            vertex_id=47,
            voxel_id=48,
            texel_id=49,
            link_key=50,
            link_channel=2,
            freshness_serial=51,
            has_uvw=True,
            uvw=(0.25, 0.5, 0.75),
            has_depth=True,
            depth=0.125,
        )
    )

    payload = result.extension_payload
    assert result.extension_payload_kind == DATOVIZ_QUERY_PAYLOAD_KIND
    assert payload.payload_version == 3
    assert payload.native_visual_id == 23
    assert payload.visual_family == DVZ_SCENE_VISUAL_FAMILY_POINT
    assert payload.resolved_target == 7
    assert payload.resolved_id == 41
    assert (
        payload.group_id,
        payload.auxiliary_id,
        payload.instance_id,
        payload.face_id,
        payload.primitive_id,
        payload.vertex_id,
        payload.voxel_id,
        payload.texel_id,
    ) == (42, 43, 44, 45, 46, 47, 48, 49)
    assert (payload.link_key, payload.link_channel, payload.freshness_serial) == (50, 2, 51)
    assert payload.uvw == (0.25, 0.5, 0.75)
    assert payload.depth == 0.125


@pytest.mark.parametrize(
    ("value_kind", "fields", "expected"),
    [
        (DVZ_QUERY_VALUE_VEC3, {"vector": (1.0, 2.0, 3.0, 4.0)}, (1.0, 2.0, 3.0)),
        (DVZ_QUERY_VALUE_CATEGORY, {"category_id": 0}, 0),
        (DVZ_QUERY_VALUE_TEXT, {"label": b"temperature\x00ignored"}, "temperature"),
    ],
)
def test_decode_preserves_typed_values(
    value_kind: int, fields: dict[str, object], expected: object
) -> None:
    result = decode_dvz_query_result(_result(value_kind=value_kind, **fields))

    assert result.value == expected


def test_decode_preserves_canonical_coordinates_and_rgba_without_z() -> None:
    result = decode_dvz_query_result(_result())

    assert result.panel_coordinate == (12.0, 34.0)
    assert result.visual_coordinate == (0.25, 0.5)
    assert result.data_coordinate == (1.0, 2.0)
    assert result.displayed_rgba == (0.1, 0.2, 0.3, 1.0)
