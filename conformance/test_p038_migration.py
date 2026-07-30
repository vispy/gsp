import json
from pathlib import Path

import pytest

from conformance.fixtures.migration.p038.migrate import migrate_document


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "migration" / "p038"


def test_p038_checked_migration_is_deterministic() -> None:
    before = json.loads((FIXTURE_ROOT / "before" / "scene.json").read_text(encoding="utf-8"))
    expected = json.loads((FIXTURE_ROOT / "after" / "scene.json").read_text(encoding="utf-8"))
    assert migrate_document(before) == expected


def test_p038_migration_rejects_duplicate_panel_ids() -> None:
    before = json.loads((FIXTURE_ROOT / "before" / "scene.json").read_text(encoding="utf-8"))
    before["scenes"][0]["panels"].append(dict(before["scenes"][0]["panels"][0]))
    with pytest.raises(ValueError, match="duplicate panel ids"):
        migrate_document(before)
