import json
from pathlib import Path

from tools.spec_traceability import REGISTRY, check


def test_repository_traceability_references_are_valid():
    assert check() == []


def write_registry(tmp_path: Path, mutate) -> Path:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    mutate(registry)
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_rejects_stale_current_destination(tmp_path):
    path = write_registry(
        tmp_path,
        lambda data: data["requirements"][0].update(destination="spec/current/capabilities.md"),
    )
    assert any("current destination does not exist" in error for error in check(path))


def test_rejects_missing_evidence_test(tmp_path):
    path = write_registry(
        tmp_path, lambda data: data["requirements"][0].update(tests=["tests/no-such-test.py"])
    )
    assert any("evidence path does not exist" in error for error in check(path))


def test_rejects_duplicate_id(tmp_path):
    def duplicate(data):
        data["requirements"][1]["id"] = data["requirements"][0]["id"]

    path = write_registry(tmp_path, duplicate)
    assert any("duplicate requirement ID" in error for error in check(path))


def test_rejects_id_missing_from_destination_chapter(tmp_path):
    def misroute(data):
        data["requirements"][0]["destination"] = "docs/specification/scene.md"

    path = write_registry(tmp_path, misroute)
    assert any("ID is not registered" in error for error in check(path))


def test_archival_source_paths_are_preserved_as_provenance(tmp_path):
    original = json.loads(REGISTRY.read_text(encoding="utf-8"))
    archival_source = "GSP_API/spec/old-chapter.md"
    original["requirements"][0]["source"] = archival_source
    path = tmp_path / "requirements.json"
    path.write_text(json.dumps(original), encoding="utf-8")
    assert check(path) == []
    reread = json.loads(path.read_text(encoding="utf-8"))
    assert reread["requirements"][0]["source"] == archival_source
