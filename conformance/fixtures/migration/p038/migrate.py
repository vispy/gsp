"""One-shot deterministic migration for checked P038 JSON fixture evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any


def migrate_document(document: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(document)
    for scene in migrated["scenes"]:
        panel_ids = [panel["id"] for panel in scene["panels"]]
        if len(panel_ids) != len(set(panel_ids)):
            raise ValueError(f"scene {scene['id']!r} has duplicate panel ids")
        placements = []
        clip_by_view: dict[str, bool] = {
            view["id"]: view.pop("clip", True) for view in scene.get("view2d", [])
        }
        for panel in scene["panels"]:
            rectangle = panel.pop("viewport_rect")
            panel.pop("figure_id")
            placements.append(
                {
                    "panel_id": panel["id"],
                    "allocation_rect": dict(
                        zip(("left", "top", "width", "height"), rectangle, strict=True)
                    ),
                }
            )
        scene["panel_layout"] = {
            "kind": "layout.panel.explicit_rects.v1",
            "placements": placements,
        }
        for attachment in scene.get("attachments", []):
            attachment["clip_scope"] = (
                "plot" if clip_by_view[attachment["view_id"]] else "render_target"
            )
    return migrated


def main() -> None:
    source, destination = map(Path, sys.argv[1:3])
    document = json.loads(source.read_text(encoding="utf-8"))
    migrated = migrate_document(document)
    destination.write_text(
        json.dumps(migrated, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
