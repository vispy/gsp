#!/usr/bin/env python3
"""Check current requirement destinations, evidence links, and stable IDs.

Test links are evidence pointers only; this checker does not infer rule-level
coverage from a test's presence in the registry.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "specs/requirements/requirements.json"
ID_PATTERN = re.compile(r"GSP-[A-Z]+-[0-9]{3}\Z")


def check(registry_path: Path = REGISTRY) -> list[str]:
    errors: list[str] = []
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"cannot read {registry_path}: {exc}"]

    requirements = registry.get("requirements")
    if not isinstance(requirements, list):
        return ["requirements must be a JSON list"]

    ids = [r.get("id") for r in requirements if isinstance(r, dict)]
    for req_id in ids:
        if not isinstance(req_id, str) or not ID_PATTERN.fullmatch(req_id):
            errors.append(f"invalid requirement ID: {req_id!r}")
    counts = Counter(req_id for req_id in ids if isinstance(req_id, str))
    for req_id, count in sorted(counts.items()):
        if count > 1:
            errors.append(f"duplicate requirement ID: {req_id} ({count} entries)")

    for req in requirements:
        if not isinstance(req, dict):
            errors.append("requirement entry must be an object")
            continue
        req_id = req.get("id", "<missing ID>")
        destination = req.get("destination")
        if not isinstance(destination, str) or not destination:
            errors.append(f"{req_id}: missing destination")
        else:
            path = (ROOT / destination).resolve()
            if ROOT not in path.parents or not path.is_file():
                errors.append(f"{req_id}: current destination does not exist: {destination}")
            elif isinstance(req_id, str) and req_id not in path.read_text(encoding="utf-8"):
                errors.append(f"{req_id}: ID is not registered in {destination}")
        tests = req.get("tests")
        if not isinstance(tests, list) or not tests:
            errors.append(f"{req_id}: tests must contain at least one evidence path")
            continue
        for test_path in tests:
            if not isinstance(test_path, str) or not test_path:
                errors.append(f"{req_id}: invalid evidence path {test_path!r}")
                continue
            resolved = (ROOT / test_path).resolve()
            if ROOT not in resolved.parents or not resolved.is_file():
                errors.append(f"{req_id}: evidence path does not exist: {test_path}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="check registry references")
    args = parser.parse_args()
    if not args.check:
        parser.error("--check is required")
    errors = check()
    if errors:
        print("Specification traceability check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(
        "Specification traceability references are valid (test links are evidence pointers, not coverage claims)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
