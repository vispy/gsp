from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_datoviz_metadata_discovery_does_not_import_datoviz() -> None:
    code = """
import sys
import gsp

assert "datoviz" not in sys.modules
infos = {item.name: item for item in gsp.discover_backends()}
assert "datoviz" in infos
assert infos["datoviz"].available is None
assert "datoviz" not in sys.modules
"""
    env = dict(os.environ)
    env["GSP_DATOVIZ_SOURCE"] = "none"
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_datoviz_probe_reports_local_current_api() -> None:
    import gsp

    source = Path("/Users/cyrille/GIT/Viz/datoviz")
    if not source.is_dir():
        return
    os.environ["GSP_DATOVIZ_SOURCE"] = str(source)
    infos = {item.name: item for item in gsp.discover_backends(probe=True)}
    assert infos["datoviz"].available is True, infos["datoviz"].diagnostics
