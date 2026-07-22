from __future__ import annotations

import sys
import subprocess

import numpy as np

import gsp
from gsp.protocol import PointVisual, View2D


def test_matplotlib_plugin_is_discoverable_without_eager_matplotlib_import():
    code = """
import sys
import gsp

assert "matplotlib.pyplot" not in sys.modules
infos = {item.name: item for item in gsp.discover_backends()}
assert "matplotlib" in infos
assert infos["matplotlib"].available is None
assert "matplotlib.pyplot" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_matplotlib_session_renders_gsp_scene():
    scene = gsp.Scene(
        id="scene:test",
        visuals=(
            PointVisual(
                id="visual:points",
                positions=np.array([[0.0, 0.0]], dtype=np.float32),
                colors=np.array([[255, 0, 0, 255]], dtype=np.uint8),
                sizes=8.0,
            ),
        ),
        view2d=View2D(
            id="view:main",
            panel_id="panel:main",
            x_range=(-1.0, 1.0),
            y_range=(-1.0, 1.0),
        ),
    )
    with gsp.open_session("matplotlib", require={"visual.points"}) as session:
        result = session.render(scene)
        assert result.axes.get_xlim() == (-1.0, 1.0)
