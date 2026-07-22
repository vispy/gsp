from __future__ import annotations

import subprocess
import sys

import numpy as np

from gsp.protocol import ColorMapId
from gsp.protocol.color_mapping import canonical_lut


def test_import_gsp_does_not_import_rendering_backends() -> None:
    code = (
        "import sys, gsp; "
        "assert not any(n == 'matplotlib' or n.startswith('matplotlib.') for n in sys.modules); "
        "assert not any(n == 'datoviz' or n.startswith('datoviz.') for n in sys.modules)"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_canonical_colormaps_are_packaged_and_read_only() -> None:
    for colormap_id in ColorMapId:
        lut = canonical_lut(colormap_id)
        assert lut.shape == (256, 4)
        assert lut.dtype == np.dtype(np.uint8)
        assert not lut.flags.writeable
        assert np.all(lut[:, 3] == 255)

