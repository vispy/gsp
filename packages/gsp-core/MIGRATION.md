# gsp-core migration provenance

Source baseline: `vispy/GSP_API@463d34d1d6560f045e5c40af594372d0fea93ab5`.

| Destination | Source | Source object | Mode |
|---|---|---|---|
| `src/gsp/protocol/**` | `src/gsp/protocol/**` | tree `b2c4798ae58aefac2ea98cf6c727eeae71572bd9` | exact copies except below |
| `src/gsp/protocol/color_mapping.py` | same path, blob `15add13a135153c7441f71ae9f72829f172f2b82` | derived rewrite | remove Matplotlib dependency; load vendored canonical data |
| `src/gsp/protocol/_colormaps.npz` | six S026 LUTs resolved by the source implementation | SHA-256 `94c38c8611860a53bc9a3d757e1b3a171dc1b02b96028601ac4ff1169a22c4e0` | derived binary data |
| `src/gsp/__init__.py` | source architecture and ADR-0035 | new | minimal core surface without legacy imports |
| `tests/**` | selected source `tests/**`, tree `6fea750cf1d6af58d45e69c8aa5d54af3269817b` | curated copies | protocol-only test subset |

The canonical LUT arrays preserve the exact 256-entry RGBA8 results produced by the accepted S026
source baseline for gray, viridis, magma, plasma, inferno, and cividis. They are protocol data, not a
runtime dependency on Matplotlib.

