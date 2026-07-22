# Local bootstrap qualification

Date: 2026-07-22

This unpublished `0.2.0a1` bootstrap was built from committed source and installed into isolated
virtual environments. Tests ran from `/tmp`, so imports resolved from installed wheels rather than
repository source trees.

| Combination | Result |
|---|---|
| `gsp-core` only | 167 tests; no provider imports; empty discovery result |
| `gsp-core` + `gsp-matplotlib` + `vispy2` | 126 adapter tests; local extra resolution passed |
| `gsp-core` + `gsp-datoviz` + `vispy2` | 150 adapter tests; local extra resolution passed |
| All four distributions | One unchanged VisPy2 scene rendered to PNG through both providers |

Strict mypy passes for 51 GSP source files and Ruff passes for all three distributions. Metadata
discovery is side-effect-free. Matplotlib and Datoviz provider modules, plus GSP and VisPy2, were
confirmed under the isolated environment's `site-packages` directory.

The Datoviz native gates used the explicit development source
`/Users/cyrille/GIT/Viz/datoviz` at commit
`be7f2a80354c25e85bab88c85f5ea7340975b569`. This is an RC3-oriented development checkpoint, not
an RC2 compatibility or publication claim. An ordinary Datoviz dependency remains blocked on a
compatible published artifact.

## Artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| `gsp_core-0.2.0a1-py3-none-any.whl` | `727ec6d12078b8abf2aa1f3eebc6373704eba6a2e17b5c98256c9e8f37e607cc` |
| `gsp_matplotlib-0.2.0a1-py3-none-any.whl` | `5935b2bb5845449d3ba63391c68d29492a89dd458ae224ead85f99e336eade6b` |
| `gsp_datoviz-0.2.0a1-py3-none-any.whl` | `675edf044d778b7d64bd66c38b39c54a63a1fe623405437ade70f69d2646d116` |
| `gsp_core-0.2.0a1.tar.gz` | `4fa75a4bce1b50f31443cc05e004a44cabe444b324956029aba38c7b30e2599c` |
| `gsp_matplotlib-0.2.0a1.tar.gz` | `d86b2a4df85588c42b2c7a8b7a94ce378621f301d60c46ac15a51234b2887bfc` |
| `gsp_datoviz-0.2.0a1.tar.gz` | `d59c9532bc68daa00626d484fba5e215e9841d99ee9cab5f88f928d2c95cb954` |
