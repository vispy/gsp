# Local bootstrap qualification

## M305 P038 canonical protocol refactor gate

Date: 2026-07-30

The unpublished P038 working tree replaces the singular panel viewport model with identity-only
panels, required versioned scene layout, per-panel resolved geometry, attachment-owned clipping,
and provider-neutral session capabilities. It deliberately provides no runtime aliases for the
superseded unpublished protocol.

| Gate | Result |
|---|---|
| Complete GSP source pytest | 802 passed |
| Strict mypy | 51 source files clean |
| Ruff | packages and conformance clean |
| JSON schema syntax | all P038 schemas valid |
| Fresh installed-wheel set with VisPy2 | 928 passed |
| Built distributions | `gsp-core`, `gsp-matplotlib`, `gsp-datoviz`, and `vispy2` wheels built and installed |

The installed-wheel gate used Matplotlib 3.11.1 and NumPy 2.5.1. Repository-local conformance
fixtures were supplied only as test data; all product imports resolved from the fresh environment.
`gsp-datoviz` remains a development-only artifact until an ordinary compatible Datoviz dependency
is available. No version, tag, push, publication, merge, or Datoviz source change was performed.

## M303 pre-release mechanical correction gate

Date: 2026-07-30

Committed GSP head: `aee00ca22f52b8168ab6d5e6ceb877b218452729`.

The intended first ordinary publication set is `gsp-core`, `gsp-matplotlib`, and VisPy2.
`gsp-datoviz` remains a separately built development artifact until a compatible Datoviz runtime
is ordinarily resolvable.

| Gate | Result |
|---|---|
| Complete GSP source pytest | 804 passed |
| Strict mypy | 51 source files clean |
| Ruff | packages and conformance clean |
| Three-package isolated qualification with VisPy2 | 628 passed, one Datoviz-only conformance module skipped |
| Wheel and sdist builds | all four projects built; Datoviz classified development-only |
| Twine | all eight wheel/sdist checks passed without warnings |
| Wheel contents | all four wheels passed |
| Licensing | SPDX `BSD-3-Clause` plus packaged LICENSE in every wheel |

Candidate wheel hashes:

| Artifact | SHA-256 |
|---|---|
| `gsp_core-0.2.0a1-py3-none-any.whl` | `82b4701800f798c0d9e98002727199ecbad4859433e9a6d063eb3f75e98c06ea` |
| `gsp_matplotlib-0.2.0a1-py3-none-any.whl` | `a04780d70dfc64648814ed0a74af29a736eef07214b634af032e667c7512268a` |
| development-only `gsp_datoviz-0.2.0a1-py3-none-any.whl` | `a940f23cc1c54b41fe204b11214396d1722f9fad7a876d01e5520e1fe2e8b83a` |

This gate performs no version, tag, push, or publication operation. P038 still blocks the
independent Panel and producer-capability protocol refactor.

## Original local bootstrap

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

## S063 live View2D interaction qualification

On 2026-07-22, all four wheels were rebuilt after the live View2D synchronization changes and
installed into clean Python 3.13 environments outside both source repositories.

| Combination | Result |
|---|---|
| `gsp-core` only | 167 passed |
| `gsp-core` + `vispy2` only | 10 passed; semantic example passed |
| `gsp-core` + `gsp-matplotlib` + `vispy2` | 140 passed |
| `gsp-core` + `gsp-datoviz` + `vispy2` | 163 passed |
| GSP source workspace | 451 passed; strict mypy and Ruff passed |
| VisPy2 source | Strict mypy and Ruff passed |
| All four installed wheels | Matplotlib and Datoviz offscreen PNGs passed from `site-packages` imports |

Equivalent canonical pan actions produced equal provider ranges and revision transitions. Existing
Datoviz retained-navigation tests confirmed no unchanged visual-buffer upload, Texture2D nearest and
linear expectations remained exact, and controller callbacks unsubscribed on close. The project
owner manually accepted the same installed-wheel VisPy2 scene in both live windows: DATA points and
grid navigation remained synchronized while the NDC overlay stayed fixed.

## Artifact SHA-256

| Artifact | SHA-256 |
|---|---|
| `gsp_core-0.2.0a1-py3-none-any.whl` | `727ec6d12078b8abf2aa1f3eebc6373704eba6a2e17b5c98256c9e8f37e607cc` |
| `gsp_matplotlib-0.2.0a1-py3-none-any.whl` | `5935b2bb5845449d3ba63391c68d29492a89dd458ae224ead85f99e336eade6b` |
| `gsp_datoviz-0.2.0a1-py3-none-any.whl` | `675edf044d778b7d64bd66c38b39c54a63a1fe623405437ade70f69d2646d116` |
| `gsp_core-0.2.0a1.tar.gz` | `4fa75a4bce1b50f31443cc05e004a44cabe444b324956029aba38c7b30e2599c` |
| `gsp_matplotlib-0.2.0a1.tar.gz` | `d86b2a4df85588c42b2c7a8b7a94ce378621f301d60c46ac15a51234b2887bfc` |
| `gsp_datoviz-0.2.0a1.tar.gz` | `d59c9532bc68daa00626d484fba5e215e9841d99ee9cab5f88f928d2c95cb954` |
