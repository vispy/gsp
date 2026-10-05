# GSP

GSP is a backend-independent graphics session protocol for scientific visualization.

This repository owns the semantic protocol, capability and diagnostic model, backend provider
interface, conformance corpus, and the first-party Matplotlib and Datoviz adapters. It is a
multi-distribution workspace; the repository root does not publish an umbrella wheel.

The repository is being curated from the historical `vispy/GSP_API` research prototype. See
`PROVENANCE.md` and `migration-manifest.json` before importing implementation material.

## Packages

- `gsp-core` imports as `gsp` and has no rendering-backend dependency.
- `gsp-matplotlib` imports as `gsp_matplotlib` and provides the reference/publication backend.
- `gsp-datoviz` imports as `gsp_datoviz` and provides the flagship GPU backend.

Backends register lazy providers in the `gsp.backends` entry-point group. `gsp.discover_backends()`
lists installed provider metadata without importing Matplotlib or Datoviz;
`gsp.discover_backends(probe=True)` performs dependency/API checks. Rendering always selects a
backend explicitly with `gsp.open_session("matplotlib")`, `gsp.open_session("datoviz")`, or a
caller-supplied ordered `prefer=` policy.

Scenes carry explicit versioned panel-allocation intent. GSP resolves it once into per-panel logical
geometry consumed by adapters; DATA transforms and interactions use each resolved plot rectangle.
Attachment clipping is explicit and producer-emission support is never a renderer capability.

Start with the [protocol and backend guide](docs/protocol-and-backends.md) for session ownership,
capability checks, queries, and backend limitations. High-level plotting journeys and reviewed
artifacts live in the VisPy2 repository.

For a guided review of project contracts and backend behavior, use the [interactive review tour](../vispy2/docs/review.md).
Run `just review-setup` once to install the Qt toolkit, then `just review`. On macOS, the recipe
configures an installed Vulkan SDK through the Datoviz source checkout's environment helper when
no Vulkan driver is explicitly selected. `GSP_DATOVIZ_SOURCE` selects that checkout, defaulting to
`../datoviz`; explicit `VK_DRIVER_FILES` or `VK_ICD_FILENAMES` settings are preserved.

The intended first ordinary publication set is `gsp-core` plus `gsp-matplotlib`; there is no
repository-root umbrella distribution. The Datoviz adapter intentionally has no ordinary Datoviz
dependency yet because the required RC3-compatible artifact is not published, so `gsp-datoviz`
remains development-only. Local development sets
`GSP_DATOVIZ_SOURCE=/path/to/datoviz` for explicit source-checkout probing.

The source repository is [vispy/gsp](https://github.com/vispy/gsp). No public package release is
configured during the bootstrap.

## Current implementation and next work

The 2026-10-05 audit covers this workspace and the tightly coupled VisPy2 producer. Both adapters
render explicit multi-panel scenes and provide lazy discovery, caller-owned sessions, capture,
navigation, and bounded queries. Core includes eleven visual families, transforms, scalar color
mapping, textures, layout and camera records, typed diagnostics, and conformance fixtures.
Semantic array records now detach writable caller storage. Both adapters expose optional retained
point-value updates with stable native resources and per-scene revisions.
Datoviz additionally exposes bounded native FACE picking when a visible DATA mesh in a View3D
panel is the scene's sole visual, with HIT/MISS/stale results and explicit native freshness.

VisPy2 provides subplot grids and mixed projections, custom panel allocations, 2D data fitting,
programmatic shared-axis groups, bars and histograms, filled bands, and reference lines/spans.
Producer links do not synchronize backend mouse navigation. The early internal refactors split
producer conversion and fitting from axes/figure APIs, and split adapter lowering, navigation,
queries, layout, and capture from session orchestration.

Datoviz qualification targets frozen main at
`066a7451195b38c5e95dcf7af7383b89ec5ec903`, the pre-RC3 engine baseline. A source checkout with RC2
package metadata is qualified by its source and native-library provenance; it is not a claim of
compatibility with a released RC2 artifact. The imminent RC3 wheel still needs its own exact-wheel
qualification. Datoviz source changes are not required for the implemented work.

The next priorities are:

1. Run the exact published RC3 wheel through the existing gallery/native gate, then enable an
   ordinary Datoviz dependency once that artifact is resolvable and qualified.
2. Complete the manual paired-window review, especially live View3D, resize, guides, and teardown.
3. Define native shared-view navigation and legend layout/query semantics before adding them.
4. Extend retained updates to other visual families through explicit bounded contracts, then
   implement the general command server and frame/batch state machine.
5. Consolidate remaining field-level specification detail and map tests to individual requirements.

Full multi-visual mesh-face picking, exact image-texel/glyph queries, Datoviz panel titles,
production remote transports, and arbitrary extensions remain deferred or unsupported. See
[qualification evidence](QUALIFICATION.md) and the [backend guide](docs/protocol-and-backends.md).
