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

Start with the [protocol and backend guide](docs/protocol-and-backends.md) for session ownership,
capability checks, queries, and backend limitations. High-level plotting journeys and reviewed
artifacts live in the VisPy2 repository.

During the unpublished bootstrap, install the built wheels together from `dist/`; there is no
repository-root umbrella distribution. The Datoviz adapter intentionally has no ordinary Datoviz
dependency yet because the required RC3-compatible artifact is not published. Local development
sets `GSP_DATOVIZ_SOURCE=/path/to/datoviz` for explicit source-checkout probing. That bootstrap is
not a release installation claim.

The source repository is [vispy/gsp](https://github.com/vispy/gsp). No public package release is
configured during the bootstrap.
