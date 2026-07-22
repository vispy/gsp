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

No GitHub remote or public release is configured during the local bootstrap.

