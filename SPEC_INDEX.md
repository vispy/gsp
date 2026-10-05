# GSP Specification Index

This repository is the canonical specification home for GSP 0.2. The chapters under `docs/specification/` are the current normative reading path; accepted detailed requirement statements are indexed in `specs/requirements/requirements.json`. Historical GSP_API paths recorded in the inventory and disposition files are provenance only and do not imply that those files exist locally or remain normative.

## GSP 0.2 specification

| Topic | Authoritative chapter
|---|---|
| Scope, conformance language, and reading order | `docs/specification/index.md`
| Sessions, commands, batches, frames, and shutdown | `docs/specification/protocol.md`
| Identifiers, panels, views, visuals, guides, and state relationships | `docs/specification/scene.md`
| Buffers, textures, locality, and virtual data | `docs/specification/resources.md`
| Eleven accepted visual families and their semantics | `docs/specification/visuals.md`
| Coordinate spaces, transforms, View2D, View3D, navigation, and layout | `docs/specification/views-layout.md`
| Capability negotiation, adaptation, and diagnostics | `docs/specification/capabilities.md`
| Panel queries, readback, payloads, and snapshot coherence | `docs/specification/queries.md`
| Transport independence, in-process exchange, debug JSON, and extensions | `docs/specification/transports-extensions.md`
| Matplotlib, Datoviz v0.4, and legacy implementation boundaries | `docs/specification/backend-profiles.md`
| Stable command, capability, diagnostic, and payload identifiers | `docs/specification/registries.md` |

The documentation website publishes these files under its **Specification** navigation. The requirement registry links each accepted requirement to a current chapter and to test files that provide domain-level evidence. Those test links do not assert that each test independently verifies each listed rule. Chapter prose and registered requirements remain subject to further consolidation where field-level detail is not yet present.

## Supporting material

`specs/requirements/source_inventory.json` and `source_dispositions.json` preserve legacy path and migration provenance. Their source paths are archival references, not a claim that legacy files are present in this checkout. `specs/requirements/requirements.json` is the machine-readable register of accepted requirements and destinations. It does not by itself establish rule-level test coverage.

When wording conflicts, use the authority order in `AGENTS.md`: charter, architecture, this index, the current specification chapters, accepted ADRs, conformance, then implementation. Accepted ADRs explain rationale. Conformance fixtures and backend evidence validate implementation claims but do not redefine protocol semantics.
