# GSP Specification Index

This repository is the canonical specification home for GSP 0.2. Copies retained in the historical GSP_API repository are archival evidence and are not current authority. The concise `docs/specification/` chapters are the target organization and public reading path, but they remain a **consolidation draft** until requirement-level traceability is complete. During this transition, accepted detailed topic specifications in this repository retain normative authority for rules not yet migrated.

## GSP 0.2 consolidation draft

| Topic | Authoritative chapter | |---|---| | Scope, conformance language, and reading order | `docs/specification/index.md` | | Sessions, commands, batches, frames, and shutdown | `docs/specification/protocol.md` | | Identifiers, panels, views, visuals, guides, and state relationships | `docs/specification/scene.md` | | Buffers, textures, locality, and virtual data | `docs/specification/resources.md` | | Eleven accepted visual families and their semantics | `docs/specification/visuals.md` | | Coordinate spaces, transforms, View2D, View3D, navigation, and layout | `docs/specification/views-layout.md` | | Capability negotiation, adaptation, and diagnostics | `docs/specification/capabilities.md` | | Panel queries, readback, payloads, and snapshot coherence | `docs/specification/queries.md` | | Transport independence, in-process exchange, debug JSON, and extensions | `docs/specification/transports-extensions.md` | | Matplotlib, Datoviz v0.4, and legacy implementation boundaries | `docs/specification/backend-profiles.md` | | Stable command, capability, diagnostic, and payload identifiers | `docs/specification/registries.md` |

The documentation website publishes these files under its **Specification** navigation. Their purpose is to provide a coherent semantic map while the detailed contracts are normalized.

## Supporting material

The topic files directly under `specs/` retain detailed validation tables, diagnostic vocabularies, fixture requirements, and accepted semantic rules while that material is folded into the target chapters. A rule remains normative in its accepted topic specification until the traceability registry records its GSP 0.2 destination or explicit disposition. The source inventory lives under `specs/requirements/`.

When wording conflicts, use the authority order in `AGENTS.md`: charter, architecture, this index, accepted detailed topic specification, then implementation. Accepted ADRs explain rationale. Conformance fixtures and backend evidence validate implementation claims but do not redefine protocol semantics.
