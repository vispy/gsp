# AGENTS.md

## Authority

Follow this order: project charter, architecture, specification, accepted ADRs, conformance, then
implementation. Source material from `GSP_API` is not authoritative merely because it exists.

## Repository boundary

- `gsp-core` must not depend on Matplotlib, Datoviz, VisPy2, legacy GSP objects, network, or Pydantic.
- Adapters depend on `gsp-core`; core never imports adapters.
- Backend discovery is lazy and metadata inspection creates no graphics resources.
- Unsupported behavior and adaptations are explicit and diagnostic-bearing.
- Do not add remotes, push, tag, publish, or import legacy Git history without owner approval.

## Migration

Record source commit/path/blob provenance for every imported file. Do not copy an archive-only
component wholesale. Validate built wheels rather than relying only on editable source imports.

