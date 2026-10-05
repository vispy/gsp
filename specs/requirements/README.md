# GSP 0.2 requirement traceability

This directory prevents semantic loss during the pre-1.0 specification and API consolidation.

## Stable identifiers

Normative requirements use `GSP-<DOMAIN>-NNN`, where `<DOMAIN>` is one of:

| Domain | Scope |
|---|---|
| `CORE` | conformance language, identifiers, ownership, and common validation |
| `LIFE` | initialization, sessions, commands, batches, frames, and shutdown |
| `SCENE` | panels, views, visuals, guides, attachments, and state relationships |
| `DATA` | buffers, textures, locality, virtual sources, and materialization |
| `VIS` | cross-cutting and family-specific visual semantics |
| `VIEW` | coordinate spaces, transforms, layout, navigation, and cameras |
| `CAP` | capability negotiation, adaptation, limits, and diagnostics |
| `QUERY` | panel queries, readback, payloads, ordering, and snapshots |
| `XPORT` | in-process, debug JSON, binary, and network transport contracts |
| `EXT` | extensions, manifests, security policy, and custom data sources |
| `PROD` | producer conformance requirements |

Identifiers are never reused. Removing or replacing a requirement records a disposition instead of
deleting its history.

## Files

- `source_inventory.json` records source classifications and provenance; legacy source paths do not promise local files or current normative authority.
- `source_dispositions.json` records migration destinations for historical source material.
- `requirements.json` is the accepted GSP 0.2 requirement registry. Destination paths point to current chapters; test paths are domain-level evidence references, not rule-level coverage claims.
- `schema.json` defines the machine-checkable registry format.

Run `uv run python tools/spec_traceability.py --check` after changing specification authority or
the registries.
