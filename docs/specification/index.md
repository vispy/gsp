# Specification overview

These chapters are the current GSP 0.2 specification and public reading path. The requirement registry at `specs/requirements/requirements.json` links accepted requirements to chapters. Some chapter sections still lack complete field-level detail; where the text and registry do not yet provide a complete contract, record that consolidation gap explicitly instead of inferring behavior from missing detail. Legacy source paths in the inventory and disposition records are provenance only.

!!! warning "Specification detail gaps"
    Do not infer that an omitted field or rule has been removed. A missing field or rule in a chapter is an open documentation gap, not evidence that the behavior was removed. A breaking removal requires an explicit disposition in the requirement registry.

## How to read it

The words **must**, **must not**, **should**, and **may** describe requirements. Protocol acceptance and backend implementation are independent:

- A protocol rule defines valid meaning and data.
- A producer may or may not expose a convenience API for it.
- A backend may advertise strict, adapted, partial, or no support.
- A query capability is separate from the corresponding rendering capability.

## Specification map

| Chapter | Defines
|---|---|
| [Protocol lifecycle](protocol.md) | Sessions, initialization, commands, batches, execution, and shutdown
| [Scene model](scene.md) | Identifiers, panels, views, visuals, guides, and state relationships
| [Resources and data](resources.md) | Buffers, textures, data locality, and virtual sources
| [Visual semantics](visuals.md) | Shared visual rules and each accepted visual family
| [Views, transforms, and layout](views-layout.md) | Coordinate spaces, View2D, View3D, navigation, and resolved layout
| [Capabilities and diagnostics](capabilities.md) | Negotiation, adaptation, support states, and failures
| [Queries and readback](queries.md) | Panel queries, result states, payloads, and snapshot coherence
| [Transports and extensions](transports-extensions.md) | Encoding independence, local transport, manifests, and data sources
| [Backend profiles](backend-profiles.md) | Current Matplotlib and Datoviz implementation posture
| [Normative registries](registries.md) | Command kinds, visual families, capabilities, diagnostics, and payload IDs |

## Conformance language

`must` and `must not` are requirements. `should` identifies a recommended default whose deviation requires documented rationale. `may` grants permission. Descriptive text does not weaken a labeled `GSP-<DOMAIN>-NNN` requirement.

A conforming implementation identifies:

1. the exact GSP protocol version;
1. the transport profile;
1. a capability snapshot;
1. the requirement set claimed for each feature scope;
1. any adaptations and limits;
1. conformance evidence mapped to stable requirement IDs.

Protocol conformance, producer conformance, renderer conformance, query conformance, and transport conformance are independent claims.

## Authority and status

These chapters describe the target GSP 0.2 organization. Backend tables describe implementation evidence and never redefine the protocol. GSP remains an experimental pre-1.0 prototype; GSP 0.2 may deliberately break prototype APIs when the change is recorded in the requirement registry, specification, tests, and migration notes together.
