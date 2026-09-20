# ADR-0036: Multi-Panel Scenes Use Explicit View Collections

Status: proposed

Date: 2026-09-20

## Context

P038 already gives a scene multiple panel identities, requires an explicit placement for every
panel, and gives resolved layout snapshots a collection of per-panel records. The Python `Scene`
snapshot nevertheless has singular `view2d` and `view3d` fields. Attachments name a `view_id`, but
validation can currently resolve that identifier only against those two singular fields. Both
initial adapters consequently reject multi-panel execution even though layout resolution itself is
multi-panel capable.

Adding panels only in a producer or backend would make view identity, queries, navigation,
snapshot freshness, clipping, and diagnostics depend on implicit ordering. The protocol needs one
model before adapters implement native grids or subplot APIs.

## Proposed decision

Replace the singular scene fields with explicit collections:

```text
Scene.views2d: tuple[View2D, ...]
Scene.views3d: tuple[View3D, ...]
```

View identifiers are unique across both collections. Each view names exactly one existing panel.
A panel has at most one primary data view across both collections in GSP 0.2. A viewless panel is
valid for visuals expressed entirely in NDC. Overlay or secondary data views remain deferred until
they have an explicit attachment, transform, query, and navigation contract.

Every DATA-space visual has an explicit `VisualAttachment`. The attachment's `panel_id` and the
referenced view's `panel_id` must match. NDC visuals may use a viewless attachment only after the
attachment schema makes `view_id` optional; no implementation infers a view from panel order.

Queries name a panel and report the resolved view snapshot used for inverse mapping. A query that
could match more than one panel or view is rejected rather than routed by creation order.
Navigation actions continue to name `view_id`; each accepted result changes only that view's
revision and projection snapshot. Layout snapshot identity remains scene-wide because one resize
or reservation change may affect several panels.

Backends advertise multi-panel execution separately from multi-panel layout inspection. Until an
adapter proves creation, clipping, guide layout, capture, query routing, resize, and teardown for
all panels, it rejects a scene containing more than one panel before allocating graphics
resources.

## Migration shape

Because the packages are unpublished alphas, the preferred migration is a deliberate schema
break rather than permanent singular aliases:

```text
view2d=None             -> views2d=()
view2d=value            -> views2d=(value,)
view3d=None             -> views3d=()
view3d=value            -> views3d=(value,)
```

The migration tool may perform this rewrite for recorded fixtures. Runtime constructors, parsers,
and adapters do not accept both shapes. VisPy2 may retain singular producer conveniences only when
they lower to the collection fields before constructing a GSP `Scene`.

## Required conformance before acceptance

- reject duplicate view IDs across the 2D and 3D collections;
- reject unknown panels and more than one primary data view for a panel;
- reject attachment/view panel mismatches;
- resolve two disjoint panel rectangles without a singular shortcut;
- route DATA transforms, queries, and navigation by explicit panel/view identity;
- preserve per-view revisions while changing a different panel's view;
- prove per-panel clipping and guide boxes after resize;
- require adapters without the multi-panel execution capability to fail before allocation;
- round-trip the new scene schema and provide a one-way fixture migration from the singular form.

## Consequences

- Core scene state becomes consistent with the already multi-panel layout model.
- VisPy2 can later expose subplot topology without embedding producer objects in GSP.
- Matplotlib and Datoviz require coordinated adapter work rather than independent panel loops.
- This proposal intentionally does not authorize implementation. Accepting it is a breaking schema
  decision and requires project-owner approval plus a numbered migration decision.

## Rejected alternatives

- parallel arrays keyed by tuple position;
- one implicit scene-wide view reused by every panel;
- allowing attachment `panel_id` and view `panel_id` to disagree;
- backend-specific subplot extensions;
- retaining singular fields alongside collections indefinitely;
- implementing adapter grids before query and navigation routing are specified.
