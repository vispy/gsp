# ADR-0036: Multi-Panel Scenes Use Explicit View Collections

Status: accepted

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
valid for visuals expressed entirely in NDC. Overlay, secondary, and multi-attachment views remain
deferred until they have an explicit transform, query, and navigation contract.

Every rendered visual has exactly one explicit `VisualAttachment`. `VisualAttachment.view_id` is
optional: DATA-space visuals require it, while NDC visuals require `view_id=None`. The attachment's
`panel_id` and the referenced view's `panel_id` must match. The attachment does not repeat the
visual's coordinate space; that semantic field remains owned by the visual. No implementation
infers a panel or view from collection order.

The attachment is the scene-level owner of visibility, cross-visual `z_order`, and clipping. A
visual-family field may order primitives within that visual, but it does not override attachment
visibility, panel membership, clipping, or cross-visual ordering.

Core exposes identifier-based lookup helpers for views, primary panel views, attachments, and
panel visual membership. These helpers reject missing or ambiguous identities rather than falling
back to tuple position.

Queries name a panel and report the resolved view snapshot used for inverse mapping. A query that
could match more than one panel or view is rejected rather than routed by creation order.
Navigation is routed to one explicit view: View3D requests name it directly, while View2D actions
use a controller whose retained binding names it. Each accepted result changes only that view's
revision and projection snapshot. Layout snapshot identity remains scene-wide because one resize
or reservation change may affect several panels.

Backends advertise multi-panel execution separately from multi-panel layout inspection. Until an
adapter proves creation, clipping, guide layout, capture, query routing, resize, and teardown for
all panels, it rejects a scene containing more than one panel before allocating graphics
resources.

## Migration shape

Because the packages are unpublished alphas, this is a deliberate schema break rather than a
compatibility layer:

```text
view2d=None             -> views2d=()
view2d=value            -> views2d=(value,)
view3d=None             -> views3d=()
view3d=value            -> views3d=(value,)
```

The one-way migration also creates exactly one attachment for each legacy visual when the source
scene has one unambiguous panel and view. NDC visuals receive a viewless attachment. A legacy
multi-panel scene without explicit attachments fails migration as ambiguous. Runtime constructors,
parsers, and adapters accept only the new form and expose no singular compatibility properties.
VisPy2 may retain singular producer conveniences only when they lower to the collection fields and
explicit attachments before constructing a GSP `Scene`.

## Required conformance before acceptance

- reject duplicate view IDs across the 2D and 3D collections;
- reject unknown panels and more than one primary data view for a panel;
- require exactly one attachment for every visual and reject duplicate attachment visual IDs;
- reject attachment/view panel mismatches;
- require DATA attachments to name the dimensionally correct view and NDC attachments to be
  viewless;
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
- Existing single-panel producers remain convenient, but convenience lowering ends before the
  backend-neutral `Scene` boundary.

## Rejected alternatives

- parallel arrays keyed by tuple position;
- one implicit scene-wide view reused by every panel;
- allowing simultaneous primary View2D and View3D records on one panel in the first slice;
- allowing attachment `panel_id` and view `panel_id` to disagree;
- backend-specific subplot extensions;
- retaining singular fields alongside collections indefinitely;
- implementing adapter grids before query and navigation routing are specified.

## Acceptance record

The project owner accepted this direction on 2026-09-20 after review of the schema, adapter,
Datoviz, and VisPy2 consequences. ADR-0036 is the numbered migration decision for this unpublished
GSP 0.2 schema break.
