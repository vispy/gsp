# ADR-0037: Owned Semantic Arrays and Bounded Retained Point Updates

Status: accepted

Date: 2026-10-05

## Context

Frozen dataclasses previously retained writable caller arrays. Editing an input array changed an
accepted scene without a revision transition. Re-rendering a whole scene also allocated a new
native figure and visual even when only point positions, colors, or sizes changed.

## Decision

Visuals, inline affine transforms, transform resources, scalar color encodings, and Texture2D
own detached, permanently read-only array values. Already owned immutable byte-backed arrays may
be shared by `dataclasses.replace`. Read-only views of writable caller storage are copied. This
rule does not change explicit BufferResource memory-view borrowing and its lifetime obligations.

An optional local `PointUpdateSession` advertises `scene.update.points.v1` and provides:

```python
session.update_point(point_visual, scene_id=None) -> int
session.scene_revision(scene_id=None) -> int
```

The visual must already belong to a visible attachment in a successfully rendered scene. Its ID,
point count, coordinate dimension, coordinate space, transform binding, and scalar color binding
stay fixed. Positions, explicit colors or scalar values/alpha, and pixel diameters may change.
Core validates the complete replacement before the adapter touches native resources. The session
replaces its semantic scene only after a successful update; the returned per-scene revision starts
at zero on first render and advances on successful replacement or another render of that ID.
These revisions are independent of navigation/view revisions.

Matplotlib retains its figure and PathCollection. Datoviz retains its scene, figure, and native
point visual, uploads changed attributes, and refreshes query metadata. If a Datoviz upload fails
after mutation may have begun, the session closes rather than exposing divergent semantic and
native state. Structural rejection leaves the session and revision unchanged.

VisPy2 keeps semantic producer state only. `Figure.update_point(session, visual)` forwards the
existing scene ID and replaces its producer visual only after session success.

## Limits and evidence

This extension is an in-process convenience, not the general command server, resource-update,
batch rollback, resize, or remote transport protocol. It has no effect on other visual families.

Ownership and structural rejection have core regressions; both adapters test retained identity,
revision, failure, and query behavior. The installed-wheel gallery harness additionally moves a
point, verifies its new pixels, a HIT at the new coordinate and a MISS at the vacated coordinate,
and checks topology rejection against the selected runtime.
