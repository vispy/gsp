# GSP protocol and backends

GSP is the semantic session layer beneath VisPy2. Producers create immutable `Scene` snapshots;
caller-owned backend sessions select a renderer, validate capabilities, render, display, and
answer queries. A figure or scene never owns a native backend object.

## Install and discover

The workspace currently publishes no umbrella wheel. Build and install the three local wheels
`gsp-core`, `gsp-matplotlib`, and `gsp-datoviz`. The Datoviz development adapter also needs an
RC3-compatible checkout selected with `GSP_DATOVIZ_SOURCE`.

This block is executable after those wheels are installed:

```python
import gsp

for provider in gsp.discover_backends(probe=True):
    print(provider.name, provider.available, sorted(provider.capabilities))
```

Discovery without `probe=True` reads entry-point metadata lazily. Probing imports the provider and
checks its runtime binding. Rendering remains explicit:

```python
import gsp

with gsp.open_session("matplotlib", require={"output.file", "visual.points"}) as session:
    print(session.backend_name, session.capabilities.snapshot_id)
```

The example intentionally does not render because GSP scenes are normally produced by VisPy2.

## Capabilities and diagnostics

`open_session(..., require={...})` rejects a provider before rendering if an ordinary capability
such as `visual.mesh` or `output.file` is missing. Versioned View3D and visual contracts are tested
with `session.capabilities.supports_view3d_capability(...)`. Capability support describes a
semantic path; provider metadata and diagnostics state whether its realization is strict, adapted,
experimental, or unsupported.

Never infer support from a backend name. Datoviz capabilities depend on the probed v0.4 binding,
and experimental live View3D navigation is advertised only when explicitly opted in and the input
surface qualifies. Matplotlib is deterministic but adapts GPU-oriented depth, raster, and
billboard semantics to publication artists.

## Session lifecycle

Use a context manager for every caller-owned session:

```python
with gsp.open_session("datoviz", require={"output.file", "visual.mesh"}) as session:
    session.render(scene, target="scene.png")
```

The block is illustrative pseudocode because `scene` construction is producer-specific. A caller
must keep the session open across non-blocking display and query operations. Closing the context
releases native resources. Interactive Datoviz applications additionally close the window to
terminate the event loop; `Ctrl-C` is the terminal fallback.

## Queries

The public query model asks what rendered contribution lies under a panel coordinate. Query
requests can target the latest scene retained by a session or an explicit retained `scene_id`.
Querying a closed session or a scene that was never rendered is a lifecycle error. Once lifecycle
and request structure are valid, unsupported visual families return a structured
`QueryResult(status=UNSUPPORTED)` rather than raising.

The qualified slice proves Matplotlib point identity `HIT`/`MISS` behavior and a bounded Datoviz
point-only panel-query path. Sphere, vector, primitive, billboard, general 3D occlusion, and
per-glyph picking are not claimed. View3D ray construction is separate from item picking.

## Backend limitations

| Concern | Matplotlib | Datoviz v0.4 |
|---|---|---|
| Output | Deterministic PNG/SVG/PDF | Offscreen PNG when capture binding qualifies |
| 3D geometry | CPU projection and painter-order adaptations | Retained DATA-space GPU path when qualified |
| Sphere | Projected-circle approximation | Raycast impostor with analytic surface depth when advertised |
| Vector | Deterministic line/marker-cap adaptation | Public dense vector visual |
| Primitive | Collection adaptation; no GPU raster parity | Public primitive topology/index binding |
| Billboard text | Projected overlay; backend fonts | Projected retained overlay; default backend font |
| Titles and guides | Native axes/title layout | Native/adapted guides; partial layout snapshot |
| Query | Bounded reference paths and structured unsupported results | Proven point-only query and separate ray context |
| Live View3D | Programmatic camera snapshots only | Experimental opt-in; human review required |

Titles, tick layout, fonts, glyph metrics, antialiasing, and output raster dimensions are
backend-native, not pixel-parity contracts. Matplotlib M283 captures are 640×480; Datoviz M283
offscreen captures use the native 800×600 target. Compare semantic content, not exact pixels.

## Evidence

The VisPy2 `examples/validate_gallery.py` harness copies the gallery scripts outside both source
trees, verifies wheel-installed imports, enforces 20-second process-group timeouts with one
Datoviz retry, and hashes fourteen captures. M283 successfully produced and visually reviewed
seven Matplotlib and seven Datoviz PNGs. One earlier Datoviz invocation hung without output;
repeated isolated reruns did not reproduce a backend defect. That event remains lifecycle-stress
evidence for M284 rather than a reason to weaken capabilities.
