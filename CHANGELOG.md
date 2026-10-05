# Changelog

This file records user-visible GSP changes. The project is not yet published; `0.2.0a1` describes
the current experimental candidate rather than a package-index release.

## Unreleased

### Protocol

- Detached semantic array payloads from caller storage and made them permanently read-only.
- Added optional `PointUpdateSession` (`scene.update.points.v1`) with stable resource identity,
  complete structural validation, and per-scene revisions.
- Exposed bounded Datoviz `MeshPickSession` (`query.mesh.single.v1`) through caller-owned sessions,
  preserving explicit native freshness, stale results, and unsupported multi-visual cases.

- Made `Panel` identity-only and moved outer allocation into required scene-level
  `layout.panel.explicit_rects.v1` intent.
- Made resolved layout explicitly per-panel with one protocol-defined edge quantizer.
- Moved rectangular clipping from `View2D` to `VisualAttachment.clip_scope`.
- Removed producer-emission identifiers from GSP capabilities; only renderer execution
  capabilities remain.

### Packaging

- Defined `gsp-core` and `gsp-matplotlib` as the intended first ordinary publication set.
- Kept `gsp-datoviz` development-only until a compatible Datoviz runtime is ordinarily resolvable.
- Added package long descriptions, SPDX licensing, project URLs, classifiers, and license files.

### Documentation

- Repaired specification tables and requirement destinations; added automated traceability checks.
- Split adapter orchestration from visual lowering, queries, capture, layout, and navigation.
- Fixed Matplotlib attachment ordering, failed-render cleanup, and point footprint queries across
  zoom, DPI, and device scale; corrected Datoviz semantic identity in individual query hits.

- Designated the fresh-root GSP repository as the canonical GSP 0.2 specification home.
- Corrected current native gallery and DATA-space image evidence.

## 0.2.0a1 — unpublished candidate

### Added

- Backend-independent semantic scene, resource, view, layout, capability, diagnostic, transport,
  query, and security records.
- Lazy backend discovery and explicit caller-owned sessions.
- Matplotlib reference/publication provider.
- Development Datoviz v0.4 retained GPU provider with capability-gated execution.
- Versioned conformance fixtures and installed-wheel qualification.

### Boundaries

- The canonical specification remains an alpha contract under active pre-publication refinement.
- Binary IPC and production remote transports are reserved.
- Comprehensive 3D, image-texel, and glyph readback remain capability-gated or unsupported.
- Datoviz requires an explicit compatible development source until an ordinary dependency exists.
