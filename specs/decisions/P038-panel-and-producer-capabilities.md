# P038 panel, layout, clipping, and producer-capability decision

Status: accepted for GSP 0.2 alpha before first publication.

## Decision

`Panel` contains only scene-scoped `id`. Every `Scene` contains a required `ExplicitPanelLayoutV1` whose exact kind is `layout.panel.explicit_rects.v1`. Allocation rectangles are normalized outer-panel intent; the core quantizer resolves them once into per-panel logical-pixel geometry. `ResolvedLayoutSnapshot` contains `ResolvedPanelLayout` entries and has no singular panel or plot shortcut.

`View2D` has no clipping field. `VisualAttachment.clip_scope` selects `plot`, `panel`, or `render_target`; transforms and interactions always use the plot rectangle.

GSP session capabilities describe renderer execution. The canonical texture capabilities remain `meshvisual.material.texture2d_unlit.v1` and `meshvisual.texture_filter.linear.v1`. Producer emission support is VisPy2-local, non-wire state using `vispy2.emit.meshvisual.material.texture2d_unlit.v1` and `vispy2.emit.meshvisual.texture_filter.linear.v1`.

## Breaking migration

The old producer panel topology and allocation are split as follows:

```text
Panel(id, figure_id, viewport_rect)
  -> Panel(id)
  + PanelPlacement(panel_id=id, allocation_rect=NormalizedRenderTargetRect(...))
```

Old view clipping migrates per attachment: enabled becomes `ClipScope.PLOT`; disabled becomes `ClipScope.RENDER_TARGET`.

The unpublished producer identifiers map for evidence only:

```text
gsp_vispy2.producer.mesh.texture2d_unlit.v1
  -> vispy2.emit.meshvisual.material.texture2d_unlit.v1
gsp_vispy2.producer.mesh.texture_filter.linear.v1
  -> vispy2.emit.meshvisual.texture_filter.linear.v1
```

There are no runtime aliases, fallback parsers, warning-based adaptations, or legacy field conversions.
