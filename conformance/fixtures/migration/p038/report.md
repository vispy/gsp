# P038 internal fixture migration report

The deterministic one-shot migration processed one scene, one panel, one view, and one attachment.

- Panel IDs were unique before producer figure identity was removed.
- The normalized outer allocation moved to `layout.panel.explicit_rects.v1`.
- Disabled view clipping became attachment `render_target` clipping.
- No identifiers were rewritten and no runtime compatibility path was created.
