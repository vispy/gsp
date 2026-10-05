"""Geometry lowering, mesh payloads, and affine adaptation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from gsp.protocol import (
    AffineTransform2DResource,
    ColorScale,
    ImageOrigin,
    ImageVisual,
    MeshColorMode,
    MeshShading,
    MeshVisual,
    View2D,
    View3D,
    View3DDiagnosticCode,
    VisualTransformBinding,
    project_view3d_data_point,
    validate_mesh_visual_flat_lambert,
)
from gsp.protocol.color_mapping import (
    resolve_color_scale,
)
from gsp.protocol.visuals import CoordinateSpace

from ._state import (
    DatovizV04Unsupported,
)
from .colors import (
    _rgba8,
    _rgba8_image,
    _rgba8_scalar_image,
    _rgba8_scalar_values,
)


def _positions_3d(
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
) -> npt.NDArray[np.float32]:
    array = np.asarray(positions, dtype=np.float32)
    if array.shape[1] == 3:
        return np.ascontiguousarray(array)
    zeros = np.zeros((array.shape[0], 1), dtype=np.float32)
    return np.ascontiguousarray(np.column_stack([array, zeros]))


def _adapt_visual_positions(
    visual_id: str,
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    transform: VisualTransformBinding | None,
    coordinate_space: CoordinateSpace,
    view: View2D | None,
    transform_resources: Mapping[str, AffineTransform2DResource] | None,
    *,
    cpu_map_data_to_view: bool = False,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    transformed = _cpu_adapt_affine_positions(visual_id, positions, transform, transform_resources)
    if coordinate_space is CoordinateSpace.NDC:
        return transformed
    if coordinate_space is CoordinateSpace.DATA:
        if cpu_map_data_to_view:
            if view is None:
                raise DatovizV04Unsupported(
                    "Datoviz data-to-view CPU adaptation requires View2D limits"
                )
            return _map_view2d_data_positions_to_view(transformed, view)
        return transformed
    raise DatovizV04Unsupported(
        f"Datoviz visual coordinate space is unsupported: {coordinate_space.value}"
    )


def _map_view2d_data_positions_to_view(
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64], view: View2D
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    array = np.asarray(positions)
    mapped = array.astype(np.float64, copy=True)
    x0, x1 = view.x_range
    y0, y1 = view.y_range
    mapped[:, 0] = ((mapped[:, 0] - x0) / (x1 - x0)) * 2.0 - 1.0
    mapped[:, 1] = ((mapped[:, 1] - y0) / (y1 - y0)) * 2.0 - 1.0
    return np.ascontiguousarray(mapped.astype(array.dtype, copy=False))


def _cpu_adapt_affine_positions(
    visual_id: str,
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    transform: VisualTransformBinding | None,
    transform_resources: Mapping[str, AffineTransform2DResource] | None,
) -> npt.NDArray[np.float32] | npt.NDArray[np.float64]:
    """Apply bounded S027 CPU transform adaptation for finite eager positions."""
    if transform is None:
        return positions
    xy = np.asarray(positions[:, :2], dtype=np.float64)
    homogeneous = np.column_stack([xy, np.ones((xy.shape[0],), dtype=np.float64)])
    matrix = _transform_binding_matrix(transform, transform_resources)
    transformed = np.asarray((homogeneous @ matrix.T)[:, :2], dtype=np.float64)
    if positions.shape[1] == 3:
        z = np.asarray(positions[:, 2:3], dtype=np.float64)
        transformed = np.column_stack([transformed, z])
    if not np.all(np.isfinite(transformed)):
        raise DatovizV04Unsupported(
            f"GSP_TRANSFORM_NONFINITE: CPU transform adaptation produced "
            f"non-finite positions for {visual_id}"
        )
    return np.ascontiguousarray(transformed.astype(positions.dtype, copy=False))


def _transform_binding_matrix(
    transform: VisualTransformBinding,
    transform_resources: Mapping[str, AffineTransform2DResource] | None,
) -> npt.NDArray[np.float64]:
    if transform.inline is not None:
        return np.asarray(transform.inline.matrix, dtype=np.float64)
    if transform.ref is None:
        raise DatovizV04Unsupported(
            "GSP_TRANSFORM_UNSUPPORTED_KIND: Datoviz transform binding is invalid"
        )
    if transform_resources is None or transform.ref.id not in transform_resources:
        if not transform.ref.required:
            return np.eye(3, dtype=np.float64)
        raise DatovizV04Unsupported(
            "GSP_TRANSFORM_MISSING_REF: Datoviz v0.4 transform CPU adapter "
            f"could not resolve named transform resource {transform.ref.id!r}"
        )
    return np.asarray(transform_resources[transform.ref.id].matrix, dtype=np.float64)


def _record_transform_adaptation(
    adaptations: dict[str, tuple[str, ...]],
    visual_id: str,
    transform: VisualTransformBinding | None,
) -> None:
    if transform is None:
        return
    adaptations[visual_id] = (
        "cpu_adapter_affine2d_eager_ndc",
        "query_inverse_unsupported",
    )


def _validate_datoviz_mesh3d_visual(visual: MeshVisual, view3d: View3D | None) -> None:
    if visual.canonical_shading() is MeshShading.FLAT_LAMBERT:
        try:
            validate_mesh_visual_flat_lambert(visual, view3d=view3d)
        except ValueError as exc:
            raise DatovizV04Unsupported(str(exc)) from exc
    if visual.transform is not None:
        raise DatovizV04Unsupported(
            f"{View3DDiagnosticCode.MESH3D_TRANSFORM_UNSUPPORTED.value}: "
            "Datoviz MeshVisual 3D path does not apply 2D affine transforms"
        )
    if visual.coordinate_space is CoordinateSpace.DATA and view3d is None:
        raise DatovizV04Unsupported(
            f"{View3DDiagnosticCode.MESH3D_REQUIRES_VIEW3D.value}: "
            "Datoviz MeshVisual DATA positions3d require View3D"
        )
    if visual.coordinate_space not in {CoordinateSpace.DATA, CoordinateSpace.NDC}:
        raise DatovizV04Unsupported(
            f"{View3DDiagnosticCode.MESH3D_COORDINATE_SPACE_UNSUPPORTED.value}: "
            f"unsupported MeshVisual 3D coordinate_space {visual.coordinate_space!r}"
        )
    if visual.color is None:
        return
    colors = _rgba8(np.asarray(visual.color))
    if bool(np.any(colors.reshape(-1, 4)[:, 3] < 255)):
        raise DatovizV04Unsupported(
            f"{View3DDiagnosticCode.MESH3D_ALPHA_NOT_STRICT.value}: "
            "Datoviz MeshVisual 3D depth path requires opaque colors"
        )


def _datoviz_mesh3d_plot_ndc_positions(
    visual: MeshVisual, *, view3d: View3D | None, aspect_ratio: float
) -> npt.NDArray[np.float64]:
    source = np.asarray(visual.positions, dtype=np.float64)
    if visual.coordinate_space is CoordinateSpace.DATA:
        if view3d is None:
            raise DatovizV04Unsupported(
                f"{View3DDiagnosticCode.MESH3D_REQUIRES_VIEW3D.value}: "
                "Datoviz MeshVisual DATA positions3d require View3D"
            )
        return np.asarray(
            [
                project_view3d_data_point(view3d, tuple(point), aspect_ratio=aspect_ratio)
                for point in source
            ],
            dtype=np.float64,
        )
    if visual.coordinate_space is CoordinateSpace.NDC:
        return source
    raise DatovizV04Unsupported(
        f"{View3DDiagnosticCode.MESH3D_COORDINATE_SPACE_UNSUPPORTED.value}: "
        f"unsupported MeshVisual 3D coordinate_space {visual.coordinate_space!r}"
    )


def _datoviz_mesh_payload(
    visual: MeshVisual,
    positions: npt.NDArray[np.float32] | npt.NDArray[np.float64],
    *,
    view3d: View3D | None = None,
    face_order: npt.NDArray[np.int64] | None = None,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.uint8], npt.NDArray[np.uint32]]:
    if visual.face_color_encoding is not None:
        raise DatovizV04Unsupported("Datoviz MeshVisual face scalar color encoding is unavailable")
    if visual.color is None:
        raise DatovizV04Unsupported("Datoviz MeshVisual requires resolved RGBA colors")
    color_mode = visual.resolved_color_mode()
    positions = np.asarray(positions, dtype=np.float32)
    faces = np.asarray(visual.faces, dtype=np.uint32)
    colors = _rgba8(np.asarray(visual.color))

    if visual.canonical_shading() is MeshShading.FLAT_LAMBERT:
        colors = _resolve_datoviz_flat_lambert_facecolors(
            visual,
            np.asarray(visual.color),
            color_mode=color_mode,
            view3d=view3d,
        )
        color_mode = MeshColorMode.FACE

    if face_order is not None:
        faces = faces[face_order]
        if color_mode is MeshColorMode.FACE:
            colors = colors.reshape(visual.faces.shape[0], 4)[face_order]

    if color_mode is MeshColorMode.UNIFORM:
        rgba = np.asarray(colors, dtype=np.uint8).reshape(1, 4)
        vertex_colors = np.repeat(rgba, positions.shape[0], axis=0)
        return (
            _positions_3d(positions),
            np.ascontiguousarray(vertex_colors),
            np.ascontiguousarray(faces.reshape(-1)),
        )
    if color_mode is MeshColorMode.VERTEX:
        return (
            _positions_3d(positions),
            np.ascontiguousarray(colors.reshape(positions.shape[0], 4)),
            np.ascontiguousarray(faces.reshape(-1)),
        )
    if color_mode is MeshColorMode.FACE:
        triangle_positions = positions[faces].reshape(-1, positions.shape[1])
        triangle_colors = np.repeat(colors.reshape(faces.shape[0], 4), 3, axis=0)
        triangle_indices = np.arange(triangle_positions.shape[0], dtype=np.uint32)
        return (
            _positions_3d(np.ascontiguousarray(triangle_positions)),
            np.ascontiguousarray(triangle_colors),
            np.ascontiguousarray(triangle_indices),
        )
    raise DatovizV04Unsupported(f"unsupported mesh color mode: {color_mode}")


def _mesh3d_face_depth_order(
    vertex_depth: npt.NDArray[np.float64], faces: npt.NDArray[np.integer]
) -> npt.NDArray[np.int64]:
    face_depth = np.mean(vertex_depth[np.asarray(faces, dtype=np.uint32)], axis=1)
    return np.argsort(-face_depth, kind="stable")


def _resolve_datoviz_flat_lambert_facecolors(
    visual: MeshVisual,
    colors: npt.NDArray[Any],
    *,
    color_mode: MeshColorMode,
    view3d: View3D | None,
) -> npt.NDArray[np.uint8]:
    try:
        validate_mesh_visual_flat_lambert(visual, view3d=view3d)
    except ValueError as exc:
        raise DatovizV04Unsupported(str(exc)) from exc
    if view3d is None:
        raise DatovizV04Unsupported("flat_lambert_requires_view3d: flat_lambert requires a View3D")
    if color_mode is MeshColorMode.UNIFORM:
        facecolors = np.repeat(
            np.asarray(colors).reshape(1, 4),
            visual.faces.shape[0],
            axis=0,
        )
    elif color_mode is MeshColorMode.FACE:
        facecolors = np.asarray(colors).reshape(visual.faces.shape[0], 4)
    else:
        raise DatovizV04Unsupported(
            "flat_lambert_unsupported: Datoviz S040 flat Lambert requires "
            "uniform or per-face RGBA base colors"
        )

    if facecolors.dtype == np.dtype(np.uint8):
        base = np.asarray(facecolors, dtype=np.float64) / 255.0
    else:
        base = np.clip(np.asarray(facecolors, dtype=np.float64), 0.0, 1.0)
    normals = np.asarray(visual.normalized_face_normals(), dtype=np.float64)
    light_factor = np.full(
        (normals.shape[0],),
        float(view3d.ambient_light_intensity),
        dtype=np.float64,
    )
    if view3d.directional_light is not None:
        light_direction = np.asarray(
            view3d.directional_light.direction_to_light,
            dtype=np.float64,
        )
        light_direction = light_direction / np.linalg.norm(light_direction)
        lambert = np.maximum(0.0, normals @ light_direction)
        light_factor = light_factor + (float(view3d.directional_light.intensity) * lambert)
    light_factor = np.clip(light_factor, 0.0, 1.0)
    resolved = base.copy()
    resolved[:, :3] = np.clip(base[:, :3] * light_factor[:, np.newaxis], 0.0, 1.0)
    resolved[:, 3] = base[:, 3]
    return np.ascontiguousarray(np.rint(resolved * 255.0).clip(0, 255).astype(np.uint8))


def _diameters_from_pixel_diameters(
    sizes: npt.NDArray[np.float32] | npt.NDArray[np.float64] | float,
    count: int,
) -> npt.NDArray[np.float32]:
    if isinstance(sizes, np.ndarray):
        diameters = np.asarray(sizes, dtype=np.float32)
    else:
        diameters = np.full((count,), float(sizes), dtype=np.float32)
    return np.ascontiguousarray(diameters.reshape(-1))


def _rgba8_image_visual(
    visual: ImageVisual, *, color_scales: Mapping[str, ColorScale] | None
) -> npt.NDArray[np.uint8]:
    image = visual.image
    if image.ndim == 2:
        if visual.color_scale_id is not None:
            scale = resolve_color_scale(color_scales, visual.color_scale_id)
            return _rgba8_scalar_values(image, scale, alpha=1.0)
        return _rgba8_scalar_image(image, visual.clim)
    return _rgba8_image(image)


def _image_positions(
    extent: tuple[float, float, float, float],
) -> npt.NDArray[np.float32]:
    left, right, bottom, top = extent
    return np.ascontiguousarray(
        np.array(
            [
                [left, bottom, 0.0],
                [left, top, 0.0],
                [right, bottom, 0.0],
                [right, top, 0.0],
            ],
            dtype=np.float32,
        )
    )


def _image_texcoords(origin: ImageOrigin) -> npt.NDArray[np.float32]:
    if origin == ImageOrigin.LOWER:
        values = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    else:
        values = [[0.0, 1.0], [0.0, 0.0], [1.0, 1.0], [1.0, 0.0]]
    return np.ascontiguousarray(np.array(values, dtype=np.float32))
