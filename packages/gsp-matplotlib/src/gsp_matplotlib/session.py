"""GSP session implementation for the Matplotlib reference provider."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from gsp import Scene
from gsp.backends import SessionRequest

from .capabilities import capability_snapshot
from .protocol_renderer import MatplotlibProtocolRenderResult, render_protocol_scene_with_layout


class MatplotlibSession:
    backend_name = "matplotlib"

    def __init__(self, *, request: SessionRequest) -> None:
        self.request = request
        self.capabilities = capability_snapshot()
        self._diagnostics: list[str] = []
        self._results: list[MatplotlibProtocolRenderResult] = []
        self._closed = False

    @property
    def diagnostics(self) -> tuple[str, ...]:
        return tuple(self._diagnostics)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")

    def render(
        self,
        scene: Scene,
        *,
        target: str | Path | None = None,
        output_dpi: float | None = None,
        **savefig_kwargs: Any,
    ) -> MatplotlibProtocolRenderResult:
        self._require_open()
        if not isinstance(scene, Scene):
            raise TypeError("render() requires a gsp.Scene")
        result = render_protocol_scene_with_layout(
            visuals=scene.visuals,
            view=scene.view2d,
            view3d=scene.view3d,
            axis_guides=scene.axis_guides,
            panel_text_guides=scene.panel_text_guides,
            colorbar_guides=scene.colorbar_guides,
            color_scales={item.id: item for item in scene.color_scales},
            transform_resources={item.id: item for item in scene.transforms},
            canvas_size=scene.canvas_size,
            output_dpi=output_dpi,
        )
        self._results.append(result)
        if target is not None:
            result.figure.savefig(target, **savefig_kwargs)
        return result

    def display(self, scene: Scene, **kwargs: Any) -> MatplotlibProtocolRenderResult:
        return self.render(scene, **kwargs)

    def run(self) -> None:
        self._require_open()
        import matplotlib.pyplot as plt

        plt.show()

    def close(self) -> None:
        if self._closed:
            return
        import matplotlib.pyplot as plt

        for result in self._results:
            plt.close(result.figure)
        self._closed = True

    def __enter__(self) -> "MatplotlibSession":
        self._require_open()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

