from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rfstudio.engine.task import Task
from rfstudio.graphics import TriangleMesh
from rfstudio.ui import console
from rfstudio.visualization import Visualizer

from .utils import RadFoamProxy


@dataclass
class Script(Task):

    load: Path = ...
    viser: Visualizer = Visualizer()

    def run(self) -> None:
        proxy = RadFoamProxy(self.load, device=self.device, split='test')
        cameras = proxy.get_cameras()
        with console.progress('Rendering', transient=True) as ptrack:
            rgbds = proxy.get_rgbds(progress_handle=ptrack)
        pts = proxy.get_pts()
        with console.progress('Fusing', transient=True) as ptrack:
            mesh = TriangleMesh.from_depth_fusion(depths=rgbds, cameras=cameras, progress_handle=ptrack)
        self.viser.show(pts=pts, fused=mesh, cameras=cameras)


if __name__ == '__main__':
    Script(cuda=0).run()
