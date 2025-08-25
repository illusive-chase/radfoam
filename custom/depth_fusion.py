from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rfstudio.engine.task import Task
from rfstudio.graphics import TriangleMesh
from rfstudio.ui import console
from rfstudio.visualization import Visualizer

from .utils import RadFoamProxy


@dataclass
class Script(Task):

    load: Path = ...
    viser: Visualizer = Visualizer()
    only_vis: bool = False
    output: Optional[Path] = None  # Path to save the fused mesh as a PLY file

    def run(self) -> None:
        proxy = RadFoamProxy.from_radfoam(self.load, device=self.device, split='test')
        cameras = proxy.get_cameras()
        pts = proxy.get_pts()

        if self.only_vis:
            self.viser.show(pts=pts, cameras=cameras)
            return

        with console.progress('Rendering', transient=True) as ptrack:
            depths = proxy.get_depths(progress_handle=ptrack)
        with console.progress('Fusing', transient=True) as ptrack:
            mesh = TriangleMesh.from_depth_fusion(depths=depths, cameras=cameras, progress_handle=ptrack, depth_trunc=8)

        # Save mesh as PLY if output path is provided
        if self.output is not None:
            mesh_path = self.output.with_suffix('.ply') if self.output.suffix != '.ply' else self.output
            mesh.to_ply(mesh_path)
            console.print(f"[green]Fused mesh saved to {mesh_path}[/green]")

        self.viser.show(pts=pts, fused=mesh, cameras=cameras)


if __name__ == '__main__':
    Script(cuda=0).run()
