from __future__ import annotations

import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import DMTet, Points, TriangleMesh
from rfstudio.io import dump_float32_image
from rfstudio.ui import console
from rfstudio.visualization import Visualizer

from .utils import RadFoamProxy


@dataclass
class Script(Task):
    """Load a checkpoint from a rfstudio-trained nerf into radfoam and dump depth images."""

    load: Path = ...
    mesh_proxy: Optional[Path] = None
    output: Optional[Path] = None
    rgb: bool = True

    num_points: int = 524_288
    bg: float = 0.0
    viser: Visualizer = Visualizer()
    debug: bool = False

    @torch.no_grad()
    def run(self) -> None:
        """Main execution function."""
        vis = {}
        if self.mesh_proxy is None:
            pts = Points(torch.randn(self.num_points) * 0.6)
        else:
            mesh = TriangleMesh.from_file(self.mesh_proxy)
            pts = mesh.uniformly_sample(self.num_points)
            pts.positions.add_(torch.randn_like(pts.positions) * 0.04)
            vis['mesh_proxy'] = mesh

        # 1. Create RadFoamProxy from the NeRF model
        with console.status("Creating RadFoam model from NeRF..."):
            proxy = RadFoamProxy.from_nerf(
                self.load,
                points=pts,
                device=self.device,
            )
            cameras = proxy.get_cameras()
            pts = proxy.get_pts().cpu()
            vis['cameras'] = cameras
            vis['cell_centers'] = pts

        console.print("RadFoam model created from NeRF.")
        gc.collect()
        torch.cuda.empty_cache()

        if self.debug:
            tet = DMTet(
                vertices=pts.positions,
                indices=proxy.model.triangulation.tets().cpu().long(),
                sdf_values= torch.zeros_like(pts.positions[..., :1]),
            )
            vis['tet'] = tet

        if self.output is None:
            self.viser.show(**vis)
            return

        # 2. Dump depth and RGB images
        assert not self.output.exists() or self.output.is_dir()
        with console.progress('Rendering Depth', transient=True) as ptrack:
            depths = proxy.get_depths(progress_handle=ptrack)
        if self.rgb:
            with console.progress('Rendering RGB', transient=True) as ptrack:
                rgbs = proxy.get_rgbas(progress_handle=ptrack).cpu().blend((self.bg, self.bg, self.bg))
                gt_rgbs = proxy.get_gt_images().cpu().blend((self.bg, self.bg, self.bg))
        self.output.mkdir(exist_ok=True, parents=True)
        with console.progress('Dumping', transient=True) as ptrack:
            for i in ptrack(range(len(depths))):
                img = torch.cat((
                    depths[i].compute_pseudo_normals(cameras=cameras[i]).visualize((self.bg, self.bg, self.bg)).item(),
                    depths[i].visualize(max_bound=8).item(),
                ), dim=1)
                if self.rgb:
                    img = torch.cat((
                        torch.cat((rgbs[i].item(), gt_rgbs[i].item()), dim=1),
                        img.cpu(),
                    ), dim=0)
                dump_float32_image(self.output / f'{i:04d}.png', img.clamp(0, 1))
        console.print(f"Images dumped to [green]{self.output}[/green]")

if __name__ == '__main__':
    Script(cuda=0).run()
