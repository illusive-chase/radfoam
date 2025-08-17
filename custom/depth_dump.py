from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rfstudio.engine.task import Task
from rfstudio.io import dump_float32_image
from rfstudio.ui import console

from .utils import RadFoamProxy


@dataclass
class Script(Task):

    load: Path = ...
    output: Path = ...
    rgb: bool = True

    def run(self) -> None:
        assert not self.output.exists() or self.output.is_dir()
        proxy = RadFoamProxy.from_radfoam(self.load, device=self.device, split='test')
        with console.progress('Rendering Depth', transient=True) as ptrack:
            depths = proxy.get_depths(progress_handle=ptrack)
        if self.rgb:
            with console.progress('Rendering RGB', transient=True) as ptrack:
                rgbs = proxy.get_rgbas(progress_handle=ptrack).blend((1, 1, 1))
                gt_rgbs = proxy.get_gt_images().blend((1, 1, 1))
        cameras = proxy.get_cameras()
        self.output.mkdir(exist_ok=True, parents=True)
        with console.progress('Dumping', transient=True) as ptrack:
            for i in ptrack(range(len(depths))):
                img = torch.cat((
                    depths[i].compute_pseudo_normals(cameras=cameras[i]).visualize((1, 1, 1)).item(),
                    depths[i].visualize(max_bound=8).item(),
                ), dim=1)
                if self.rgb:
                    img = torch.cat((
                        torch.cat((rgbs[i].item(), gt_rgbs[i].item()), dim=1),
                        img,
                    ), dim=0)
                dump_float32_image(self.output / f'{i:04d}.png', img.clamp(0, 1))

if __name__ == '__main__':
    Script(cuda=0).run()
