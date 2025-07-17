from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from rfstudio.engine.task import Task
from rfstudio.graphics import Points
from rfstudio.visualization import Visualizer


@dataclass
class Script(Task):

    load: Path = ...
    viser: Visualizer = Visualizer()

    def run(self) -> None:
        assert self.load.suffix == '.pt'
        state_dict = torch.load(self.load)
        pts = Points(
            positions=state_dict['xyz'],
            colors=state_dict['color_dc'],
        )
        self.viser.show(pts=pts)

if __name__ == '__main__':
    Script(cuda=0).run()
