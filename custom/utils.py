from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import configargparse
import torch
from rfstudio.graphics import Cameras, DepthImages, Points, RGBAImages
from rfstudio.graphics.math import safe_normalize

import radfoam
from configs import DatasetParams, ModelParams, OptimizationParams, PipelineParams
from data_loader import DataHandler
from radfoam_model.scene import RadFoamScene


class RadFoamProxy:

    def __init__(self, config: Path, *, device: Optional[torch.device] = None, split: str = 'test') -> None:
        assert config.name == 'config.yaml'
        parser = configargparse.ArgParser()

        model_params = ModelParams(parser)
        dataset_params = DatasetParams(parser)
        PipelineParams(parser)
        OptimizationParams(parser)

        # Add argument to specify a custom config file
        parser.add_argument(
            "-c", "--config", is_config_file=True, help="Path to config file"
        )

        # Parse arguments
        args = parser.parse_args(["-c", str(config)])
        checkpoint = args.config.replace("/config.yaml", "")
        model = RadFoamScene(model_params.extract(args), device=device)
        model.load_pt(f"{checkpoint}/model.pt")

        dataset_args = dataset_params.extract(args)
        data_handler = DataHandler(dataset_args, rays_per_batch=0, device=device)
        data_handler.reload(split=split, downsample=min(dataset_args.downsample))
        points, _, _, _ = model.get_trace_data()
        start_points = model.get_starting_point(
            data_handler.rays[:, 0, 0].to(device), points, model.aabb_tree
        )

        self.model = model
        self.start_points = start_points
        self.data_handler = data_handler
        self.device = device

    def get_gt_images(self) -> RGBAImages:
        return RGBAImages(torch.cat((self.data_handler.rgbs, self.data_handler.alphas), dim=-1))
    
    def get_cameras(self, *, colmap2blender: bool = True) -> Cameras:
        fx = torch.full((self.data_handler.c2ws.shape[0],), fill_value=self.data_handler.fx)
        fy = torch.full((self.data_handler.c2ws.shape[0],), fill_value=self.data_handler.fy)
        c2w = self.data_handler.c2ws[:, :3, :]
        if colmap2blender:
            c2w[:, :, 1:3] *= -1
        return Cameras(
            c2w=c2w,
            fx=fx,
            fy=fy,
            cx=torch.full_like(fx, fill_value=self.data_handler.img_wh[0] / 2),
            cy=torch.full_like(fx, fill_value=self.data_handler.img_wh[1] / 2),
            width=torch.full_like(fx, dtype=torch.long, fill_value=self.data_handler.img_wh[0]),
            height=torch.full_like(fx, dtype=torch.long, fill_value=self.data_handler.img_wh[1]),
            near=torch.full_like(fx, fill_value=1e-3),
            far=torch.full_like(fx, fill_value=1e3),
        ).to(self.device)
    
    def get_rgbas(self, *, progress_handle: Optional[Callable] = None) -> RGBAImages:
        rays = self.data_handler.rays
        rng = range(rays.shape[0])
        if progress_handle is not None:
            rng = progress_handle(rng)
        ray_batch_fetcher = radfoam.BatchFetcher(rays, batch_size=1, shuffle=False)
        outputs = []
        for i in rng:
            ray_batch = ray_batch_fetcher.next()[0]
            output, _, _, _, _ = self.model(ray_batch, self.start_points[i])
            outputs.append(output)
        return RGBAImages(torch.stack(output)).to(self.device)
    
    def get_rgbds(self, *, progress_handle: Optional[Callable] = None) -> DepthImages:
        cameras = self.get_cameras(colmap2blender=False)
        rays = self.data_handler.rays
        rng = range(rays.shape[0])
        if progress_handle is not None:
            rng = progress_handle(rng)
        ray_batch_fetcher = radfoam.BatchFetcher(rays, batch_size=1, shuffle=False)
        outputs = []
        for i in rng:
            ray_batch = ray_batch_fetcher.next()[0]
            depth = self.model.forward_expected_depth(
                ray_batch,
                cameras.c2w[i, :3, 3],
                safe_normalize(-cameras.c2w[i, :3, 2]),
                self.start_points[i],
            )
            outputs.append(depth)
        return DepthImages(torch.stack(outputs))

    def get_pts(self) -> Points:
        return Points(
            positions=self.model.primal_points,
            colors=self.model.att_dc,
        )
