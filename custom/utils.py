from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import configargparse
import torch
import torch.nn as nn
from rfstudio.data import MeshViewSynthesisDataset, MultiViewDataset, SfMDataset
from rfstudio.engine.train import TrainTask
from rfstudio.graphics import Cameras, DepthImages, Points, RGBAImages
from rfstudio.graphics.math import rgb2sh, safe_normalize, sh2rgb
from rfstudio.model import VanillaNeRF, VanillaNeuS
from torch import Tensor

import radfoam
from configs import DatasetParams, ModelParams, OptimizationParams, PipelineParams
from data_loader import DataHandler
from data_loader.blender import get_ray_directions
from radfoam_model.scene import RadFoamScene
from radfoam_model.utils import inverse_softplus


def dataset_rfstudio2radfoam(
    dataset: Union[MultiViewDataset, SfMDataset, MeshViewSynthesisDataset],
    *,
    split: str,
    colmap2blender: bool = True,
) -> DataHandler:
    cameras = dataset.get_inputs(split=split)[...]
    assert cameras.has_all_same_resolution
    images = dataset.get_gt_outputs(split=split)[...]
    datahandler = DataHandler(None, rays_per_batch=0)
    h, w = images[0].item().shape[:2]
    datahandler.img_wh = (w, h)
    datahandler.fx = cameras.fx.unique().item()
    datahandler.fy = cameras.fy.unique().item()
    datahandler.c2ws = torch.cat((
        cameras.c2w,
        torch.tensor([0, 0, 0, 1]).to(cameras.c2w).expand(cameras.c2w.shape[0], 1, 4),
    ), dim=1)
    if colmap2blender:
        datahandler.c2ws[:, :, 1:3] *= -1
    cam_ray_dirs = get_ray_directions(
        h, w, [datahandler.fx, datahandler.fy]
    ).to(datahandler.c2ws.device)

    world_ray_dirs = torch.einsum(
        "ij,bkj->bik",
        cam_ray_dirs,
        datahandler.c2ws[:, :3, :3],
    )
    world_ray_origins = datahandler.c2ws[:, None, :3, 3] + torch.zeros_like(cam_ray_dirs)
    world_rays = torch.cat([world_ray_origins, world_ray_dirs], dim=-1)
    world_rays = world_rays.reshape(-1, h, w, 6)

    datahandler.rays = world_rays.cpu()
    datahandler.c2ws = datahandler.c2ws.cpu()

    if isinstance(images, RGBAImages):
        datahandler.rgbs = torch.stack([img for img in images.blend((1, 1, 1))]).cpu()
        datahandler.alphas = torch.stack([img[..., 3:] for img in images]).cpu()
    else:
        datahandler.rgbs = torch.stack([img for img in images]).cpu()
        datahandler.alphas = torch.ones_like(datahandler.rgbs[..., :1]).cpu()
    datahandler.batch_size = 0
    return datahandler


@dataclass
class RadFoamProxy:

    model: RadFoamScene
    start_points: Tensor
    data_handler: DataHandler
    device: Optional[torch.device]

    @classmethod
    def from_radfoam(cls, config: Path, *, device: Optional[torch.device] = None, split: str = 'test') -> RadFoamProxy:
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
        return cls(model=model, start_points=start_points, data_handler=data_handler, device=device)

    @classmethod
    @torch.no_grad()
    def from_nerf(
        cls,
        load: Path,
        *,
        points: Points,
        sh_degree: int = 1,
        activation_scale: float = 1.0,
        device: Optional[torch.device] = None,
        split: str = 'test',
        step: Optional[int] = None,
    ) -> RadFoamProxy:
        """Create a RadFoamProxy from a trained NeRF model."""
        train_task = TrainTask.load_from_script(load, step=step)
        dataset = train_task.dataset
        dataset.to(device)
        assert isinstance(dataset, (MultiViewDataset, SfMDataset, MeshViewSynthesisDataset))
        nerf_model = train_task.model
        assert isinstance(nerf_model, VanillaNeRF)
        nerf_model = nerf_model.to(device)
        nerf_model.eval()

        points = points.positions.view(-1, 3).to(device)

        fine_field = nerf_model.fine_field
        view_dirs = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(points)

        queried_colors = []
        queried_densities = []
        for points_chunk, view_dirs_chunk in zip(torch.split(points, 1024), torch.split(view_dirs, 1024)):
            encoded_pos = fine_field.position_encoding(points_chunk)
            density_embedding = fine_field.base_mlp(encoded_pos)
            density = fine_field.density_head(density_embedding)

            encoded_dir = fine_field.direction_encoding(view_dirs_chunk)
            color = fine_field.color_head(fine_field.head_mlp(torch.cat((
                encoded_dir,
                density_embedding,
            ), dim=-1)))
            queried_colors.append(color)
            queried_densities.append(density)

        queried_colors = rgb2sh(torch.cat(queried_colors, dim=0))
        queried_densities = torch.cat(queried_densities, dim=0)

        class DummyArgs:
            def __init__(self, sh_degree, num_points, activation_scale):
                self.sh_degree = sh_degree
                self.init_points = num_points
                self.final_points = num_points
                self.activation_scale = activation_scale

        radfoam_args = DummyArgs(sh_degree, points.shape[0], activation_scale)
        radfoam_model = RadFoamScene(radfoam_args, device=device)
        radfoam_model.triangulation = radfoam.Triangulation(points)
        perm = radfoam_model.triangulation.permutation().to(torch.long)

        radfoam_model.primal_points = nn.Parameter(points[perm])
        radfoam_model.att_dc = nn.Parameter(queried_colors[perm])
        radfoam_model.att_sh = nn.Parameter(torch.zeros_like(radfoam_model.att_sh))

        pre_activation_density = inverse_softplus(queried_densities / activation_scale, beta=10)
        radfoam_model.density = nn.Parameter(pre_activation_density[perm])
        
        radfoam_model.update_triangulation(rebuild=False)

        data_handler = dataset_rfstudio2radfoam(dataset, split=split)
        
        model_points, _, _, _ = radfoam_model.get_trace_data()
        start_points = radfoam_model.get_starting_point(
            data_handler.rays[:, 0, 0].to(device), model_points, radfoam_model.aabb_tree
        )

        return cls(model=radfoam_model, start_points=start_points, data_handler=data_handler, device=device)

    @classmethod
    @torch.no_grad()
    def from_neus(
        cls,
        load: Path,
        *,
        points: Points,
        sh_degree: int = 1,
        activation_scale: float = 1.0,
        device: Optional[torch.device] = None,
        split: str = 'test',
        step: Optional[int] = None,
    ) -> RadFoamProxy:
        """Create a RadFoamProxy from a trained NeuS model.

        Colors are inferred using the NeuS color head with a fixed view direction; densities are approximated from the SDF
        as a sharp surface occupancy around the zero level-set.
        """
        train_task = TrainTask.load_from_script(load, step=step)
        dataset = train_task.dataset
        dataset.to(device)
        assert isinstance(dataset, (MultiViewDataset, SfMDataset, MeshViewSynthesisDataset))
        neus_model = train_task.model
        assert isinstance(neus_model, VanillaNeuS)
        neus_model = neus_model.to(device)
        neus_model.eval()

        pts = points.positions.view(-1, 3).to(device)

        # Query SDF, geometric features, and gradients at points in chunks to avoid OOM
        sdf_field = neus_model.sdf_field
        
        queried_sdf_vals = []
        queried_geom_feats = []
        queried_gradients = []
        queried_colors = []
        
        # Process in chunks to avoid OOM during gradient computation
        for pts_chunk in torch.split(pts, 1024):
            sdf_vals_chunk, geom_feats_chunk, gradients_chunk = sdf_field.sdf_mlp.get_sdf_gradient(pts_chunk)
            
            # Fixed viewing direction per point (approximation)
            view_dirs_chunk = torch.tensor([0.0, 0.0, -1.0], device=device).expand_as(pts_chunk)
            enc_dir_chunk = sdf_field.direction_encoding(view_dirs_chunk / view_dirs_chunk.norm(dim=-1, keepdim=True))
            
            # Color head expects positions, gradients, encoded_dir, and geom_feats
            colors_chunk = sdf_field.color_mlp(torch.cat((
                pts_chunk,
                gradients_chunk,
                enc_dir_chunk,
                geom_feats_chunk,
            ), dim=-1))  # [chunk_size, 3] in [0,1]
            
            queried_sdf_vals.append(sdf_vals_chunk)
            queried_geom_feats.append(geom_feats_chunk)
            queried_gradients.append(gradients_chunk)
            queried_colors.append(colors_chunk)
        
        sdf_vals = torch.cat(queried_sdf_vals, dim=0)
        geom_feats = torch.cat(queried_geom_feats, dim=0)
        gradients = torch.cat(queried_gradients, dim=0)
        colors = torch.cat(queried_colors, dim=0)

        # Build RadFoam scene
        class DummyArgs:
            def __init__(self, sh_degree, num_points, activation_scale):
                self.sh_degree = sh_degree
                self.init_points = num_points
                self.final_points = num_points
                self.activation_scale = activation_scale

        radfoam_args = DummyArgs(sh_degree, pts.shape[0], activation_scale)
        # use NeuS renderer
        dummy_colors = torch.ones_like(pts)
        radfoam_model = RadFoamScene(radfoam_args, points=pts, points_colors=dummy_colors, device=device, use_neus_renderer=True)
        perm = radfoam_model.triangulation.permutation().to(torch.long)

        radfoam_model.primal_points = nn.Parameter(pts[perm])
        radfoam_model.att_dc = nn.Parameter(rgb2sh(colors)[perm])
        radfoam_model.att_sh = nn.Parameter(torch.zeros_like(radfoam_model.att_sh))

        # store SDF values directly
        radfoam_model.density = nn.Parameter(sdf_vals[perm])
        # Set the deviation parameter
        radfoam_model.deviation = nn.Parameter(sdf_field.deviation.params.clone())
        
        # Store the NeuS model for gradient computation during rendering
        radfoam_model.neus_model = neus_model
        radfoam_model.sdf_field = sdf_field

        radfoam_model.update_triangulation(rebuild=False)

        data_handler = dataset_rfstudio2radfoam(dataset, split=split)
        model_points, _, _, _ = radfoam_model.get_trace_data()
        start_points = radfoam_model.get_starting_point(
            data_handler.rays[:, 0, 0].to(device), model_points, radfoam_model.aabb_tree
        )

        return cls(model=radfoam_model, start_points=start_points, data_handler=data_handler, device=device)

    def get_gt_images(self) -> RGBAImages:
        return RGBAImages(torch.cat((self.data_handler.rgbs, self.data_handler.alphas), dim=-1)).to(self.device)
    
    def get_cameras(self, *, colmap2blender: bool = True) -> Cameras:
        c2w = self.data_handler.c2ws[:, :3, :].clone()
        if colmap2blender:
            c2w[:, :, 1:3] *= -1
        fx = torch.full((self.data_handler.c2ws.shape[0],), fill_value=self.data_handler.fx).to(c2w.device)
        fy = torch.full((self.data_handler.c2ws.shape[0],), fill_value=self.data_handler.fy).to(c2w.device)
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
        return RGBAImages(torch.stack(outputs)).to(self.device)
    
    def get_depths(self, *, progress_handle: Optional[Callable] = None) -> DepthImages:
        cameras = self.get_cameras(colmap2blender=True)
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
            colors=sh2rgb(self.model.att_dc).clamp(0, 1),
        )
