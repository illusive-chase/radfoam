import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from radfoam_model.scene import RadFoamScene
from scipy.spatial import Voronoi
from poly import colored_slice


def main(
    pt_path,
    out_dir="scan_frames",
    video_path="scan_video.mp4",
    plane_axis="x",
    n_steps=100,
    img_size=512,
    start_step=0,
):
    os.makedirs(out_dir, exist_ok=True)

    sh_degree_of_saved_data = 3
    args = type("Args", (), {
        "sh_degree": sh_degree_of_saved_data,
        "init_points": 30000,          # <-- Added this required attribute
        "final_points": 100000,        # <-- Added this required attribute
        "activation_scale": 1.0,       # <-- Added this required attribute
    })()

    scene = RadFoamScene(args=args)

    scene.load_pt(pt_path)
    points = scene.primal_points.detach().cpu().numpy()
    adjacency = scene.point_adjacency.cpu().numpy()
    adjacency_offsets = scene.point_adjacency_offsets.cpu().numpy()
    N = points.shape[0]

    # Get scan range
    axis = {'x':0, 'y':1, 'z':2}[plane_axis]
    min_val, max_val = points[:, axis].min(), points[:, axis].max()
    scan_vals = np.linspace(min_val, max_val, n_steps)
    
    vor = Voronoi(points)
    xmin, xmax = np.min(points[:, 0]), np.max(points[:, 0])
    ymin, ymax = np.min(points[:, 1]), np.max(points[:, 1])
    cell_colors = [np.random.rand(3) * 0.7 + 0.3 for _ in range(N)]

    frame_paths = []
    for idx in range(start_step, n_steps):
        val = scan_vals[idx]
        print(f"Processing frame {idx+1}/{n_steps} at {plane_axis}={val:.2f}")
        colored_slice(
            vor_3d=vor,
            bound=[xmin, xmax, ymin, ymax],
            plane_axis=plane_axis,
            plane_value=val,
            cell_colors=cell_colors,
            path=out_dir,
            id=idx
        )
        frame_path = Path(out_dir) / f"frame_{idx:04d}.png"
        frame_paths.append(str(frame_path))

    # Create video
    frame = cv2.imread(frame_paths[0])
    height, width, _ = frame.shape
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (width, height))
    for frame_path in frame_paths:
        frame = cv2.imread(frame_path)
        video.write(frame)
    video.release()
    print(f"Video saved to {video_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", type=str, required=True, help="Path to trained scene .pt file")
    parser.add_argument("--out_dir", type=str, default="scan_frames")
    parser.add_argument("--video_path", type=str, default="scan_video.mp4")
    parser.add_argument("--plane_axis", type=str, default="x", choices=["x", "y", "z"])
    parser.add_argument("--n_steps", type=int, default=100)
    parser.add_argument("--start_step", type=int, default=0, help="Start step for the scan")
    args = parser.parse_args()
    main(
        pt_path=args.pt_path,
        out_dir=args.out_dir,
        video_path=args.video_path,
        plane_axis=args.plane_axis,
        n_steps=args.n_steps,
        start_step=args.start_step,
    )