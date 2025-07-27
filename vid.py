#!/usr/bin/env python3
# filepath: /root/autodl-tmp/data/pingxing/radfoam/make_video.py
import os
import cv2
import glob
import argparse
from pathlib import Path

import re

def extract_number(path):
    # Extracts the largest integer from the filename
    nums = re.findall(r'\d+', os.path.basename(path))
    return int(nums[-1]) if nums else -1


def create_video_from_frames(
    frames_dir="scan_frames",
    output_path="scan_video.mp4",
    fps=20,
    pattern="frame_*.png"
):
    """Create a video from a series of image frames."""
    # Get all frame files and sort them numerically
    
    frame_paths = sorted(glob.glob(f"{frames_dir}/{pattern}"), key=extract_number)
    
    if not frame_paths:
        print(f"No frames found in {frames_dir} matching pattern {pattern}")
        return
    
    print(f"Found {len(frame_paths)} frames")
    
    # Read the first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        print(f"Failed to read frame: {frame_paths[0]}")
        return
    
    height, width, _ = first_frame.shape
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Add frames to the video
    for i, frame_path in enumerate(frame_paths):
        print(f"Processing frame {i+1}/{len(frame_paths)}: {frame_path}", end="\r")
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"\nWarning: Could not read frame {frame_path}")
    
    # Release the video writer
    video.release()
    print(f"\nVideo saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create video from image frames")
    parser.add_argument("--frames_dir", type=str, default="scan_frames",
                        help="Directory containing frame images")
    parser.add_argument("--output", type=str, default="scan_video.mp4",
                        help="Output video file path")
    parser.add_argument("--fps", type=int, default=20,
                        help="Frames per second in output video")
    parser.add_argument("--pattern", type=str, default="voronoi*.png",
                        help="File pattern to match frame images")
    
    args = parser.parse_args()
    
    create_video_from_frames(
        frames_dir=args.frames_dir,
        output_path=args.output,
        fps=args.fps,
        pattern=args.pattern
    )