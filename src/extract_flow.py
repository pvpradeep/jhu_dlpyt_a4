"""
===============================================================================
UCF50 Dense Optical Flow Extraction Pipeline (Two-Stream Motion Stream)
===============================================================================

Description:
------------
This script precomputes dense optical flow fields across pre-extracted contiguous 
video frame clips in the UCF50 dataset. It extracts spatial motion vectors (u, v) 
between consecutive RGB frames and converts them into 3-channel BGR motion maps, 
enabling a dedicated Motion Stream for Decoupled Two-Stream LRCN Action Recognition.

Technical Details & Mathematical Mapping:
------------------------------------------
1. Algorithm: Gunnar Farneback Dense Optical Flow (cv2.calcOpticalFlowFarneback).
   - Parameters: pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2.
2. Coordinate Transformation:
   - Converts Cartesian displacement vectors (u, v) to Polar coordinates (magnitude, angle).
3. HSV Motion Visualizer Encoding:
   - Hue (H): Represents motion direction/angle (mapped to [0, 180] degrees).
   - Saturation (S): Fixed at max saturation (255) for pure hue rendering.
   - Value (V): Normalized motion magnitude (scaled to [0, 255] via NORM_MINMAX).
4. Output Format:
   - Converted back to 3-channel BGR (`cv2.COLOR_HSV2BGR`) and written as standard JPEG 
     images (`frame0.jpg` through `frame15.jpg`).
5. Temporal Alignment:
   - Frame pair (t, t+1) produces flow map `frame_t`. 
   - The final flow map (`frame14.jpg`) is duplicated to `frame15.jpg` to preserve 
     exact 16-frame sequence lengths expected by the VideoDataset loader.

Directory Structure Assumptions:
--------------------------------
Input (RGB Frames):
  data/UCF50_frames_contig_5clips/
  ├── [Category_Name]/
  │   └── [Video_Clip_ID]/
  │       ├── frame0.jpg
  │       └── ... (16 frames)

Output (Dense Flow Maps):
  data/UCF50_flow_contig_5clips/
  ├── [Category_Name]/
  │   └── [Video_Clip_ID]/
  │       ├── frame0.jpg
  │       └── ... (16 motion map frames)

Usage:
------
$ python src/extract_flow.py

Target Pipeline Context:
------------------------
Generates motion inputs for `v3_flow` / `v4_flow` models. Fused via soft voting 
logits (0.5 * Logits_RGB + 0.5 * Logits_Flow) during test evaluation to eliminate 
motion-blind errors on dynamic classes (e.g., JumpingJack, ThrowDiscus, JumpRope).
===============================================================================
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

def compute_farneback_flow(prev_frame, next_frame):
    gray1 = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    
    flow = cv2.calcOpticalFlowFarneback(
        gray1, gray2, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros_like(prev_frame)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def process_dataset(src_dir, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    categories = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d)) and not d.startswith('.')]
    
    for cat in tqdm(categories, desc="Categories"):
        cat_src, cat_dst = os.path.join(src_dir, cat), os.path.join(dst_dir, cat)
        os.makedirs(cat_dst, exist_ok=True)
        clips = [d for d in os.listdir(cat_src) if os.path.isdir(os.path.join(cat_src, d)) and not d.startswith('.')]
        
        for clip in clips:
            clip_src, clip_dst = os.path.join(cat_src, clip), os.path.join(cat_dst, clip)
            os.makedirs(clip_dst, exist_ok=True)
            
            frame_files = sorted([f for f in os.listdir(clip_src) if f.endswith(('.jpg', '.png'))])
            if len(frame_files) < 2:
                continue
            frames = [cv2.imread(os.path.join(clip_src, f)) for f in frame_files]
            
            for t in range(len(frames) - 1):
                flow_img = compute_farneback_flow(frames[t], frames[t+1])
                cv2.imwrite(os.path.join(clip_dst, f"frame{t}.jpg"), flow_img)
            cv2.imwrite(os.path.join(clip_dst, f"frame{len(frames)-1}.jpg"), flow_img)

if __name__ == "__main__":
    process_dataset("data/UCF50_frames_contig_5clips", "data/UCF50_flow_contig_5clips")