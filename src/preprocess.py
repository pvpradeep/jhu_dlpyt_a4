"""
Module: preprocess.py

Extracts frames from UCF50 video dataset and saves them as JPEG images.
Each video becomes a directory containing its frames.

Usage:
    python preprocess.py --video_dir ../data/UCF50 --output_dir ../data/UCF50_frames --n_frames 16
"""

import os
import argparse
from tqdm import tqdm
from utils import get_frames, store_frames


def preprocess_dataset(video_dir, output_dir, n_frames=16):
    """
    Extract frames from all videos in the dataset.

    Args:
        video_dir (str): Path to raw video dataset (e.g., ../data/UCF50)
        output_dir (str): Path to save extracted frames (e.g., ../data/UCF50_frames)
        n_frames (int): Number of frames to extract per video
    """
    # Get all action categories
    categories = sorted([
        d for d in os.listdir(video_dir) 
        if os.path.isdir(os.path.join(video_dir, d))
    ])
    
    print(f"Found {len(categories)} action categories")
    print(f"Extracting {n_frames} frames per video...")
    
    total_videos = 0
    failed_videos = []
    
    for category in tqdm(categories, desc="Categories"):
        category_path = os.path.join(video_dir, category)
        output_category_path = os.path.join(output_dir, category)
        
        # Create output category directory
        os.makedirs(output_category_path, exist_ok=True)
        
        # Get all video files
        videos = [f for f in os.listdir(category_path) if f.endswith('.avi')]
        
        for video_file in videos:
            video_path = os.path.join(category_path, video_file)
            
            # Create output directory for this video's frames
            video_name = os.path.splitext(video_file)[0]  # Remove .avi extension
            frames_output_path = os.path.join(output_category_path, video_name)
            
            # Skip if already processed
            if os.path.exists(frames_output_path) and len(os.listdir(frames_output_path)) > 0:
                total_videos += 1
                continue
            
            os.makedirs(frames_output_path, exist_ok=True)
            
            # Extract frames
            frames, v_len = get_frames(video_path, n_frames=n_frames)
            
            if len(frames) == 0:
                failed_videos.append(video_path)
                continue
            
            # Save frames
            store_frames(frames, frames_output_path)
            total_videos += 1
    
    print(f"\n✅ Processed {total_videos} videos")
    
    if failed_videos:
        print(f"⚠️ Failed to process {len(failed_videos)} videos:")
        for v in failed_videos[:10]:  # Show first 10
            print(f"   - {v}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from UCF50 videos")
    parser.add_argument(
        "--video_dir", 
        type=str, 
        default="../data/UCF50",
        help="Path to raw video directory"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="../data/UCF50_frames",
        help="Path to save extracted frames"
    )
    parser.add_argument(
        "--n_frames", 
        type=int, 
        default=16,
        help="Number of frames to extract per video"
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.video_dir):
        raise FileNotFoundError(f"Video directory not found: {args.video_dir}")
    
    print(f"Video source: {args.video_dir}")
    print(f"Output destination: {args.output_dir}")
    print(f"Frames per video: {args.n_frames}")
    print("-" * 50)
    
    preprocess_dataset(args.video_dir, args.output_dir, args.n_frames)


if __name__ == "__main__":
    main()