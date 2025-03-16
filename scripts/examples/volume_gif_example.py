#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example of creating GIFs from 3D volumes using VolumeGifCreator

This script demonstrates how to use the VolumeGifCreator class to create
GIFs from a sample of 2D slices that form a 3D volume.
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the VolumeGifCreator class - now directly from the visualization folder
from scripts.visualization.nuclei_visualizer import VolumeGifCreator
import config

def main():
    """
    Run a simple example of creating a GIF from a 3D volume.
    """
    # Check if data directory exists
    data_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
    sample_dir = os.path.join(data_dir, 'nuclei_sample_1a_v1', '1515')  # Use sample 1515 as an example
    
    print(f"Data directory: {data_dir}")
    print(f"Sample directory: {sample_dir}")
    
    if not os.path.exists(sample_dir):
        print(f"Error: Sample directory not found: {sample_dir}")
        print("Please adjust the sample ID or check your data paths.")
        return 1
    
    # Create VolumeGifCreator
    creator = VolumeGifCreator(
        output_dir='results/example_gifs',
        batch_size=5,         # Process 5 frames at a time to manage memory
        max_frames=50,        # Limit to 50 frames max
        dpi=80,               # Lower DPI for faster rendering
        figsize=(6, 6)        # Medium-sized frames
    )
    
    # Create GIF from the sample
    print("\nExample 1: Creating GIF from a single sample")
    gif_path = creator.create_gif_from_folder(
        folder_path=sample_dir,
        output_filename="example_volume.gif",
        duration=150,         # 150ms between frames (faster animation)
        include_masks=True    # Include mask overlays if available
    )
    
    if gif_path:
        print(f"Successfully created GIF: {gif_path}")
    else:
        print("Failed to create GIF.")
    
    # Create montage of slices
    print("\nExample 2: Creating a montage of slices")
    montage_path = creator.create_volume_montage(
        sample_dirs=[sample_dir],  # Just using one sample for this example
        output_filename="example_montage.png",
        slices_per_sample=5,       # Show 5 representative slices
        include_masks=True         # Include mask overlays if available
    )
    
    if montage_path:
        print(f"Successfully created montage: {montage_path}")
    else:
        print("Failed to create montage.")
    
    print("\nExample completed!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 