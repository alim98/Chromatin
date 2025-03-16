#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to test GIF creation functionality
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm  # Import tqdm for progress bars
import gc

def create_gif(max_frames=80):
    """
    Create a simple GIF from synthetic images
    
    Args:
        max_frames (int): Maximum number of frames to include in the GIF
    """
    print("Testing GIF creation...")
    
    # Create output directory
    output_dir = os.path.join('results', 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    
    # Create frames directory
    frames_dir = os.path.join(output_dir, 'gif_test_frames')
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Output directory: {output_dir}")
    print(f"Frames directory: {frames_dir}")
    
    # Generate more images than max_frames to test downsampling
    num_frames = 120  # Generate 120 frames, will be downsampled
    print(f"Generating {num_frames} frames for testing...")
    
    # Process frames in batches to avoid memory issues
    batch_size = 10  # Process 10 frames at a time
    num_batches = (min(num_frames, max_frames) + batch_size - 1) // batch_size
    
    all_frame_paths = []
    
    # Downsample if needed
    if num_frames > max_frames:
        indices = np.linspace(0, num_frames - 1, max_frames, dtype=int)
    else:
        indices = np.arange(num_frames)
    
    # Process frames in batches
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(indices))
        
        print(f"Processing batch {batch_idx+1}/{num_batches} (frames {start_idx+1}-{end_idx})...")
        batch_frame_paths = []
        
        # Process frames in current batch
        for i in tqdm(range(start_idx, end_idx), desc=f"Creating frames (batch {batch_idx+1})"):
            img_idx = indices[i]
            # Create a simple image
            plt.figure(figsize=(4, 4), dpi=80)
            
            # Create a gradient with moving pattern
            data = np.zeros((64, 64))
            for y in range(64):
                for x in range(64):
                    data[y, x] = 0.5 + 0.5 * np.sin((x + y + img_idx*10) / 10.0)
            
            plt.imshow(data, cmap='viridis')
            plt.title(f"Frame {img_idx+1}")
            plt.colorbar()
            
            # Save the frame
            frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
            plt.savefig(frame_path)
            plt.close()
            
            batch_frame_paths.append(frame_path)
            all_frame_paths.append(frame_path)
            
        # Force garbage collection between batches
        gc.collect()
    
    # Create GIF with PIL (more memory efficient)
    gif_path = os.path.join(output_dir, 'test_animation.gif')
    print(f"Creating GIF: {gif_path}")
    
    try:
        # Open first image to get size
        with Image.open(all_frame_paths[0]) as img:
            # Load frames one by one
            print("Processing frames for GIF...")
            frames = []
            for frame_path in tqdm(all_frame_paths, desc="Processing frames"):
                try:
                    frame = Image.open(frame_path)
                    frames.append(frame.copy())
                    frame.close()
                except Exception as e:
                    print(f"Error processing frame {frame_path}: {e}")
            
            # Save the GIF
            if frames:
                print(f"Saving GIF with {len(frames)} frames...")
                with tqdm(total=1, desc="Saving GIF") as pbar:
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        optimize=False,
                        duration=300,
                        loop=0
                    )
                    pbar.update(1)
                
                if os.path.exists(gif_path):
                    file_size = os.path.getsize(gif_path)
                    print(f"GIF created successfully: {gif_path} ({file_size/1024:.1f} KB)")
                    return True
                else:
                    print(f"Error: GIF file was not created")
                    return False
            else:
                print("Error: No frames could be processed")
                return False
        
    except Exception as e:
        print(f"Error creating GIF: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup
        print("Cleaning up temporary files...")
        # Get all files in frames directory
        all_frame_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')]
        for frame_path in tqdm(all_frame_files, desc="Cleaning up"):
            try:
                os.remove(frame_path)
            except Exception as e:
                print(f"Warning: Could not remove {frame_path}: {e}")
                
        try:
            os.rmdir(frames_dir)
            print(f"Removed directory: {frames_dir}")
        except Exception as e:
            print(f"Warning: Could not remove directory {frames_dir}: {e}")
            
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GIF creation with downsampling')
    parser.add_argument('--max-frames', type=int, default=80, help='Maximum number of frames for the GIF')
    parser.add_argument('--batch-size', type=int, default=10, help='Batch size for processing frames')
    args = parser.parse_args()
    
    success = create_gif(max_frames=args.max_frames)
    print(f"GIF creation {'successful' if success else 'failed'}")
    exit(0 if success else 1) 