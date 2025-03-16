#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volume GIF Creator Script

This script creates GIFs from 3D volumes (folders of 2D slice images).
It can process a single sample or multiple samples.
"""

import os
import sys
import argparse
import random

# Add parent directory to path to import our config module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the same directory
from nuclei_visualizer import VolumeGifCreator
# Import config for data paths
import config

def main():
    """
    Process 3D volumes and create GIFs.
    
    Example usage:
    python volume_gif_creator.py --sample /path/to/sample_dir
    python volume_gif_creator.py --sample /path/to/sample_dir --no-masks
    python volume_gif_creator.py --batch /path/to/parent_dir --pattern "sample_*"
    python volume_gif_creator.py --montage /path/to/parent_dir --pattern "class2_*"
    python volume_gif_creator.py --random 5 --class 2
    """
    parser = argparse.ArgumentParser(description='Create GIFs from 3D volumes')
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--sample', type=str, help='Path to a single sample directory')
    group.add_argument('--batch', type=str, help='Path to parent directory containing multiple sample folders')
    group.add_argument('--montage', type=str, help='Create a montage of slices from multiple samples')
    group.add_argument('--random', type=int, metavar='N', help='Process N random samples')
    
    # Processing options
    parser.add_argument('--pattern', type=str, default='*', help='Pattern to match sample directories (for batch mode)')
    parser.add_argument('--class', type=int, dest='class_id', help='Filter random samples by class ID')
    parser.add_argument('--output-dir', type=str, default='results/visualizations', help='Output directory for GIFs')
    parser.add_argument('--max-frames', type=int, default=80, help='Maximum number of frames per GIF')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of frames to process at once')
    parser.add_argument('--duration', type=int, default=200, help='Duration of each frame in milliseconds')
    parser.add_argument('--no-masks', dest='include_masks', action='store_false', help='Disable mask overlays')
    parser.add_argument('--slices', type=int, default=3, help='Number of slices per sample for montage mode')
    parser.add_argument('--csv-path', type=str, help='Path to CSV file with class information (for random mode)')
    parser.add_argument('--data-dir', type=str, help='Root directory for the dataset (for random mode)')
    
    args = parser.parse_args()
    
    # Create the GIF creator
    creator = VolumeGifCreator(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_frames=args.max_frames,
        dpi=80,
        figsize=(6, 6)
    )
    
    # Process based on the selected mode
    if args.sample:
        # Single sample mode
        print(f"Processing sample: {args.sample}")
        
        # Check if directory exists
        if not os.path.exists(args.sample) or not os.path.isdir(args.sample):
            print(f"Error: Sample directory not found: {args.sample}")
            return 1
            
        # Create GIF
        gif_path = creator.create_gif_from_folder(
            folder_path=args.sample,
            duration=args.duration,
            include_masks=args.include_masks
        )
        
        if gif_path:
            print(f"Successfully created GIF: {gif_path}")
            return 0
        else:
            print("Failed to create GIF.")
            return 1
            
    elif args.batch:
        # Batch mode - process multiple samples
        print(f"Processing samples in: {args.batch}")
        print(f"Matching pattern: {args.pattern}")
        
        # Check if directory exists
        if not os.path.exists(args.batch) or not os.path.isdir(args.batch):
            print(f"Error: Parent directory not found: {args.batch}")
            return 1
            
        # Find matching directories
        import glob
        sample_dirs = []
        pattern = os.path.join(args.batch, args.pattern)
        
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                sample_dirs.append(path)
                
        if not sample_dirs:
            print(f"Error: No matching sample directories found with pattern: {args.pattern}")
            return 1
            
        print(f"Found {len(sample_dirs)} sample directories")
        
        # Process each sample
        success_count = 0
        for sample_dir in sample_dirs:
            sample_id = os.path.basename(sample_dir)
            print(f"\nProcessing sample: {sample_id}")
            
            gif_path = creator.create_gif_from_folder(
                folder_path=sample_dir,
                output_filename=f"volume_{sample_id}.gif",
                duration=args.duration,
                include_masks=args.include_masks
            )
            
            if gif_path:
                success_count += 1
                
        print(f"\nProcessed {len(sample_dirs)} samples with {success_count} successful GIFs")
        return 0 if success_count > 0 else 1
        
    elif args.montage:
        # Montage mode - create a grid of slices from multiple samples
        print(f"Creating montage from samples in: {args.montage}")
        print(f"Matching pattern: {args.pattern}")
        
        # Check if directory exists
        if not os.path.exists(args.montage) or not os.path.isdir(args.montage):
            print(f"Error: Parent directory not found: {args.montage}")
            return 1
            
        # Find matching directories
        import glob
        sample_dirs = []
        pattern = os.path.join(args.montage, args.pattern)
        
        for path in glob.glob(pattern):
            if os.path.isdir(path):
                sample_dirs.append(path)
                
        if not sample_dirs:
            print(f"Error: No matching sample directories found with pattern: {args.pattern}")
            return 1
            
        print(f"Found {len(sample_dirs)} sample directories")
        
        # Create montage
        montage_path = creator.create_volume_montage(
            sample_dirs=sample_dirs,
            output_filename="volume_montage.png",
            slices_per_sample=args.slices,
            include_masks=args.include_masks
        )
        
        if montage_path:
            print(f"Successfully created montage: {montage_path}")
            return 0
        else:
            print("Failed to create montage.")
            return 1
    
    elif args.random:
        # Random samples mode
        num_samples = args.random
        print(f"Creating GIFs for {num_samples} random samples")
        
        if args.class_id is not None:
            print(f"Filtering to class ID: {args.class_id}")
            
        # Use default paths from config if not provided
        data_dir = args.data_dir
        csv_path = args.csv_path
        
        if not data_dir:
            data_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
        
        if not csv_path:
            csv_path = os.path.abspath(config.CLASS_CSV_PATH)
            
        print(f"Using data directory: {data_dir}")
        print(f"Using CSV file: {csv_path}")
        
        # Validate paths
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found: {data_dir}")
            return 1
            
        if not os.path.exists(csv_path):
            print(f"Error: CSV file not found: {csv_path}")
            return 1
            
        # Read CSV file to get sample IDs
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            print(f"Found {len(df)} samples in CSV file")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return 1
            
        # Filter by class if requested
        if args.class_id is not None:
            df = df[df['class_id'] == args.class_id]
            print(f"Filtered to {len(df)} samples in class {args.class_id}")
            
        # Check if we have any samples after filtering
        if len(df) == 0:
            print("No samples found matching the filter criteria.")
            return 1
            
        # Get unique sample IDs
        unique_samples = df['sample_id'].unique()
        print(f"Found {len(unique_samples)} unique sample IDs")
        
        # Adjust the path to include the nuclei_sample_1a_v1 subdirectory
        nuclei_dir = os.path.join(data_dir, 'nuclei_sample_1a_v1')
        
        # Check which sample directories actually exist in the nuclei_dir
        existing_sample_dirs = []
        if os.path.exists(nuclei_dir):
            for d in os.listdir(nuclei_dir):
                sample_path = os.path.join(nuclei_dir, d)
                if os.path.isdir(sample_path):
                    try:
                        # Try to convert directory name to integer (sample ID)
                        sample_id = int(d)
                        if sample_id in unique_samples:
                            existing_sample_dirs.append(sample_path)
                    except ValueError:
                        # Directory name is not a valid sample ID number, skip it
                        continue
        else:
            print(f"Error: Nuclei directory not found: {nuclei_dir}")
            return 1
            
        if not existing_sample_dirs:
            print(f"Error: No matching sample directories found in {nuclei_dir}")
            return 1
            
        print(f"Found {len(existing_sample_dirs)} existing sample directories")
        
        # Randomly select samples
        if len(existing_sample_dirs) <= num_samples:
            selected_samples = existing_sample_dirs
            print(f"Requesting {num_samples} samples but only {len(existing_sample_dirs)} available. Using all available samples.")
        else:
            selected_samples = random.sample(existing_sample_dirs, num_samples)
            
        print(f"Selected {len(selected_samples)} random samples")
        
        # Process each sample
        success_count = 0
        for sample_dir in selected_samples:
            sample_id = os.path.basename(sample_dir)
            print(f"\nProcessing sample: {sample_id}")
            
            # Get class info for this sample
            sample_row = df[df['sample_id'] == int(sample_id)]
            class_info = ""
            if not sample_row.empty:
                class_id = sample_row.iloc[0]['class_id']
                class_name = sample_row.iloc[0]['class_name']
                class_info = f"_class{class_id}_{class_name}"
                
            gif_path = creator.create_gif_from_folder(
                folder_path=sample_dir,
                output_filename=f"volume_{sample_id}{class_info}.gif",
                duration=args.duration,
                include_masks=args.include_masks
            )
            
            if gif_path:
                success_count += 1
                
        # Create a montage if at least 2 samples were successful
        if success_count >= 2:
            print("\nCreating montage of all processed samples...")
            montage_path = creator.create_volume_montage(
                sample_dirs=selected_samples,
                output_filename="random_samples_montage.png",
                slices_per_sample=args.slices,
                include_masks=args.include_masks
            )
            
            if montage_path:
                print(f"Successfully created montage: {montage_path}")
                
        print(f"\nProcessed {len(selected_samples)} random samples with {success_count} successful GIFs")
        return 0 if success_count > 0 else 1

if __name__ == "__main__":
    sys.exit(main()) 