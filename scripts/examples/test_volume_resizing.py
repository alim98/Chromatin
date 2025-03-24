#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Volume Resizing Comparison Generator

This script generates comparison GIFs between original and resized 3D volumes.
It processes multiple samples and creates side-by-side GIFs to evaluate how well
the resizing preserves important structures.
"""

import os
import sys
import argparse
import numpy as np
import torch
import random
import shutil
from tqdm import tqdm
import time
import gc
import pandas as pd
import glob

# Try to import psutil for memory tracking
try:
    import psutil
    MEMORY_TRACKING = True
except ImportError:
    MEMORY_TRACKING = False

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import our dataloader
from dataloader.nuclei_dataloader import NucleiDataset, get_nuclei_dataloader
import config

def main():
    """Generate comparison GIFs between original and resized 3D volumes."""
    # Track total execution time
    start_time = time.time()
    
    parser = argparse.ArgumentParser(description='Generate comparison GIFs between original and resized 3D volumes')
    parser.add_argument('--sample', type=str, help='Specific sample ID to process')
    parser.add_argument('--class', type=int, dest='class_id', help='Filter samples by class ID')
    parser.add_argument('--output-dir', type=str, default='results/visualizations/resizing_comparisons', 
                        help='Output directory for visualizations')
    parser.add_argument('--target-shape', type=str, default='80,80,80', 
                        help='Target shape for resizing (z,y,x)')
    parser.add_argument('--data-dir', type=str, 
                        help='Root directory for the dataset')
    parser.add_argument('--csv-path', type=str, 
                        help='Path to CSV file with class information')
    parser.add_argument('--no-mask-priority', dest='mask_priority', action='store_false',
                        help='Disable prioritizing regions with masks during resizing')
    parser.add_argument('--preserve-resolution', action='store_true',
                        help='Preserve original resolution using crop/pad instead of scaling')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug output for resizing process')
    parser.add_argument('--max-frames', type=int, default=80,
                        help='Maximum number of frames in output GIFs')
    parser.add_argument('--count', type=int, default=20,
                        help='Number of samples to process (max 20)')
    parser.add_argument('--clean', action='store_true',
                        help='Clean output directory before generating new comparisons')
    
    args = parser.parse_args()
    
    # Limit count to max 20
    sample_count = min(20, args.count)
    print(f"Will generate {sample_count} comparison examples")
    
    # Parse target shape
    target_shape = tuple(map(int, args.target_shape.split(',')))
    print(f"Target shape: {target_shape}")
    
    # Use default paths from config if not provided
    data_dir = args.data_dir if args.data_dir else os.path.abspath(config.DATA_ROOT)
    nuclei_dir = os.path.join(os.path.dirname(data_dir), 'nuclei_sample_1a_v1')
    csv_path = args.csv_path if args.csv_path else os.path.abspath(config.CLASS_CSV_PATH)
    
    print(f"Using data directory: {nuclei_dir}")
    print(f"Using CSV file: {csv_path}")
    
    # Memory usage at start
    if MEMORY_TRACKING:
        process = psutil.Process(os.getpid())
        memory_start = process.memory_info().rss / 1024 / 1024  # Convert to MB
        print(f"Initial memory usage: {memory_start:.1f} MB")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(nuclei_dir):
        print(f"Error: Data directory not found: {nuclei_dir}")
        print("Please check your paths and try again.")
        return 1
        
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file not found: {csv_path}")
        print("Will proceed without class information.")
    
    # Clean output directory if requested
    if args.clean and os.path.exists(output_dir):
        print(f"\n=== Cleaning output directory ===")
        clean_start = time.time()
        
        item_count = 0
        for item in tqdm(os.listdir(output_dir), desc="Removing items"):
            item_path = os.path.join(output_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
            item_count += 1
        
        clean_time = time.time() - clean_start
        print(f"Removed {item_count} items in {clean_time:.1f} seconds")
    
    # Create summary directory for comparison GIFs
    summary_dir = os.path.join(output_dir, "comparisons")
    os.makedirs(summary_dir, exist_ok=True)
    
    # Create dataset with specified parameters
    sample_ids = [args.sample] if args.sample else None
    filter_by_class = args.class_id
    
    print("\n=== Finding Valid Samples ===")
    sample_finding_start = time.time()
    
    print(f"Mask priority: {'Enabled' if args.mask_priority else 'Disabled'}")
    print(f"Preserve resolution: {'Enabled' if args.preserve_resolution else 'Disabled'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    if sample_ids:
        print(f"Processing specific sample: {sample_ids[0]}")
    elif filter_by_class is not None:
        print(f"Filtering by class: {filter_by_class}")
    
    try:
        # OPTIMIZATION: Instead of loading all volumes, just scan the CSV and directories
        # to find valid samples without loading any data
        valid_samples = []
        
        # Load CSV data
        print("Loading class information from CSV...")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Filter by class if needed
            if filter_by_class is not None:
                if isinstance(filter_by_class, int):
                    filter_by_class = [filter_by_class]
                df = df[df['class_id'].isin(filter_by_class)]
                print(f"Filtered to {len(df)} samples with class(es): {filter_by_class}")
            
            # Filter by sample ID if specified
            if sample_ids:
                sample_ids = [str(sid) for sid in sample_ids]
                df = df[df['sample_id'].astype(str).isin(sample_ids)]
                print(f"Filtered to {len(df)} specified sample(s)")
            
            # Check which samples have valid directories
            print("Checking for valid sample directories...")
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating samples"):
                sample_id = str(row['sample_id'])
                sample_dir = os.path.join(nuclei_dir, sample_id)
                
                # Check if raw and mask directories exist
                raw_dir = os.path.join(sample_dir, 'raw')
                mask_dir = os.path.join(sample_dir, 'mask')
                
                if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                    # Check if there are any image files
                    raw_files = glob.glob(os.path.join(raw_dir, '*.tif'))
                    if raw_files:
                        # This is a valid sample
                        valid_samples.append({
                            'sample_id': sample_id,
                            'class_id': row['class_id'],
                            'class_name': row['class_name']
                        })
        else:
            # No CSV, just scan directories
            print("No CSV file found. Scanning directories directly...")
            for sample_dir in tqdm(os.listdir(nuclei_dir), desc="Scanning directories"):
                if not os.path.isdir(os.path.join(nuclei_dir, sample_dir)):
                    continue
                    
                # Check if this is a sample directory with raw and mask subdirectories
                raw_dir = os.path.join(nuclei_dir, sample_dir, 'raw')
                mask_dir = os.path.join(nuclei_dir, sample_dir, 'mask')
                
                if os.path.exists(raw_dir) and os.path.exists(mask_dir):
                    # Check if there are any image files
                    raw_files = glob.glob(os.path.join(raw_dir, '*.tif'))
                    if raw_files:
                        # This is a valid sample
                        valid_samples.append({
                            'sample_id': sample_dir,
                            'class_id': -1,  # Unknown class
                            'class_name': 'Unknown'
                        })
        
        sample_finding_time = time.time() - sample_finding_start
        
        if not valid_samples:
            print("\nNo valid samples found. Please check your data directory and filters.")
            return 1
        
        total_samples = len(valid_samples)
        print(f"\nFound {total_samples} valid samples in {sample_finding_time:.1f} seconds.")
        
        # Memory usage after sample finding
        if MEMORY_TRACKING:
            memory_after_finding = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage after finding samples: {memory_after_finding:.1f} MB " +
                 f"(Δ: {memory_after_finding - memory_start:+.1f} MB)")
        
        # Sample selection logic
        print("\n=== Selecting Samples ===")
        selection_start = time.time()
        
        # If we have more samples than needed, select a diverse set
        selected_samples = []
        if total_samples <= sample_count:
            # Process all samples if we have fewer than requested
            selected_samples = valid_samples
            print(f"Using all {total_samples} samples (fewer than requested {sample_count})")
        else:
            # Try to get a diverse set of classes
            class_samples = {}
            
            # Group samples by class
            print("Grouping samples by class...")
            for sample in valid_samples:
                class_id = sample['class_id']
                if class_id not in class_samples:
                    class_samples[class_id] = []
                class_samples[class_id].append(sample)
            
            if len(class_samples) > 1:  # Only if we have multiple classes
                # We have class information, select evenly from classes
                print(f"Found {len(class_samples)} classes")
                classes = list(class_samples.keys())
                samples_per_class = max(1, sample_count // len(classes))
                print(f"Selecting ~{samples_per_class} samples per class...")
                
                for class_id in classes:
                    # Take a sample from each class
                    class_samples_list = class_samples[class_id]
                    class_name = class_samples_list[0]['class_name']
                    
                    num_to_take = min(samples_per_class, len(class_samples_list))
                    if len(class_samples_list) <= samples_per_class:
                        # Take all samples if we have fewer than needed
                        selected = class_samples_list
                    else:
                        # Random sample if we have more than needed
                        selected = random.sample(class_samples_list, samples_per_class)
                    
                    selected_samples.extend(selected)
                    print(f"  Class {class_id} ({class_name}): {len(selected)}/{len(class_samples_list)} samples")
                    
                    # Stop if we have enough samples
                    if len(selected_samples) >= sample_count:
                        break
                
                # If we still need more samples, add randomly
                if len(selected_samples) < sample_count:
                    needed = sample_count - len(selected_samples)
                    print(f"Need {needed} more samples to reach target count...")
                    
                    # Get all samples not yet selected
                    remaining = [s for s in valid_samples if s not in selected_samples]
                    if remaining:
                        # Add random samples to reach the desired count
                        additional = random.sample(remaining, min(needed, len(remaining)))
                        selected_samples.extend(additional)
                        print(f"Added {len(additional)} random samples")
            else:
                # No multiple classes, select randomly
                print("Selecting random samples...")
                selected_samples = random.sample(valid_samples, min(sample_count, len(valid_samples)))
                print(f"Selected {len(selected_samples)} random samples")
        
        # Ensure we don't process more than requested
        selected_samples = selected_samples[:sample_count]
        
        selection_time = time.time() - selection_start
        print(f"Sample selection completed in {selection_time:.1f} seconds")
        
        # Memory after sample selection
        if MEMORY_TRACKING:
            memory_after_selection = process.memory_info().rss / 1024 / 1024
            print(f"Memory usage after sample selection: {memory_after_selection:.1f} MB " +
                 f"(Δ: {memory_after_selection - memory_after_finding:+.1f} MB)")
        
        # Create an HTML file for comparison visualization
        html_path = os.path.join(output_dir, "comparison_gallery.html")
        print(f"\n=== Creating HTML Gallery ===")
        print(f"Gallery will be saved to {html_path}")
        
        # Write HTML gallery header
        with open(html_path, 'w') as html_file:
            html_file.write("""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Volume Resizing Comparisons</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .comparison { 
                        display: flex; 
                        margin-bottom: 30px;
                        border: 1px solid #ccc;
                        padding: 15px;
                        border-radius: 5px;
                    }
                    .sample { flex: 1; margin: 10px; text-align: center; }
                    h2 { color: #333; }
                    .metadata { 
                        background-color: #f5f5f5;
                        padding: 10px;
                        border-radius: 5px;
                        margin-top: 10px;
                        font-size: 14px;
                        text-align: left;
                    }
                    .gif-container { margin: 10px 0; }
                    .summary { 
                        background-color: #e9f7ef; 
                        padding: 15px; 
                        border-radius: 5px;
                        margin-bottom: 20px;
                    }
                    .error { color: #c0392b; }
                    .success { color: #27ae60; }
                </style>
            </head>
            <body>
                <h1>Volume Resizing Comparisons</h1>
                <div class="summary">
                    <p><strong>Date:</strong> %s</p>
                    <p><strong>Target shape:</strong> %s</p>
                    <p><strong>Mask priority:</strong> %s</p>
                    <p><strong>Preserve resolution:</strong> %s</p>
                    <p><strong>Debug mode:</strong> %s</p>
                    <p><strong>Total samples:</strong> %d</p>
                </div>
            """ % (time.strftime("%Y-%m-%d %H:%M:%S"), str(target_shape), 
                  "Enabled" if args.mask_priority else "Disabled",
                  "Enabled" if args.preserve_resolution else "Disabled",
                  "Enabled" if args.debug else "Disabled",
                  len(selected_samples)))
        
            # Process selected samples
            print(f"\n=== Processing {len(selected_samples)} Samples ===")
            processing_start = time.time()
            
            success_count = 0
            error_count = 0
            
            # Use tqdm for the overall progress bar
            for i, sample_info in enumerate(tqdm(selected_samples, desc="Generating comparisons")):
                sample_start = time.time()
                sample_id = sample_info['sample_id']
                class_id = sample_info['class_id']
                class_name = sample_info['class_name']
                
                # Print sample information
                print(f"\n=== Processing Sample {i+1}/{len(selected_samples)} ===")
                print(f"Sample ID: {sample_id}")
                print(f"Class: {class_id} ({class_name})")
                
                # Create a dataset just for this sample
                try:
                    print(f"Loading volume for sample {sample_id}...")
                    
                    # Create a dataset with just this sample
                    dataset = NucleiDataset(
                        root_dir=nuclei_dir,
                        sample_ids=[sample_id],
                        class_csv_path=csv_path,
                        load_volumes=True,  # Enable 3D volume loading
                        target_shape=target_shape,
                        mask_priority=args.mask_priority,
                        preserve_resolution=args.preserve_resolution,
                        debug=args.debug,
                        return_paths=True
                    )
                    
                    if len(dataset) == 0:
                        print(f"Error: Could not load sample {sample_id}")
                        error_count += 1
                        continue
                    
                    # Get the sample data
                    sample = dataset[0]  # There should be only one sample
                    original_shape = sample['metadata']['original_shape']
                    
                    print(f"Original shape: {original_shape}")
                    print(f"Target shape: {target_shape}")
                    
                    # Create visualization GIFs of original and resized volumes
                    print(f"Creating visualizations...")
                    original_gif, resized_gif = dataset.visualize_volume(
                        idx=0,  # There's only one sample in the dataset
                        output_dir=output_dir,
                        max_frames=args.max_frames
                    )
                    
                    sample_time = time.time() - sample_start
                    print(f"Processing time: {sample_time:.1f} seconds")
                    
                    if original_gif and resized_gif:
                        # Copy GIFs to summary directory with numbered prefixes for easy browsing
                        original_name = f"{i+1:02d}_{os.path.basename(original_gif)}"
                        resized_name = f"{i+1:02d}_{os.path.basename(resized_gif)}"
                        
                        original_copy = os.path.join(summary_dir, original_name)
                        resized_copy = os.path.join(summary_dir, resized_name)
                        
                        shutil.copy(original_gif, original_copy)
                        shutil.copy(resized_gif, resized_copy)
                        
                        print(f"Copied comparison GIFs to {summary_dir}")
                        
                        # Get file sizes
                        original_size = os.path.getsize(original_copy) / 1024  # KB
                        resized_size = os.path.getsize(resized_copy) / 1024  # KB
                        
                        # Calculate shape details and resolution changes
                        res_change_z = original_shape[0] / target_shape[0]
                        res_change_y = original_shape[1] / target_shape[1]
                        res_change_x = original_shape[2] / target_shape[2]
                        
                        # Calculate volume change
                        original_volume = original_shape[0] * original_shape[1] * original_shape[2]
                        target_volume = target_shape[0] * target_shape[1] * target_shape[2]
                        volume_ratio = original_volume / target_volume
                        
                        # Class info for HTML
                        class_info = f"Class: {class_id} ({class_name})"
                        
                        # Add to HTML gallery
                        html_file.write(f"""
                        <div class="comparison">
                            <div class="sample">
                                <h2>Original Volume</h2>
                                <div class="gif-container">
                                    <img src="comparisons/{original_name}" alt="Original Volume">
                                </div>
                                <div class="metadata">
                                    <p><strong>Sample ID:</strong> {sample_id}</p>
                                    <p><strong>Shape:</strong> {original_shape[0]}×{original_shape[1]}×{original_shape[2]}</p>
                                    <p><strong>File size:</strong> {original_size:.1f} KB</p>
                                    <p><strong>{class_info}</strong></p>
                                </div>
                            </div>
                            <div class="sample">
                                <h2>Resized Volume</h2>
                                <div class="gif-container">
                                    <img src="comparisons/{resized_name}" alt="Resized Volume">
                                </div>
                                <div class="metadata">
                                    <p><strong>Sample ID:</strong> {sample_id}</p>
                                    <p><strong>Target shape:</strong> {target_shape[0]}×{target_shape[1]}×{target_shape[2]}</p>
                                    <p><strong>File size:</strong> {resized_size:.1f} KB</p>
                                    <p><strong>Resize method:</strong> {'Crop/Pad (Preserves Resolution)' if args.preserve_resolution else 'Scaling'}</p>
                                    <p><strong>Resolution change:</strong></p>
                                    <ul>
                                        <li>Z-axis: {res_change_z:.2f}×</li>
                                        <li>Y-axis: {res_change_y:.2f}×</li>
                                        <li>X-axis: {res_change_x:.2f}×</li>
                                        <li>Volume: {volume_ratio:.2f}×</li>
                                    </ul>
                                    <p class="success">Processing time: {sample_time:.1f} seconds</p>
                                </div>
                            </div>
                        </div>
                        """)
                        
                        success_count += 1
                    else:
                        print(f"Error: GIF creation failed for sample {sample_id}")
                        html_file.write(f"""
                        <div class="comparison">
                            <div class="sample">
                                <h2>Error Processing Sample</h2>
                                <div class="metadata">
                                    <p><strong>Sample ID:</strong> {sample_id}</p>
                                    <p><strong>{class_info}</strong></p>
                                    <p class="error">Failed to create GIFs for this sample</p>
                                </div>
                            </div>
                        </div>
                        """)
                        error_count += 1
                
                except Exception as e:
                    print(f"Error processing sample {sample_id}: {e}")
                    import traceback
                    traceback.print_exc()
                    
                    # Class info for HTML
                    class_info = f"Class: {class_id} ({class_name})"
                    
                    html_file.write(f"""
                    <div class="comparison">
                        <div class="sample">
                            <h2>Error Processing Sample</h2>
                            <div class="metadata">
                                <p><strong>Sample ID:</strong> {sample_id}</p>
                                <p><strong>{class_info}</strong></p>
                                <p class="error">Error: {str(e)}</p>
                            </div>
                        </div>
                    </div>
                    """)
                    error_count += 1
                
                # Memory check after each sample
                if MEMORY_TRACKING:
                    gc.collect()  # Force garbage collection
                    memory_current = process.memory_info().rss / 1024 / 1024
                    print(f"Memory usage: {memory_current:.1f} MB " +
                         f"(Δ: {memory_current - memory_after_selection:+.1f} MB)")
            
            # Calculate processing statistics
            processing_time = time.time() - processing_start
            avg_time = processing_time / len(selected_samples) if selected_samples else 0
            
            # Add summary to HTML
            html_file.write(f"""
            <div class="summary">
                <h2>Processing Summary</h2>
                <p><strong>Total samples processed:</strong> {len(selected_samples)}</p>
                <p><strong>Successful:</strong> {success_count}</p>
                <p><strong>Failed:</strong> {error_count}</p>
                <p><strong>Total processing time:</strong> {processing_time:.1f} seconds</p>
                <p><strong>Average time per sample:</strong> {avg_time:.1f} seconds</p>
                <p><strong>Generated on:</strong> {time.strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            """)
            
            # Close the HTML file
            html_file.write("""
            </body>
            </html>
            """)
        
        # Final output
        total_time = time.time() - start_time
        print(f"\n=== Process Complete ===")
        print(f"Successfully processed {success_count}/{len(selected_samples)} samples " +
             f"({error_count} errors)")
        print(f"Total execution time: {total_time:.1f} seconds")
        print(f"Comparison gallery: {html_path}")
        print(f"Individual GIFs available in: {summary_dir}")
        
        # Final memory usage
        if MEMORY_TRACKING:
            memory_final = process.memory_info().rss / 1024 / 1024
            print(f"Final memory usage: {memory_final:.1f} MB " +
                 f"(Δ: {memory_final - memory_start:+.1f} MB)")
        
        return 0
    
    except Exception as e:
        print(f"\n=== Process Failed ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 