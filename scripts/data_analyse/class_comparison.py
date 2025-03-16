#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compare and analyze chromatin classes in more detail.
This script provides deeper analysis comparing different chromatin classes,
including sample count distributions, statistical tests, and visualizations.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# Add parent directory to path to import from dataloader and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataloader.nuclei_dataloader import get_nuclei_dataloader
import config

def load_data(csv_path, root_dir, selected_classes=None, ignore_unclassified=True):
    """
    Load class data from CSV file and return relevant DataFrames and statistics
    
    Args:
        csv_path (str): Path to CSV file with class information
        root_dir (str): Root directory for the dataset
        selected_classes (list): List of class IDs to focus on (if None, use all)
        ignore_unclassified (bool): Whether to ignore unclassified samples
        
    Returns:
        tuple: (df, class_counts, class_stats) - DataFrames with class info and statistics
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Filter unclassified if needed
    if ignore_unclassified:
        df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
    
    # Filter selected classes if specified
    if selected_classes is not None:
        df = df[df['class_id'].isin(selected_classes)]
    
    # Count samples per class
    class_counts = df.groupby(['class_id', 'class_name']).size().reset_index(name='count')
    class_counts = class_counts.sort_values('count', ascending=False)
    
    # Calculate percentages
    total_samples = class_counts['count'].sum()
    class_counts['percentage'] = (class_counts['count'] / total_samples * 100).round(2)
    
    # Generate class statistics
    class_stats = pd.DataFrame()
    
    # Try to get dataloader to add slice counts if root_dir exists
    if os.path.exists(root_dir):
        try:
            dataloader = get_nuclei_dataloader(
                root_dir=root_dir,
                class_csv_path=csv_path,
                batch_size=1,
                num_workers=0
            )
            
            # Group samples by class and count slices
            slice_data = defaultdict(list)
            
            for sample in dataloader.dataset.samples:
                sample_id = sample[2]  # Extract sample_id
                if sample_id in dataloader.dataset.sample_to_class:
                    class_id = dataloader.dataset.sample_to_class[sample_id]['class_id']
                    slice_data[class_id].append(sample_id)
            
            # Calculate statistics for each class
            stats_list = []
            for class_id, samples in slice_data.items():
                # Count unique samples in this class
                unique_samples = set(samples)
                sample_count = len(unique_samples)
                
                # Count slices per sample
                slice_counts = {sample: samples.count(sample) for sample in unique_samples}
                
                if slice_counts:
                    stats_list.append({
                        'class_id': class_id,
                        'sample_count': sample_count,
                        'slice_count': sum(slice_counts.values()),
                        'avg_slices_per_sample': sum(slice_counts.values()) / sample_count,
                        'min_slices': min(slice_counts.values()),
                        'max_slices': max(slice_counts.values()),
                        'median_slices': np.median(list(slice_counts.values())),
                    })
            
            if stats_list:
                class_stats = pd.DataFrame(stats_list)
                
                # Add class names
                class_name_map = dict(zip(df['class_id'], df['class_name']))
                class_stats['class_name'] = class_stats['class_id'].map(class_name_map)
                
                # Sort by count
                class_stats = class_stats.sort_values('sample_count', ascending=False)
            
        except Exception as e:
            print(f"Warning: Couldn't create dataloader statistics: {e}")
    
    return df, class_counts, class_stats

def visualize_class_comparisons(class_counts, class_stats=None, output_dir=None):
    """
    Create visualizations comparing different classes
    
    Args:
        class_counts (DataFrame): DataFrame with class count information
        class_stats (DataFrame): DataFrame with additional class statistics (optional)
        output_dir (str): Directory to save output files (if None, use default)
    """
    if output_dir is None:
        output_dir = config.ANALYSIS_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Top classes bar chart
    plt.figure(figsize=(14, 8))
    sns.barplot(x='class_name', y='count', data=class_counts)
    plt.title('Number of Samples per Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_count_comparison.png'), dpi=300)
    plt.close()
    
    # Percentage distribution
    plt.figure(figsize=(10, 10))
    explode = [0.1 if i == 0 else 0 for i in range(len(class_counts))]  # Explode largest class
    plt.pie(
        class_counts['count'], 
        labels=class_counts['class_name'], 
        autopct='%1.1f%%',
        explode=explode,
        shadow=True
    )
    plt.title('Percentage Distribution of Classes')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_percentage_distribution.png'), dpi=300)
    plt.close()
    
    # If we have additional stats, create more visualizations
    if class_stats is not None and not class_stats.empty:
        # Distribution of slices per sample across classes
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='class_name', y='avg_slices_per_sample', data=class_stats)
        plt.title('Distribution of Slices per Sample Across Classes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'class_slice_distribution.png'), dpi=300)
        plt.close()
        
        # Heatmap of min, max, median slices
        if len(class_stats) > 1:  # Only if we have multiple classes
            heat_data = class_stats.set_index('class_name')[['min_slices', 'median_slices', 'max_slices']]
            plt.figure(figsize=(10, 8))
            sns.heatmap(heat_data, annot=True, cmap='viridis', fmt='.1f')
            plt.title('Slice Count Statistics by Class')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_slice_stats_heatmap.png'), dpi=300)
            plt.close()

def compare_specific_classes(df, root_dir, class_ids, output_dir=None):
    """
    Perform a detailed comparison between specific classes
    
    Args:
        df (DataFrame): DataFrame with class information
        root_dir (str): Root directory for the dataset
        class_ids (list): List of class IDs to compare
        output_dir (str): Directory to save output files
    """
    if output_dir is None:
        output_dir = config.ANALYSIS_OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter for selected classes
    filtered_df = df[df['class_id'].isin(class_ids)]
    class_names = {class_id: name for class_id, name in zip(
        filtered_df['class_id'], filtered_df['class_name']
    )}
    
    # Count samples in each class
    counts = filtered_df.groupby('class_id')['sample_id'].nunique()
    
    # Create a comparison plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        [class_names.get(cid, f"Class {cid}") for cid in counts.index], 
        counts.values
    )
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2., 
            height + 0.1,
            f'{height}',
            ha='center', va='bottom'
        )
    
    plt.title(f'Comparison of Selected Classes')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'specific_class_comparison.png'), dpi=300)
    plt.close()
    
    # If root_dir exists, try to load some samples for visual comparison
    if os.path.exists(root_dir):
        try:
            # Create a figure to show sample images from each class
            plt.figure(figsize=(15, 5 * len(class_ids)))
            
            for i, class_id in enumerate(class_ids):
                class_name = class_names.get(class_id, f"Class {class_id}")
                
                # Create a temporary dataloader to get a few samples from this class
                dataloader = get_nuclei_dataloader(
                    root_dir=root_dir,
                    class_csv_path=None,  # Don't need CSV as we're filtering explicitly
                    batch_size=1,
                    num_workers=0,
                    filter_by_class=class_id,
                    sample_ids=filtered_df[filtered_df['class_id'] == class_id]['sample_id'].unique()[:3]
                )
                
                # Get up to 3 samples to show
                sample_count = min(3, len(dataloader.dataset))
                
                if sample_count > 0:
                    for j in range(sample_count):
                        if j < len(dataloader.dataset):
                            sample = dataloader.dataset[j]
                            plt.subplot(len(class_ids), 3, i*3 + j + 1)
                            
                            # Convert tensor to numpy and adjust for display
                            img = sample['image'].numpy().transpose(1, 2, 0)
                            img = (img + 1) / 2  # Adjust from [-1,1] to [0,1]
                            
                            plt.imshow(img.squeeze(), cmap='gray')
                            plt.title(f"{class_name}\nSample {sample['metadata']['sample_id']}, Slice {sample['metadata']['slice_num']}")
                            plt.axis('off')
                
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'class_sample_images.png'), dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"Warning: Couldn't create sample image comparison: {e}")
            
def main():
    # Use paths from config.py
    default_csv_path = os.path.abspath(config.CLASS_CSV_PATH)
    default_root_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
    default_output_dir = config.ANALYSIS_OUTPUT_DIR
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser(description='Compare chromatin classes.')
    parser.add_argument('--csv', dest='csv_path', default=default_csv_path,
                        help=f'Path to CSV file with class information')
    parser.add_argument('--data', dest='root_dir', default=default_root_dir,
                        help=f'Root directory of the nuclei dataset')
    parser.add_argument('--output', dest='output_dir', default=default_output_dir,
                        help=f'Directory to save analysis outputs')
    parser.add_argument('--classes', dest='selected_classes', nargs='+', type=int,
                        help='Specific class IDs to analyze (space-separated)')
    parser.add_argument('--include-unclassified', action='store_true',
                        help='Include unclassified samples in the analysis')
    
    args = parser.parse_args()
    
    print(f"Using CSV file: {args.csv_path}")
    print(f"Using data directory: {args.root_dir}")
    print(f"Saving outputs to: {args.output_dir}")
    
    if args.selected_classes:
        print(f"Focusing on classes: {args.selected_classes}")
    
    try:
        # Load data
        df, class_counts, class_stats = load_data(
            args.csv_path, 
            args.root_dir,
            selected_classes=args.selected_classes,
            ignore_unclassified=not args.include_unclassified
        )
        
        # Create visualizations
        visualize_class_comparisons(class_counts, class_stats, args.output_dir)
        
        # If specific classes were selected, do a detailed comparison
        if args.selected_classes and len(args.selected_classes) >= 2:
            compare_specific_classes(df, args.root_dir, args.selected_classes, args.output_dir)
        
        print(f"Analysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 