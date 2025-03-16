#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Analyze the distribution of chromatin classes from the dataset.
This script examines class distributions, generates statistics, and visualizes the data.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add parent directory to path to import from dataloader and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataloader.nuclei_dataloader import get_nuclei_dataloader
import config

def load_class_data(csv_path):
    """
    Load class data from CSV file and return a DataFrame
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def analyze_class_distribution(df, include_unclassified=False):
    """
    Analyze and visualize the distribution of chromatin classes
    
    Args:
        df (DataFrame): DataFrame containing class information
        include_unclassified (bool): Whether to include unclassified samples in analysis
    """
    # Filter out unclassified if needed
    if not include_unclassified:
        df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
    
    # Count samples per class
    class_counts = df.groupby(['class_id', 'class_name']).size().reset_index(name='count')
    
    # Sort by count (descending)
    class_counts = class_counts.sort_values('count', ascending=False)
    
    # Print statistics
    print("\n--- Class Distribution Statistics ---")
    print(f"Total number of samples: {df['sample_id'].nunique()}")
    print(f"Total number of classes: {class_counts.shape[0]}")
    print("\nSamples per class:")
    for _, row in class_counts.iterrows():
        print(f"  Class {row['class_id']} ({row['class_name']}): {row['count']} samples")
    
    # Calculate percentages
    total_samples = class_counts['count'].sum()
    class_counts['percentage'] = (class_counts['count'] / total_samples * 100).round(2)
    
    print("\nClass distribution percentages:")
    for _, row in class_counts.iterrows():
        print(f"  Class {row['class_id']} ({row['class_name']}): {row['percentage']}%")
    
    # Create visualizations
    plt.figure(figsize=(12, 8))
    
    # Bar plot
    plt.subplot(2, 1, 1)
    sns.barplot(x='class_name', y='count', data=class_counts)
    plt.title('Number of Samples per Chromatin Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Pie chart
    plt.subplot(2, 1, 2)
    plt.pie(class_counts['count'], labels=class_counts['class_name'], autopct='%1.1f%%')
    plt.title('Percentage Distribution of Chromatin Classes')
    plt.axis('equal')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = config.ANALYSIS_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
    
    # Save statistics to CSV
    class_counts.to_csv(os.path.join(output_dir, 'class_distribution_stats.csv'), index=False)
    
    print(f"\nVisualization saved to {os.path.join(output_dir, 'class_distribution.png')}")
    print(f"Statistics saved to {os.path.join(output_dir, 'class_distribution_stats.csv')}")
    
    return class_counts

def analyze_dataset_statistics(root_dir, csv_path):
    """
    Analyze detailed statistics about the dataset
    """
    # Create a temporary dataloader to get sample and slice counts
    try:
        dataloader = get_nuclei_dataloader(
            root_dir=root_dir,
            class_csv_path=csv_path,
            batch_size=1,
            num_workers=0  # Use 0 to avoid multiprocessing complexity for analysis
        )
        
        dataset = dataloader.dataset
        
        # Count slices per sample
        sample_slice_counts = Counter([item[2] for item in dataset.samples])
        
        # Calculate statistics
        total_samples = len(sample_slice_counts)
        total_slices = len(dataset)
        avg_slices_per_sample = total_slices / total_samples if total_samples > 0 else 0
        
        # Print dataset statistics
        print("\n--- Dataset Statistics ---")
        print(f"Total number of samples with images: {total_samples}")
        print(f"Total number of image slices: {total_slices}")
        print(f"Average slices per sample: {avg_slices_per_sample:.2f}")
        
        # Analyze samples with most and least slices
        if sample_slice_counts:
            min_slices = min(sample_slice_counts.values())
            max_slices = max(sample_slice_counts.values())
            
            print(f"Minimum slices per sample: {min_slices}")
            print(f"Maximum slices per sample: {max_slices}")
            
            # Get samples with min and max slices
            samples_with_min = [sample for sample, count in sample_slice_counts.items() if count == min_slices]
            samples_with_max = [sample for sample, count in sample_slice_counts.items() if count == max_slices]
            
            print(f"Samples with fewest slices ({min_slices}): {samples_with_min[:5]}{'...' if len(samples_with_min) > 5 else ''}")
            print(f"Samples with most slices ({max_slices}): {samples_with_max[:5]}{'...' if len(samples_with_max) > 5 else ''}")
    
    except Exception as e:
        print(f"Error analyzing dataset statistics: {e}")

def main():
    # Use paths from config.py
    default_csv_path = os.path.abspath(config.CLASS_CSV_PATH)
    default_root_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
    
    # Allow command-line overrides
    import argparse
    parser = argparse.ArgumentParser(description='Analyze chromatin class distribution.')
    parser.add_argument('--csv', dest='csv_path', default=default_csv_path,
                        help=f'Path to CSV file with class information (default: {default_csv_path})')
    parser.add_argument('--data', dest='root_dir', default=default_root_dir,
                        help=f'Root directory of the nuclei dataset (default: {default_root_dir})')
    parser.add_argument('--include-unclassified', action='store_true',
                        help='Include unclassified samples in the analysis')
    args = parser.parse_args()
    
    print(f"Using CSV file: {args.csv_path}")
    print(f"Using data directory: {args.root_dir}")
    
    try:
        # Load class data
        df = load_class_data(args.csv_path)
        
        # Analyze class distribution
        analyze_class_distribution(df, include_unclassified=args.include_unclassified)
        
        # Analyze dataset statistics
        analyze_dataset_statistics(args.root_dir, args.csv_path)
        
        plt.show()  # Show plots interactively if run in interactive mode
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 