#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Shape Analyzer Module

This module provides a class for analyzing the size and shape of samples in the dataset.
It can:
- Calculate the number of samples per class
- Analyze the dimensions (x*y*z) of each sample
- Generate statistics about sample sizes
- Visualize distribution of sample dimensions
"""

import os
import sys
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Add parent directory to path to import from config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import config

class ShapeAnalyzer:
    """
    Class for analyzing the size and shape of samples in the dataset.
    """
    def __init__(self, data_dir=None, csv_path=None):
        """
        Initialize the shape analyzer.
        
        Args:
            data_dir (str, optional): Root directory for the dataset
            csv_path (str, optional): Path to CSV file with class information
        """
        # Use default paths from config if not provided
        self.data_dir = data_dir if data_dir else os.path.abspath(os.path.dirname(config.DATA_ROOT))
        self.csv_path = csv_path if csv_path else os.path.abspath(config.CLASS_CSV_PATH)
        
        # Adjust to include nuclei subdirectory
        self.nuclei_dir = os.path.join(self.data_dir, 'nuclei_sample_1a_v1')
        
        # Initialize containers for analysis results
        self.class_counts = {}
        self.class_names = {}
        self.sample_shapes = {}
        self.class_statistics = {}
        
        # Validate paths
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory not found: {self.data_dir}")
            
        if not os.path.exists(self.csv_path):
            raise ValueError(f"CSV file not found: {self.csv_path}")
            
        if not os.path.exists(self.nuclei_dir):
            raise ValueError(f"Nuclei directory not found: {self.nuclei_dir}")
            
        # Load class data from CSV
        self._load_class_data()
        
    def _load_class_data(self):
        """Load class information from the CSV file."""
        try:
            df = pd.read_csv(self.csv_path)
            print(f"Loaded {len(df)} sample records from {self.csv_path}")
            
            # Count samples per class
            class_counts = df['class_id'].value_counts().to_dict()
            self.class_counts = {int(k): v for k, v in class_counts.items()}
            
            # Get class names
            class_names = df.groupby('class_id')['class_name'].first().to_dict()
            self.class_names = {int(k): v for k, v in class_names.items()}
            
            # Map samples to their classes
            self.sample_to_class = {}
            for _, row in df.iterrows():
                self.sample_to_class[int(row['sample_id'])] = {
                    'class_id': int(row['class_id']),
                    'class_name': row['class_name']
                }
                
        except Exception as e:
            print(f"Error loading class data: {e}")
            raise
            
    def analyze_shapes(self, limit_per_class=None, verbose=True):
        """
        Analyze the shape (dimensions) of all samples in the dataset.
        
        Args:
            limit_per_class (int, optional): Limit number of samples to analyze per class
            verbose (bool): Whether to print progress information
            
        Returns:
            dict: Dictionary containing analysis results
        """
        if verbose:
            print("Analyzing sample shapes...")
            
        # Reset containers
        self.sample_shapes = {}
        self.class_statistics = defaultdict(dict)
        
        # Prepare a dictionary to collect samples by class
        samples_by_class = defaultdict(list)
        
        # Find existing sample directories
        for dirname in os.listdir(self.nuclei_dir):
            sample_dir = os.path.join(self.nuclei_dir, dirname)
            if os.path.isdir(sample_dir):
                try:
                    # Try to convert directory name to integer (sample ID)
                    sample_id = int(dirname)
                    if sample_id in self.sample_to_class:
                        class_id = self.sample_to_class[sample_id]['class_id']
                        samples_by_class[class_id].append((sample_id, sample_dir))
                except ValueError:
                    # Directory name is not a valid sample ID number, skip it
                    continue
        
        if verbose:
            print(f"Found samples in {len(samples_by_class)} classes")
            
        # Apply limit per class if specified
        if limit_per_class:
            for class_id in samples_by_class:
                if len(samples_by_class[class_id]) > limit_per_class:
                    samples_by_class[class_id] = samples_by_class[class_id][:limit_per_class]
        
        # Process each class
        for class_id, samples in samples_by_class.items():
            if verbose:
                print(f"\nAnalyzing {len(samples)} samples from class {class_id} ({self.class_names.get(class_id, 'Unknown')})")
            
            # Initialize lists to collect shape information
            x_dimensions = []
            y_dimensions = []
            z_dimensions = []
            volumes = []
            
            # Process each sample in this class
            for sample_id, sample_dir in tqdm(samples, desc=f"Class {class_id}", disable=not verbose):
                # Get raw directory
                raw_dir = os.path.join(sample_dir, 'raw')
                if not os.path.exists(raw_dir):
                    if verbose:
                        print(f"Warning: Raw directory not found for sample {sample_id}")
                    continue
                
                # Find image files
                image_files = []
                for ext in ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']:
                    image_files.extend(glob.glob(os.path.join(raw_dir, ext)))
                
                if not image_files:
                    if verbose:
                        print(f"Warning: No image files found for sample {sample_id}")
                    continue
                
                # Get number of slices (z dimension)
                z_dim = len(image_files)
                
                # Get x and y dimensions from first image
                try:
                    with Image.open(image_files[0]) as img:
                        x_dim, y_dim = img.size
                        
                        # Store dimensions
                        self.sample_shapes[sample_id] = {
                            'x': x_dim,
                            'y': y_dim,
                            'z': z_dim,
                            'volume': x_dim * y_dim * z_dim,
                            'class_id': class_id,
                            'class_name': self.class_names.get(class_id, 'Unknown')
                        }
                        
                        # Collect for statistics
                        x_dimensions.append(x_dim)
                        y_dimensions.append(y_dim)
                        z_dimensions.append(z_dim)
                        volumes.append(x_dim * y_dim * z_dim)
                        
                except Exception as e:
                    if verbose:
                        print(f"Error processing sample {sample_id}: {e}")
                    continue
            
            # Calculate statistics for this class
            if x_dimensions:
                self.class_statistics[class_id] = {
                    'count': len(x_dimensions),
                    'x_min': min(x_dimensions),
                    'x_max': max(x_dimensions),
                    'x_mean': np.mean(x_dimensions),
                    'x_std': np.std(x_dimensions),
                    'y_min': min(y_dimensions),
                    'y_max': max(y_dimensions),
                    'y_mean': np.mean(y_dimensions),
                    'y_std': np.std(y_dimensions),
                    'z_min': min(z_dimensions),
                    'z_max': max(z_dimensions),
                    'z_mean': np.mean(z_dimensions),
                    'z_std': np.std(z_dimensions),
                    'volume_min': min(volumes),
                    'volume_max': max(volumes),
                    'volume_mean': np.mean(volumes),
                    'volume_std': np.std(volumes)
                }
        
        return {
            'class_counts': self.class_counts,
            'class_names': self.class_names,
            'sample_shapes': self.sample_shapes,
            'class_statistics': dict(self.class_statistics)
        }
    
    def print_class_summary(self):
        """Print a summary of the number of samples per class."""
        print("\n=== Class Summary ===")
        print(f"Total classes: {len(self.class_counts)}")
        print(f"Total samples: {sum(self.class_counts.values())}")
        print("\nSamples per class:")
        
        for class_id in sorted(self.class_counts.keys()):
            count = self.class_counts[class_id]
            name = self.class_names.get(class_id, 'Unknown')
            print(f"Class {class_id} ({name}): {count} samples")
        
        print("====================")
    
    def print_shape_statistics(self):
        """Print statistics about sample shapes for each class."""
        if not self.class_statistics:
            print("No shape statistics available. Run analyze_shapes() first.")
            return
            
        print("\n=== Shape Statistics by Class ===")
        
        for class_id in sorted(self.class_statistics.keys()):
            stats = self.class_statistics[class_id]
            name = self.class_names.get(class_id, 'Unknown')
            
            print(f"\nClass {class_id} ({name}) - {stats['count']} samples:")
            print(f"  X dimension: {stats['x_mean']:.1f} ± {stats['x_std']:.1f} pixels (range: {stats['x_min']}-{stats['x_max']})")
            print(f"  Y dimension: {stats['y_mean']:.1f} ± {stats['y_std']:.1f} pixels (range: {stats['y_min']}-{stats['y_max']})")
            print(f"  Z dimension: {stats['z_mean']:.1f} ± {stats['z_std']:.1f} slices (range: {stats['z_min']}-{stats['z_max']})")
            print(f"  Volume: {stats['volume_mean']/1e6:.2f} ± {stats['volume_std']/1e6:.2f} million voxels")
            
        print("=============================")
    
    def plot_class_distribution(self, output_file=None, figsize=(10, 6)):
        """Plot the distribution of samples across classes."""
        plt.figure(figsize=figsize)
        
        # Sort classes by number of samples
        sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
        class_ids = [str(c[0]) for c in sorted_classes]
        counts = [c[1] for c in sorted_classes]
        
        # Create bar chart
        bars = plt.bar(class_ids, counts)
        
        # Add labels
        plt.xlabel('Class ID')
        plt.ylabel('Number of Samples')
        plt.title('Distribution of Samples by Class')
        plt.xticks(rotation=45)
        
        # Add class names as annotations
        for i, (class_id, count) in enumerate(sorted_classes):
            name = self.class_names.get(int(class_id), 'Unknown')
            plt.annotate(f"{name}", 
                        xy=(i, count), 
                        xytext=(0, 5),
                        textcoords='offset points',
                        ha='center', va='bottom',
                        fontsize=8, rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved class distribution plot to {output_file}")
            
        plt.show()
    
    def plot_dimension_distributions(self, output_file=None, figsize=(15, 10)):
        """Plot distributions of X, Y, and Z dimensions across classes."""
        if not self.class_statistics:
            print("No shape statistics available. Run analyze_shapes() first.")
            return
            
        # Setup figure and axes
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Get classes with statistics
        class_ids = sorted(self.class_statistics.keys())
        
        # Plot X dimension
        axs[0, 0].bar(
            [str(c) for c in class_ids],
            [self.class_statistics[c]['x_mean'] for c in class_ids],
            yerr=[self.class_statistics[c]['x_std'] for c in class_ids],
            capsize=5
        )
        axs[0, 0].set_title('X Dimension by Class')
        axs[0, 0].set_xlabel('Class ID')
        axs[0, 0].set_ylabel('Pixels')
        axs[0, 0].set_xticklabels([str(c) for c in class_ids], rotation=45)
        
        # Plot Y dimension
        axs[0, 1].bar(
            [str(c) for c in class_ids],
            [self.class_statistics[c]['y_mean'] for c in class_ids],
            yerr=[self.class_statistics[c]['y_std'] for c in class_ids],
            capsize=5
        )
        axs[0, 1].set_title('Y Dimension by Class')
        axs[0, 1].set_xlabel('Class ID')
        axs[0, 1].set_ylabel('Pixels')
        axs[0, 1].set_xticklabels([str(c) for c in class_ids], rotation=45)
        
        # Plot Z dimension
        axs[1, 0].bar(
            [str(c) for c in class_ids],
            [self.class_statistics[c]['z_mean'] for c in class_ids],
            yerr=[self.class_statistics[c]['z_std'] for c in class_ids],
            capsize=5
        )
        axs[1, 0].set_title('Z Dimension (Slices) by Class')
        axs[1, 0].set_xlabel('Class ID')
        axs[1, 0].set_ylabel('Number of Slices')
        axs[1, 0].set_xticklabels([str(c) for c in class_ids], rotation=45)
        
        # Plot Volume
        axs[1, 1].bar(
            [str(c) for c in class_ids],
            [self.class_statistics[c]['volume_mean']/1e6 for c in class_ids],  # Convert to millions
            yerr=[self.class_statistics[c]['volume_std']/1e6 for c in class_ids],
            capsize=5
        )
        axs[1, 1].set_title('Volume by Class')
        axs[1, 1].set_xlabel('Class ID')
        axs[1, 1].set_ylabel('Million Voxels')
        axs[1, 1].set_xticklabels([str(c) for c in class_ids], rotation=45)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            print(f"Saved dimension distribution plot to {output_file}")
            
        plt.show()
        
    def export_statistics_to_csv(self, output_file='shape_statistics.csv'):
        """Export shape statistics to a CSV file."""
        if not self.class_statistics:
            print("No shape statistics available. Run analyze_shapes() first.")
            return
            
        # Create records for the CSV
        records = []
        
        for class_id, stats in self.class_statistics.items():
            record = {
                'class_id': class_id,
                'class_name': self.class_names.get(class_id, 'Unknown'),
                'sample_count': stats['count'],
                'x_mean': stats['x_mean'],
                'x_std': stats['x_std'],
                'x_min': stats['x_min'],
                'x_max': stats['x_max'],
                'y_mean': stats['y_mean'],
                'y_std': stats['y_std'],
                'y_min': stats['y_min'],
                'y_max': stats['y_max'],
                'z_mean': stats['z_mean'],
                'z_std': stats['z_std'],
                'z_min': stats['z_min'],
                'z_max': stats['z_max'],
                'volume_mean': stats['volume_mean'],
                'volume_std': stats['volume_std'],
                'volume_min': stats['volume_min'],
                'volume_max': stats['volume_max']
            }
            records.append(record)
            
        # Create dataframe and export
        df = pd.DataFrame(records)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        df.to_csv(output_file, index=False)
        print(f"Exported statistics to {output_file}")
        
        return df
        
    def export_sample_shapes_to_csv(self, output_file='sample_shapes.csv'):
        """Export individual sample shapes to a CSV file."""
        if not self.sample_shapes:
            print("No sample shapes available. Run analyze_shapes() first.")
            return
            
        # Create dataframe from the sample shapes dictionary
        records = []
        
        for sample_id, shape_info in self.sample_shapes.items():
            record = {
                'sample_id': sample_id,
                'class_id': shape_info['class_id'],
                'class_name': shape_info['class_name'],
                'x_dimension': shape_info['x'],
                'y_dimension': shape_info['y'],
                'z_dimension': shape_info['z'],
                'volume': shape_info['volume']
            }
            records.append(record)
            
        df = pd.DataFrame(records)
        
        # Ensure output directory exists
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        df.to_csv(output_file, index=False)
        print(f"Exported sample shapes to {output_file}")
        
        return df


def main():
    """Run the shape analyzer from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze the size and shape of samples in the dataset')
    parser.add_argument('--data-dir', type=str, help='Root directory for the dataset')
    parser.add_argument('--csv-path', type=str, help='Path to CSV file with class information')
    parser.add_argument('--limit', type=int, help='Limit number of samples to analyze per class')
    parser.add_argument('--output-dir', type=str, default='results/analysis', help='Directory for output files')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir, exist_ok=True)
            
        print("Initializing Shape Analyzer...")
        analyzer = ShapeAnalyzer(data_dir=args.data_dir, csv_path=args.csv_path)
        
        # Print class information
        analyzer.print_class_summary()
        
        # Analyze shapes
        print("\nAnalyzing sample shapes...")
        analyzer.analyze_shapes(limit_per_class=args.limit)
        
        # Print shape statistics
        analyzer.print_shape_statistics()
        
        # Export data to CSV
        analyzer.export_statistics_to_csv(os.path.join(args.output_dir, 'class_shape_statistics.csv'))
        analyzer.export_sample_shapes_to_csv(os.path.join(args.output_dir, 'sample_shapes.csv'))
        
        # Create plots
        print("\nGenerating plots...")
        analyzer.plot_class_distribution(os.path.join(args.output_dir, 'class_distribution.png'))
        analyzer.plot_dimension_distributions(os.path.join(args.output_dir, 'dimension_distributions.png'))
        
        print(f"\nAnalysis complete. Results saved to {args.output_dir}")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 