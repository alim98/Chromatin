#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Example script for analyzing the size and shape of samples in the dataset.

This script demonstrates how to use the ShapeAnalyzer class to:
1. Count samples per class
2. Analyze the dimensions (x*y*z) of samples
3. Generate statistics and plots
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from scripts.data_analysis.shape_analyzer import ShapeAnalyzer
import config

def main():
    """Run a simple example of shape analysis."""
    # Create output directory
    output_dir = 'results/examples/shape_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== Sample Shape Analysis Example ===")
    
    # Create the shape analyzer
    analyzer = ShapeAnalyzer()
    
    # Print a summary of the classes in the dataset
    print("\nStep 1: Print class summary")
    analyzer.print_class_summary()
    
    # Analyze shapes - limiting to 5 samples per class for speed
    print("\nStep 2: Analyze sample shapes (limiting to 5 samples per class for speed)")
    analyzer.analyze_shapes(limit_per_class=5)
    
    # Print shape statistics
    print("\nStep 3: Print shape statistics by class")
    analyzer.print_shape_statistics()
    
    # Export statistics to CSV
    print("\nStep 4: Export statistics to CSV")
    analyzer.export_statistics_to_csv(output_file=os.path.join(output_dir, 'class_shape_statistics.csv'))
    analyzer.export_sample_shapes_to_csv(output_file=os.path.join(output_dir, 'sample_shapes.csv'))
    
    # Plot class distribution
    print("\nStep 5: Generate plots")
    analyzer.plot_class_distribution(output_file=os.path.join(output_dir, 'class_distribution.png'))
    
    # Plot dimension distributions
    analyzer.plot_dimension_distributions(output_file=os.path.join(output_dir, 'dimension_distributions.png'))
    
    print(f"\nExample complete! Results saved to {output_dir}")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 