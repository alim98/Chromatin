# Analysis Scripts

Documentation for the analysis scripts created to examine chromatin class distributions and properties.

## Table of Contents
- [analyze_class_distribution.py](#analyze_class_distributionpy)
- [class_comparison.py](#class_comparisonpy)
- [Usage Examples](#usage-examples)
- [Output Examples](#output-examples)

## analyze_class_distribution.py

This script analyzes the distribution of chromatin classes in the dataset.

### Purpose

- Calculate and display class distribution statistics
- Generate visualizations of class distributions
- Analyze sample and slice counts within the dataset

### Key Features

1. **Class Distribution Analysis**
   - Counts samples per class
   - Calculates percentage distributions
   - Handles optional inclusion/exclusion of unclassified samples

2. **Data Visualization**
   - Bar chart of sample counts per class
   - Pie chart showing percentage distribution
   - Saves high-resolution images

3. **Dataset Statistics**
   - Counts total samples and slices
   - Calculates average slices per sample
   - Identifies samples with min/max slice counts

### Implementation

The script follows this workflow:
1. Reads and optionally filters the CSV file
2. Calculates statistics on class distributions
3. Generates visualizations
4. Uses the dataloader to analyze image statistics

## class_comparison.py

This script provides detailed comparisons between chromatin classes.

### Purpose

- Compare specific classes in detail
- Generate visualizations for comparing class properties
- Provide statistical analysis of classes

### Key Features

1. **Class Filtering**
   - Select specific classes for comparison
   - Filter unclassified samples
   - Focused analysis on classes of interest

2. **Enhanced Visualizations**
   - Comparison bar charts
   - Distribution pie charts with visual enhancements (exploded slices, shadows)
   - Sample image visualization

3. **Statistical Analysis**
   - Slice count statistics across classes
   - Min/max/median calculations
   - Visual statistical comparisons (heatmaps, boxplots)

### Implementation

The script supports:
1. Loading and filtering data
2. Creating multiple visualization types
3. Performing class-specific comparisons
4. Sampling and displaying actual images from selected classes

## Usage Examples

### Basic Class Distribution Analysis

```bash
python scripts/data_analyse/analyze_class_distribution.py
```

### Include Unclassified Samples

```bash
python scripts/data_analyse/analyze_class_distribution.py --include-unclassified
```

### Compare Specific Classes

```bash
python scripts/data_analyse/class_comparison.py --classes 2 18 4
```

### Use Custom Paths

```bash
python scripts/data_analyse/analyze_class_distribution.py --csv path/to/custom.csv --data path/to/data
```

## Output Examples

All outputs are saved to the `ANALYSIS_OUTPUT_DIR` specified in `config.py`.

### Files Generated

1. **analyze_class_distribution.py**
   - `class_distribution.png` - Visual representation of class distributions
   - `class_distribution_stats.csv` - CSV file with class statistics

2. **class_comparison.py**
   - `class_count_comparison.png` - Bar chart comparing class counts
   - `class_percentage_distribution.png` - Pie chart of class percentages
   - `class_slice_distribution.png` - Boxplot of slice distribution (if available)
   - `class_slice_stats_heatmap.png` - Heatmap of min/max/median slice counts
   - `specific_class_comparison.png` - Bar chart of selected classes
   - `class_sample_images.png` - Sample images from selected classes 