# Dataloader Optimizations

This document details the optimizations made to the nuclei dataloader to improve performance and usability.

## Table of Contents
- [Ignoring Unclassified Samples](#ignoring-unclassified-samples)
- [CSV-First Approach](#csv-first-approach)
- [Implementation Details](#implementation-details)

## Ignoring Unclassified Samples

### Problem

The dataset contains a large number of samples (82.58%) marked as "Unclassified" (class_id = 19) in the CSV file. 
These unclassified samples were being loaded unnecessarily, consuming resources and complicating analysis.

### Solution

The dataloader was modified to optionally ignore unclassified samples:

1. Added an `ignore_unclassified` parameter (default: True) to the `load_chromatin_classes` function
2. Added filtering logic that skips samples where:
   - `class_name` is "Unclassified" OR
   - `class_id` is 19

### Benefit

- Reduced memory usage by approximately 80% when unclassified samples are ignored
- Cleaner dataset for training and analysis
- Faster loading times

## CSV-First Approach

### Problem

The original dataloader implementation first scanned all directories and then filtered them based on classes,
which was inefficient, especially when only specific classes were needed.

### Solution

Refactored the dataloader to:

1. First read and filter the CSV file
2. Determine the needed sample IDs from CSV data
3. Only scan the directories that match these IDs

### Implementation

```python
# First, determine which samples to include based on CSV
filtered_sample_ids = None
if class_csv_path and os.path.exists(class_csv_path):
    # Read CSV file directly
    df = pd.read_csv(class_csv_path)
    
    # Filter by class_id if needed
    if filter_by_class is not None:
        df = df[df['class_id'].isin(filter_by_class)]
    
    # Filter out unclassified samples if needed
    if ignore_unclassified:
        df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
    
    # Convert to list of sample_ids
    filtered_sample_ids = [str(sid) for sid in df['sample_id'].unique()]
```

### Benefit

- Significantly faster loading when filtering is requested
- Better handling of large datasets with many directories
- More efficient resource usage
- Optimized for common use cases (selecting specific classes)

## Implementation Details

### Key Changes to NucleiDataset.__init__

1. Added `ignore_unclassified` parameter with default value of `True`
2. Changed directory scanning logic:
   ```python
   if filtered_sample_ids is not None:
       # Only get directories that match the filtered sample IDs
       sample_dirs = [sid for sid in filtered_sample_ids if os.path.isdir(os.path.join(root_dir, sid))]
   else:
       # If no filtering, get all directories
       sample_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
   ```

### Usage Examples

```python
# Load only specific classes, ignoring unclassified samples
dataloader = get_nuclei_dataloader(
    root_dir="path/to/data",
    class_csv_path="path/to/chromatin_classes_and_samples.csv",
    filter_by_class=[2, 18, 4]  # Get only classes 2, 18, and 4
)

# Include unclassified samples if needed
dataloader_with_all = get_nuclei_dataloader(
    root_dir="path/to/data",
    class_csv_path="path/to/chromatin_classes_and_samples.csv",
    ignore_unclassified=False
) 