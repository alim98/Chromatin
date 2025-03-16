# Volume GIF Creator

The `VolumeGifCreator` class provides specialized functionality for creating GIFs from 3D volumes represented as sequences of 2D slice images. This documentation explains how to use the class to visualize 3D datasets efficiently.

## Overview

This tool is designed to:

1. Create animated GIFs from a series of 2D slices that form a 3D volume
2. Create static montages showing representative slices from one or more volumes
3. Handle memory constraints through batch processing
4. Provide progress indication during the creation process
5. Support both image-only volumes and image+mask combinations
6. Generate visualizations from random samples for dataset exploration

## Installation

The `VolumeGifCreator` class is part of the nuclei visualization module. No additional installation is required beyond the core dependencies:

- numpy
- matplotlib
- PIL (Pillow)
- tqdm (for progress bars)
- pandas (for random sample selection)

## Basic Usage

### Creating a GIF from a Sample Directory

```python
from scripts.visualization.nuclei_visualizer import VolumeGifCreator

# Create a VolumeGifCreator instance
creator = VolumeGifCreator(
    output_dir='results/visualizations',
    batch_size=5,         # Process 5 frames at a time
    max_frames=80,        # Limit to 80 frames maximum
    dpi=80,               # Lower DPI to save memory
    figsize=(6, 6)        # Size of output frames
)

# Create a GIF from a sample directory
gif_path = creator.create_gif_from_folder(
    folder_path='path/to/sample/directory',
    output_filename='volume.gif',   # Optional, will be auto-generated if not provided
    duration=200,                   # Milliseconds per frame
    include_masks=True,             # Include mask overlays if available
    mask_alpha=0.3                  # Transparency of mask overlay
)

print(f"GIF created at: {gif_path}")
```

### Creating a Montage of Multiple Samples

```python
# List of sample directories
sample_dirs = [
    'path/to/sample1',
    'path/to/sample2',
    'path/to/sample3'
]

# Create a montage with representative slices from each sample
montage_path = creator.create_volume_montage(
    sample_dirs=sample_dirs,
    output_filename='volume_montage.png',
    slices_per_sample=3,            # Number of representative slices per sample
    include_masks=True              # Include mask overlays if available
)

print(f"Montage created at: {montage_path}")
```

## Command-line Interface

The package includes a command-line interface for creating GIFs and montages:

```bash
# Create a GIF from a single sample
python scripts/visualization/volume_gif_creator.py --sample path/to/sample_dir

# Create GIFs from multiple samples matching a pattern
python scripts/visualization/volume_gif_creator.py --batch path/to/parent_dir --pattern "sample_*"

# Create a montage of slices from multiple samples
python scripts/visualization/volume_gif_creator.py --montage path/to/parent_dir --pattern "class2_*"

# Create GIFs from 5 random samples
python scripts/visualization/volume_gif_creator.py --random 5

# Create GIFs from 3 random samples of class 2
python scripts/visualization/volume_gif_creator.py --random 3 --class 2
```

### Command-line Options

| Option | Description |
| ------ | ----------- |
| `--sample DIR` | Path to a single sample directory |
| `--batch DIR` | Path to parent directory containing multiple sample folders |
| `--montage DIR` | Create a montage of slices from multiple samples |
| `--random N` | Process N random samples from the dataset |
| `--class ID` | Filter random samples by class ID |
| `--pattern PATTERN` | Pattern to match sample directories (for batch mode) |
| `--output-dir DIR` | Output directory for GIFs (default: results/visualizations) |
| `--max-frames N` | Maximum number of frames per GIF (default: 80) |
| `--batch-size N` | Number of frames to process at once (default: 5) |
| `--duration MS` | Duration of each frame in milliseconds (default: 200) |
| `--no-masks` | Disable mask overlays |
| `--slices N` | Number of slices per sample for montage mode (default: 3) |
| `--csv-path PATH` | Path to CSV file with class information (for random mode) |
| `--data-dir DIR` | Root directory for the dataset (for random mode) |

## Directory Structure

The `VolumeGifCreator` class expects samples to be organized in either of these structures:

1. **Standard structure (with masks)**:
   ```
   sample_directory/
     ├── raw/
     │    ├── slice1.tif
     │    ├── slice2.tif
     │    └── ...
     └── mask/
          ├── slice1.tif
          ├── slice2.tif
          └── ...
   ```

2. **Simple structure (without masks)**:
   ```
   sample_directory/
     ├── slice1.tif
     ├── slice2.tif
     └── ...
   ```

## Random Sample Selection

The random sample selection feature allows you to:

1. Create GIFs from a specified number of random samples from your dataset
2. Filter the random selection to samples of a specific class
3. Automatically generate a montage showing representative slices from all processed samples

This is particularly useful for exploring and understanding your dataset. The tool automatically:

1. Reads the sample and class information from a CSV file
2. Filters samples by class if requested
3. Randomly selects the specified number of samples
4. Creates individual GIFs for each sample
5. Generates a montage to see all samples side by side

The random selection requires the class CSV file (by default at the path defined in `config.CLASS_CSV_PATH`) with columns:
- `sample_id`: Unique identifier for each sample
- `class_id`: Numeric class identifier
- `class_name`: Human-readable class name

If you use custom paths, specify them with `--csv-path` and `--data-dir`.

## Memory Management

The `VolumeGifCreator` class is designed to handle large volumes efficiently:

1. **Batch Processing**: Processes frames in small batches to avoid loading all frames into memory at once
2. **Garbage Collection**: Performs garbage collection between batches to free memory
3. **Frame Downsampling**: Automatically downsamples frames if the total exceeds `max_frames`
4. **Low DPI**: Uses lower DPI values for faster processing and lower memory usage

## Examples

See the `scripts/examples/volume_gif_example.py` script for a complete example of using the `VolumeGifCreator` class.

## Troubleshooting

If you encounter memory issues:

1. Reduce the `batch_size` parameter (e.g., from 5 to 3 or 2)
2. Lower the `max_frames` parameter to include fewer frames in the final GIF
3. Reduce the `dpi` parameter to create smaller images
4. Decrease the `figsize` parameter to create smaller frames

If no GIF is created:

1. Check that the sample directory exists and contains image files
2. Verify the directory structure (raw/mask subdirectories or direct images)
3. Ensure the images are in a supported format (.tif, .png, .jpg)
4. Check console output for specific error messages 