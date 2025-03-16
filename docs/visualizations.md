# Nuclei Visualization Module

This document details the visualization module developed for the Chromatin project, which provides tools for visualizing nuclei images, masks, and creating animations.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Usage Examples](#usage-examples)
- [Implementation Details](#implementation-details)
- [Output Types](#output-types)
- [Improved Visualization Capabilities](#improved-visualization-capabilities)

## Overview

The `NucleiVisualizer` class provides a comprehensive set of visualization tools specifically designed for the nuclei dataset. It supports visualization of individual slices, multiple slices in a grid layout, and creation of animated GIFs from sequences of slices.

## Features

1. **Single Image Visualization**
   - Display raw nuclei images
   - Optional mask overlay
   - Customizable appearance

2. **Multi-Image Grid Visualization**
   - Arrange multiple slices in a customizable grid
   - Support for titles and annotations
   - Automatic layout calculation

3. **GIF Animation**
   - Create animated GIFs from slice sequences
   - Control frame rate and looping
   - Include mask overlays in animations
   - Progress tracking for long operations

4. **Sample-Based Visualization**
   - Visualize all slices from a specific sample
   - Generate both static grids and animated GIFs
   - Include class information when available

5. **Class-Based Visualization**
   - Visualize samples from a specific chromatin class
   - Compare multiple samples with the same class
   - Utilize the optimized dataloader for efficient sample selection

6. **Random Sample Visualization**
   - Visualize randomly selected samples from the dataset
   - Option to filter random selection by class
   - Creates both individual sample visualizations and an overview grid
   - Useful for dataset exploration and quality assessment

## Usage Examples

### Command Line Interface

The module includes a command-line interface for easy use:

```bash
# Visualize a specific sample (all slices)
python scripts/visualization/nuclei_visualizer.py --sample 1234

# Visualize a specific sample with limited slices
python scripts/visualization/nuclei_visualizer.py --sample 1234 --slices 5

# Visualize samples from a specific class
python scripts/visualization/nuclei_visualizer.py --class 2 --samples 3 --slices 5

# Visualize random samples
python scripts/visualization/nuclei_visualizer.py --random --samples 5 --slices 3

# Visualize random samples from a specific class
python scripts/visualization/nuclei_visualizer.py --random --filter-class 2 --samples 4

# Disable GIF creation
python scripts/visualization/nuclei_visualizer.py --sample 1234 --no-gif

# Disable mask overlay
python scripts/visualization/nuclei_visualizer.py --sample 1234 --no-mask
```

### Python API

The module can also be imported and used programmatically:

```python
from scripts.visualization.nuclei_visualizer import NucleiVisualizer

# Create visualizer instance
visualizer = NucleiVisualizer()

# Visualize a specific sample
result = visualizer.visualize_sample_slices(
    sample_id="1234",
    num_slices=10,
    create_gif=True,
    show_masks=True
)

# Visualize samples from a specific class
results = visualizer.visualize_class_samples(
    class_id=2,
    num_samples=3,
    slices_per_sample=5,
    create_gifs=True
)

# Visualize random samples
results = visualizer.visualize_random_samples(
    num_samples=5,
    slices_per_sample=3,
    filter_by_class=None,  # Optional: specify class ID to filter
    create_gifs=True,
    show_masks=True
)

# Access results
grid_path = result['grid']  # Path to grid visualization
gif_path = result['gif']    # Path to animated GIF
```

## Implementation Details

### Image Preparation

The visualizer intelligently handles different image formats:
- Converts PyTorch tensors to NumPy arrays
- Handles channel-first vs. channel-last formats
- Normalizes image ranges appropriately
- Manages dimensionality (squeezes single-channel images)

### GIF Creation Process

The GIF creation process follows these steps:
1. Create temporary frames for each slice
2. Save each frame to disk
3. Use imageio to create an animated GIF
4. Clean up temporary files

### Integration with Dataloader

The visualizer leverages the optimized dataloader to:
- Efficiently load samples from specific classes
- Maintain the slice ordering
- Access class information for display
- Filter samples by class or ID

### Random Sample Selection

The random sample selection process:
1. Reads the CSV file to get all available sample IDs
2. Optionally filters by class
3. Randomly selects the requested number of samples
4. Creates individual visualizations for each sample
5. Generates an overview grid with representative slices from each sample

## Output Types

1. **Single Image Files**
   - PNG format
   - Customizable DPI
   - Option to display immediately or just save

2. **Grid Visualizations**
   - PNG format
   - Automatic or custom layout
   - Title and annotation support

3. **Animated GIFs**
   - Configurable frame rate
   - Optional looping
   - Includes frame numbering
   - Mask overlay support

4. **Overview Visualizations**
   - Composite view of multiple samples
   - Representative slices from each sample
   - Class information included in titles

## Configuration

All outputs are saved to the `VISUALIZATION_OUTPUT_DIR` specified in `config.py` (default: `results/visualizations/`). 

Additional customization options include:
- Color map for mask overlays
- Transparency level for masks
- Output image resolution (DPI)
- Custom output directory 

## Improved Visualization Capabilities

The visualization system has been enhanced with several improvements:

1. **Direct Loading Method**: The system now includes a fallback mechanism that directly loads images when the dataloader fails. This ensures more reliable visualization even when there are issues with file naming or format.

2. **Better Error Handling**: Detailed error messages and diagnostics help identify issues with sample directories, file matching, and image loading.

3. **File Matching Diagnostics**: A dedicated tool for checking file matching between raw and mask directories helps diagnose issues with sample visualization.

4. **Robust GIF Creation**: The GIF creation process has been improved to handle various image formats and ensure all necessary directories exist.

### Using the File Matching Diagnostic Tool

To check if a sample has matching raw and mask files:

```bash
python scripts/visualization/nuclei_visualizer.py --check-sample 2150
```

This will provide detailed information about:
- Number of raw and mask files
- Number of matching files
- Files that exist in raw but not in mask directories (and vice versa)
- Potential filename format mismatches

### Troubleshooting Visualization Issues

If you encounter issues with visualization:

1. First, check if the sample directory exists:
   ```bash
   python scripts/visualization/nuclei_visualizer.py --check-sample <sample_id>
   ```

2. If files exist but don't match, check the file naming patterns in raw and mask directories.

3. For class-specific visualization issues, verify that the sample IDs in the CSV file match the directory names.

4. If GIF creation fails, try using the `--no-gif` option to skip GIF creation:
   ```bash
   python scripts/visualization/nuclei_visualizer.py --random --samples 2 --no-gif
   ``` 