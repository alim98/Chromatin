# Volume Resizing Logic

This document explains how 3D volumes are loaded, processed, and resized for neural network input in the chromatin dataset.

## Overview

The volume resizing functionality allows loading 3D volumes from directories of 2D slices and resizing them to a consistent target shape (default: 80×80×80) while preserving important structures marked by masks.

## How It Works

### 1. Volume Loading Process

Volumes are loaded from directories containing 2D slices:

```
sample_id/
├── raw/
│   ├── 001.tif
│   ├── 002.tif
│   └── ...
└── mask/
    ├── 001.tif
    ├── 002.tif
    └── ...
```

The `_load_volume` method:
- Finds all TIF files in the raw directory
- Loads the corresponding mask files
- Stacks them into 3D numpy arrays
- Returns volume and mask arrays with shape (depth, height, width)

### 2. Mask-Prioritized Resizing

The key innovation is **mask prioritization** during resizing, which ensures important structures are preserved:

1. When `mask_priority=True` and mask contains positive values:
   - Find the bounding box of positive mask values (where nuclei are marked)
   - Add padding around this region of interest (default: 5 pixels)
   - Crop the volume and mask to this bounded region
   - Resize the cropped volume to the target shape

2. When `mask_priority=False` or no mask regions are found:
   - Resize the entire volume directly to the target shape

This approach focuses the resizing on regions of interest, preserving small structures that might otherwise be lost during downsampling.

### 3. Resizing Implementation

The resizing uses `skimage.transform.resize` with these parameters:

```python
# For volume (raw image)
resized_volume = resize(volume, target_shape, 
                      order=1,             # Bilinear interpolation
                      mode='constant',     # Pad with zeros
                      anti_aliasing=True)  # Apply anti-aliasing

# For mask
resized_mask = resize(mask, target_shape,
                    order=0,             # Nearest neighbor interpolation
                    mode='constant',     # Pad with zeros 
                    anti_aliasing=False) # No anti-aliasing for masks
```

- Different interpolation orders are used for volumes vs. masks
  - `order=1` for volumes (bilinear) preserves gradients
  - `order=0` for masks (nearest-neighbor) preserves binary/categorical values

## Output Format

The dataloader with `load_volumes=True` outputs a dictionary for each sample:

```python
{
    'original_volume': tensor,   # Shape: (1, original_depth, original_height, original_width)
    'original_mask': tensor,     # Shape: (1, original_depth, original_height, original_width)
    'volume': tensor,            # Shape: (1, 80, 80, 80) - the resized volume
    'mask': tensor,              # Shape: (1, 80, 80, 80) - the resized mask
    'label': int,                # Class ID of the sample
    'metadata': {
        'sample_id': str,
        'original_shape': tuple, # Original shape (depth, height, width)
        'target_shape': tuple,   # Target shape (80, 80, 80)
        'class_id': int,         # If class information is available
        'class_name': str        # If class information is available
    }
}
```

When used in a dataloader with a batch size of N, the batch would have a shape of `(N, 1, 80, 80, 80)`.

## Using Resized Volumes with Models

To use the resized volumes with a 3D CNN model:

```python
dataloader = get_nuclei_dataloader(
    root_dir='path/to/nuclei',
    batch_size=8,
    load_volumes=True,          # Enable 3D volume loading
    target_shape=(80, 80, 80),  # Set target shape
    mask_priority=True          # Enable mask prioritization
)

# Get a batch
batch = next(iter(dataloader))

# Extract inputs and labels
inputs = batch['volume']   # Shape: (batch_size, 1, 80, 80, 80)
labels = batch['label']    # Shape: (batch_size)

# Forward pass
outputs = model(inputs)
```

## Visualization

The dataloader includes a `visualize_volume` method to create GIFs comparing original and resized volumes:

```python
dataset = NucleiDataset(
    root_dir='path/to/nuclei',
    load_volumes=True,
    target_shape=(80, 80, 80),
    mask_priority=True
)

# Visualize the first sample
original_gif, resized_gif = dataset.visualize_volume(
    idx=0,
    output_dir='results/visualizations',
    max_frames=80
)
```

## Best Practices

1. **Choose an appropriate target shape** based on your model's input requirements and memory constraints.

2. **Enable mask prioritization** when structures of interest are marked by masks, especially if they occupy a small portion of the volume.

3. **Customize padding** in the bounding box calculation based on the size of structures and desired context.

4. **Use appropriate batch sizes** - 3D volumes require more memory than 2D images.

5. **Verify results with visualization** to ensure important structures are preserved after resizing. 