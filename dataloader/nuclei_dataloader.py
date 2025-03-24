import os
import glob
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import resize
from tqdm import tqdm
import sys
import gc
import time

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config


def load_chromatin_classes(csv_path, ignore_unclassified=True):
    """
    Load chromatin class information from CSV file.
    
    Args:
        csv_path (str): Path to the CSV file containing chromatin class information.
        ignore_unclassified (bool): Whether to ignore entries with class_name "Unclassified".
        
    Returns:
        dict: Dictionary mapping sample_id to class information (id and name).
    """
    if not os.path.exists(csv_path):
        print(f"Warning: Chromatin class CSV file not found: {csv_path}")
        return {}
    
    df = pd.read_csv(csv_path)
    
    # Create a dictionary mapping sample_id to class information
    sample_to_class = {}
    for _, row in df.iterrows():
        # Skip unclassified samples if ignore_unclassified is True
        if ignore_unclassified and (row['class_name'] == 'Unclassified' or row['class_id'] == 19):
            continue
            
        sample_to_class[str(row['sample_id'])] = {
            'class_id': row['class_id'],
            'class_name': row['class_name']
        }
    
    return sample_to_class


class NucleiDataset(Dataset):
    """
    Dataset class for loading nuclei images and their corresponding masks.
    """
    def __init__(self, 
                 root_dir,
                 transform=None,
                 mask_transform=None,
                 sample_ids=None,
                 slice_range=None,
                 return_paths=False,
                 class_csv_path=None,
                 filter_by_class=None,
                 ignore_unclassified=True,
                 load_volumes=False,
                 target_shape=(80, 80, 80),
                 mask_priority=True,
                 preserve_resolution=False,
                 high_quality_resize=True,
                 debug=False):
        """
        Args:
            root_dir (str): Root directory of the nuclei dataset.
            transform (callable, optional): Transform to be applied on the raw images.
            mask_transform (callable, optional): Transform to be applied on the mask images.
            sample_ids (list, optional): List of sample IDs to include. If None, includes all samples.
            slice_range (tuple, optional): Range of slice numbers to include, e.g., (1, 50).
            return_paths (bool): Whether to return the file paths along with the data.
            class_csv_path (str, optional): Path to CSV file containing chromatin class information.
            filter_by_class (int or list, optional): Class ID or list of class IDs to include.
            ignore_unclassified (bool): Whether to ignore unclassified samples.
            load_volumes (bool): Whether to load 3D volumes instead of 2D slices.
            target_shape (tuple): Target shape for volume resizing (z, y, x).
            mask_priority (bool): Whether to prioritize regions with masks during resizing.
            preserve_resolution (bool): Whether to preserve original resolution using crop/pad instead of scaling.
            high_quality_resize (bool): Whether to use high-quality resize methods instead of basic ones.
            debug (bool): Whether to print debug statements during processing.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_paths = return_paths
        self.load_volumes = load_volumes
        self.target_shape = target_shape
        self.mask_priority = mask_priority
        self.preserve_resolution = preserve_resolution
        self.high_quality_resize = high_quality_resize
        self.debug = debug
        
        # First, determine which samples to include based on CSV (if provided)
        filtered_sample_ids = None
        self.sample_to_class = {}
        
        if class_csv_path and os.path.exists(class_csv_path):
            # Read CSV file directly
            df = pd.read_csv(class_csv_path)
            
            # Filter by class_id if needed
            if filter_by_class is not None:
                if isinstance(filter_by_class, int):
                    filter_by_class = [filter_by_class]
                
                df = df[df['class_id'].isin(filter_by_class)]
            
            # Filter out unclassified samples if needed
            if ignore_unclassified:
                df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
            
            # Convert to list of sample_ids
            filtered_sample_ids = [str(sid) for sid in df['sample_id'].unique()]
            
            # Create the sample_to_class mapping
            for _, row in df.iterrows():
                self.sample_to_class[str(row['sample_id'])] = {
                    'class_id': row['class_id'],
                    'class_name': row['class_name']
                }
        
        # Merge with user-provided sample_ids if needed
        if sample_ids is not None:
            sample_ids = [str(sid) for sid in sample_ids]
            if filtered_sample_ids is not None:
                # Intersection of filtered_sample_ids and sample_ids
                filtered_sample_ids = [sid for sid in filtered_sample_ids if sid in sample_ids]
            else:
                filtered_sample_ids = sample_ids
        
        # Now get only the necessary directories (either filtered or all)
        if filtered_sample_ids is not None:
            # Only get directories that match the filtered sample IDs
            # This avoids scanning the entire root_dir
            sample_dirs = [sid for sid in filtered_sample_ids if os.path.isdir(os.path.join(root_dir, sid))]
        else:
            # If no filtering, get all directories (original behavior)
            sample_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Different approach for volume loading vs. slice loading
        if load_volumes:
            # For volume loading, we just need the sample directories
            self.samples = []
            for sample_dir in sample_dirs:
                sample_path = os.path.join(root_dir, sample_dir)
                raw_dir = os.path.join(sample_path, 'raw')
                mask_dir = os.path.join(sample_path, 'mask')
                
                # Skip if either directory doesn't exist
                if not (os.path.exists(raw_dir) and os.path.exists(mask_dir)):
                    continue
                
                self.samples.append(sample_dir)
        else:    
            # For slice loading, we need all image pairs
            self.samples = []
            
            # Collect all image pairs
            for sample_dir in sample_dirs:
                sample_path = os.path.join(root_dir, sample_dir)
                raw_dir = os.path.join(sample_path, 'raw')
                mask_dir = os.path.join(sample_path, 'mask')
                
                # Skip if either directory doesn't exist
                if not (os.path.exists(raw_dir) and os.path.exists(mask_dir)):
                    continue
                    
                raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
                
                for raw_file in raw_files:
                    file_name = os.path.basename(raw_file)
                    mask_file = os.path.join(mask_dir, file_name)
                    
                    # Skip if mask doesn't exist
                    if not os.path.exists(mask_file):
                        continue
                        
                    # Extract slice number and filter by slice_range if specified
                    slice_num = int(os.path.splitext(file_name)[0])
                    if slice_range is not None:
                        if slice_num < slice_range[0] or slice_num > slice_range[1]:
                            continue
                    
                    self.samples.append((raw_file, mask_file, sample_dir, slice_num))
    
    def __len__(self):
        return len(self.samples)
    
    def _load_volume(self, sample_id):
        """
        Load a 3D volume and its mask from 2D slices.
        
        Args:
            sample_id (str): Sample ID to load.
            
        Returns:
            tuple: (volume, mask) as numpy arrays, or (None, None) if loading fails.
        """
        sample_path = os.path.join(self.root_dir, sample_id)
        raw_dir = os.path.join(sample_path, 'raw')
        mask_dir = os.path.join(sample_path, 'mask')
        
        # Find all image files
        raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
        
        if not raw_files:
            print(f"No image files found for sample {sample_id}")
            return None, None
        
        # Load the first image to get dimensions
        first_img = np.array(Image.open(raw_files[0]))
        height, width = first_img.shape
        depth = len(raw_files)
        
        # Initialize volume arrays
        volume = np.zeros((depth, height, width), dtype=np.float32)
        mask_volume = np.zeros((depth, height, width), dtype=np.float32)
        
        # Load each slice
        for i, raw_file in enumerate(raw_files):
            # Get corresponding mask file
            file_name = os.path.basename(raw_file)
            mask_file = os.path.join(mask_dir, file_name)
            
            # Skip if mask doesn't exist
            if not os.path.exists(mask_file):
                continue
            
            # Load raw and mask images
            raw_img = np.array(Image.open(raw_file))
            mask_img = np.array(Image.open(mask_file))
            
            # Add to volume
            volume[i] = raw_img
            mask_volume[i] = mask_img
        
        return volume, mask_volume
    
    def _center_crop_volume(self, volume, mask=None):
        """
        Extract a center crop from the volume at original resolution.
        
        This is a more direct approach than _resize_preserve_resolution, ensuring 
        no scaling whatsoever, just a direct center crop.
        
        Args:
            volume (ndarray): 3D volume to crop
            mask (ndarray, optional): Corresponding mask
            
        Returns:
            tuple: (cropped_volume, cropped_mask)
        """
        if self.debug:
            print("\n=== CENTER_CROP_VOLUME DEBUG ===")
            print(f"Original volume shape: {volume.shape}")
            print(f"Target shape: {self.target_shape}")
        
        original_shape = volume.shape
        target_shape = self.target_shape
        
        # Find the center point (either mask center of mass or volume center)
        if mask is not None and self.mask_priority and np.any(mask > 0):
            # Use center of mass of the mask
            from scipy import ndimage
            center_of_mass = ndimage.center_of_mass(mask > 0)
            center_z, center_y, center_x = [int(c) for c in center_of_mass]
            if self.debug:
                print(f"Using mask center of mass: ({center_z}, {center_y}, {center_x})")
        else:
            # Use geometric center of the volume
            center_z = original_shape[0] // 2
            center_y = original_shape[1] // 2
            center_x = original_shape[2] // 2
            if self.debug:
                print(f"Using volume geometric center: ({center_z}, {center_y}, {center_x})")
        
        # Calculate crop boundaries
        # Ensure we don't go out of bounds, and crop equal amounts on both sides of center
        # If the target shape is larger than the original in any dimension, we'll just take the whole original
        crop_z_size = min(target_shape[0], original_shape[0])
        crop_y_size = min(target_shape[1], original_shape[1])
        crop_x_size = min(target_shape[2], original_shape[2])
        
        # Calculate the starting indices for the crop
        z_start = max(0, center_z - crop_z_size // 2)
        y_start = max(0, center_y - crop_y_size // 2)
        x_start = max(0, center_x - crop_x_size // 2)
        
        # Adjust if we're too close to the end
        if z_start + crop_z_size > original_shape[0]:
            z_start = original_shape[0] - crop_z_size
        if y_start + crop_y_size > original_shape[1]:
            y_start = original_shape[1] - crop_y_size
        if x_start + crop_x_size > original_shape[2]:
            x_start = original_shape[2] - crop_x_size
        
        # Ensure indices are valid
        z_start = max(0, z_start)
        y_start = max(0, y_start)
        x_start = max(0, x_start)
        
        # Calculate end indices
        z_end = min(original_shape[0], z_start + crop_z_size)
        y_end = min(original_shape[1], y_start + crop_y_size)
        x_end = min(original_shape[2], x_start + crop_x_size)
        
        if self.debug:
            print(f"Crop region: z[{z_start}:{z_end}], y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
            print(f"Crop dimensions: {z_end-z_start}×{y_end-y_start}×{x_end-x_start}")
        
        # Extract the center crop
        cropped_volume = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        cropped_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end] if mask is not None else None
        
        # Create output arrays of target shape (may need padding)
        output_volume = np.zeros(target_shape, dtype=volume.dtype)
        output_mask = np.zeros(target_shape, dtype=mask.dtype) if mask is not None else None
        
        # Place the cropped volume in the center of the output volume
        crop_shape = cropped_volume.shape
        
        # Calculate padding offsets
        z_offset = (target_shape[0] - crop_shape[0]) // 2
        y_offset = (target_shape[1] - crop_shape[1]) // 2
        x_offset = (target_shape[2] - crop_shape[2]) // 2
        
        if self.debug:
            print(f"Cropped shape: {crop_shape}")
            print(f"Padding offsets: ({z_offset}, {y_offset}, {x_offset})")
        
        # Insert cropped volume into output volume
        output_volume[
            z_offset:z_offset + crop_shape[0],
            y_offset:y_offset + crop_shape[1],
            x_offset:x_offset + crop_shape[2]
        ] = cropped_volume
        
        # Insert cropped mask into output mask
        if cropped_mask is not None:
            output_mask[
                z_offset:z_offset + crop_shape[0],
                y_offset:y_offset + crop_shape[1],
                x_offset:x_offset + crop_shape[2]
            ] = cropped_mask
        
        if self.debug:
            print(f"Final output shape: {output_volume.shape}")
            print("=== END CENTER_CROP_VOLUME DEBUG ===\n")
        
        return output_volume, output_mask

    def _resize_multi_resolution(self, volume, mask=None):
        """
        Resize a volume while preserving maximum resolution by maintaining 
        multiple resolution levels.
        
        Instead of just cropping or downsampling, this method:
        1. Stores the full volume at a lower resolution
        2. Stores a high-resolution crop of the most important region
        
        Args:
            volume (ndarray): 3D volume to resize
            mask (ndarray, optional): Corresponding mask
            
        Returns:
            tuple: (resized_volume_dict, resized_mask_dict) where each is a dictionary 
                   containing 'full_res' and 'region_of_interest' entries
        """
        if self.debug:
            print("\n=== RESIZE_MULTI_RESOLUTION DEBUG ===")
            print(f"Original volume shape: {volume.shape}")
            print(f"Target shape: {self.target_shape}")
        
        original_shape = volume.shape
        target_shape = self.target_shape
        
        # --- Full resolution downsampled version ---
        # Calculate the scaling factors for each dimension
        scale_factors = [t/o for t, o in zip(target_shape, original_shape)]
        if self.debug:
            print(f"Scaling factors: {scale_factors}")
        
        # Resize the full volume to target shape (lower resolution but complete coverage)
        from skimage.transform import resize
        full_volume = resize(volume, target_shape, order=1, mode='constant', anti_aliasing=True)
        
        if mask is not None:
            full_mask = resize(mask, target_shape, order=0, mode='constant', anti_aliasing=False)
        else:
            full_mask = None
        
        # --- High resolution region of interest ---
        # Find the center point (either mask center of mass or volume center)
        if mask is not None and self.mask_priority and np.any(mask > 0):
            # Use center of mass of the mask
            from scipy import ndimage
            center_of_mass = ndimage.center_of_mass(mask > 0)
            center_z, center_y, center_x = [int(c) for c in center_of_mass]
            if self.debug:
                print(f"Using mask center of mass: ({center_z}, {center_y}, {center_x})")
        else:
            # Use geometric center of the volume
            center_z = original_shape[0] // 2
            center_y = original_shape[1] // 2
            center_x = original_shape[2] // 2
            if self.debug:
                print(f"Using volume geometric center: ({center_z}, {center_y}, {center_x})")
        
        # Calculate roi size based on the smallest dimension of target_shape
        roi_size = min(target_shape) // 2
        
        # Calculate ROI boundaries around the center
        z_start = max(0, center_z - roi_size)
        y_start = max(0, center_y - roi_size)
        x_start = max(0, center_x - roi_size)
        
        z_end = min(original_shape[0], z_start + roi_size * 2)
        y_end = min(original_shape[1], y_start + roi_size * 2)
        x_end = min(original_shape[2], x_start + roi_size * 2)
        
        # Adjust start if end was limited
        z_start = max(0, z_end - roi_size * 2)
        y_start = max(0, y_end - roi_size * 2)
        x_start = max(0, x_end - roi_size * 2)
        
        if self.debug:
            roi_shape = (z_end-z_start, y_end-y_start, x_end-x_start)
            print(f"ROI boundaries: z[{z_start}:{z_end}], y[{y_start}:{y_end}], x[{x_start}:{x_end}]")
            print(f"ROI shape: {roi_shape}")
        
        # Extract ROI
        roi_volume = volume[z_start:z_end, y_start:y_end, x_start:x_end]
        roi_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end] if mask is not None else None
        
        # Package results into a dictionary structure
        result_volume = {
            'full': full_volume,  
            'roi': roi_volume,
            'roi_coords': (z_start, z_end, y_start, y_end, x_start, x_end),
            'original_shape': original_shape,
            'scale_factors': scale_factors
        }
        
        if mask is not None:
            result_mask = {
                'full': full_mask,
                'roi': roi_mask
            }
        else:
            result_mask = None
        
        if self.debug:
            print(f"Full volume shape: {full_volume.shape}")
            if roi_volume is not None:
                print(f"ROI volume shape: {roi_volume.shape}")
            print("=== END RESIZE_MULTI_RESOLUTION DEBUG ===\n")
        
        return result_volume, result_mask

    def _resize_preserve_resolution(self, volume, mask=None):
        """
        Resize volume while preserving original resolution using cropping and padding.
        
        This approach maintains the original resolution by:
        1. Finding the center of mass of the mask (or center of volume if no mask)
        2. Cropping a region around this center that fits within the target shape
        3. Padding with zeros if the volume is smaller than the target shape
        
        Args:
            volume (ndarray): 3D volume to resize
            mask (ndarray, optional): Corresponding mask
            
        Returns:
            tuple: (resized_volume, resized_mask)
        """
        if self.debug:
            print("\n=== RESIZE_PRESERVE_RESOLUTION DEBUG ===")
            print("Using improved resolution preservation method")
        
        # Create a uniform-quality downsample that covers the full volume
        # This is better than just cropping, as it preserves all the data
        from skimage.transform import resize
        
        # Calculate scaling factors
        scale_factors = [t/o for t, o in zip(self.target_shape, volume.shape)]
        min_scale = min(scale_factors)
        
        if self.debug:
            print(f"Original volume shape: {volume.shape}")
            print(f"Target shape: {self.target_shape}")
            print(f"Scale factors: {scale_factors}")
            print(f"Minimum scale factor: {min_scale}")
        
        # If we need to downsample, use a higher quality method
        if min_scale < 1.0:
            # Use a higher order interpolation for better quality
            resized_volume = resize(volume, self.target_shape, 
                                   order=3, mode='constant', 
                                   anti_aliasing=True, preserve_range=True)
            
            if mask is not None:
                # For mask, use nearest neighbor to preserve binary values
                resized_mask = resize(mask, self.target_shape,
                                     order=0, mode='constant',
                                     anti_aliasing=False, preserve_range=True)
            else:
                resized_mask = None
        else:
            # If we're actually upsampling or keeping the same size, use center crop/pad
            return self._center_crop_volume(volume, mask)
        
        if self.debug:
            print(f"Resized volume shape: {resized_volume.shape}")
            print("=== END RESIZE_PRESERVE_RESOLUTION DEBUG ===\n")
            
        return resized_volume, resized_mask
    
    def _resize_maintain_aspect_ratio(self, volume, mask=None):
        """
        Resize a volume while maintaining its original aspect ratio.
        
        This approach:
        1. Calculates a target shape that preserves the original aspect ratio
        2. Resizes the volume using high-quality interpolation
        3. Adds padding with edge values rather than zeros if necessary
        
        Args:
            volume (ndarray): 3D volume to resize
            mask (ndarray, optional): Corresponding mask
            
        Returns:
            tuple: (resized_volume, resized_mask)
        """
        if self.debug:
            print("\n=== RESIZE_MAINTAIN_ASPECT_RATIO DEBUG ===")
            print(f"Original volume shape: {volume.shape}")
            print(f"Target cube shape: {self.target_shape}")
        
        original_shape = volume.shape
        target_size = self.target_shape[0]  # Use the first dimension as reference
        
        # Calculate new shape maintaining aspect ratio
        scaling_factor = target_size / max(original_shape)
        aspect_shape = tuple(int(dim * scaling_factor) for dim in original_shape)
        
        # Ensure minimum size of 1 for each dimension
        aspect_shape = tuple(max(1, dim) for dim in aspect_shape)
        
        if self.debug:
            print(f"Scaling factor: {scaling_factor}")
            print(f"Aspect-preserved shape: {aspect_shape}")
        
        # Resize with high quality interpolation
        from skimage.transform import resize
        
        # Find the min and max values for better padding later
        min_val = volume.min()
        
        # Resize while maintaining aspect ratio
        resized_volume = resize(volume, aspect_shape, 
                            order=3, mode='edge',  # Use edge padding mode
                            anti_aliasing=True, preserve_range=True)
        
        if mask is not None:
            resized_mask = resize(mask, aspect_shape,
                              order=0, mode='constant',
                              anti_aliasing=False, preserve_range=True)
        else:
            resized_mask = None
        
        # Create output arrays of target shape
        output_shape = self.target_shape
        padded_volume = np.full(output_shape, min_val, dtype=volume.dtype)  # Use min value instead of zeros
        padded_mask = np.zeros(output_shape, dtype=mask.dtype) if mask is not None else None
        
        # Calculate padding offsets
        z_offset = (output_shape[0] - aspect_shape[0]) // 2
        y_offset = (output_shape[1] - aspect_shape[1]) // 2
        x_offset = (output_shape[2] - aspect_shape[2]) // 2
        
        # Place the resized volume in the center of the output volume
        padded_volume[
            z_offset:z_offset + aspect_shape[0],
            y_offset:y_offset + aspect_shape[1],
            x_offset:x_offset + aspect_shape[2]
        ] = resized_volume
        
        # Place the resized mask in the center of the output mask if it exists
        if padded_mask is not None and resized_mask is not None:
            padded_mask[
                z_offset:z_offset + aspect_shape[0],
                y_offset:y_offset + aspect_shape[1],
                x_offset:x_offset + aspect_shape[2]
            ] = resized_mask
        
        if self.debug:
            print(f"Final output shape: {padded_volume.shape}")
            print("=== END RESIZE_MAINTAIN_ASPECT_RATIO DEBUG ===\n")
        
        return padded_volume, padded_mask

    def _resize_volume(self, volume, mask=None):
        """
        Resize a 3D volume to the target shape, with optional mask prioritization.
        
        Args:
            volume (ndarray): 3D volume to resize
            mask (ndarray, optional): Corresponding mask
            
        Returns:
            tuple: (resized_volume, resized_mask)
        """
        if self.debug:
            print("\n=== RESIZE_VOLUME DEBUG ===")
            print(f"preserve_resolution flag: {self.preserve_resolution}")
            print(f"high_quality_resize flag: {self.high_quality_resize}")
            print(f"Original volume shape: {volume.shape}")
            print(f"Target shape: {self.target_shape}")
            
        # Use the aspect ratio preserving method which works better for thin slice issues
        if self.high_quality_resize:
            if self.debug:
                print("Using aspect-ratio preserving high quality resize")
                print("=== END RESIZE_VOLUME DEBUG ===\n")
            return self._resize_maintain_aspect_ratio(volume, mask)
        
        # Otherwise use the original implementation
        if self.preserve_resolution:
            if self.debug:
                print("Using PRESERVE_RESOLUTION path (crop and pad)")
                print("=== END RESIZE_VOLUME DEBUG ===\n")
            return self._resize_preserve_resolution(volume, mask)
        
        # Otherwise, use the original scaling method
        if self.debug:
            print("Using SCALING path")
            
        from skimage.transform import resize
        
        # If mask is provided and has positive values, prioritize that region
        if mask is not None and self.mask_priority and np.any(mask > 0):
            if self.debug:
                print("Using mask prioritization")
                
            # Find bounding box of mask
            z_indices, y_indices, x_indices = np.where(mask > 0)
            
            # Get min and max indices with padding
            padding = 5  # Add some padding around the masked region
            z_min, z_max = max(0, np.min(z_indices) - padding), min(volume.shape[0], np.max(z_indices) + padding)
            y_min, y_max = max(0, np.min(y_indices) - padding), min(volume.shape[1], np.max(y_indices) + padding)
            x_min, x_max = max(0, np.min(x_indices) - padding), min(volume.shape[2], np.max(x_indices) + padding)
            
            if self.debug:
                print(f"Crop region: z[{z_min}:{z_max}], y[{y_min}:{y_max}], x[{x_min}:{x_max}]")
                
            # Crop volume and mask to bounding box
            cropped_volume = volume[z_min:z_max, y_min:y_max, x_min:x_max]
            cropped_mask = mask[z_min:z_max, y_min:y_max, x_min:x_max]
            
            if self.debug:
                print(f"Cropped shape: {cropped_volume.shape}")
                print(f"Resizing to: {self.target_shape}")
                
            # Resize the cropped volume
            resized_volume = resize(cropped_volume, self.target_shape,
                                   order=1, mode='constant', anti_aliasing=True)
            
            # Resize the cropped mask (use order=0 to preserve binary values)
            resized_mask = resize(cropped_mask, self.target_shape,
                                 order=0, mode='constant', anti_aliasing=False)
        
        else:
            if self.debug:
                print("Resizing entire volume (no mask prioritization)")
                
            # Resize the entire volume
            resized_volume = resize(volume, self.target_shape,
                                  order=1, mode='constant', anti_aliasing=True)
            
            if mask is not None:
                resized_mask = resize(mask, self.target_shape,
                                    order=0, mode='constant', anti_aliasing=False)
            else:
                resized_mask = None
        
        if self.debug:
            print(f"Resized volume shape: {resized_volume.shape}")
            if resized_mask is not None:
                print(f"Resized mask shape: {resized_mask.shape}")
            print("=== END RESIZE_VOLUME DEBUG ===\n")
                
        return resized_volume, resized_mask
    
    def __getitem__(self, idx):
        if self.load_volumes:
            # Volume mode - load and process 3D volume
            sample_id = self.samples[idx]
            
            # Load volume
            volume, mask = self._load_volume(sample_id)
            
            # If loading failed, return a placeholder or raise an error
            if volume is None or mask is None:
                raise ValueError(f"Failed to load volume for sample {sample_id}")
            
            # Get original shape
            original_shape = volume.shape
            
            # Resize volume if necessary
            if original_shape != self.target_shape:
                resized_volume, resized_mask = self._resize_volume(volume, mask)
            else:
                resized_volume, resized_mask = volume, mask
            
            # Convert to torch tensors
            if self.transform:
                volume_tensor = self.transform(volume)
                resized_volume_tensor = self.transform(resized_volume)
            else:
                # Default conversion to tensor
                volume_tensor = torch.from_numpy(volume).float().unsqueeze(0)  # Add channel dimension
                resized_volume_tensor = torch.from_numpy(resized_volume).float().unsqueeze(0)
            
            if mask is not None:
                if self.mask_transform:
                    mask_tensor = self.mask_transform(mask)
                    resized_mask_tensor = self.mask_transform(resized_mask)
                else:
                    # Default conversion to tensor
                    mask_tensor = torch.from_numpy(mask).float().unsqueeze(0)
                    resized_mask_tensor = torch.from_numpy(resized_mask).float().unsqueeze(0)
            else:
                mask_tensor = None
                resized_mask_tensor = None
            
            # Initialize label with default value (-1 indicating no label)
            label = -1
            
            # Create metadata dictionary
            metadata = {
                'sample_id': sample_id,
                'original_shape': original_shape,
                'target_shape': self.target_shape
            }
            
            # Add class information if available
            if sample_id in self.sample_to_class:
                class_id = self.sample_to_class[sample_id]['class_id']
                class_name = self.sample_to_class[sample_id]['class_name']
                
                # Add class info to metadata
                metadata['class_id'] = class_id
                metadata['class_name'] = class_name
                
                # Set the label to the class_id
                label = class_id
            
            # Include paths if requested
            if self.return_paths:
                sample_path = os.path.join(self.root_dir, sample_id)
                metadata['sample_path'] = sample_path
            
            return {
                'original_volume': volume_tensor,
                'original_mask': mask_tensor,
                'volume': resized_volume_tensor,
                'mask': resized_mask_tensor,
                'label': label,
                'metadata': metadata
            }
        else:
            # Slice mode (original behavior)
            raw_path, mask_path, sample_id, slice_num = self.samples[idx]
            
            # Load raw image and mask
            raw_img = Image.open(raw_path)
            mask_img = Image.open(mask_path)
            
            # Apply transforms if specified
            if self.transform:
                raw_img = self.transform(raw_img)
            else:
                # Default transform to convert to tensor
                raw_img = transforms.ToTensor()(raw_img)
                
            if self.mask_transform:
                mask_img = self.mask_transform(mask_img)
            else:
                # Default transform to convert to tensor
                mask_img = transforms.ToTensor()(mask_img)
            
            # Create metadata dictionary
            metadata = {
                'sample_id': sample_id,
                'slice_num': slice_num
            }
            
            # Initialize label with default value (-1 indicating no label)
            label = -1
            
            # Add class information if available
            if sample_id in self.sample_to_class:
                class_id = self.sample_to_class[sample_id]['class_id']
                class_name = self.sample_to_class[sample_id]['class_name']
                
                # Add class info to metadata
                metadata['class_id'] = class_id
                metadata['class_name'] = class_name
                
                # Set the label to the class_id
                label = class_id
            
            if self.return_paths:
                metadata['raw_path'] = raw_path
                metadata['mask_path'] = mask_path
                
            return {
                'image': raw_img,
                'mask': mask_img,
                'label': label,  # Add class_id as a top-level key
                'metadata': metadata
            }
    
    def visualize_volume(self, idx, output_dir='results/visualizations/volume_comparisons', max_frames=80):
        """
        Create GIFs to visualize original and resized volumes for a specific sample.
        
        Args:
            idx (int): Index of the sample in the dataset
            output_dir (str): Directory to save visualization files
            max_frames (int): Maximum number of frames to include in GIFs
            
        Returns:
            tuple: (original_gif_path, resized_gif_path) Paths to the created GIF files
        """
        # Import visualization module for GIF creation
        from scripts.visualization.nuclei_visualizer import VolumeGifCreator
        
        # Start timing
        start_time = time.time()
        
        print(f"\n=== Volume Visualization Process Started ===")
        print(f"[Step 1/7] Loading sample data...")
        
        # Get the sample data
        if not self.load_volumes:
            raise ValueError("visualize_volume can only be used when load_volumes=True")
        
        # Get the sample data
        sample = self.__getitem__(idx)
        sample_id = sample['metadata']['sample_id']
        
        print(f"Sample ID: {sample_id}")
        if 'class_name' in sample['metadata']:
            print(f"Class: {sample['metadata']['class_name']} (ID: {sample['metadata']['class_id']})")
        
        # Get original and target shapes
        original_shape = sample['metadata']['original_shape']
        target_shape = sample['metadata']['target_shape']
        print(f"Original shape: {original_shape} → Resized shape: {target_shape}")
        print(f"Resize method: {'Preserve Resolution (crop/pad)' if self.preserve_resolution else 'Scaling'}")
        
        print(f"[Step 2/7] Creating directories...")
        # Create subdirectories for original and resized volumes
        original_dir = os.path.join(output_dir, 'original')
        resized_dir = os.path.join(output_dir, 'resized')
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(resized_dir, exist_ok=True)
        
        # Get original volume and mask
        original_volume = sample['original_volume'].numpy()[0]  # Remove channel dimension
        original_mask = sample['original_mask'].numpy()[0] if sample['original_mask'] is not None else None
        
        # Get resized volume and mask
        resized_volume = sample['volume'].numpy()[0]  # Remove channel dimension
        resized_mask = sample['mask'].numpy()[0] if sample['mask'] is not None else None
        
        # Create temporary directories to store the slice images
        original_slices_dir = os.path.join(output_dir, f'temp_{sample_id}_original')
        resized_slices_dir = os.path.join(output_dir, f'temp_{sample_id}_resized')
        
        print(f"Creating temporary directories:")
        for dir_path in [
            original_slices_dir, 
            os.path.join(original_slices_dir, 'raw'),
            resized_slices_dir, 
            os.path.join(resized_slices_dir, 'raw')
        ]:
            os.makedirs(dir_path, exist_ok=True)
            print(f"  - {dir_path}")
        
        if original_mask is not None:
            os.makedirs(os.path.join(original_slices_dir, 'mask'), exist_ok=True)
            print(f"  - {os.path.join(original_slices_dir, 'mask')}")
        
        if resized_mask is not None:
            os.makedirs(os.path.join(resized_slices_dir, 'mask'), exist_ok=True)
            print(f"  - {os.path.join(resized_slices_dir, 'mask')}")
        
        # Memory usage tracking
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # Convert to MB
            print(f"Memory usage before processing: {memory_before:.1f} MB")
        except ImportError:
            print("psutil not available, memory tracking disabled")
            memory_before = 0
        
        try:
            # Processing original volume
            print(f"\n[Step 3/7] Processing original volume ({original_volume.shape[0]} slices)...")
            
            # Save original volume slices as images
            for i in tqdm(range(original_volume.shape[0]), desc="Saving original slices", unit="slice"):
                # Normalize slice for visualization (0-255)
                slice_img = ((original_volume[i] - original_volume.min()) / 
                             (original_volume.max() - original_volume.min() + 1e-8) * 255).astype(np.uint8)
                
                # Save raw slice
                slice_path = os.path.join(original_slices_dir, 'raw', f'{i:04d}.tif')
                Image.fromarray(slice_img).save(slice_path)
                
                # Save mask slice if available
                if original_mask is not None:
                    mask_img = (original_mask[i] * 255).astype(np.uint8)
                    mask_path = os.path.join(original_slices_dir, 'mask', f'{i:04d}.tif')
                    Image.fromarray(mask_img).save(mask_path)
                
                # Call garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
            
            # Processing resized volume
            print(f"\n[Step 4/7] Processing resized volume ({resized_volume.shape[0]} slices)...")
            
            # Save resized volume slices as images
            for i in tqdm(range(resized_volume.shape[0]), desc="Saving resized slices", unit="slice"):
                # Normalize slice for visualization (0-255)
                slice_img = ((resized_volume[i] - resized_volume.min()) / 
                             (resized_volume.max() - resized_volume.min() + 1e-8) * 255).astype(np.uint8)
                
                # Save raw slice
                slice_path = os.path.join(resized_slices_dir, 'raw', f'{i:04d}.tif')
                Image.fromarray(slice_img).save(slice_path)
                
                # Save mask slice if available
                if resized_mask is not None:
                    mask_img = (resized_mask[i] * 255).astype(np.uint8)
                    mask_path = os.path.join(resized_slices_dir, 'mask', f'{i:04d}.tif')
                    Image.fromarray(mask_img).save(mask_path)
                
                # Call garbage collection periodically
                if i % 20 == 0:
                    gc.collect()
            
            # Memory usage after processing slices
            try:
                if memory_before > 0:
                    memory_after_slices = process.memory_info().rss / 1024 / 1024
                    print(f"Memory usage after processing slices: {memory_after_slices:.1f} MB " +
                          f"(Δ: {memory_after_slices - memory_before:.1f} MB)")
            except:
                pass
            
            # Create GIFs using VolumeGifCreator
            print(f"\n[Step 5/7] Creating GIFs...")
            
            print(f"Initializing GIF creator with batch_size={5}, max_frames={max_frames}")
            gif_creator = VolumeGifCreator(
                output_dir=output_dir,
                batch_size=5,
                max_frames=max_frames,
                dpi=80,
                figsize=(6, 6)
            )
            
            # Create GIF of original volume
            print(f"\n[Step 6/7] Creating GIF for original volume...")
            original_output_filename = f"{sample_id}_original.gif"
            original_gif = gif_creator.create_gif_from_folder(
                folder_path=original_slices_dir,
                output_filename=original_output_filename,
                duration=150,
                include_masks=(original_mask is not None)
            )
            
            # Check if the original GIF was created successfully
            if original_gif:
                print(f"✓ Original GIF created: {os.path.basename(original_gif)}")
                if os.path.exists(original_gif):
                    file_size = os.path.getsize(original_gif) / 1024  # KB
                    print(f"  Size: {file_size:.1f} KB")
            else:
                print(f"✗ Failed to create original GIF")
            
            # Create GIF of resized volume
            print(f"\n[Step 7/7] Creating GIF for resized volume...")
            resized_output_filename = f"{sample_id}_resized.gif"
            resized_gif = gif_creator.create_gif_from_folder(
                folder_path=resized_slices_dir,
                output_filename=resized_output_filename,
                duration=150,
                include_masks=(resized_mask is not None)
            )
            
            # Check if the resized GIF was created successfully
            if resized_gif:
                print(f"✓ Resized GIF created: {os.path.basename(resized_gif)}")
                if os.path.exists(resized_gif):
                    file_size = os.path.getsize(resized_gif) / 1024  # KB
                    print(f"  Size: {file_size:.1f} KB")
            else:
                print(f"✗ Failed to create resized GIF")
            
            # Final memory usage
            try:
                if memory_before > 0:
                    memory_final = process.memory_info().rss / 1024 / 1024
                    print(f"Final memory usage: {memory_final:.1f} MB " +
                          f"(Δ: {memory_final - memory_before:.1f} MB)")
            except:
                pass
            
            # Report total time
            end_time = time.time()
            duration = end_time - start_time
            print(f"\nTotal processing time: {duration:.1f} seconds")
            print(f"=== Volume Visualization Process Completed ===\n")
            
            return original_gif, resized_gif
            
        except Exception as e:
            print(f"\n✗ ERROR: Error creating visualizations: {e}")
            import traceback
            traceback.print_exc()
            return None, None
            
        finally:
            print(f"\nCleaning up temporary directories...")
            # Clean up temporary directories
            import shutil
            
            for dir_path in [original_slices_dir, resized_slices_dir]:
                try:
                    if os.path.exists(dir_path):
                        print(f"Removing {dir_path}...")
                        shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"Warning: Could not clean up directory {dir_path}: {e}")


def get_nuclei_dataloader(root_dir, 
                          batch_size=8, 
                          shuffle=True, 
                          num_workers=4,
                          transform=None,
                          mask_transform=None,
                          sample_ids=None,
                          slice_range=None,
                          return_paths=False,
                          class_csv_path=None,
                          filter_by_class=None,
                          ignore_unclassified=True,
                          load_volumes=False,
                          target_shape=(80, 80, 80),
                          mask_priority=True,
                          preserve_resolution=False,
                          high_quality_resize=True,
                          debug=False):
    """
    Create a DataLoader for the nuclei dataset.
    
    Args:
        root_dir (str): Root directory of the nuclei dataset.
        batch_size (int): Batch size for the dataloader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        transform (callable, optional): Transform to be applied on raw images.
        mask_transform (callable, optional): Transform to be applied on mask images.
        sample_ids (list, optional): List of sample IDs to include.
        slice_range (tuple, optional): Range of slice numbers to include.
        return_paths (bool): Whether to return file paths with the data.
        class_csv_path (str, optional): Path to CSV file containing chromatin class information.
        filter_by_class (int or list, optional): Class ID or list of class IDs to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        load_volumes (bool): Whether to load 3D volumes instead of 2D slices.
        target_shape (tuple): Target shape for volume resizing (z, y, x).
        mask_priority (bool): Whether to prioritize regions with masks during resizing.
        preserve_resolution (bool): Whether to preserve original resolution using crop/pad instead of scaling.
        high_quality_resize (bool): Whether to use high-quality resize methods instead of basic ones.
        debug (bool): Whether to print debug statements during processing.
        
    Returns:
        DataLoader: PyTorch DataLoader for the nuclei dataset.
    """
    # Create default transforms if none provided
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
        ])
    
    if mask_transform is None:
        mask_transform = transforms.Compose([
            transforms.ToTensor()
        ])
    
    # Create dataset
    dataset = NucleiDataset(
        root_dir=root_dir,
        transform=transform,
        mask_transform=mask_transform,
        sample_ids=sample_ids,
        slice_range=slice_range,
        return_paths=return_paths,
        class_csv_path=class_csv_path,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        load_volumes=load_volumes,
        target_shape=target_shape,
        mask_priority=mask_priority,
        preserve_resolution=preserve_resolution,
        high_quality_resize=high_quality_resize,
        debug=debug
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return dataloader


# Custom transforms for data augmentation
class NucleiTransforms:
    """
    Collection of transforms specifically designed for nuclei images and masks.
    """
    @staticmethod
    def get_train_transforms(img_size=256):
        """
        Get transforms for training data.
        
        Args:
            img_size (int): Target image size after transforms.
            
        Returns:
            dict: Dictionary with 'image' and 'mask' transforms.
        """
        image_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        mask_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        return {
            'image': image_transforms,
            'mask': mask_transforms
        }
    
    @staticmethod
    def get_val_transforms(img_size=256):
        """
        Get transforms for validation data.
        
        Args:
            img_size (int): Target image size after transforms.
            
        Returns:
            dict: Dictionary with 'image' and 'mask' transforms.
        """
        image_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        mask_transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor()
        ])
        
        return {
            'image': image_transforms,
            'mask': mask_transforms
        } 