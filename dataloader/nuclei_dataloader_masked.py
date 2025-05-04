import os
import sys
import glob
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gc
from tqdm import tqdm
from scipy import ndimage
from skimage.measure import label, regionprops

class MaskedNucleiDataset(Dataset):
    """
    Dataset class for loading 3D nuclei volumes, extracting subvolumes only from masked areas.
    This implementation uses true lazy loading and processes cells based on a CSV file.
    """
    def __init__(self, 
                 root_dir,
                 transform=None,
                 mask_transform=None,
                 sample_ids=None,
                 class_csv_path=None,
                 filter_by_class=None,
                 ignore_unclassified=True,
                 load_volumes=True, 
                 crop_size=(80, 80, 80),  # Size of each cropped volume
                 target_size=(80, 80, 80),
                 sample_percent=100,  # Percentage of samples to load per class (1-100)
                 mask_threshold=0.5,  # Threshold for mask values to be considered "masked"
                 min_masked_ratio=0.5,  # Minimum ratio of masked voxels in a subvolume
                 scan_step=40,  # Step size for scanning potential crop locations
                 debug=False):
        """
        Args:
            root_dir (str): Root directory of the nuclei dataset.
            transform (callable, optional): Transform to be applied on the raw volumes.
            mask_transform (callable, optional): Transform to be applied on the mask volumes.
            sample_ids (list, optional): List of sample IDs to include.
            class_csv_path (str, optional): Path to CSV file containing chromatin class information.
            filter_by_class (int or list, optional): Class ID or list of class IDs to include.
            ignore_unclassified (bool): Whether to ignore unclassified samples.
            load_volumes (bool): Whether to load 3D volumes (True by default).
            crop_size (tuple): Size of each 3D cropped volume (depth, height, width).
            target_size (tuple): Target size for the volumes (depth, height, width) for deep learning models.
            sample_percent (int): Percentage of samples to load per class (1-100).
            mask_threshold (float): Threshold for mask values to be considered "masked".
            min_masked_ratio (float): Minimum ratio of masked voxels in a subvolume (0-1).
            scan_step (int): Step size for scanning potential crop locations.
            debug (bool): Whether to print debug statements during processing.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.load_volumes = load_volumes
        self.crop_size = crop_size
        self.target_size = target_size
        self.debug = debug
        self.sample_percent = min(max(1, sample_percent), 100)  # Clamp between 1-100
        self.mask_threshold = mask_threshold
        self.min_masked_ratio = min_masked_ratio
        self.scan_step = scan_step
        
        # Dictionary to cache loaded volumes for reuse
        self.volume_cache = {}
        self.max_cache_size = 10  # Maximum number of volumes to keep in cache
        
        # Process samples based on CSV file
        self.samples = []  # Will contain all samples to process
        
        if class_csv_path and os.path.exists(class_csv_path):
            try:
                # Load the CSV file
                df = pd.read_csv(class_csv_path)
                
                # Remove unclassified samples first
                df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
                
                # Filter by class if specified
                if filter_by_class is not None:
                    if isinstance(filter_by_class, int):
                        filter_by_class = [filter_by_class]
                    df = df[df['class_id'].isin(filter_by_class)]
                
                # Filter by sample IDs if specified
                if sample_ids is not None:
                    sample_ids = [str(sid).zfill(4) for sid in sample_ids]  # Ensure 4-digit format
                    df = df[df['sample_id'].astype(str).apply(lambda x: x.zfill(4)).isin(sample_ids)]
                
                # Apply percentage-based sampling per class
                if self.sample_percent < 100:
                    sampled_df = pd.DataFrame()
                    for class_id, group in df.groupby('class_id'):
                        # Calculate number to keep
                        num_to_keep = max(1, int(len(group) * self.sample_percent / 100.0))
                        # Randomly sample
                        random.seed(42)  # For reproducibility
                        sampled_group = group.sample(n=num_to_keep, random_state=42)
                        sampled_df = pd.concat([sampled_df, sampled_group])
                        print(f"Class {class_id}: Sampled {num_to_keep}/{len(group)} samples ({self.sample_percent}%)")
                    df = sampled_df
                
                # For each sample, determine the possible crops and add to samples list
                for _, row in df.iterrows():
                    sample_id = str(row['sample_id']).zfill(4)  # Ensure 4-digit format
                    class_id = int(row['class_id'])
                    class_name = row['class_name']
                    
                    # Skip unclassified samples explicitly
                    if class_name == 'Unclassified' or class_id == 19:
                        continue
                    
                    # Check if this sample directory exists (try both formats)
                    padded_dir = os.path.join(root_dir, sample_id)
                    unpadded_dir = os.path.join(root_dir, sample_id.lstrip('0'))
                    
                    sample_dir = None
                    if os.path.isdir(padded_dir):
                        sample_dir = padded_dir
                    elif os.path.isdir(unpadded_dir) and sample_id.lstrip('0') != '':
                        sample_dir = unpadded_dir
                    
                    if sample_dir is None:
                        print(f"Warning: Sample directory not found: {padded_dir}")
                        continue
                    
                    # Check for raw and mask directories
                    raw_dir = os.path.join(sample_dir, 'raw')
                    mask_dir = os.path.join(sample_dir, 'mask')
                    
                    if not (os.path.exists(raw_dir) and os.path.exists(mask_dir)):
                        print(f"Warning: Raw or mask directory not found for sample {sample_id}")
                        continue
                    
                    # Get the dimensions without loading the entire volume
                    raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
                    mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
                    
                    if not raw_files or not mask_files:
                        print(f"Warning: No raw or mask files found for sample {sample_id}")
                        continue
                    
                    # Load a few mask slices to find potential crop locations based on masked regions
                    try:
                        # Get dimensions from first image
                        first_img = np.array(Image.open(raw_files[0]))
                        height, width = first_img.shape
                        depth = len(raw_files)
                        
                        # We'll analyze the mask to find regions with good mask coverage
                        # Sample a subset of the mask to speed up analysis
                        num_slices_to_check = min(len(mask_files), 20)  # Check a reasonable number of slices
                        slice_indices = np.linspace(0, len(mask_files)-1, num_slices_to_check, dtype=int)
                        
                        # Create a downsampled 3D representation of where masks are present
                        mask_presence = np.zeros((len(slice_indices), 
                                                  height // self.scan_step + 1, 
                                                  width // self.scan_step + 1), dtype=bool)
                        
                        for i, slice_idx in enumerate(slice_indices):
                            if slice_idx < len(mask_files):
                                mask_img = np.array(Image.open(mask_files[slice_idx]))
                                # Downsample to reduce computation
                                for h_idx in range(0, height, self.scan_step):
                                    for w_idx in range(0, width, self.scan_step):
                                        # Note if this region has mask values above threshold
                                        h_end = min(h_idx + self.scan_step, height)
                                        w_end = min(w_idx + self.scan_step, width)
                                        region = mask_img[h_idx:h_end, w_idx:w_end]
                                        if np.mean(region) > self.mask_threshold:
                                            h_ds = h_idx // self.scan_step
                                            w_ds = w_idx // self.scan_step
                                            mask_presence[i, h_ds, w_ds] = True
                        
                        # Find connected regions in the downsampled mask
                        labeled_regions = label(mask_presence)
                        regions = regionprops(labeled_regions)
                        
                        # For each significant region, generate crop positions
                        crop_positions = []
                        
                        # First try to find crop positions based on region analysis
                        for region in regions:
                            if region.area > 3:  # Only consider regions with some minimum size
                                # Get the bounding box of this region
                                min_d, min_h, min_w, max_d, max_h, max_w = region.bbox
                                # Scale back to original dimensions
                                min_h *= self.scan_step
                                min_w *= self.scan_step
                                max_h *= self.scan_step
                                max_w *= self.scan_step
                                min_d = slice_indices[min_d]
                                max_d = slice_indices[max_d-1] if max_d < len(slice_indices) else depth-1
                                
                                # Ensure we don't exceed volume bounds
                                max_d = min(max_d, depth-1)
                                max_h = min(max_h, height-1)
                                max_w = min(max_w, width-1)
                                
                                # Create a crop centered on this region, ensuring it fits within bounds
                                d_center = (min_d + max_d) // 2
                                h_center = (min_h + max_h) // 2
                                w_center = (min_w + max_w) // 2
                                
                                # Compute crop start positions (centered on the region)
                                d_start = max(0, d_center - self.crop_size[0] // 2)
                                h_start = max(0, h_center - self.crop_size[1] // 2)
                                w_start = max(0, w_center - self.crop_size[2] // 2)
                                
                                # Ensure crop doesn't exceed volume bounds
                                if d_start + self.crop_size[0] > depth:
                                    d_start = max(0, depth - self.crop_size[0])
                                if h_start + self.crop_size[1] > height:
                                    h_start = max(0, height - self.crop_size[1])
                                if w_start + self.crop_size[2] > width:
                                    w_start = max(0, width - self.crop_size[2])
                                
                                crop_positions.append((d_start, h_start, w_start))
                        
                        # If we found good crop positions, add them to samples
                        if crop_positions:
                            for crop_idx, crop_position in enumerate(crop_positions):
                                self.samples.append({
                                    'sample_id': sample_id,
                                    'class_id': class_id,
                                    'class_name': class_name,
                                    'raw_dir': raw_dir,
                                    'mask_dir': mask_dir,
                                    'crop_idx': crop_idx,
                                    'crop_position': crop_position,
                                    'depth': depth,
                                    'height': height,
                                    'width': width
                                })
                        else:
                            # If no good positions found, fall back to a simpler approach
                            # Try middle of the volume as a fallback
                            d_start = max(0, depth // 2 - self.crop_size[0] // 2)
                            h_start = max(0, height // 2 - self.crop_size[1] // 2)
                            w_start = max(0, width // 2 - self.crop_size[2] // 2)
                            
                            crop_position = (d_start, h_start, w_start)
                            self.samples.append({
                                'sample_id': sample_id,
                                'class_id': class_id,
                                'class_name': class_name,
                                'raw_dir': raw_dir,
                                'mask_dir': mask_dir,
                                'crop_idx': 0,
                                'crop_position': crop_position,
                                'depth': depth,
                                'height': height,
                                'width': width
                            })
                            
                    except Exception as e:
                        print(f"Error processing sample {sample_id}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                print(f"Total number of samples (crops) to process: {len(self.samples)}")
                
            except Exception as e:
                print(f"Error loading class CSV: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Class CSV file not provided or not found: {class_csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_volume_slice(self, sample_metadata, crop_position):
        """
        Load only the necessary slice of a 3D volume based on crop position.
        
        Args:
            sample_metadata (dict): Sample metadata including directories
            crop_position (tuple): (d_start, h_start, w_start) for the crop
            
        Returns:
            tuple: (volume_crop, mask_crop) as numpy arrays
        """
        d_start, h_start, w_start = crop_position
        d_end = d_start + self.crop_size[0]
        h_end = h_start + self.crop_size[1]
        w_end = w_start + self.crop_size[2]
        
        # Ensure crop dimensions are within volume bounds
        d_end = min(d_end, sample_metadata['depth'])
        h_end = min(h_end, sample_metadata['height'])
        w_end = min(w_end, sample_metadata['width'])
        
        # Adjusting crop size if needed
        actual_crop_size = (d_end - d_start, h_end - h_start, w_end - w_start)
        
        # Prepare empty arrays for the crop
        volume_crop = np.zeros(self.crop_size, dtype=np.float32)
        mask_crop = np.zeros(self.crop_size, dtype=np.float32)
        
        # Load only the slices needed for this crop
        raw_files = sorted(glob.glob(os.path.join(sample_metadata['raw_dir'], '*.tif')))
        mask_files = sorted(glob.glob(os.path.join(sample_metadata['mask_dir'], '*.tif')))
        
        for d_idx, d in enumerate(range(d_start, d_end)):
            if d >= len(raw_files) or d >= len(mask_files):
                continue
            
            # Load the raw and mask slices
            raw_img = np.array(Image.open(raw_files[d]))
            mask_img = np.array(Image.open(mask_files[d]))
            
            # Extract only the region we need
            raw_slice = raw_img[h_start:h_end, w_start:w_end]
            mask_slice = mask_img[h_start:h_end, w_start:w_end]
            
            # Handle padding if the actual crop is smaller than the target crop size
            actual_h, actual_w = raw_slice.shape
            volume_crop[d_idx, :actual_h, :actual_w] = raw_slice
            mask_crop[d_idx, :actual_h, :actual_w] = mask_slice
        
        # Verify that this crop has sufficient mask coverage
        mask_ratio = np.mean(mask_crop > self.mask_threshold)
        if mask_ratio < self.min_masked_ratio:
            # If mask coverage is insufficient, adjust the volume crop to use only masked regions
            # Zero out areas outside the mask
            volume_crop = volume_crop * (mask_crop > self.mask_threshold)
        
        return volume_crop, mask_crop
    
    def __getitem__(self, idx):
        """
        Get a specific sample by index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            dict: Dictionary containing the sample data
        """
        try:
            # Get the sample metadata
            sample = self.samples[idx]
            sample_id = sample['sample_id']
            
            # Load only the required crop of the volume
            crop_position = sample['crop_position']
            volume_crop, mask_crop = self._load_volume_slice(sample, crop_position)
            
            # Check for NaN or Inf values
            if np.isnan(volume_crop).any() or np.isinf(volume_crop).any():
                print(f"WARNING: Found NaN or Inf values in volume for sample {sample_id}")
                volume_crop = np.nan_to_num(volume_crop, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Apply transforms
            if self.transform is not None:
                try:
                    crop_tensor = self.transform(volume_crop)
                    
                    # Ensure tensor has 5D shape (1, 1, D, H, W)
                    if crop_tensor.dim() != 5:
                        if crop_tensor.dim() == 3:  # (D, H, W)
                            crop_tensor = crop_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, D, H, W)
                        elif crop_tensor.dim() == 4:  # (C, D, H, W) or (B, D, H, W)
                            crop_tensor = crop_tensor.unsqueeze(0)  # -> (1, C, D, H, W)
                except Exception as e:
                    print(f"Error in transform for sample {sample_id}: {e}")
                    crop_tensor = torch.from_numpy(volume_crop).float().unsqueeze(0).unsqueeze(0)
            else:
                crop_tensor = torch.from_numpy(volume_crop).float().unsqueeze(0).unsqueeze(0)
            
            # Do the same for mask
            if self.mask_transform is not None:
                try:
                    mask_tensor = self.mask_transform(mask_crop)
                    
                    # Ensure tensor has 5D shape (1, 1, D, H, W)
                    if mask_tensor.dim() != 5:
                        if mask_tensor.dim() == 3:  # (D, H, W)
                            mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, D, H, W)
                        elif mask_tensor.dim() == 4:  # (C, D, H, W) or (B, D, H, W)
                            mask_tensor = mask_tensor.unsqueeze(0)  # -> (1, C, D, H, W)
                except Exception as e:
                    print(f"Error in mask transform for sample {sample_id}: {e}")
                    mask_tensor = torch.from_numpy(mask_crop).float().unsqueeze(0).unsqueeze(0)
            else:
                mask_tensor = torch.from_numpy(mask_crop).float().unsqueeze(0).unsqueeze(0)
            
            # Validate tensor shapes
            assert crop_tensor.dim() == 5, f"Expected 5D tensor for crop, got {crop_tensor.dim()}D with shape {crop_tensor.shape}"
            assert mask_tensor.dim() == 5, f"Expected 5D tensor for mask, got {mask_tensor.dim()}D with shape {mask_tensor.shape}"
            
            # Create metadata
            metadata = {
                'sample_id': sample_id,
                'crop_idx': sample['crop_idx'], 
                'class_name': sample['class_name'],
                'tensor_shape': crop_tensor.shape
            }
            
            # Return the sample
            return {
                'volume': crop_tensor,
                'mask': mask_tensor,
                'label': sample['class_id'],
                'metadata': metadata
            }
            
        except Exception as e:
            print(f"Error in __getitem__ for idx={idx}: {e}")
            import traceback
            traceback.print_exc()
            raise

def custom_collate_fn(batch):
    """
    Custom collate function to handle samples of different sizes.
    
    Args:
        batch (list): List of samples
        
    Returns:
        dict: Dictionary containing batched samples
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if not batch:
        return None
    
    # Initialize empty dict for batched samples
    batched_samples = {
        'volume': [],
        'mask': [],
        'label': [],
        'metadata': []
    }
    
    # Add each sample to the batched samples
    for sample in batch:
        batched_samples['volume'].append(sample['volume'])
        batched_samples['mask'].append(sample['mask'])
        batched_samples['label'].append(sample['label'])
        batched_samples['metadata'].append(sample['metadata'])
    
    return batched_samples

def get_masked_nuclei_dataloader(root_dir, 
                                batch_size=8, 
                                shuffle=True, 
                                num_workers=4,
                                transform=None,
                                mask_transform=None,
                                sample_ids=None,
                                class_csv_path=None,
                                filter_by_class=None,
                                ignore_unclassified=True,
                                target_size=(80, 80, 80),
                                load_volumes=True,
                                sample_percent=100,  # Percentage of samples to load per class
                                crop_size=(80, 80, 80),  # Size of each cropped volume
                                mask_threshold=0,  # Threshold for mask values
                                min_masked_ratio=0.5,  # Minimum ratio of masked voxels
                                scan_step=40,  # Step size for scanning
                                pin_memory=False,  # Disable pin memory by default
                                debug=False):
    """
    Create a DataLoader for the nuclei dataset that only extracts subvolumes from masked areas.
    
    Args:
        root_dir (str): Root directory of the nuclei dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes for data loading.
        transform (callable, optional): Transform to be applied on the raw volumes.
        mask_transform (callable, optional): Transform to be applied on the mask volumes.
        sample_ids (list, optional): List of sample IDs to include.
        class_csv_path (str, optional): Path to CSV file containing chromatin class information.
        filter_by_class (int or list, optional): Class ID or list of class IDs to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        target_size (tuple): Target size for the volumes (depth, height, width) for deep learning models.
        load_volumes (bool): Whether to load 3D volumes (True by default).
        sample_percent (int): Percentage of samples to load per class (1-100).
        crop_size (tuple): Size of each 3D cropped volume (depth, height, width).
        mask_threshold (float): Threshold for mask values to be considered "masked".
        min_masked_ratio (float): Minimum ratio of masked voxels in a subvolume (0-1).
        scan_step (int): Step size for scanning potential crop locations.
        pin_memory (bool): Whether to use pinned memory for faster CUDA transfers.
        debug (bool): Whether to print debug statements during processing.
        
    Returns:
        DataLoader: DataLoader for the nuclei dataset.
    """
    dataset = MaskedNucleiDataset(
        root_dir=root_dir,
        transform=transform,
        mask_transform=mask_transform,
        sample_ids=sample_ids,
        class_csv_path=class_csv_path,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        load_volumes=load_volumes,
        crop_size=crop_size,
        target_size=target_size,
        sample_percent=sample_percent,
        mask_threshold=mask_threshold,
        min_masked_ratio=min_masked_ratio,
        scan_step=scan_step,
        debug=debug
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Add the collate function as an attribute for later use
    dataloader.collate_fn = custom_collate_fn
    # Don't set dataset attribute, it's already set by DataLoader initialization
    
    return dataloader 