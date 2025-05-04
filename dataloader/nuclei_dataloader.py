import os
import glob
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
import torchvision.transforms as transforms
from skimage.transform import resize
from tqdm import tqdm
import sys
import gc
import time
import random

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
    
    sample_to_class = {}
    for _, row in df.iterrows():
        if ignore_unclassified and (row['class_name'] == 'Unclassified' or row['class_id'] == 19):
            continue
        sample_to_class[str(row['sample_id'])] = {
            'class_id': row['class_id'],
            'class_name': row['class_name']
        }
    
    return sample_to_class


def custom_collate_fn(batch):
    """
    Custom collate function that handles samples of different sizes.
    
    Args:
        batch (list): List of samples from the dataset.
        
    Returns:
        dict: Dictionary containing lists of tensors and batch metadata.
    """
    result = {}
    if not batch:
        return result
    
    keys = batch[0].keys()
    
    # Check that all volumes have same shape before stacking
    if 'volume' in keys:
        volumes = [sample['volume'] for sample in batch]
        # Print shape information for debugging
        if len(volumes) > 0:
            # Expected shape is (1, 1, D, H, W)
            expected_dim = 5
            first_shape = volumes[0].shape
            
            # Check and fix inconsistent volume shapes
            for i, vol in enumerate(volumes):
                # Check dimensions
                if vol.dim() != expected_dim:
                    print(f"WARNING: Volume at index {i} has {vol.dim()} dimensions instead of {expected_dim}")
                    # Try to fix by adding missing dimensions
                    if vol.dim() < expected_dim:
                        # Add missing dimensions
                        while vol.dim() < expected_dim:
                            vol = vol.unsqueeze(0)
                        volumes[i] = vol
                    elif vol.dim() > expected_dim:
                        # Too many dimensions, try to remove extras
                        while vol.dim() > expected_dim and vol.shape[0] == 1:
                            vol = vol.squeeze(0)
                        volumes[i] = vol
                
                # Check shape matches first tensor
                if volumes[i].shape != first_shape:
                    print(f"WARNING: Inconsistent volume shapes in batch: {first_shape} vs {volumes[i].shape} at index {i}")
                    # Don't try to fix different shapes - it's safer to skip the batch
                    # Try to handle by resizing all to the first shape
                    # In production this would need more robust handling
                    raise ValueError(f"Inconsistent volume shapes in batch: {volumes[i].shape} != {first_shape}")
    
    for key in keys:
        if key == 'metadata':
            result[key] = {}
            metadata_keys = batch[0][key].keys()
            
            for metadata_key in metadata_keys:
                result[key][metadata_key] = [sample[key][metadata_key] for sample in batch]
        else:
            # Before stacking, check if the tensors contain NaN or Inf values
            if key in ['volume', 'mask'] and all(torch.is_tensor(sample[key]) for sample in batch):
                for i, sample in enumerate(batch):
                    # Check for NaN/Inf
                    if torch.isnan(sample[key]).any() or torch.isinf(sample[key]).any():
                        print(f"WARNING: NaN/Inf values in '{key}' at batch index {i}")
                        # Fix NaN/Inf values
                        batch[i][key] = torch.nan_to_num(sample[key], nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Check dimensions for tensors - we expect 5D (B,C,D,H,W)
                    if sample[key].dim() != 5:
                        print(f"WARNING: {key} tensor at index {i} has {sample[key].dim()} dimensions instead of 5")
                        # This shouldn't happen since we fixed it in __getitem__, but just in case
                        tensor = sample[key]
                        if tensor.dim() < 5:
                            # Add missing dimensions
                            while tensor.dim() < 5:
                                tensor = tensor.unsqueeze(0)
                        elif tensor.dim() > 5:
                            # Too many dimensions
                            while tensor.dim() > 5 and tensor.shape[0] == 1:
                                tensor = tensor.squeeze(0)
                        batch[i][key] = tensor

            # For volume and mask, stack the tensors
            if key in ['volume', 'mask'] and all(torch.is_tensor(sample[key]) for sample in batch):
                try:
                    # Try to stack tensors along batch dimension (dim=0)
                    result[key] = torch.cat([sample[key] for sample in batch], dim=0)
                except RuntimeError as e:
                    print(f"ERROR stacking {key} tensors: {e}")
                    # Collect shape information to help debug
                    shapes = [sample[key].shape for sample in batch]
                    print(f"Tensor shapes: {shapes}")
                    # Fall back to returning the list
                    result[key] = [sample[key] for sample in batch]
            else:
                # For other keys, just return the list
                result[key] = [sample[key] for sample in batch]
    
    return result


class NucleiDataset(Dataset):
    """
    Dataset class for loading 3D nuclei volumes, cropping them into smaller patches, and assigning labels.
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
                    if not raw_files:
                        print(f"Warning: No raw files found for sample {sample_id}")
                        continue
                    
                    # Get dimensions from first image
                    try:
                        first_img = np.array(Image.open(raw_files[0]))
                        height, width = first_img.shape
                        depth = len(raw_files)
                        
                        # Calculate how many crops we can get from this sample
                        depth_crops = max(1, (depth - self.crop_size[0] + 1) // self.crop_size[0])
                        height_crops = max(1, (height - self.crop_size[1] + 1) // self.crop_size[1])
                        width_crops = max(1, (width - self.crop_size[2] + 1) // self.crop_size[2])
                        
                        # For each potential crop, add a sample entry
                        for d_idx in range(depth_crops):
                            for h_idx in range(height_crops):
                                for w_idx in range(width_crops):
                                    crop_idx = (d_idx, h_idx, w_idx)
                                    crop_position = (
                                        d_idx * self.crop_size[0],
                                        h_idx * self.crop_size[1],
                                        w_idx * self.crop_size[2]
                                    )
                                    
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
                    except Exception as e:
                        print(f"Error processing sample {sample_id}: {e}")
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
        for d_idx, d in enumerate(range(d_start, d_end)):
            if d >= len(raw_files):
                continue
                
            file_name = os.path.basename(raw_files[d])
            mask_file = os.path.join(sample_metadata['mask_dir'], file_name)
            
            if not os.path.exists(mask_file):
                continue
            
            # Load the full slice
            raw_img = np.array(Image.open(raw_files[d]))
            mask_img = np.array(Image.open(mask_file))
            
            # Extract only the region we need
            raw_slice = raw_img[h_start:h_end, w_start:w_end]
            mask_slice = mask_img[h_start:h_end, w_start:w_end]
            
            # Handle padding if the actual crop is smaller than the target crop size
            actual_h, actual_w = raw_slice.shape
            volume_crop[d_idx, :actual_h, :actual_w] = raw_slice
            mask_crop[d_idx, :actual_h, :actual_w] = mask_slice
        
        return volume_crop, mask_crop
    
    def _manage_cache(self, sample_id):
        """
        Manage the volume cache to prevent memory issues.
        
        Args:
            sample_id (str): ID of the sample being added
            
        Returns:
            None
        """
        # If cache is too large, remove the oldest item
        if len(self.volume_cache) >= self.max_cache_size:
            # Get the oldest item (first key)
            oldest_key = next(iter(self.volume_cache))
            del self.volume_cache[oldest_key]
            # Force garbage collection
            gc.collect()
    
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


def get_nuclei_dataloader(root_dir, 
                          batch_size=8, 
                          shuffle=True, 
                          num_workers=4,
                          transform=None,
                          mask_transform=None,
                          sample_ids=None,
                          return_paths=False,
                          class_csv_path=None,
                          filter_by_class=None,
                          ignore_unclassified=True,
                          target_size=(80, 80, 80),
                          load_volumes=True,
                          sample_percent=100,  # Percentage of samples to load per class
                          crop_size=(80, 80, 80),  # Size of each cropped volume
                          pin_memory=False,  # Disable pin memory by default
                          debug=False):
    """
    Create a DataLoader for the nuclei dataset.
    
    Args:
        root_dir (str): Root directory of the nuclei dataset.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes for data loading.
        transform (callable, optional): Transform to be applied on the raw volumes.
        mask_transform (callable, optional): Transform to be applied on the mask volumes.
        sample_ids (list, optional): List of sample IDs to include.
        return_paths (bool): Whether to return file paths in the DataLoader.
        class_csv_path (str, optional): Path to CSV file containing chromatin class information.
        filter_by_class (int or list, optional): Class ID or list of class IDs to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        target_size (tuple): Target size for the volumes (depth, height, width) for deep learning models.
        load_volumes (bool): Whether to load 3D volumes (True by default).
        sample_percent (int): Percentage of samples to load per class (1-100).
        crop_size (tuple): Size of each 3D cropped volume (depth, height, width).
        pin_memory (bool): Whether to use pinned memory for faster CUDA transfers.
        debug (bool): Whether to print debug statements during processing.
        
    Returns:
        DataLoader: DataLoader for the nuclei dataset.
    """
    # Check if path exists
    if not os.path.exists(root_dir):
        raise ValueError(f"Dataset directory does not exist: {root_dir}")
    
    # Create dataset
    dataset = NucleiDataset(
        root_dir=root_dir,
        transform=transform,
        mask_transform=mask_transform,
        sample_ids=sample_ids,
        class_csv_path=class_csv_path,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        load_volumes=load_volumes,
        target_size=target_size,
        crop_size=crop_size,  # Pass through crop_size parameter
        sample_percent=sample_percent,
        debug=debug
    )
    
    # Create dataloader with safety settings for CUDA errors
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory,  # Set to False to avoid CUDA memory issues
        persistent_workers=False if num_workers == 0 else True,  # Keep workers alive between iterations if using workers
        drop_last=False  # Don't drop the last incomplete batch
    )
    
    return dataloader
