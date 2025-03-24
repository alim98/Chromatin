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


def custom_collate_fn(batch):
    """
    Custom collate function that handles samples of different sizes.
    Instead of stacking tensors of different sizes, it returns lists of tensors.
    
    Args:
        batch (list): List of samples from the dataset.
        
    Returns:
        dict: Dictionary containing lists of tensors and batch metadata.
    """
    # Initialize dictionaries to store batch data
    result = {}
    
    # Check if we have samples in the batch
    if not batch:
        return result
    
    # Get keys from the first sample
    keys = batch[0].keys()
    
    # Process each key
    for key in keys:
        if key == 'metadata':
            # For metadata, create a dictionary with each field as a list
            result[key] = {}
            metadata_keys = batch[0][key].keys()
            
            for metadata_key in metadata_keys:
                result[key][metadata_key] = [sample[key][metadata_key] for sample in batch]
        else:
            # For other keys (like 'image', 'mask', 'volume', etc.), collect tensors in a list
            result[key] = [sample[key] for sample in batch]
    
    return result


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
                 max_crops_per_volume=8,
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
            max_crops_per_volume (int): Maximum number of crops to extract per volume.
            debug (bool): Whether to print debug statements during processing.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.return_paths = return_paths
        self.load_volumes = load_volumes
        self.max_crops_per_volume = max_crops_per_volume
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

            # Standard approach returns the original volume
            resized_volume = volume  # No resize anymore
            resized_mask = mask
            
            # Convert to torch tensors
            if self.transform:
                volume_tensor = self.transform(resized_volume)
            else:
                # Default conversion to tensor
                volume_tensor = torch.from_numpy(resized_volume).float().unsqueeze(0)  # Add channel dimension
            
            if resized_mask is not None:
                if self.mask_transform:
                    mask_tensor = self.mask_transform(resized_mask)
                else:
                    # Default conversion to tensor
                    mask_tensor = torch.from_numpy(resized_mask).float().unsqueeze(0)
            else:
                mask_tensor = None
            
            # Initialize label with default value (-1 indicating no label)
            label = -1
            
            # Create metadata dictionary
            metadata = {
                'sample_id': sample_id,
                'original_shape': original_shape
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
                'volume': volume_tensor,
                'mask': mask_tensor,
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
                          max_crops_per_volume=8,
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
        max_crops_per_volume (int): Maximum number of crops to extract per volume.
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
        max_crops_per_volume=max_crops_per_volume,
        debug=debug
    )
    
    # Create dataloader with custom collate function to handle different sized tensors
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn  # Use custom collate function
    )
    
    return dataloader

