import os
import glob
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import sys
from tqdm import tqdm
import open3d as o3d
from PIL import Image


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.pointcloud import load_mask_volume, create_pointcloud_from_mask, create_mesh_from_mask

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
    Custom collate function for mesh/point cloud data.
    
    Args:
        batch (list): List of samples from the dataset.
        
    Returns:
        dict: Dictionary containing lists of tensors and batch metadata.
    """
    result = {}
    if not batch:
        return result
    
    keys = batch[0].keys()
    
    for key in keys:
        if key == 'metadata':
            result[key] = {}
            metadata_keys = batch[0][key].keys()
            
            for metadata_key in metadata_keys:
                result[key][metadata_key] = [sample[key][metadata_key] for sample in batch]
        elif key == 'points':
            # Handle point cloud data
            max_points = max([sample[key].shape[0] for sample in batch])
            batch_size = len(batch)
            
            # Create padded tensors
            padded_points = torch.zeros(batch_size, max_points, 3)
            padded_masks = torch.zeros(batch_size, max_points)
            
            for i, sample in enumerate(batch):
                num_points = sample[key].shape[0]
                padded_points[i, :num_points, :] = sample[key]
                padded_masks[i, :num_points] = 1  # Mask to identify real points
            
            result[key] = padded_points
            result['point_masks'] = padded_masks
        elif key == 'vertices':
            # Handle mesh vertices
            max_vertices = max([sample[key].shape[0] for sample in batch])
            batch_size = len(batch)
            
            # Create padded tensors
            padded_vertices = torch.zeros(batch_size, max_vertices, 3)
            padded_vertex_masks = torch.zeros(batch_size, max_vertices)
            
            for i, sample in enumerate(batch):
                num_vertices = sample[key].shape[0]
                padded_vertices[i, :num_vertices, :] = sample[key]
                padded_vertex_masks[i, :num_vertices] = 1  # Mask to identify real vertices
            
            result[key] = padded_vertices
            result['vertex_masks'] = padded_vertex_masks
        elif key == 'faces':
            # Handle mesh faces
            max_faces = max([sample[key].shape[0] for sample in batch])
            batch_size = len(batch)
            
            # Create padded tensors with -1 padding (invalid indices)
            padded_faces = torch.ones(batch_size, max_faces, 3, dtype=torch.long) * -1
            padded_face_masks = torch.zeros(batch_size, max_faces)
            
            for i, sample in enumerate(batch):
                num_faces = sample[key].shape[0]
                padded_faces[i, :num_faces, :] = sample[key]
                padded_face_masks[i, :num_faces] = 1  # Mask to identify real faces
            
            result[key] = padded_faces
            result['face_masks'] = padded_face_masks
        else:
            # Handle other data (like labels)
            result[key] = [sample[key] for sample in batch]
    
    return result


class MeshDataset(Dataset):
    """
    Dataset for loading triangle mesh data from mask volumes.
    """
    def __init__(self, 
                 root_dir,
                 class_csv_path=None,
                 sample_ids=None,
                 filter_by_class=None,
                 ignore_unclassified=True,
                 max_points=10000,
                 voxel_size=1.0,
                 sample_percent=100,
                 cache_dir=None,
                 use_mesh=True,
                 smoothing_iterations=1,
                 debug=False):
        """
        Args:
            root_dir (str): Root directory containing the samples.
            class_csv_path (str, optional): Path to CSV file with class information.
            sample_ids (list, optional): List of sample IDs to include.
            filter_by_class (int or list, optional): Class ID(s) to include.
            ignore_unclassified (bool): Whether to ignore unclassified samples.
            max_points (int): Maximum number of points to include in each point cloud.
            voxel_size (float): Size of each voxel in the point cloud.
            sample_percent (int): Percentage of samples to load per class (1-100).
            cache_dir (str, optional): Directory to cache processed point clouds.
            use_mesh (bool): Whether to generate meshes (True) or point clouds (False).
            smoothing_iterations (int): Number of iterations for mesh smoothing (if use_mesh=True).
            debug (bool): Whether to print debug information.
        """
        self.root_dir = root_dir
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.debug = debug
        self.sample_percent = min(max(1, sample_percent), 100)
        self.cache_dir = cache_dir
        self.use_mesh = use_mesh
        self.smoothing_iterations = smoothing_iterations
        
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize empty sample list
        self.samples = []
        
        # Load class information from CSV if provided
        if class_csv_path and os.path.exists(class_csv_path):
            try:
                # Read the CSV file
                df = pd.read_csv(class_csv_path)
                
                # Filter out unclassified samples if requested
                if ignore_unclassified:
                    df = df[(df['class_name'] != 'Unclassified') & (df['class_id'] != 19)]
                
                # Filter by class if specified
                if filter_by_class is not None:
                    if isinstance(filter_by_class, int):
                        filter_by_class = [filter_by_class]
                    df = df[df['class_id'].isin(filter_by_class)]
                
                # Filter by sample ID if specified
                if sample_ids is not None:
                    sample_ids = [str(sid).zfill(4) for sid in sample_ids]
                    df = df[df['sample_id'].astype(str).apply(lambda x: x.zfill(4)).isin(sample_ids)]
                
                # Sample a percentage of each class if requested
                if self.sample_percent < 100:
                    sampled_df = pd.DataFrame()
                    for class_id, group in df.groupby('class_id'):
                        num_to_keep = max(1, int(len(group) * self.sample_percent / 100.0))
                        sampled_group = group.sample(n=num_to_keep, random_state=42)
                        sampled_df = pd.concat([sampled_df, sampled_group])
                        if self.debug:
                            print(f"Class {class_id}: Sampled {num_to_keep}/{len(group)} samples ({self.sample_percent}%)")
                    df = sampled_df
                
                # Create sample entries for each valid sample
                for _, row in df.iterrows():
                    sample_id = str(row['sample_id']).zfill(4)
                    class_id = int(row['class_id'])
                    class_name = row['class_name']
                    
                    # Skip unclassified samples if requested
                    if ignore_unclassified and (class_name == 'Unclassified' or class_id == 19):
                        continue
                    
                    # Find the sample directory (with or without leading zeros)
                    padded_dir = os.path.join(root_dir, sample_id)
                    unpadded_dir = os.path.join(root_dir, sample_id.lstrip('0'))
                    
                    sample_dir = None
                    if os.path.isdir(padded_dir):
                        sample_dir = padded_dir
                    elif os.path.isdir(unpadded_dir) and sample_id.lstrip('0') != '':
                        sample_dir = unpadded_dir
                    
                    if sample_dir is None:
                        print(f"Warning: Sample directory not found for {sample_id}")
                        continue
                    
                    # Check if mask directory exists
                    mask_dir = os.path.join(sample_dir, 'mask')
                    if not os.path.exists(mask_dir):
                        print(f"Warning: Mask directory not found for sample {sample_id}")
                        continue
                    
                    # Add sample to the list
                    self.samples.append({
                        'sample_id': sample_id,
                        'class_id': class_id,
                        'class_name': class_name,
                        'mask_dir': mask_dir
                    })
                
                print(f"Total number of samples to process: {len(self.samples)}")
                
            except Exception as e:
                print(f"Error loading class CSV: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Warning: Class CSV file not provided or not found: {class_csv_path}")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_or_generate_pointcloud(self, sample):
        """
        Load from cache or generate a point cloud for a sample.
        
        Args:
            sample (dict): Sample information
            
        Returns:
            torch.Tensor: Point cloud tensor of shape (N, 3)
        """
        sample_id = sample['sample_id']
        mask_dir = sample['mask_dir']
        
        # Check if cached point cloud exists
        cache_file = None
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, f"{sample_id}_points.pt")
            if os.path.exists(cache_file):
                try:
                    if self.debug:
                        print(f"Loading cached point cloud for sample {sample_id}")
                    return torch.load(cache_file)
                except Exception as e:
                    print(f"Error loading cached point cloud: {e}")
                    
        # Generate point cloud if not cached
        try:
            # Load the mask volume
            mask_volume = load_mask_volume(mask_dir)
            
            # Create point cloud
            pcd = create_pointcloud_from_mask(
                mask_volume, 
                threshold=0.5, 
                voxel_size=self.voxel_size,
                sample_rate=1.0,
                max_points=self.max_points
            )
            
            # Convert to tensor
            points = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
            
            # Cache the point cloud
            if cache_file:
                torch.save(points, cache_file)
                if self.debug:
                    print(f"Cached point cloud for sample {sample_id}")
            
            return points
        
        except Exception as e:
            print(f"Error generating point cloud for sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            
            return torch.zeros((0, 3), dtype=torch.float32)
    
    def _load_or_generate_mesh(self, sample):
        """
        Load from cache or generate a triangle mesh for a sample.
        
        Args:
            sample (dict): Sample information
            
        Returns:
            tuple: (vertices, faces) as torch.Tensor objects
        """
        sample_id = sample['sample_id']
        mask_dir = sample['mask_dir']
        
        # Check if cached mesh exists
        vertices_cache_file = None
        faces_cache_file = None
        if self.cache_dir:
            vertices_cache_file = os.path.join(self.cache_dir, f"{sample_id}_vertices.pt")
            faces_cache_file = os.path.join(self.cache_dir, f"{sample_id}_faces.pt")
            if os.path.exists(vertices_cache_file) and os.path.exists(faces_cache_file):
                try:
                    if self.debug:
                        print(f"Loading cached mesh for sample {sample_id}")
                    vertices = torch.load(vertices_cache_file)
                    faces = torch.load(faces_cache_file)
                    return vertices, faces
                except Exception as e:
                    print(f"Error loading cached mesh: {e}")
        
        # Generate mesh if not cached
        try:
            # Load the mask volume
            mask_volume = load_mask_volume(mask_dir)
            
            # Create mesh
            mesh = create_mesh_from_mask(
                mask_volume, 
                threshold=0.5, 
                voxel_size=self.voxel_size,
                smoothing_iterations=self.smoothing_iterations
            )
            
            # Convert to tensors
            vertices = torch.tensor(np.asarray(mesh.vertices), dtype=torch.float32)
            faces = torch.tensor(np.asarray(mesh.triangles), dtype=torch.long)
            
            # Cache the mesh
            if vertices_cache_file and faces_cache_file:
                torch.save(vertices, vertices_cache_file)
                torch.save(faces, faces_cache_file)
                if self.debug:
                    print(f"Cached mesh for sample {sample_id}")
            
            return vertices, faces
        
        except Exception as e:
            print(f"Error generating mesh for sample {sample_id}: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty tensors on error
            return torch.zeros((0, 3), dtype=torch.float32), torch.zeros((0, 3), dtype=torch.long)
    
    def __getitem__(self, idx):
        """
        Get a sample by index.
        
        Args:
            idx (int): Index
            
        Returns:
            dict: Sample dictionary with points/vertices/faces, label, and metadata
        """
        try:
            sample = self.samples[idx]
            sample_id = sample['sample_id']
            
            if self.use_mesh:
                # Get mesh data
                vertices, faces = self._load_or_generate_mesh(sample)
                
                metadata = {
                    'sample_id': sample_id,
                    'class_name': sample['class_name'],
                    'num_vertices': vertices.shape[0],
                    'num_faces': faces.shape[0]
                }
                
                return {
                    'vertices': vertices,
                    'faces': faces,
                    'label': sample['class_id'],
                    'metadata': metadata
                }
            else:
                # Get point cloud data
                points = self._load_or_generate_pointcloud(sample)
                
                metadata = {
                    'sample_id': sample_id,
                    'class_name': sample['class_name'],
                    'num_points': points.shape[0]
                }
                
                return {
                    'points': points,
                    'label': sample['class_id'],
                    'metadata': metadata
                }
            
        except Exception as e:
            print(f"Error in __getitem__ for idx={idx}: {e}")
            import traceback
            traceback.print_exc()
            raise


def get_mesh_dataloader(root_dir,
                         batch_size=8,
                         shuffle=True,
                         num_workers=4,
                         class_csv_path=None,
                         sample_ids=None,
                         filter_by_class=None,
                         ignore_unclassified=True,
                         max_points=10000,
                         voxel_size=1.0,
                         sample_percent=100,
                         cache_dir=None,
                         pin_memory=False,
                         use_mesh=True,
                         smoothing_iterations=1,
                         debug=False):
    """
    Create a DataLoader for the mesh/pointcloud dataset.
    
    Args:
        root_dir (str): Root directory containing the samples.
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of worker processes.
        class_csv_path (str, optional): Path to CSV file with class information.
        sample_ids (list, optional): List of sample IDs to include.
        filter_by_class (int or list, optional): Class ID(s) to include.
        ignore_unclassified (bool): Whether to ignore unclassified samples.
        max_points (int): Maximum number of points to include in each point cloud.
        voxel_size (float): Size of each voxel in the point cloud.
        sample_percent (int): Percentage of samples to load per class (1-100).
        cache_dir (str, optional): Directory to cache processed point clouds.
        pin_memory (bool): Whether to pin memory for faster GPU transfer.
        use_mesh (bool): Whether to generate meshes (True) or point clouds (False).
        smoothing_iterations (int): Number of iterations for mesh smoothing (if use_mesh=True).
        debug (bool): Whether to print debug information.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader for the mesh dataset.
    """
    dataset = MeshDataset(
        root_dir=root_dir,
        class_csv_path=class_csv_path,
        sample_ids=sample_ids,
        filter_by_class=filter_by_class,
        ignore_unclassified=ignore_unclassified,
        max_points=max_points,
        voxel_size=voxel_size,
        sample_percent=sample_percent,
        cache_dir=cache_dir,
        use_mesh=use_mesh,
        smoothing_iterations=smoothing_iterations,
        debug=debug
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=pin_memory
    )
    
    return dataloader


if __name__ == "__main__":
    # Example usage
    dataloader = get_mesh_dataloader(
        root_dir="data/nuclei_sample_1a_v1",
        batch_size=4,
        class_csv_path="chromatin_classes_and_samples.csv",
        max_points=5000,
        cache_dir="data/pointclouds_cache",
        use_mesh=True,
        debug=True
    )
    
    # Print the first batch
    for batch in dataloader:
        if 'vertices' in batch:
            print(f"Batch of meshes:")
            print(f"  Vertices: {batch['vertices'].shape}")
            print(f"  Faces: {batch['faces'].shape}")
            print(f"  Labels: {batch['label']}")
            print(f"  Sample IDs: {batch['metadata']['sample_id']}")
        else:
            print(f"Batch of point clouds:")
            print(f"  Points: {batch['points'].shape}")
            print(f"  Point masks: {batch['point_masks'].shape}")
            print(f"  Labels: {batch['label']}")
            print(f"  Sample IDs: {batch['metadata']['sample_id']}")
        break
    
    
    
    
    
    
    
    
    
    
    