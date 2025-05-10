import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent directory to path to import from dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader.mesh_dataloader import get_mesh_dataloader

def test_point_cloud_loader():
    """Test the point cloud dataloader"""
    print("Testing point cloud dataloader...")
    
    # Create a dataloader for point clouds
    loader = get_mesh_dataloader(
        root_dir="data/nuclei_sample_1a_v1",
        class_csv_path="chromatin_classes_and_samples.csv",
        use_mesh=False,  # Use point clouds
        max_points=5000,
        cache_dir="data/pointclouds_cache",
        batch_size=2,
        num_workers=0,
        debug=True
    )
    
    # Get the first batch
    try:
        batch = next(iter(loader))
        
        # Print batch information
        print(f"Batch size: {len(batch['label'])}")
        print(f"Points shape: {batch['points'].shape}")
        print(f"Point masks shape: {batch['point_masks'].shape}")
        print(f"Sample IDs: {batch['metadata']['sample_id']}")
        
        # Extract valid points from first sample
        points = batch['points'][0]
        point_mask = batch['point_masks'][0]
        valid_points = points[point_mask.bool()].cpu().numpy()
        
        print(f"Valid points: {valid_points.shape[0]}")
        
        return True
    except Exception as e:
        print(f"Error testing point cloud loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mesh_loader():
    """Test the mesh dataloader"""
    print("\nTesting mesh dataloader...")
    
    # Create a dataloader for meshes
    loader = get_mesh_dataloader(
        root_dir="data/nuclei_sample_1a_v1",
        class_csv_path="chromatin_classes_and_samples.csv",
        use_mesh=True,  # Use meshes
        smoothing_iterations=1,
        cache_dir="data/pointclouds_cache",
        batch_size=2,
        num_workers=0,
        debug=True
    )
    
    # Get the first batch
    try:
        batch = next(iter(loader))
        
        # Print batch information
        print(f"Batch size: {len(batch['label'])}")
        
        if 'vertices' in batch:
            print(f"Vertices shape: {batch['vertices'].shape}")
            print(f"Faces shape: {batch['faces'].shape}")
            print(f"Vertex masks shape: {batch['vertex_masks'].shape}")
            print(f"Face masks shape: {batch['face_masks'].shape}")
            print(f"Sample IDs: {batch['metadata']['sample_id']}")
            
            # Extract valid vertices and faces from first sample
            vertices = batch['vertices'][0]
            faces = batch['faces'][0]
            vertex_mask = batch['vertex_masks'][0]
            face_mask = batch['face_masks'][0]
            
            valid_vertices = vertices[vertex_mask.bool()].cpu().numpy()
            valid_faces = faces[face_mask.bool()].cpu().numpy()
            
            print(f"Valid vertices: {valid_vertices.shape[0]}")
            print(f"Valid faces: {valid_faces.shape[0]}")
            
            return True
        else:
            print("Error: No vertices or faces in batch")
            return False
    except Exception as e:
        print(f"Error testing mesh loader: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test both loaders
    pc_success = test_point_cloud_loader()
    mesh_success = test_mesh_loader()
    
    # Print summary
    print("\nSummary:")
    print(f"Point cloud loader: {'SUCCESS' if pc_success else 'FAILED'}")
    print(f"Mesh loader: {'SUCCESS' if mesh_success else 'FAILED'}")
    
    if pc_success and mesh_success:
        print("\nBoth dataloaders are working correctly!")
    else:
        print("\nSome tests failed, please check the error messages above.") 