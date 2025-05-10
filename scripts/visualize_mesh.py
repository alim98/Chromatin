import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import torch

# Add parent directory to path to import from dataloader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataloader.mesh_dataloader import get_mesh_dataloader
from utils.pointcloud import create_mesh_from_mask, load_mask_volume

def visualize_mesh_sample(sample_id, root_dir, class_csv_path, output_path=None, smoothing_iterations=1):
    """
    Generate and visualize a mesh for a specific sample.
    
    Args:
        sample_id (int or str): ID of the sample to visualize
        root_dir (str): Root directory containing the samples
        class_csv_path (str): Path to CSV file with class information
        output_path (str, optional): Path to save the mesh file (if None, just visualize)
        smoothing_iterations (int): Number of iterations for mesh smoothing
    """
    # Create a dataloader with just the sample we want
    dataloader = get_mesh_dataloader(
        root_dir=root_dir,
        class_csv_path=class_csv_path,
        sample_ids=[sample_id],
        max_points=10000,  # Not used for mesh
        cache_dir="data/pointclouds_cache",
        use_mesh=True,
        smoothing_iterations=smoothing_iterations,
        batch_size=1,
        num_workers=0,
        debug=True
    )
    
    # Get the first batch (should only contain our sample)
    batch = next(iter(dataloader))
    
    if 'vertices' not in batch:
        print(f"Error: No mesh data found for sample {sample_id}")
        return
    
    # Get the mesh data
    vertices = batch['vertices'][0]
    faces = batch['faces'][0]
    vertex_mask = batch['vertex_masks'][0]
    face_mask = batch['face_masks'][0]
    
    # Extract only valid vertices and faces
    valid_vertices = vertices[vertex_mask.bool()].cpu().numpy()
    valid_faces = faces[face_mask.bool()].cpu().numpy()
    
    # Create an Open3D mesh for visualization
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(valid_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(valid_faces)
    
    # Assign colors
    min_coords = np.min(valid_vertices, axis=0)
    max_coords = np.max(valid_vertices, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1
    colors = (valid_vertices - min_coords) / range_coords
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    
    # Compute normals for rendering
    mesh.compute_vertex_normals()
    
    # Save the mesh if requested
    if output_path:
        o3d.io.write_triangle_mesh(output_path, mesh)
        print(f"Mesh saved to {output_path}")
    
    # Visualize
    print(f"Visualizing mesh for sample {sample_id}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.triangles)}")
    o3d.visualization.draw_geometries([mesh], window_name=f"Sample {sample_id} Mesh")

def compare_pointcloud_and_mesh(sample_id, root_dir, class_csv_path):
    """
    Compare a point cloud and mesh visualization for the same sample.
    
    Args:
        sample_id (int or str): ID of the sample to visualize
        root_dir (str): Root directory containing the samples
        class_csv_path (str): Path to CSV file with class information
    """
    # Get the point cloud
    pc_loader = get_mesh_dataloader(
        root_dir=root_dir,
        class_csv_path=class_csv_path,
        sample_ids=[sample_id],
        max_points=10000,
        cache_dir="data/pointclouds_cache",
        use_mesh=False,
        batch_size=1,
        num_workers=0,
        debug=True
    )
    
    # Get the mesh
    mesh_loader = get_mesh_dataloader(
        root_dir=root_dir,
        class_csv_path=class_csv_path,
        sample_ids=[sample_id],
        max_points=10000,  # Not used for mesh
        cache_dir="data/pointclouds_cache",
        use_mesh=True,
        batch_size=1,
        num_workers=0,
        debug=True
    )
    
    # Get first batch from each loader
    pc_batch = next(iter(pc_loader))
    mesh_batch = next(iter(mesh_loader))
    
    # Extract point cloud
    points = pc_batch['points'][0]
    point_mask = pc_batch['point_masks'][0]
    valid_points = points[point_mask.bool()].cpu().numpy()
    
    # Extract mesh
    vertices = mesh_batch['vertices'][0]
    faces = mesh_batch['faces'][0]
    vertex_mask = mesh_batch['vertex_masks'][0]
    face_mask = mesh_batch['face_masks'][0]
    valid_vertices = vertices[vertex_mask.bool()].cpu().numpy()
    valid_faces = faces[face_mask.bool()].cpu().numpy()
    
    # Create Open3D objects
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)
    
    # Assign colors to point cloud
    min_coords = np.min(valid_points, axis=0)
    max_coords = np.max(valid_points, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1
    colors = (valid_points - min_coords) / range_coords
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(valid_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(valid_faces)
    
    # Assign colors to mesh
    min_coords = np.min(valid_vertices, axis=0)
    max_coords = np.max(valid_vertices, axis=0)
    range_coords = max_coords - min_coords
    range_coords[range_coords == 0] = 1
    colors = (valid_vertices - min_coords) / range_coords
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    
    # Visualize point cloud
    print(f"Visualizing point cloud for sample {sample_id}")
    print(f"Points: {len(pcd.points)}")
    o3d.visualization.draw_geometries([pcd], window_name=f"Sample {sample_id} Point Cloud")
    
    # Visualize mesh
    print(f"Visualizing mesh for sample {sample_id}")
    print(f"Vertices: {len(mesh.vertices)}, Faces: {len(mesh.triangles)}")
    o3d.visualization.draw_geometries([mesh], window_name=f"Sample {sample_id} Mesh")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize mesh data from nuclei samples")
    parser.add_argument("--sample_id", type=int, default=1, help="ID of the sample to visualize")
    parser.add_argument("--root_dir", type=str, default="data/nuclei_sample_1a_v1", 
                        help="Root directory containing the samples")
    parser.add_argument("--class_csv", type=str, default="chromatin_classes_and_samples.csv",
                        help="Path to CSV file with class information")
    parser.add_argument("--output", type=str, default=None, 
                        help="Path to save the mesh (optional)")
    parser.add_argument("--smooth", type=int, default=1,
                        help="Number of smoothing iterations (default: 1)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare point cloud and mesh visualization")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_pointcloud_and_mesh(args.sample_id, args.root_dir, args.class_csv)
    else:
        visualize_mesh_sample(args.sample_id, args.root_dir, args.class_csv, 
                              args.output, args.smooth) 