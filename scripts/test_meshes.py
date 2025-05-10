import os
import sys
import argparse
import numpy as np
import trimesh
import glob
from tqdm import tqdm
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.mesh_dataloader_v2 import get_mesh_dataloader_v2

def validate_mesh_file(mesh_path):
    """
    Load and validate a mesh file
    
    Args:
        mesh_path (str): Path to the mesh file
        
    Returns:
        dict: Dictionary with validation results
    """
    results = {
        'path': mesh_path,
        'filename': os.path.basename(mesh_path),
        'size_bytes': os.path.getsize(mesh_path),
        'loadable': False,
        'num_vertices': 0,
        'num_faces': 0,
        'is_watertight': False,
        'has_vertex_normals': False,
        'has_face_normals': False,
        'errors': []
    }
    
    try:
        # Try to load the mesh
        mesh = trimesh.load(mesh_path)
        results['loadable'] = True
        
        # Basic properties
        results['num_vertices'] = len(mesh.vertices)
        results['num_faces'] = len(mesh.faces)
        results['is_watertight'] = mesh.is_watertight
        results['is_winding_consistent'] = mesh.is_winding_consistent
        results['has_vertex_normals'] = mesh.vertex_normals is not None and len(mesh.vertex_normals) > 0
        results['has_face_normals'] = mesh.face_normals is not None and len(mesh.face_normals) > 0
        
    except Exception as e:
        results['errors'].append(str(e))
    
    return results

def validate_point_cloud_file(pc_path):
    """
    Load and validate a point cloud file
    
    Args:
        pc_path (str): Path to the point cloud file
        
    Returns:
        dict: Dictionary with validation results
    """
    results = {
        'path': pc_path,
        'filename': os.path.basename(pc_path),
        'size_bytes': os.path.getsize(pc_path),
        'loadable': False,
        'num_points': 0,
        'has_normals': False,
        'errors': []
    }
    
    try:
        # Try to load the point cloud
        pc = np.load(pc_path)
        results['loadable'] = True
        
        # Basic properties
        results['num_points'] = len(pc)
        results['shape'] = str(pc.shape)
        results['has_normals'] = pc.shape[1] > 3  # Check if point cloud includes normals
        
    except Exception as e:
        results['errors'].append(str(e))
    
    return results

def test_mesh_files(mesh_dir, pc_dir=None, output_csv=None):
    """
    Test all mesh and point cloud files in the specified directories
    
    Args:
        mesh_dir (str): Directory containing mesh files
        pc_dir (str, optional): Directory containing point cloud files
        output_csv (str, optional): Path to save results as CSV
        
    Returns:
        pandas.DataFrame: DataFrame with test results
    """
    # Find all mesh files
    mesh_files = glob.glob(os.path.join(mesh_dir, "*.ply"))
    
    if not mesh_files:
        print(f"No mesh files found in {mesh_dir}")
        return None
    
    print(f"Testing {len(mesh_files)} mesh files...")
    
    # Test each mesh
    mesh_results = []
    for mesh_file in tqdm(mesh_files, desc="Testing meshes"):
        result = validate_mesh_file(mesh_file)
        mesh_results.append(result)
    
    mesh_df = pd.DataFrame(mesh_results)
    
    # Test point clouds if available
    pc_results = []
    if pc_dir and os.path.exists(pc_dir):
        pc_files = glob.glob(os.path.join(pc_dir, "*.npy"))
        
        if pc_files:
            print(f"Testing {len(pc_files)} point cloud files...")
            
            for pc_file in tqdm(pc_files, desc="Testing point clouds"):
                result = validate_point_cloud_file(pc_file)
                pc_results.append(result)
    
    pc_df = pd.DataFrame(pc_results) if pc_results else None
    
    # Save results to CSV if requested
    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        mesh_output = output_csv.replace('.csv', '_meshes.csv')
        mesh_df.to_csv(mesh_output, index=False)
        print(f"Mesh validation results saved to {mesh_output}")
        
        if pc_df is not None:
            pc_output = output_csv.replace('.csv', '_pointclouds.csv')
            pc_df.to_csv(pc_output, index=False)
            print(f"Point cloud validation results saved to {pc_output}")
    
    # Print summary
    print("\nMesh Validation Summary:")
    print(f"Total meshes: {len(mesh_df)}")
    print(f"Loadable: {mesh_df['loadable'].sum()} ({mesh_df['loadable'].sum() / len(mesh_df) * 100:.1f}%)")
    print(f"Watertight: {mesh_df['is_watertight'].sum()} ({mesh_df['is_watertight'].sum() / len(mesh_df) * 100:.1f}%)")
    
    if pc_df is not None:
        print("\nPoint Cloud Validation Summary:")
        print(f"Total point clouds: {len(pc_df)}")
        print(f"Loadable: {pc_df['loadable'].sum()} ({pc_df['loadable'].sum() / len(pc_df) * 100:.1f}%)")
    
    return mesh_df, pc_df

def test_mesh_dataloader(root_dir, class_csv_path=None, precomputed_dir=None, cache_dir=None, num_samples=5):
    """
    Test the mesh dataloader by loading a few samples
    
    Args:
        root_dir (str): Root directory containing the samples
        class_csv_path (str, optional): Path to CSV file with class information
        precomputed_dir (str, optional): Directory with pre-processed meshes
        cache_dir (str, optional): Directory to cache processed meshes
        num_samples (int): Number of samples to test
    """
    print(f"\nTesting Mesh Dataloader with {num_samples} samples...")
    
    # Test with mesh return type
    print("\nTesting mesh return type:")
    mesh_loader = get_mesh_dataloader_v2(
        root_dir=root_dir,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        class_csv_path=class_csv_path,
        precomputed_dir=precomputed_dir,
        generate_on_load=True,
        return_type='mesh',
        cache_dir=cache_dir,
        sample_percent=10,  # Use only 10% of samples for faster testing
        debug=True
    )
    
    if mesh_loader is None or len(mesh_loader.dataset) == 0:
        print("No samples found in the dataloader!")
        return
    
    print(f"Dataloader contains {len(mesh_loader.dataset)} samples")
    
    # Try to load a few batches
    print("Loading sample batches...")
    for i, batch in enumerate(mesh_loader):
        if i >= num_samples:
            break
            
        print(f"\nBatch {i+1}:")
        print(f"  Vertices: {batch['vertices'].shape}")
        print(f"  Faces: {batch['faces'].shape}")
        print(f"  Vertex masks: {batch['vertex_masks'].shape}")
        print(f"  Face masks: {batch['face_masks'].shape}")
        
        if batch['metadata'] and 'class_id' in batch['metadata']:
            print(f"  Class IDs: {batch['metadata']['class_id']}")
    
    # Test with point cloud return type
    print("\nTesting point cloud return type:")
    pc_loader = get_mesh_dataloader_v2(
        root_dir=root_dir,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        class_csv_path=class_csv_path,
        precomputed_dir=precomputed_dir,
        generate_on_load=True,
        return_type='pointcloud',
        num_points=1024,
        cache_dir=cache_dir,
        sample_percent=10,  # Use only 10% of samples for faster testing
        debug=True
    )
    
    # Try to load a few batches
    print("Loading sample batches...")
    for i, batch in enumerate(pc_loader):
        if i >= num_samples:
            break
            
        print(f"\nBatch {i+1}:")
        print(f"  Points: {batch['points'].shape}")
        
        if batch['metadata'] and 'class_id' in batch['metadata']:
            print(f"  Class IDs: {batch['metadata']['class_id']}")

def main():
    parser = argparse.ArgumentParser(description="Test and validate mesh files and dataloader")
    parser.add_argument("--mesh_dir", type=str, default=os.path.join(config.RESULTS_DIR, "meshes", "meshes"),
                       help="Directory containing mesh files")
    parser.add_argument("--pc_dir", type=str, default=os.path.join(config.RESULTS_DIR, "meshes", "pointclouds"),
                       help="Directory containing point cloud files")
    parser.add_argument("--output_csv", type=str, default=os.path.join(config.ANALYSIS_OUTPUT_DIR, "mesh_validation.csv"),
                       help="Path to save validation results as CSV")
    parser.add_argument("--test_dataloader", action="store_true", 
                       help="Test the mesh dataloader")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                       help="Directory with pre-processed meshes")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory to cache processed meshes")
    
    args = parser.parse_args()
    
    # Test mesh and point cloud files
    if os.path.exists(args.mesh_dir):
        mesh_df, pc_df = test_mesh_files(args.mesh_dir, args.pc_dir, args.output_csv)
    else:
        print(f"Mesh directory not found: {args.mesh_dir}")
    
    # Test the dataloader if requested
    if args.test_dataloader:
        test_mesh_dataloader(
            root_dir=config.DATA_ROOT,
            class_csv_path=config.CLASS_CSV_PATH,
            precomputed_dir=args.precomputed_dir or os.path.join(config.RESULTS_DIR, "meshes"),
            cache_dir=args.cache_dir
        )

if __name__ == "__main__":
    main() 