import os
import sys
import argparse
import numpy as np
import pandas as pd
import trimesh
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

def analyze_mesh(mesh_path):
    """
    Analyze properties of a mesh
    
    Args:
        mesh_path (str): Path to the mesh file
        
    Returns:
        dict: Dictionary of mesh properties
    """
    try:
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        
        # Basic mesh properties
        properties = {
            'num_vertices': len(mesh.vertices),
            'num_faces': len(mesh.faces),
            'volume': mesh.volume,
            'surface_area': mesh.area,
            'euler_number': mesh.euler_number,
            'is_watertight': mesh.is_watertight,
            'is_winding_consistent': mesh.is_winding_consistent,
            'genus': mesh.genus,
        }
        
        # Calculate bounding box
        bounds = mesh.bounds
        properties['bbox_size_x'] = bounds[1][0] - bounds[0][0]
        properties['bbox_size_y'] = bounds[1][1] - bounds[0][1]
        properties['bbox_size_z'] = bounds[1][2] - bounds[0][2]
        properties['bbox_volume'] = properties['bbox_size_x'] * properties['bbox_size_y'] * properties['bbox_size_z']
        
        # Calculate shape descriptors
        properties['compactness'] = (mesh.area**1.5) / (36 * np.pi * mesh.volume)
        properties['sphericity'] = (36 * np.pi * mesh.volume**2)**(1/3) / mesh.area
        
        # Calculate convex hull properties
        convex_hull = mesh.convex_hull
        properties['convex_hull_volume'] = convex_hull.volume
        properties['convexity'] = convex_hull.volume / mesh.volume if mesh.volume > 0 else 0
        
        # Principal moments of inertia
        inertia = mesh.moment_inertia
        if inertia is not None:
            eigenvalues = np.linalg.eigvalsh(inertia)
            eigenvalues = sorted(eigenvalues, reverse=True)
            properties['moment_1'] = eigenvalues[0]
            properties['moment_2'] = eigenvalues[1]
            properties['moment_3'] = eigenvalues[2]
            
            # Calculate elongation using principal axes
            properties['elongation'] = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
            properties['flatness'] = eigenvalues[1] / eigenvalues[2] if eigenvalues[2] > 0 else 0
        
        return properties
    
    except Exception as e:
        print(f"Error analyzing mesh {mesh_path}: {e}")
        return None

def analyze_point_cloud(pc_path):
    """
    Analyze properties of a point cloud
    
    Args:
        pc_path (str): Path to the point cloud file
        
    Returns:
        dict: Dictionary of point cloud properties
    """
    try:
        # Load the point cloud
        point_cloud = np.load(pc_path)
        
        # Get just the points (first 3 columns)
        points = point_cloud[:, :3] if point_cloud.shape[1] > 3 else point_cloud
        
        # Basic point cloud properties
        properties = {
            'num_points': len(points),
        }
        
        # Calculate bounding box
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        properties['bbox_size_x'] = max_coords[0] - min_coords[0]
        properties['bbox_size_y'] = max_coords[1] - min_coords[1]
        properties['bbox_size_z'] = max_coords[2] - min_coords[2]
        properties['bbox_volume'] = properties['bbox_size_x'] * properties['bbox_size_y'] * properties['bbox_size_z']
        
        # Calculate point cloud centroid
        centroid = np.mean(points, axis=0)
        properties['centroid_x'] = centroid[0]
        properties['centroid_y'] = centroid[1]
        properties['centroid_z'] = centroid[2]
        
        # Calculate average distance from centroid
        distances = np.sqrt(np.sum((points - centroid)**2, axis=1))
        properties['mean_distance_from_centroid'] = np.mean(distances)
        properties['std_distance_from_centroid'] = np.std(distances)
        
        # Calculate covariance matrix and principal components
        centered_points = points - centroid
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        properties['pca_eigenvalue_1'] = eigenvalues[0]
        properties['pca_eigenvalue_2'] = eigenvalues[1]
        properties['pca_eigenvalue_3'] = eigenvalues[2]
        
        # Calculate elongation and flatness
        properties['elongation'] = eigenvalues[0] / eigenvalues[1] if eigenvalues[1] > 0 else 0
        properties['flatness'] = eigenvalues[1] / eigenvalues[2] if eigenvalues[2] > 0 else 0
        
        return properties
    
    except Exception as e:
        print(f"Error analyzing point cloud {pc_path}: {e}")
        return None

def analyze_all_meshes(mesh_dir, pc_dir=None, output_csv=None):
    """
    Analyze all meshes in a directory
    
    Args:
        mesh_dir (str): Directory containing mesh files
        pc_dir (str, optional): Directory containing corresponding point cloud files
        output_csv (str, optional): Path to save results as CSV
        
    Returns:
        pandas.DataFrame: DataFrame containing mesh analysis results
    """
    # Find all mesh files
    mesh_files = sorted(glob.glob(os.path.join(mesh_dir, "*.ply")))
    
    if not mesh_files:
        print(f"No mesh files found in {mesh_dir}")
        return None
    
    print(f"Found {len(mesh_files)} mesh files")
    
    # Analyze each mesh
    results = []
    for mesh_file in tqdm(mesh_files, desc="Analyzing meshes"):
        # Extract sample ID from filename
        sample_id = os.path.basename(mesh_file).split('_')[0]
        
        # Analyze mesh
        mesh_props = analyze_mesh(mesh_file)
        
        if mesh_props:
            result = {'sample_id': sample_id}
            result.update(mesh_props)
            
            # Analyze corresponding point cloud if available
            if pc_dir:
                pc_file = os.path.join(pc_dir, f"{sample_id}_pc.npy")
                if os.path.exists(pc_file):
                    pc_props = analyze_point_cloud(pc_file)
                    if pc_props:
                        # Add prefix to point cloud properties
                        pc_result = {f"pc_{k}": v for k, v in pc_props.items()}
                        result.update(pc_result)
            
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Save to CSV if requested
    if output_csv and len(df) > 0:
        df.to_csv(output_csv, index=False)
        print(f"Results saved to {output_csv}")
    
    return df

def plot_histograms(df, output_dir):
    """
    Plot histograms of mesh properties
    
    Args:
        df (pandas.DataFrame): DataFrame containing mesh analysis results
        output_dir (str): Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Select columns for histograms
    columns = [
        'num_vertices', 'num_faces', 'volume', 'surface_area',
        'compactness', 'sphericity', 'elongation', 'flatness'
    ]
    
    for col in columns:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            plt.hist(df[col].dropna(), bins=30, alpha=0.7)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{col}_histogram.png'))
            plt.close()
    
    # Create scatter plots for relationships
    scatter_pairs = [
        ('volume', 'surface_area'),
        ('compactness', 'sphericity'),
        ('elongation', 'flatness'),
        ('num_vertices', 'num_faces')
    ]
    
    for x_col, y_col in scatter_pairs:
        if x_col in df.columns and y_col in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df[x_col], df[y_col], alpha=0.7)
            plt.title(f'{y_col} vs {x_col}')
            plt.xlabel(x_col)
            plt.ylabel(y_col)
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{y_col}_vs_{x_col}.png'))
            plt.close()
    
    print(f"Plots saved to {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Analyze properties of nuclei meshes")
    parser.add_argument("--mesh_dir", type=str, default=os.path.join(config.RESULTS_DIR, "meshes", "meshes"),
                        help="Directory containing mesh files")
    parser.add_argument("--pc_dir", type=str, default=os.path.join(config.RESULTS_DIR, "meshes", "pointclouds"),
                        help="Directory containing point cloud files")
    parser.add_argument("--output_csv", type=str, default=os.path.join(config.ANALYSIS_OUTPUT_DIR, "mesh_analysis.csv"),
                        help="Path to save analysis results as CSV")
    parser.add_argument("--plot_dir", type=str, default=os.path.join(config.VISUALIZATION_OUTPUT_DIR, "mesh_analysis"),
                        help="Directory to save plots")
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.mesh_dir):
        print(f"Mesh directory not found: {args.mesh_dir}")
        return
    
    # Analyze meshes
    df = analyze_all_meshes(args.mesh_dir, args.pc_dir, args.output_csv)
    
    if df is not None and len(df) > 0:
        # Plot histograms
        plot_histograms(df, args.plot_dir)
        
        # Print summary statistics
        print("\nSummary Statistics:")
        print(df.describe())
    else:
        print("No analysis results to plot")

if __name__ == "__main__":
    main() 