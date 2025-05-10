import os
import sys
import argparse
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from utils.mesh_utils import process_all_samples

def main():
    parser = argparse.ArgumentParser(description="Generate triangle meshes and point clouds for nuclei samples")
    parser.add_argument("--root_dir", type=str, default=config.DATA_ROOT, 
                        help="Root directory containing the nuclei samples")
    parser.add_argument("--output_dir", type=str, default=os.path.join(config.RESULTS_DIR, "meshes"),
                        help="Directory to save the generated meshes and point clouds")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[1.0, 1.0, 1.0],
                        help="Size of each voxel in the output mesh (dz, dy, dx)")
    parser.add_argument("--smooth", type=int, default=10, 
                        help="Number of iterations for Taubin smoothing")
    parser.add_argument("--decimate", type=int, default=5000,
                        help="Target number of faces after decimation")
    parser.add_argument("--points", type=int, default=1024,
                        help="Number of points to sample from each mesh surface")
    parser.add_argument("--sample_ids", type=int, nargs="+", default=None,
                        help="Specific sample IDs to process (optional)")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to process")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Processing samples from {args.root_dir}")
    print(f"Output will be saved to {args.output_dir}")
    print(f"Settings: Smoothing={args.smooth}, Decimation={args.decimate}, Points={args.points}")
    
    # Process all samples
    results = process_all_samples(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        voxel_size=tuple(args.voxel_size),
        smooth_iterations=args.smooth,
        decimate_target=args.decimate,
        sample_points=args.points,
        sample_ids=args.sample_ids,
        max_samples=args.max_samples
    )
    
    # Print summary
    if results:
        print("\nProcessing complete!")
        print(f"Processed {len(results)} samples")
        print(f"Meshes saved to {os.path.join(args.output_dir, 'meshes')}")
        print(f"Point clouds saved to {os.path.join(args.output_dir, 'pointclouds')}")
    else:
        print("\nNo samples were processed. Check your input directory and sample IDs.")

if __name__ == "__main__":
    main() 