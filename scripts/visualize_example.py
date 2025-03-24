import os
import sys
import argparse
import torch
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.nuclei_dataloader import get_nuclei_dataloader
from scripts.visualize import NucleiVisualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize nuclei data from dataloader')
    parser.add_argument('--data_dir', type=str, default=config.DATA_ROOT, 
                        help='Path to nuclei dataset')
    parser.add_argument('--class_csv', type=str, default=config.CLASS_CSV_PATH,
                        help='Path to class CSV file')
    parser.add_argument('--output_dir', type=str, default=config.VISUALIZATION_OUTPUT_DIR,
                        help='Directory to save visualizations')
    parser.add_argument('--mode', type=str, choices=['2d', '3d'], default='2d',
                        help='Visualization mode: 2d (slices) or 3d (volumes)')
    parser.add_argument('--class_id', type=int, nargs='+', default=None,
                        help='Filter by class ID(s). If not specified, all classes will be included.')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for the dataloader')
    parser.add_argument('--num_samples', type=int, default=5,
                        help='Number of samples to visualize')
    parser.add_argument('--slice_range', type=int, nargs=2, default=None,
                        help='Range of slice numbers to include (for 2D mode only)')
    parser.add_argument('--show', action='store_true',
                        help='Show visualizations (in addition to saving)')
    parser.add_argument('--max_crops', type=int, default=8,
                        help='Maximum number of crops per volume')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create transform
    transform = transforms.Compose([
        transforms.ToTensor()
        # Removed normalization for visualization
    ])
    
    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Create dataloader based on mode
    if args.mode == '2d':
        # 2D slices mode
        dataloader = get_nuclei_dataloader(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            transform=transform,
            mask_transform=mask_transform,
            class_csv_path=args.class_csv,
            filter_by_class=args.class_id,
            slice_range=args.slice_range,
            return_paths=True,
            load_volumes=False
        )
        
        print(f"Created 2D dataloader with {len(dataloader.dataset)} samples")
        
    else:
        # 3D volumes mode
        dataloader = get_nuclei_dataloader(
            root_dir=args.data_dir,
            batch_size=args.batch_size,
            transform=transform,
            mask_transform=mask_transform,
            class_csv_path=args.class_csv,
            filter_by_class=args.class_id,
            return_paths=True,
            load_volumes=True,
            max_crops_per_volume=args.max_crops
        )
        
        print(f"Created 3D dataloader with {len(dataloader.dataset)} samples")
    
    # Create visualizer
    visualizer = NucleiVisualizer(output_dir=args.output_dir, cmap='gray')
    
    # Visualize samples
    num_visualized = 0
    
    for batch_idx, batch in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}")
        
        if args.mode == '2d':
            # Visualize 2D slices
            visualizer.visualize_batch(batch, max_samples=args.batch_size, show=args.show)
            
        elif args.mode == '3d':
            # Visualize 3D volumes
            for i in range(min(args.batch_size, len(batch['volume']))):
                # Extract the i-th sample
                sample = {}
                for key in batch.keys():
                    if key == 'metadata':
                        sample[key] = {k: batch[key][k][i] for k in batch[key]}
                    else:
                        sample[key] = batch[key][i]
                
                # Visualize middle slice from volume
                visualizer.visualize_slice(sample, show=args.show)
                
                # Create animation for z-axis slicing
                sample_id = sample['metadata'].get('sample_id', f'sample_{i}')
                save_path = os.path.join(args.output_dir, f"{sample_id}_volume_animation.gif")
                visualizer.visualize_volume(sample, save_path=save_path, show=args.show,
                                            axis='z', frames=20)
                    
        # Count the number of samples visualized in this batch
        batch_size = len(batch['volume' if 'volume' in batch else 'image'])
        num_visualized += batch_size
        
        # Break if we've visualized enough samples
        if num_visualized >= args.num_samples:
            break
    
    print(f"Finished visualizing {num_visualized} samples. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main() 