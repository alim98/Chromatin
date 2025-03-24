import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataloader.nuclei_dataloader import get_nuclei_dataloader
import config

def create_improved_resize_comparison(sample_id, output_dir=None):
    """
    Create a comparison of different resize methods focusing on the aspect ratio issue.
    
    Args:
        sample_id (str): ID of the sample to process
        output_dir (str): Output directory for visualizations
    """
    if output_dir is None:
        output_dir = os.path.join("results", "visualizations", "improved_resize_comparison")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating improved resize comparison for sample {sample_id}")
    
    # Different configurations to compare
    configs = [
        {
            'name': 'original_issue',
            'target_shape': (80, 80, 80),
            'preserve_resolution': True,
            'high_quality_resize': False,
            'description': 'Original Issue (Center Crop)'
        },
        {
            'name': 'aspect_ratio_maintained',
            'target_shape': (80, 80, 80),
            'preserve_resolution': False,
            'high_quality_resize': True,
            'description': 'Aspect Ratio Maintained'
        },
        {
            'name': 'larger_target',
            'target_shape': (128, 128, 128),
            'preserve_resolution': False,
            'high_quality_resize': True,
            'description': 'Larger Target (128³)'
        }
    ]
    
    # Process with each configuration
    for config_idx, config in enumerate(configs):
        print(f"\nMethod {config_idx+1}/{len(configs)}: {config['name']}")
        
        # Create a dataloader with this sample and config
        dataloader = get_nuclei_dataloader(
            root_dir=os.path.join("data", "nuclei_sample_1a_v1"),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            sample_ids=[sample_id],
            load_volumes=True,
            target_shape=config['target_shape'],
            preserve_resolution=config['preserve_resolution'],
            high_quality_resize=config['high_quality_resize'],
            mask_priority=True,
            debug=True
        )
        
        # Get the dataset to call visualization
        dataset = dataloader.dataset
        
        # Visualize the sample
        method_name = config['name']
        viz_output_dir = os.path.join(output_dir, method_name)
        os.makedirs(viz_output_dir, exist_ok=True)
        
        try:
            # Use dataset's built-in visualization
            original_gif, resized_gif = dataset.visualize_volume(
                idx=0,
                output_dir=viz_output_dir,
                max_frames=80
            )
            
            print(f"✓ Visualization created for {method_name}")
            print(f"  - Original: {original_gif}")
            print(f"  - Resized: {resized_gif}")
            
        except Exception as e:
            print(f"✗ Error creating visualization for {method_name}: {e}")
    
    # Create a simple HTML to view the results
    create_html_report(sample_id, configs, output_dir)
    
    print(f"\nComparison completed. Results in {output_dir}")
    return output_dir

def create_html_report(sample_id, configs, output_dir):
    """Create a simple HTML report to view the results."""
    html_path = os.path.join(output_dir, "improved_resize_comparison.html")
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Improved Resize Comparison - Sample {sample_id}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            h1, h2 {{ color: #333; }}
            .comparison-container {{ margin-bottom: 40px; }}
            .method-container {{ 
                margin-bottom: 30px;
                background-color: white;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                padding: 20px;
            }}
            .method-header {{ 
                display: flex;
                justify-content: space-between;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 20px;
            }}
            .method-title {{ font-size: 1.5em; font-weight: bold; }}
            .method-details {{ color: #666; }}
            .visualization {{ 
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
            }}
            .viz-container {{ 
                flex: 1;
                min-width: 300px;
                text-align: center;
            }}
            .viz-container img {{ 
                max-width: 100%; 
                border: 1px solid #ddd;
                border-radius: 4px;
            }}
            .viz-label {{ font-weight: bold; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <h1>Improved Resize Comparison</h1>
        <p>Sample ID: {sample_id}</p>
        
        <div class="comparison-container">
    """
    
    for config in configs:
        method_name = config['name']
        description = config['description']
        target_shape = 'x'.join(str(x) for x in config['target_shape'])
        
        # Construct paths to the GIFs
        original_gif_path = os.path.join(method_name, f"{sample_id}_original.gif")
        resized_gif_path = os.path.join(method_name, f"{sample_id}_resized.gif")
        
        html_content += f"""
            <div class="method-container">
                <div class="method-header">
                    <div class="method-title">{description}</div>
                    <div class="method-details">
                        Target Shape: {target_shape} | 
                        High Quality: {"Yes" if config['high_quality_resize'] else "No"} | 
                        Preserve Resolution: {"Yes" if config['preserve_resolution'] else "No"}
                    </div>
                </div>
                
                <div class="visualization">
                    <div class="viz-container">
                        <div class="viz-label">Original Volume</div>
                        <img src="{original_gif_path}" alt="Original Volume">
                    </div>
                    
                    <div class="viz-container">
                        <div class="viz-label">Resized Volume</div>
                        <img src="{resized_gif_path}" alt="Resized Volume">
                    </div>
                </div>
            </div>
        """
    
    html_content += """
        </div>
    </body>
    </html>
    """
    
    with open(html_path, "w") as f:
        f.write(html_content)
    
    print(f"HTML report created: {html_path}")
    return html_path

def main():
    parser = argparse.ArgumentParser(description='Compare improved resize methods')
    parser.add_argument('--sample_id', type=str, default='2160',
                        help='Sample ID to process')
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.join("results", "visualizations", "improved_resize_comparison"),
                        help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Run comparison
    create_improved_resize_comparison(
        sample_id=args.sample_id,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main() 