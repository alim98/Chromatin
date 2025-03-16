#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Nuclei Visualization Module

This module provides classes and functions to visualize nuclei data, including:
- Single slice visualization
- Multi-slice visualization
- GIF creation from multiple slices
- Overlay masks on raw images
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio
from PIL import Image
import torch
import matplotlib.animation as animation
from IPython.display import HTML, display
import glob
from tqdm import tqdm  # Import tqdm for progress bars
import gc  # Import garbage collector

# Add parent directory to path to import from dataloader and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from dataloader.nuclei_dataloader import get_nuclei_dataloader
import config

# TODO: Fix bug in GIF creation functionality - currently only uses 5 slices 
# instead of all available slices when creating GIFs. This limits the 
# visualization quality and completeness of 3D nuclei structures.

class VolumeGifCreator:
    """
    Specialized class for creating GIFs from 3D volumes.
    
    This class is designed to efficiently convert a series of 2D slices
    from a sample folder into an animated GIF that visualizes the 3D structure.
    """
    def __init__(self, output_dir='results/visualizations', batch_size=5, 
                 max_frames=80, dpi=80, figsize=(6, 6)):
        """
        Initialize the VolumeGifCreator.
        
        Args:
            output_dir (str): Directory to save GIFs and temporary files
            batch_size (int): Number of frames to process at once (to manage memory)
            max_frames (int): Maximum number of frames to include in GIFs
            dpi (int): DPI for saved frames (lower values use less memory)
            figsize (tuple): Figure size for frames (smaller uses less memory)
        """
        self.output_dir = output_dir
        self.batch_size = batch_size
        self.max_frames = max_frames
        self.dpi = dpi
        self.figsize = figsize
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def create_gif_from_folder(self, folder_path, output_filename=None, 
                               duration=200, include_masks=True, mask_alpha=0.3):
        """
        Create a GIF from a folder containing 2D slices.
        
        Args:
            folder_path (str): Path to folder containing 2D slices
            output_filename (str, optional): Name for output GIF file
            duration (int): Duration of each frame in milliseconds
            include_masks (bool): Whether to include masks in the visualization
            mask_alpha (float): Alpha/transparency for mask overlays
            
        Returns:
            str: Path to the created GIF file
        """
        print(f"\n=== Creating GIF from 3D Volume ===")
        print(f"Source folder: {folder_path}")
        
        # Auto-detect structure
        sample_id = os.path.basename(folder_path)
        if not output_filename:
            output_filename = f"volume_{sample_id}.gif"
            
        # Check for raw/mask structure
        raw_dir = os.path.join(folder_path, 'raw')
        mask_dir = os.path.join(folder_path, 'mask')
        
        if os.path.exists(raw_dir) and os.path.isdir(raw_dir):
            print(f"Found raw images directory: {raw_dir}")
            image_dir = raw_dir
            
            if include_masks and os.path.exists(mask_dir) and os.path.isdir(mask_dir):
                print(f"Found mask directory: {mask_dir}")
            else:
                include_masks = False
                print("No mask directory found or masks disabled")
        else:
            # Assume the folder itself contains the images
            image_dir = folder_path
            include_masks = False
            print(f"Using folder directly for images: {image_dir}")
            
        # Find all image files
        image_files = self._find_image_files(image_dir)
        if not image_files:
            print(f"Error: No image files found in {image_dir}")
            return None
            
        print(f"Found {len(image_files)} image files")
        
        # Find matching mask files if needed
        mask_files = []
        if include_masks:
            mask_files = self._find_matching_masks(image_files, mask_dir)
            if not mask_files:
                print("Warning: No matching mask files found. Continuing without masks.")
                include_masks = False
            else:
                print(f"Found {len(mask_files)} matching mask files")
                
        # Downsample if needed
        if len(image_files) > self.max_frames:
            print(f"Downsampling from {len(image_files)} to {self.max_frames} frames...")
            indices = np.linspace(0, len(image_files) - 1, self.max_frames, dtype=int)
            image_files = [image_files[i] for i in indices]
            if include_masks:
                mask_files = [mask_files[i] for i in indices]
            print(f"Downsampled to {len(image_files)} frames")
            
        # Create GIF
        return self._create_gif(image_files, mask_files if include_masks else None, 
                               output_filename, duration, mask_alpha)
    
    def _find_image_files(self, directory):
        """Find all image files in a directory and sort them."""
        import glob
        
        # Look for common image file extensions
        extensions = ['*.tif', '*.tiff', '*.png', '*.jpg', '*.jpeg']
        image_files = []
        
        for ext in extensions:
            found_files = glob.glob(os.path.join(directory, ext))
            image_files.extend(found_files)
            
        # Sort the files to ensure correct sequence
        return sorted(image_files)
    
    def _find_matching_masks(self, image_files, mask_dir):
        """Find mask files that match the image files."""
        mask_files = []
        
        for img_file in image_files:
            base_name = os.path.basename(img_file)
            mask_file = os.path.join(mask_dir, base_name)
            
            if os.path.exists(mask_file):
                mask_files.append(mask_file)
            else:
                # Try without leading zeros
                name_without_ext = os.path.splitext(base_name)[0]
                name_without_zeros = name_without_ext.lstrip('0')
                
                # Look for files with matching numeric value
                for f in os.listdir(mask_dir):
                    f_name_without_ext = os.path.splitext(f)[0]
                    if f_name_without_ext.lstrip('0') == name_without_zeros:
                        mask_files.append(os.path.join(mask_dir, f))
                        break
                else:
                    # No match found
                    return []
                    
        return mask_files
    
    def _create_gif(self, image_files, mask_files, output_filename, duration, mask_alpha):
        """Create a GIF from the image files and mask files."""
        import matplotlib.pyplot as plt
        from PIL import Image
        import numpy as np
        
        # Create temporary directory for frames
        tmp_dir = os.path.join(self.output_dir, 'tmp_frames')
        os.makedirs(tmp_dir, exist_ok=True)
        
        all_frame_paths = []
        
        try:
            # Process frames in batches
            num_batches = (len(image_files) + self.batch_size - 1) // self.batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(image_files))
                
                print(f"Processing batch {batch_idx+1}/{num_batches} (frames {start_idx+1}-{end_idx})...")
                
                # Process current batch
                for i in tqdm(range(start_idx, end_idx), desc=f"Creating frames (batch {batch_idx+1})"):
                    # Load image
                    img = self._load_image(image_files[i])
                    
                    # Load mask if available
                    mask = None
                    if mask_files and i < len(mask_files):
                        mask = self._load_image(mask_files[i])
                    
                    # Create figure
                    plt.figure(figsize=self.figsize, dpi=self.dpi)
                    
                    # Display image
                    plt.imshow(img, cmap='gray')
                    
                    # Overlay mask if available
                    if mask is not None:
                        cmap = plt.cm.jet
                        mask_rgba = cmap(mask)
                        mask_rgba[..., 3] = mask_alpha  # Set alpha
                        plt.imshow(mask_rgba, alpha=mask_alpha)
                    
                    # Add slice number
                    slice_num = os.path.splitext(os.path.basename(image_files[i]))[0]
                    plt.title(f"Slice {slice_num}")
                    
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Save frame
                    frame_path = os.path.join(tmp_dir, f'frame_{i:04d}.png')
                    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    all_frame_paths.append(frame_path)
                
                # Force garbage collection between batches
                gc.collect()
            
            # Create GIF using PIL
            gif_path = os.path.join(self.output_dir, output_filename)
            
            if not all_frame_paths:
                print("Error: No frames were created")
                return None
                
            print(f"Creating GIF with {len(all_frame_paths)} frames...")
            
            # Load frames and create GIF
            frames = []
            for frame_path in tqdm(all_frame_paths, desc="Processing frames"):
                try:
                    frame = Image.open(frame_path)
                    frames.append(frame.copy())
                    frame.close()
                except Exception as e:
                    print(f"  Error processing frame {frame_path}: {e}")
            
            if frames:
                print(f"Saving GIF to {gif_path}...")
                with tqdm(total=1, desc="Saving GIF") as pbar:
                    frames[0].save(
                        gif_path,
                        save_all=True,
                        append_images=frames[1:],
                        optimize=False,
                        duration=duration,
                        loop=0
                    )
                    pbar.update(1)
                
                gif_size = os.path.getsize(gif_path) / 1024  # KB
                print(f"Created GIF: {gif_path} ({gif_size:.1f} KB)")
                return gif_path
            else:
                print("Error: No frames could be processed")
                return None
                
        except Exception as e:
            print(f"Error creating GIF: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            # Clean up temporary frames
            print("Cleaning up temporary files...")
            for frame_path in all_frame_paths:
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception:
                    pass
                    
            # Try to remove temporary directory
            try:
                if os.path.exists(tmp_dir):
                    os.rmdir(tmp_dir)
            except Exception:
                pass
                
            print("=== GIF Creation Complete ===\n")
            
    def _load_image(self, image_path):
        """Load an image file and convert to numpy array."""
        from PIL import Image
        import numpy as np
        
        try:
            with Image.open(image_path) as img:
                return np.array(img)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return np.zeros((10, 10))  # Return empty image
            
    def create_volume_montage(self, sample_dirs, output_filename=None, 
                              slices_per_sample=3, include_masks=True):
        """
        Create a montage of selected slices from multiple volume samples.
        
        Args:
            sample_dirs (list): List of sample directory paths
            output_filename (str, optional): Name for output montage file
            slices_per_sample (int): Number of slices to show per sample
            include_masks (bool): Whether to include masks in the visualization
            
        Returns:
            str: Path to the created montage file
        """
        import matplotlib.pyplot as plt
        
        if not output_filename:
            output_filename = "volume_samples_montage.png"
            
        print(f"\n=== Creating Volume Montage ===")
        print(f"Number of samples: {len(sample_dirs)}")
        
        all_images = []
        all_masks = []
        all_titles = []
        
        # Process each sample
        for sample_dir in sample_dirs:
            sample_id = os.path.basename(sample_dir)
            print(f"Processing sample {sample_id}...")
            
            # Check for raw/mask structure
            raw_dir = os.path.join(sample_dir, 'raw')
            mask_dir = os.path.join(sample_dir, 'mask')
            
            if os.path.exists(raw_dir) and os.path.isdir(raw_dir):
                print(f"Found raw images directory: {raw_dir}")
                image_dir = raw_dir
                
                if include_masks and os.path.exists(mask_dir) and os.path.isdir(mask_dir):
                    print(f"Found mask directory: {mask_dir}")
                    use_masks = True
                else:
                    use_masks = False
                    print("No mask directory found or masks disabled")
            else:
                # Assume the folder itself contains the images
                image_dir = sample_dir
                use_masks = False
                print(f"Using folder directly for images: {image_dir}")
                
            # Find all image files
            image_files = self._find_image_files(image_dir)
            if not image_files:
                print(f"Error: No image files found in {image_dir}")
                continue
                
            # Find matching mask files if needed
            mask_files = []
            if use_masks:
                mask_files = self._find_matching_masks(image_files, mask_dir)
                if not mask_files:
                    print("Warning: No matching mask files found. Continuing without masks.")
                    use_masks = False
            
            # Select representative slices
            if len(image_files) <= slices_per_sample:
                selected_indices = range(len(image_files))
            else:
                indices = np.linspace(0, len(image_files) - 1, slices_per_sample, dtype=int)
                selected_indices = indices
                
            # Load selected slices
            for idx in selected_indices:
                # Load image
                img = self._load_image(image_files[idx])
                all_images.append(img)
                
                # Load mask if available
                if use_masks and idx < len(mask_files):
                    mask = self._load_image(mask_files[idx])
                    all_masks.append(mask)
                elif use_masks:
                    all_masks.append(None)
                    
                # Create title
                slice_num = os.path.splitext(os.path.basename(image_files[idx]))[0]
                all_titles.append(f"Sample {sample_id}\nSlice {slice_num}")
                
        # Create montage
        if not all_images:
            print("Error: No images were loaded")
            return None
            
        print(f"Creating montage with {len(all_images)} total slices...")
        
        # Determine grid size
        num_images = len(all_images)
        cols = min(slices_per_sample, num_images)
        rows = (len(sample_dirs) + cols - 1) // cols
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Add images to grid
        for i, ax in enumerate(axes):
            if i < num_images:
                ax.imshow(all_images[i], cmap='gray')
                
                if include_masks and i < len(all_masks) and all_masks[i] is not None:
                    mask = all_masks[i]
                    cmap = plt.cm.jet
                    mask_rgba = cmap(mask)
                    mask_rgba[..., 3] = 0.3  # Set alpha
                    ax.imshow(mask_rgba, alpha=0.3)
                    
                if i < len(all_titles):
                    ax.set_title(all_titles[i])
                    
                ax.axis('off')
            else:
                ax.axis('off')
                
        plt.tight_layout()
        
        # Save montage
        output_path = os.path.join(self.output_dir, output_filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Montage saved to: {output_path}")
        return output_path


class NucleiVisualizer:
    """
    Class for visualizing nuclei images and masks.
    """
    def __init__(self, output_dir=None, cmap='viridis', mask_alpha=0.3, dpi=150):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations.
                If None, uses VISUALIZATION_OUTPUT_DIR from config.
            cmap (str): Colormap to use for masks (default: 'viridis')
            mask_alpha (float): Alpha/transparency for mask overlays (default: 0.3)
            dpi (int): DPI for saved images (default: 150)
        """
        self.output_dir = output_dir if output_dir else config.VISUALIZATION_OUTPUT_DIR
        self.cmap = cmap
        self.mask_alpha = mask_alpha
        self.dpi = dpi
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Track created visualizations
        self.generated_files = []
    
    def _prepare_image(self, image):
        """
        Prepare image for visualization (convert tensor to numpy if needed).
        
        Args:
            image (tensor or numpy array): Image to prepare
            
        Returns:
            numpy array: Prepared image
        """
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
            
            # Handle different tensor shapes
            if image.shape[0] in [1, 3]:  # Channels first (C, H, W)
                image = image.transpose(1, 2, 0)
            
            # If normalized to [-1, 1] range, convert to [0, 1]
            if image.min() < 0:
                image = (image + 1) / 2
        
        # Squeeze single channels to 2D
        if len(image.shape) == 3 and image.shape[-1] == 1:
            image = image.squeeze(-1)
            
        return image
    
    def visualize_single(self, image, mask=None, title=None, filename=None, show=True):
        """
        Visualize a single image and optionally its mask.
        
        Args:
            image (tensor or numpy array): Image to visualize
            mask (tensor or numpy array, optional): Mask to overlay
            title (str, optional): Title for the plot
            filename (str, optional): Filename to save the visualization
            show (bool): Whether to display the plot (default: True)
            
        Returns:
            str: Path to saved file if filename is provided, None otherwise
        """
        image = self._prepare_image(image)
        
        plt.figure(figsize=(8, 8))
        
        # Plot image
        plt.imshow(image, cmap='gray')
        
        # Overlay mask if provided
        if mask is not None:
            mask = self._prepare_image(mask)
            plt.imshow(mask, cmap=self.cmap, alpha=self.mask_alpha)
        
        if title:
            plt.title(title)
            
        plt.axis('off')
        plt.tight_layout()
        
        # Save if filename is provided
        if filename:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.generated_files.append(save_path)
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return save_path if filename else None
    
    def visualize_multiple(self, images, masks=None, titles=None, filename=None, 
                          rows=None, cols=None, figsize=(12, 12), show=True):
        """
        Visualize multiple images in a grid.
        
        Args:
            images (list): List of images to visualize
            masks (list, optional): List of masks to overlay
            titles (list, optional): List of titles for each image
            filename (str, optional): Filename to save the visualization
            rows (int, optional): Number of rows in the grid
            cols (int, optional): Number of columns in the grid
            figsize (tuple): Figure size (default: (12, 12))
            show (bool): Whether to display the plot (default: True)
            
        Returns:
            str: Path to saved file if filename is provided, None otherwise
        """
        n_images = len(images)
        
        # Determine grid layout
        if rows is None and cols is None:
            cols = min(4, n_images)
            rows = (n_images + cols - 1) // cols
        elif rows is None:
            rows = (n_images + cols - 1) // cols
        elif cols is None:
            cols = (n_images + rows - 1) // rows
            
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        if rows * cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, ax in enumerate(axes):
            if i < n_images:
                image = self._prepare_image(images[i])
                ax.imshow(image, cmap='gray')
                
                if masks is not None and i < len(masks) and masks[i] is not None:
                    mask = self._prepare_image(masks[i])
                    ax.imshow(mask, cmap=self.cmap, alpha=self.mask_alpha)
                    
                if titles is not None and i < len(titles):
                    ax.set_title(titles[i])
                    
                ax.axis('off')
            else:
                # Hide unused subplots
                ax.axis('off')
                
        plt.tight_layout()
        
        # Save if filename is provided
        if filename:
            save_path = os.path.join(self.output_dir, filename)
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            self.generated_files.append(save_path)
            
        if show:
            plt.show()
        else:
            plt.close()
            
        return save_path if filename else None
    
    def create_gif(self, images, masks=None, filename='animation.gif', duration=200, max_frames=80):
        """
        Create a GIF animation from a list of images.
        
        Args:
            images (list): List of images to include in the GIF
            masks (list, optional): List of masks to overlay on the images
            filename (str): Name of the output GIF file
            duration (int): Duration of each frame in milliseconds
            max_frames (int): Maximum number of frames to include in the GIF (default: 80)
            
        Returns:
            str: Path to the created GIF file
        """
        import os
        import imageio
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
        
        print("\n=== GIF Creation Details ===")
        print(f"Number of original images: {len(images)}")
        
        # Downsample frames if there are too many
        if len(images) > max_frames:
            print(f"Downsampling from {len(images)} to {max_frames} frames...")
            
            # Calculate indices for downsampling
            indices = np.linspace(0, len(images) - 1, max_frames, dtype=int)
            
            # Select frames and masks at those indices
            images = [images[i] for i in indices]
            if masks is not None:
                masks = [masks[i] for i in indices]
                
            print(f"Downsampled to {len(images)} frames")
        
        print(f"Output filename: {filename}")
        print(f"Duration: {duration}ms per frame")
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join('results', 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(output_dir)}")
        
        # Create temporary directory for frames
        frames_dir = os.path.join(output_dir, 'tmp_frames')
        os.makedirs(frames_dir, exist_ok=True)
        print(f"Temporary frames directory: {os.path.abspath(frames_dir)}")
        
        # Use PIL directly for GIF creation (more memory efficient)
        try:
            from PIL import Image
            
            # Process frames in batches to avoid memory issues
            batch_size = 5  # Process 5 frames at a time
            num_batches = (len(images) + batch_size - 1) // batch_size
            
            all_frame_paths = []
            
            # Create frames in batches
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(images))
                
                print(f"Processing batch {batch_idx+1}/{num_batches} (frames {start_idx+1}-{end_idx})...")
                batch_frame_paths = []
                
                # Process frames in current batch
                for i in tqdm(range(start_idx, end_idx), desc=f"Creating frames (batch {batch_idx+1})"):
                    # Convert tensor to numpy array if needed
                    if isinstance(images[i], torch.Tensor):
                        img_np = images[i].permute(1, 2, 0).cpu().numpy()
                        
                        # Denormalize if needed
                        if img_np.min() < 0:
                            img_np = (img_np + 1) / 2
                            
                        # Ensure values are in [0, 1]
                        img_np = np.clip(img_np, 0, 1)
                    else:
                        img_np = images[i]
                
                    # Create a smaller figure with lower DPI to reduce memory usage
                    plt.figure(figsize=(6, 6), dpi=80)
                    
                    # Display image
                    plt.imshow(img_np.squeeze(), cmap='gray')
                    
                    # Overlay mask if provided
                    if masks is not None and i < len(masks) and masks[i] is not None:
                        mask = masks[i]
                        if isinstance(mask, torch.Tensor):
                            mask_np = mask.permute(1, 2, 0).cpu().numpy()
                        else:
                            mask_np = mask
                            
                        # Create a colormap for the mask
                        cmap = plt.cm.jet
                        mask_rgba = cmap(mask_np.squeeze())
                        mask_rgba[..., 3] = 0.3  # Set alpha
                        
                        plt.imshow(mask_rgba, alpha=0.5)
                    
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Save frame
                    frame_path = os.path.join(frames_dir, f'frame_{i:04d}.png')
                    plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    
                    batch_frame_paths.append(frame_path)
                    all_frame_paths.append(frame_path)
                
                # Force garbage collection between batches
                gc.collect()
            
            # Check if frames were created
            if not all_frame_paths:
                print("Error: No frames were created for the GIF")
                return None
                
            # Create GIF using PIL (more memory efficient)
            gif_path = os.path.join(output_dir, filename)
            print(f"Creating GIF at: {gif_path}")
            
            # Open first image to get dimensions
            with Image.open(all_frame_paths[0]) as img:
                gif_frames = []
                
                # Load frames one by one
                print("Processing frames for GIF...")
                for frame_path in tqdm(all_frame_paths, desc="Processing frames"):
                    try:
                        frame = Image.open(frame_path)
                        # Resize to reduce memory usage and file size
                        # frame = frame.resize((frame.width // 2, frame.height // 2), Image.LANCZOS)
                        gif_frames.append(frame.copy())
                        frame.close()
                    except Exception as e:
                        print(f"  Error processing frame {frame_path}: {e}")
                
                # Save the GIF
                if gif_frames:
                    print(f"Saving GIF with {len(gif_frames)} frames...")
                    with tqdm(total=1, desc="Saving GIF") as pbar:
                        gif_frames[0].save(
                            gif_path,
                            save_all=True,
                            append_images=gif_frames[1:],
                            optimize=False,  # Set to True for smaller file, but slower
                            duration=duration,
                            loop=0
                        )
                        pbar.update(1)
                        
                    if os.path.exists(gif_path):
                        print(f"Created GIF: {gif_path} - Size: {os.path.getsize(gif_path)/1024:.1f} KB")
                    else:
                        print(f"Error: GIF file was not created at {gif_path}")
                else:
                    print("Error: No frames could be processed for the GIF")
                    return None
            
            # Clean up temporary frames
            print("Cleaning up temporary frames...")
            for frame_path in tqdm(all_frame_paths, desc="Cleaning up"):
                try:
                    if os.path.exists(frame_path):
                        os.remove(frame_path)
                except Exception as e:
                    print(f"  Warning: Could not remove temporary frame {frame_path}: {e}")
                    
            return gif_path
            
        except Exception as e:
            print(f"Error creating GIF: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            print("=== GIF Creation Complete ===\n")
    
    def visualize_sample_slices(self, sample_id, root_dir=None, class_csv_path=None, 
                               num_slices=None, create_gif=True, show_masks=True, max_frames=80):
        """
        Visualize slices from a specific sample.
        
        Args:
            sample_id (str): ID of the sample to visualize
            root_dir (str, optional): Root directory for the dataset
            class_csv_path (str, optional): Path to CSV with class information
            num_slices (int, optional): Number of slices to visualize (None = all)
            create_gif (bool): Whether to create a GIF from the slices
            show_masks (bool): Whether to show masks with the images
            max_frames (int): Maximum number of frames to include in GIFs (default: 80)
            
        Returns:
            dict: Dictionary containing paths to created visualizations
        """
        # Use default paths from config if not provided
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
        if class_csv_path is None:
            class_csv_path = os.path.abspath(config.CLASS_CSV_PATH)
        
        # First try using the dataloader
        try:
            print("Attempting to load sample using dataloader...")
            # Adjust the sample directory path to account for the nuclei_sample_1a_v1 subfolder
            nuclei_dir = os.path.join(root_dir, 'nuclei_sample_1a_v1')
            
            # Check if the nuclei_sample_1a_v1 directory exists
            if not os.path.exists(nuclei_dir):
                print(f"Error: Nuclei sample directory not found: {nuclei_dir}")
                return {'sample_id': sample_id, 'num_slices': 0, 'error': 'nuclei_dir_not_found'}
                
            # Check if sample directory exists in the nuclei_sample_1a_v1 folder
            sample_dir = os.path.join(nuclei_dir, str(sample_id))
            if not os.path.exists(sample_dir):
                print(f"Error: Sample directory not found: {sample_dir}")
                return {'sample_id': sample_id, 'num_slices': 0, 'error': 'directory_not_found'}
            
            # Check if raw and mask directories exist
            raw_dir = os.path.join(sample_dir, 'raw')
            mask_dir = os.path.join(sample_dir, 'mask')
            
            if not os.path.exists(raw_dir):
                print(f"Error: Raw image directory not found: {raw_dir}")
                return {'sample_id': sample_id, 'num_slices': 0, 'error': 'raw_dir_not_found'}
                
            if not os.path.exists(mask_dir):
                print(f"Error: Mask directory not found: {mask_dir}")
                return {'sample_id': sample_id, 'num_slices': 0, 'error': 'mask_dir_not_found'}
                
            # Create a dataloader for this sample
            dataloader = get_nuclei_dataloader(
                root_dir=nuclei_dir,  # Use the nuclei_dir instead of root_dir
                class_csv_path=class_csv_path,
                batch_size=1,
                shuffle=False,  # Important to keep slices in order
                num_workers=0,
                sample_ids=[sample_id]
            )
            
            dataset = dataloader.dataset
            total_slices = len(dataset)
            
            if total_slices == 0:
                print(f"Error: No slices found for sample {sample_id} using dataloader")
                print("Falling back to direct loading method...")
                return self._visualize_sample_direct(sample_id, root_dir, num_slices, create_gif, show_masks, max_frames)
                
            # Determine how many slices to visualize
            if num_slices is None or num_slices > total_slices:
                num_slices = total_slices
                
            print(f"Visualizing {num_slices} of {total_slices} slices for sample {sample_id}")
            
            # Get class information if available
            class_info = ""
            if sample_id in dataset.sample_to_class:
                class_id = dataset.sample_to_class[sample_id]['class_id']
                class_name = dataset.sample_to_class[sample_id]['class_name']
                class_info = f" (Class {class_id}: {class_name})"
                
            # Collect slices
            images = []
            masks = []
            titles = []
            
            for i in range(min(num_slices, total_slices)):
                sample = dataset[i]
                image = sample['image']
                mask = sample['mask'] if show_masks else None
                slice_num = sample['metadata']['slice_num']
                
                images.append(image)
                if show_masks:
                    masks.append(mask)
                titles.append(f"Sample {sample_id}{class_info}\nSlice {slice_num}")
            
            # Create grid visualization
            grid_filename = f"sample_{sample_id}_slices.png"
            grid_path = self.visualize_multiple(
                images, 
                masks=masks if show_masks else None,
                titles=titles,
                filename=grid_filename,
                show=True
            )
            
            # Create GIF if requested
            gif_path = None
            if create_gif and num_slices > 1:
                gif_filename = f"sample_{sample_id}_animation.gif"
                gif_path = self.create_gif(
                    images,
                    masks=masks if show_masks else None,
                    filename=gif_filename,
                    duration=300,
                    max_frames=max_frames
                )
            
            return {
                'grid': grid_path,
                'gif': gif_path,
                'sample_id': sample_id,
                'class_info': class_info.strip() if class_info else None,
                'num_slices': num_slices
            }
            
        except Exception as e:
            print(f"Error using dataloader: {e}")
            print("Falling back to direct loading method...")
            return self._visualize_sample_direct(sample_id, root_dir, num_slices, create_gif, show_masks, max_frames)
            
    def _visualize_sample_direct(self, sample_id, root_dir=None, num_slices=None, create_gif=True, show_masks=True, max_frames=80):
        """
        Visualize sample slices using direct loading method.
        
        Args:
            sample_id (str): ID of the sample to visualize
            root_dir (str, optional): Root directory for the dataset
            num_slices (int, optional): Number of slices to visualize (None = all)
            create_gif (bool): Whether to create a GIF from the slices
            show_masks (bool): Whether to show masks with the images
            max_frames (int): Maximum number of frames to include in GIFs (default: 80)
            
        Returns:
            dict: Dictionary containing paths to created visualizations
        """
        # Load sample data directly
        sample_data = self.direct_load_sample(sample_id, root_dir, num_slices)
        
        if 'error' in sample_data:
            print(f"Error loading sample {sample_id}: {sample_data.get('error')}")
            return {'sample_id': sample_id, 'num_slices': 0, 'error': sample_data.get('error')}
            
        images = sample_data['images']
        masks = sample_data['masks'] if show_masks else None
        slice_nums = sample_data['slice_nums']
        class_info = sample_data.get('class_info', '')
        
        # Create titles for each slice
        titles = [f"Sample {sample_id}\n{class_info}\nSlice {slice_num}" for slice_num in slice_nums]
        
        # Create grid visualization
        grid_filename = f"sample_{sample_id}_slices.png"
        grid_path = self.visualize_multiple(
            images, 
            masks=masks if show_masks else None,
            titles=titles,
            filename=grid_filename,
            show=True
        )
        
        # Create GIF if requested
        gif_path = None
        if create_gif and len(images) > 1:
            gif_filename = f"sample_{sample_id}_animation.gif"
            gif_path = self.create_gif(
                images,
                masks=masks if show_masks else None,
                filename=gif_filename,
                duration=300,
                max_frames=max_frames
            )
        
        return {
            'grid': grid_path,
            'gif': gif_path,
            'sample_id': sample_id,
            'class_info': class_info,
            'num_slices': len(images)
        }
    
    def visualize_class_samples(self, class_id, num_samples=3, slices_per_sample=5, root_dir=None, 
                               class_csv_path=None, create_gifs=True, show_masks=True, max_frames=80):
        """
        Visualize random samples from a specific class.
        
        Args:
            class_id (int): ID of the class to visualize
            num_samples (int): Number of random samples to visualize
            slices_per_sample (int): Number of slices to visualize per sample
            root_dir (str, optional): Root directory for the dataset
            class_csv_path (str, optional): Path to CSV with class information
            create_gifs (bool): Whether to create GIFs for each sample
            show_masks (bool): Whether to show masks with the images
            max_frames (int): Maximum number of frames to include in GIFs (default: 80)
            
        Returns:
            list: List of dictionaries containing paths to created visualizations
        """
        # Just use the visualize_random_samples method with a class filter
        return self.visualize_random_samples(
            num_samples=num_samples,
            slices_per_sample=slices_per_sample,
            root_dir=root_dir,
            class_csv_path=class_csv_path,
            filter_by_class=class_id,
            create_gifs=create_gifs,
            show_masks=show_masks,
            max_frames=max_frames
        )
    
    def visualize_random_samples(self, num_samples=5, slices_per_sample=5, root_dir=None, class_csv_path=None,
                                 filter_by_class=None, create_gifs=True, show_masks=True, max_frames=80):
        """
        Visualize randomly selected samples from the dataset.
        
        Args:
            num_samples (int): Number of random samples to visualize
            slices_per_sample (int): Number of slices to visualize per sample
            root_dir (str, optional): Root directory for the dataset
            class_csv_path (str, optional): Path to CSV with class information
            filter_by_class (int or list, optional): Filter random selection to specific class(es)
            create_gifs (bool): Whether to create GIFs for each sample
            show_masks (bool): Whether to show masks with the images
            max_frames (int): Maximum number of frames to include in GIFs (default: 80)
            
        Returns:
            list: List of dictionaries containing paths to created visualizations
        """
        import pandas as pd
        import random
        
        # Use default paths from config if not provided
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
        if class_csv_path is None:
            class_csv_path = os.path.abspath(config.CLASS_CSV_PATH)
        
        # Adjust the path to include the nuclei_sample_1a_v1 subdirectory
        nuclei_dir = os.path.join(root_dir, 'nuclei_sample_1a_v1')
        
        print(f"Using data directory: {root_dir}")
        print(f"Using nuclei directory: {nuclei_dir}")
        print(f"Using class CSV file: {class_csv_path}")
        
        # Check if the nuclei_sample_1a_v1 directory exists
        if not os.path.exists(nuclei_dir):
            print(f"Error: Nuclei sample directory not found: {nuclei_dir}")
            return []
        
        # Read the CSV file to get sample IDs
        try:
            df = pd.read_csv(class_csv_path)
            print(f"Found {len(df)} samples in CSV file")
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return []
        
        # Filter by class if requested
        if filter_by_class is not None:
            if isinstance(filter_by_class, int):
                filter_by_class = [filter_by_class]
            df = df[df['class_id'].isin(filter_by_class)]
            print(f"Filtered to {len(df)} samples in classes {filter_by_class}")
            
        # Check if we have any samples after filtering
        if len(df) == 0:
            print("No samples found matching the filter criteria.")
            return []
            
        # Get unique sample IDs
        unique_samples = df['sample_id'].unique()
        print(f"Found {len(unique_samples)} unique sample IDs")
        
        # Check which sample directories actually exist in the nuclei_dir
        existing_sample_dirs = []
        for d in os.listdir(nuclei_dir):
            if os.path.isdir(os.path.join(nuclei_dir, d)):
                try:
                    # Try to convert directory name to integer (sample ID)
                    sample_id = int(d)
                    if sample_id in unique_samples:
                        existing_sample_dirs.append(sample_id)
                except ValueError:
                    # Directory name is not a valid sample ID number, skip it
                    continue
                    
        if not existing_sample_dirs:
            print(f"Error: No matching sample directories found in {nuclei_dir}")
            return []
            
        print(f"Found {len(existing_sample_dirs)} sample directories that match CSV sample IDs")
        
        # If filter_by_class is specified, find samples that both match the class and exist as directories
        if filter_by_class is not None:
            filtered_df = df[df['class_id'].isin(filter_by_class)]
            filtered_samples = filtered_df['sample_id'].unique()
            available_samples = [s for s in filtered_samples if s in existing_sample_dirs]
            
            if not available_samples:
                print(f"Error: No samples of classes {filter_by_class} have directories in {nuclei_dir}")
                return []
                
            sample_pool = available_samples
            print(f"Found {len(sample_pool)} samples that match class filter and have directories")
        else:
            sample_pool = existing_sample_dirs
        
        # Randomly select samples
        if len(sample_pool) <= num_samples:
            selected_samples = sample_pool
            print(f"Requesting {num_samples} samples but only {len(sample_pool)} available. Using all available samples.")
        else:
            selected_samples = random.sample(list(sample_pool), num_samples)
            
        print(f"Selected {len(selected_samples)} random samples: {selected_samples}")
        
        # Visualize each sample
        visualizations = []
        for sample_id in selected_samples:
            print(f"\nVisualizing sample {sample_id}...")
            sample_viz = self.visualize_sample_slices(
                sample_id=sample_id,
                root_dir=root_dir,  # This will be adjusted inside visualize_sample_slices
                class_csv_path=class_csv_path,
                num_slices=slices_per_sample,
                create_gif=create_gifs,
                show_masks=show_masks,
                max_frames=max_frames
            )
            
            if sample_viz and 'error' not in sample_viz:
                visualizations.append(sample_viz)
            elif 'error' in sample_viz:
                print(f"Skipping sample {sample_id} due to error: {sample_viz.get('error')}")
            else:
                print(f"Skipping sample {sample_id} due to unknown error")
                
        # Create an overview of all visualized samples if multiple samples were successfully visualized
        if len(visualizations) > 1:
            print(f"\nCreating overview of {len(visualizations)} samples...")
            # Collect one representative image from each sample
            images = []
            masks = []
            titles = []
            
            for viz in visualizations:
                # Try to get a representative image from each sample
                sample_id = viz['sample_id']
                
                # Load sample data directly
                sample_data = self.direct_load_sample(sample_id, root_dir, num_slices=1)
                
                if 'error' not in sample_data and sample_data.get('num_slices', 0) > 0:
                    images.append(sample_data['images'][0])
                    if show_masks:
                        masks.append(sample_data['masks'][0])
                    class_info = viz.get('class_info', '')
                    titles.append(f"Sample {sample_id}\n{class_info}")
            
            if images:
                # Create a grid of representative images
                overview_filename = f"random_samples_overview.png"
                overview_path = self.visualize_multiple(
                    images,
                    masks=masks if show_masks else None,
                    titles=titles,
                    filename=overview_filename,
                    show=True
                )
                
                # Add the overview to the visualizations
                visualizations.append({
                    'type': 'overview',
                    'grid': overview_path,
                    'num_samples': len(images)
                })
        
        if not visualizations:
            print("No valid visualizations were created. Check for errors above.")
            
        return visualizations

    def check_file_matching(self, sample_id, root_dir=None):
        """
        Check if there are matching raw and mask files for a sample.
        
        Args:
            sample_id (str): ID of the sample to check
            root_dir (str, optional): Root directory for the dataset
            
        Returns:
            dict: Dictionary with information about the files
        """
        import glob
        import os
        
        # Use default paths from config if not provided
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
            
        # Adjust the sample directory path to account for the nuclei_sample_1a_v1 subfolder
        nuclei_dir = os.path.join(root_dir, 'nuclei_sample_1a_v1')
        
        # Check if the nuclei_sample_1a_v1 directory exists
        if not os.path.exists(nuclei_dir):
            print(f"Error: Nuclei sample directory not found: {nuclei_dir}")
            return {'error': 'nuclei_dir_not_found'}
            
        # Check if sample directory exists in the nuclei_sample_1a_v1 folder
        sample_dir = os.path.join(nuclei_dir, str(sample_id))
        if not os.path.exists(sample_dir):
            print(f"Error: Sample directory not found: {sample_dir}")
            return {'error': 'directory_not_found'}
        
        # Check if raw and mask directories exist
        raw_dir = os.path.join(sample_dir, 'raw')
        mask_dir = os.path.join(sample_dir, 'mask')
        
        if not os.path.exists(raw_dir):
            print(f"Error: Raw image directory not found: {raw_dir}")
            return {'error': 'raw_dir_not_found'}
            
        if not os.path.exists(mask_dir):
            print(f"Error: Mask directory not found: {mask_dir}")
            return {'error': 'mask_dir_not_found'}
            
        # Get all .tif files in raw and mask directories
        raw_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(raw_dir, '*.tif'))])
        mask_files = sorted([os.path.basename(f) for f in glob.glob(os.path.join(mask_dir, '*.tif'))])
        
        print(f"Found {len(raw_files)} raw files and {len(mask_files)} mask files")
        
        # Find matching files
        matching_files = [f for f in raw_files if f in mask_files]
        print(f"Found {len(matching_files)} matching files between raw and mask")
        
        # Find files in raw but not in mask
        raw_only = [f for f in raw_files if f not in mask_files]
        if raw_only:
            print(f"Files in raw but not in mask ({len(raw_only)}):")
            for i, f in enumerate(raw_only[:10]):
                print(f"  {f}")
            if len(raw_only) > 10:
                print(f"  ... and {len(raw_only) - 10} more")
        else:
            print("All raw files have matching mask files")
        
        # Find files in mask but not in raw
        mask_only = [f for f in mask_files if f not in raw_files]
        if mask_only:
            print(f"Files in mask but not in raw ({len(mask_only)}):")
            for i, f in enumerate(mask_only[:10]):
                print(f"  {f}")
            if len(mask_only) > 10:
                print(f"  ... and {len(mask_only) - 10} more")
        else:
            print("All mask files have matching raw files")
            
        # Check if there's a mismatch in file numbering (e.g., 0237.tif vs 237.tif)
        if len(matching_files) == 0:
            print("\nChecking for filename format mismatches...")
            # Strip leading zeros from filenames and compare
            raw_files_no_zeros = [f.lstrip('0') for f in raw_files]
            mask_files_no_zeros = [f.lstrip('0') for f in mask_files]
            
            # Find matches after stripping zeros
            matches_after_strip = set(raw_files_no_zeros).intersection(set(mask_files_no_zeros))
            if matches_after_strip:
                print(f"Found {len(matches_after_strip)} matches after stripping leading zeros")
                print("Example matches:")
                for i, f in enumerate(list(matches_after_strip)[:5]):
                    raw_idx = raw_files_no_zeros.index(f)
                    mask_idx = mask_files_no_zeros.index(f)
                    print(f"  Raw: {raw_files[raw_idx]} -> Mask: {mask_files[mask_idx]}")
                
                print("\nThis suggests a filename format mismatch between raw and mask files.")
                print("The dataloader expects exact filename matches.")
        
        return {
            'sample_id': sample_id,
            'raw_files': len(raw_files),
            'mask_files': len(mask_files),
            'matching_files': len(matching_files),
            'raw_only': len(raw_only),
            'mask_only': len(mask_only)
        }

    def direct_load_sample(self, sample_id, root_dir=None, num_slices=None):
        """
        Load sample images directly without using the dataloader.
        
        Args:
            sample_id (str): ID of the sample to load
            root_dir (str, optional): Root directory for the dataset
            num_slices (int, optional): Number of slices to load (None = all)
            
        Returns:
            dict: Dictionary containing loaded images and masks
        """
        import glob
        import os
        import random
        from PIL import Image
        import numpy as np
        import torch
        import torchvision.transforms as transforms
        
        # Use default paths from config if not provided
        if root_dir is None:
            root_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
            
        # Adjust the sample directory path to account for the nuclei_sample_1a_v1 subfolder
        nuclei_dir = os.path.join(root_dir, 'nuclei_sample_1a_v1')
        
        # Check if the nuclei_sample_1a_v1 directory exists
        if not os.path.exists(nuclei_dir):
            print(f"Error: Nuclei sample directory not found: {nuclei_dir}")
            return {'error': 'nuclei_dir_not_found'}
            
        # Check if sample directory exists in the nuclei_sample_1a_v1 folder
        sample_dir = os.path.join(nuclei_dir, str(sample_id))
        if not os.path.exists(sample_dir):
            print(f"Error: Sample directory not found: {sample_dir}")
            return {'error': 'directory_not_found'}
        
        # Check if raw and mask directories exist
        raw_dir = os.path.join(sample_dir, 'raw')
        mask_dir = os.path.join(sample_dir, 'mask')
        
        if not os.path.exists(raw_dir):
            print(f"Error: Raw image directory not found: {raw_dir}")
            return {'error': 'raw_dir_not_found'}
            
        if not os.path.exists(mask_dir):
            print(f"Error: Mask directory not found: {mask_dir}")
            return {'error': 'mask_dir_not_found'}
            
        # Get all .tif files in raw and mask directories
        raw_files = sorted(glob.glob(os.path.join(raw_dir, '*.tif')))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
        
        # Match raw and mask files by basename
        raw_basenames = [os.path.basename(f) for f in raw_files]
        mask_basenames = [os.path.basename(f) for f in mask_files]
        
        # Find matching files
        matching_files = []
        for i, raw_file in enumerate(raw_files):
            basename = os.path.basename(raw_file)
            if basename in mask_basenames:
                mask_idx = mask_basenames.index(basename)
                matching_files.append((raw_file, mask_files[mask_idx]))
        
        if not matching_files:
            print(f"Error: No matching raw and mask files found for sample {sample_id}")
            return {'error': 'no_matching_files'}
            
        print(f"Found {len(matching_files)} matching file pairs")
        
        # Determine how many slices to load
        if num_slices is None or num_slices > len(matching_files):
            num_slices = len(matching_files)
            
        # Select slices to load
        if num_slices < len(matching_files):
            # Randomly select slices
            indices = sorted(random.sample(range(len(matching_files)), num_slices))
            selected_files = [matching_files[i] for i in indices]
        else:
            selected_files = matching_files
            
        print(f"Loading {len(selected_files)} slices")
        
        # Load images and masks
        images = []
        masks = []
        slice_nums = []
        
        to_tensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        
        for raw_file, mask_file in selected_files:
            try:
                # Load raw image
                raw_img = Image.open(raw_file)
                raw_tensor = normalize(to_tensor(raw_img))
                
                # Load mask image
                mask_img = Image.open(mask_file)
                mask_tensor = to_tensor(mask_img)
                
                # Extract slice number from filename
                basename = os.path.basename(raw_file)
                slice_num = int(os.path.splitext(basename)[0])
                
                # Add to lists
                images.append(raw_tensor)
                masks.append(mask_tensor)
                slice_nums.append(slice_num)
                
            except Exception as e:
                print(f"Error loading files {raw_file} and {mask_file}: {e}")
                continue
                
        if not images:
            print(f"Error: Failed to load any images for sample {sample_id}")
            return {'error': 'loading_failed'}
            
        print(f"Successfully loaded {len(images)} image pairs")
        
        # Get class information from CSV if available
        class_info = None
        try:
            import pandas as pd
            csv_path = os.path.abspath(config.CLASS_CSV_PATH)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                sample_row = df[df['sample_id'] == int(sample_id)]
                if not sample_row.empty:
                    class_id = sample_row.iloc[0]['class_id']
                    class_name = sample_row.iloc[0]['class_name']
                    class_info = f"Class {class_id}: {class_name}"
        except Exception as e:
            print(f"Error getting class information: {e}")
            
        return {
            'images': images,
            'masks': masks,
            'slice_nums': slice_nums,
            'sample_id': sample_id,
            'class_info': class_info,
            'num_slices': len(images)
        }


def main():
    """
    Run the nuclei visualization from the command line.
    
    Example usage:
    python nuclei_visualizer.py --sample 3445 --slices 5
    python nuclei_visualizer.py --class 2 --samples 3 --slices 5
    python nuclei_visualizer.py --random --samples 4 --slices 3
    python nuclei_visualizer.py --check-sample 2150
    python nuclei_visualizer.py --test-gif --slices 3
    python nuclei_visualizer.py --sample 1515 --max-frames 50
    """
    import sys
    import argparse
    import os
    
    # Create argument parser
    parser = argparse.ArgumentParser(description='Visualize nuclei samples')
    parser.add_argument('--sample', type=str, help='ID of a specific sample to visualize')
    parser.add_argument('--class', dest='class_id', type=int, help='Class ID to visualize samples from')
    parser.add_argument('--random', action='store_true', help='Randomly select samples to visualize')
    parser.add_argument('--filter-class', type=int, help='Filter random samples by class ID')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples to visualize when using --class or --random')
    parser.add_argument('--slices', type=int, default=5, help='Number of slices to visualize per sample')
    parser.add_argument('--data-dir', type=str, help='Root directory for the dataset')
    parser.add_argument('--csv-path', type=str, help='Path to the CSV file with class information')
    parser.add_argument('--no-gif', dest='create_gif', action='store_false', help='Do not create GIF animations')
    parser.add_argument('--no-mask', dest='show_mask', action='store_false', help='Do not show masks with images')
    parser.add_argument('--check-sample', type=str, help='Check if a sample has matching raw and mask files')
    parser.add_argument('--test-gif', action='store_true', help='Test GIF creation with synthetic images')
    parser.add_argument('--max-frames', type=int, default=80, help='Maximum number of frames to include in GIFs (default: 80)')
    parser.set_defaults(create_gif=True, show_mask=True)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Check environment and data paths
    data_dir = args.data_dir
    csv_path = args.csv_path
    
    if not data_dir:
        data_dir = os.path.abspath(os.path.dirname(config.DATA_ROOT))
    
    if not csv_path:
        csv_path = os.path.abspath(config.CLASS_CSV_PATH)
    
    # Verify data directory exists
    if not os.path.exists(data_dir) and not args.test_gif:
        print(f"Error: Data directory not found: {data_dir}")
        print(f"Please check your configuration or specify a valid data directory with --data-dir")
        return 1
    
    # Verify CSV file exists if we need class information
    if (args.class_id is not None or args.filter_class is not None) and not os.path.exists(csv_path):
        print(f"Error: Class CSV file not found: {csv_path}")
        print(f"Please check your configuration or specify a valid CSV file with --csv-path")
        return 1
        
    print(f"Using data directory: {data_dir}")
    print(f"Using class CSV path: {csv_path}")
    
    # Create visualizer
    print("Creating NucleiVisualizer...")
    try:
        visualizer = NucleiVisualizer()
    except Exception as e:
        print(f"Error creating visualizer: {e}")
        return 1
    
    # Test GIF creation with synthetic images if requested
    if args.test_gif:
        print("Testing GIF creation with synthetic images...")
        import torch
        import numpy as np
        
        # Generate some synthetic images (much smaller for speed)
        num_slices = args.slices
        images = []
        masks = []
        
        for i in range(num_slices):
            # Create a simple gradient image (smaller size)
            img = torch.zeros(1, 64, 64)
            for y in range(64):
                for x in range(64):
                    # Create a moving pattern
                    img[0, y, x] = 0.5 + 0.5 * np.sin((x + y + i*10) / 10.0)
            
            # Create a simple mask
            mask = torch.zeros(1, 64, 64)
            mask[0, 20:44, 20:44] = 1.0
            
            images.append(img)
            masks.append(mask)
            
        print(f"Generated {len(images)} synthetic images of size 64x64")
            
        # Create a GIF from these images
        gif_path = visualizer.create_gif(
            images,
            masks=masks if args.show_mask else None,
            filename="test_animation.gif",
            duration=300,
            max_frames=args.max_frames
        )
        
        if gif_path:
            print(f"GIF creation test successful! GIF saved to: {gif_path}")
            return 0
        else:
            print("GIF creation test failed!")
            return 1
            
    # Check if the user is requesting to check a sample
    if args.check_sample:
        print(f"Checking file matching for sample {args.check_sample}...")
        result = visualizer.check_file_matching(args.check_sample, root_dir=data_dir)
        if 'error' in result:
            print(f"Check completed with error: {result.get('error')}")
            return 1
        else:
            print(f"Check completed. Summary:")
            print(f"  Raw files: {result.get('raw_files', 0)}")
            print(f"  Mask files: {result.get('mask_files', 0)}")
            print(f"  Matching files: {result.get('matching_files', 0)}")
            print(f"  Files only in raw: {result.get('raw_only', 0)}")
            print(f"  Files only in mask: {result.get('mask_only', 0)}")
            return 0
        
    # Perform visualization based on arguments
    try:
        if args.sample:
            print(f"Visualizing specific sample: {args.sample}")
            result = visualizer.visualize_sample_slices(
                sample_id=args.sample,
                root_dir=data_dir,
                class_csv_path=csv_path,
                num_slices=args.slices,
                create_gif=args.create_gif,
                show_masks=args.show_mask,
                max_frames=args.max_frames
            )
            
            if 'error' in result:
                print(f"Visualization completed with error: {result.get('error')}")
                print("Please check the messages above for more details.")
                return 1
            elif result.get('num_slices', 0) == 0:
                print(f"No slices were visualized for sample {args.sample}")
                print("Please check the messages above for more details.")
                return 1
            else:
                print(f"Visualization for sample {args.sample} completed successfully")
                print(f"Created {result.get('num_slices', 0)} slice visualizations")
                if result.get('grid'):
                    print(f"Grid image: {result.get('grid')}")
                if result.get('gif'):
                    print(f"GIF animation: {result.get('gif')}")
                return 0
                
        elif args.class_id is not None:
            print(f"Visualizing samples from class: {args.class_id}")
            results = visualizer.visualize_class_samples(
                class_id=args.class_id,
                num_samples=args.samples,
                slices_per_sample=args.slices,
                root_dir=data_dir,
                class_csv_path=csv_path,
                create_gifs=args.create_gif,
                show_masks=args.show_mask,
                max_frames=args.max_frames
            )
            
            if not results:
                print(f"No samples were visualized for class {args.class_id}")
                print("Please check the messages above for more details.")
                return 1
            else:
                print(f"Visualization for class {args.class_id} completed successfully")
                print(f"Visualized {len(results)} samples")
                return 0
                
        elif args.random:
            print("Visualizing random samples")
            filter_class = args.filter_class
            if filter_class is not None:
                print(f"Filtering random samples to class: {filter_class}")
                
            results = visualizer.visualize_random_samples(
                num_samples=args.samples,
                slices_per_sample=args.slices,
                root_dir=data_dir,
                class_csv_path=csv_path,
                filter_by_class=filter_class,
                create_gifs=args.create_gif,
                show_masks=args.show_mask,
                max_frames=args.max_frames
            )
            
            if not results:
                print("No random samples were visualized")
                print("Please check the messages above for more details.")
                return 1
            else:
                print(f"Random sample visualization completed successfully")
                print(f"Visualized {len(results)} samples")
                return 0
        else:
            print("Error: You must specify one of --sample, --class, --random, --test-gif, or --check-sample")
            parser.print_help()
            return 1
            
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 