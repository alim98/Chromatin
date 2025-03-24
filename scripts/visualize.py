import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch
from skimage.transform import resize
from PIL import Image

# Add parent directory to path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config
from dataloader.nuclei_dataloader import get_nuclei_dataloader


class NucleiVisualizer:
    """
    A class for visualizing nuclei data from the NucleiDataset.
    Supports visualization of both 2D slices and 3D volumes.
    """
    def __init__(self, output_dir=None, cmap='gray', mask_cmap='hot', alpha=0.7, figsize=(12, 10)):
        """
        Initialize the visualizer.
        
        Args:
            output_dir (str, optional): Directory to save visualizations. Defaults to config.VISUALIZATION_OUTPUT_DIR.
            cmap (str): Colormap for raw images.
            mask_cmap (str): Colormap for mask overlays.
            alpha (float): Alpha value for mask overlay transparency.
            figsize (tuple): Figure size for plots.
        """
        self.output_dir = output_dir if output_dir else config.VISUALIZATION_OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.cmap = cmap
        self.mask_cmap = mask_cmap
        self.alpha = alpha
        self.figsize = figsize
        
        # Create custom colormaps for mask overlays
        red_cmap = self._create_transparent_cmap('red')
        self.mask_cmaps = {
            'red': red_cmap,
            'blue': self._create_transparent_cmap('blue'),
            'green': self._create_transparent_cmap('green'),
            'hot': plt.cm.hot,
            'binary': plt.cm.binary
        }

    def _create_transparent_cmap(self, color):
        """
        Create a transparent colormap for overlay visualization.
        
        Args:
            color (str): Base color for the colormap.
            
        Returns:
            LinearSegmentedColormap: A matplotlib colormap.
        """
        if color == 'red':
            color_tuple = (1, 0, 0)
        elif color == 'green':
            color_tuple = (0, 1, 0)
        elif color == 'blue':
            color_tuple = (0, 0, 1)
        else:
            color_tuple = (1, 0, 0)  # Default to red
            
        # Create colormap with transparency
        cdict = {
            'red': [(0, 0, 0), (1, color_tuple[0], color_tuple[0])],
            'green': [(0, 0, 0), (1, color_tuple[1], color_tuple[1])],
            'blue': [(0, 0, 0), (1, color_tuple[2], color_tuple[2])],
            'alpha': [(0, 0, 0), (1, self.alpha, self.alpha)]
        }
        
        return LinearSegmentedColormap(f'transparent_{color}', cdict)

    def _tensor_to_numpy(self, tensor):
        """
        Convert a PyTorch tensor to a NumPy array.
        
        Args:
            tensor (torch.Tensor): Input tensor.
            
        Returns:
            np.ndarray: NumPy array.
        """
        if tensor is None:
            return None
            
        if isinstance(tensor, torch.Tensor):
            # Move to CPU if on GPU
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
                
            # Remove batch and channel dimensions if present
            if tensor.ndim > 3:
                tensor = tensor.squeeze(0)  # Remove batch dimension
            if tensor.ndim > 2:
                tensor = tensor.squeeze(0)  # Remove channel dimension
                
            return tensor.numpy()
        elif isinstance(tensor, np.ndarray):
            # Remove batch and channel dimensions if present
            if tensor.ndim > 3:
                tensor = tensor.squeeze(0)  # Remove batch dimension
            if tensor.ndim > 2:
                tensor = tensor.squeeze(0)  # Remove channel dimension
                
            return tensor
        else:
            raise TypeError(f"Unsupported type: {type(tensor)}")

    def _normalize_array(self, array):
        """
        Normalize array to [0, 1] for visualization.
        
        Args:
            array (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Normalized array.
        """
        if array is None:
            return None
            
        # Check if already normalized
        if array.min() >= 0 and array.max() <= 1:
            return array
            
        # Handle empty array
        if array.max() == array.min():
            return np.zeros_like(array)
            
        return (array - array.min()) / (array.max() - array.min())

    def visualize_slice(self, data_item, save_path=None, show=True, title=None):
        """
        Visualize a 2D slice from the dataset.
        
        Args:
            data_item (dict): Data item from the dataloader (__getitem__ output).
            save_path (str, optional): Path to save the visualization.
            show (bool): Whether to display the plot.
            title (str, optional): Title for the plot.
            
        Returns:
            tuple: Figure and axes objects.
        """
        # Extract data from data_item
        if 'image' in data_item:
            # 2D slice data
            image = self._tensor_to_numpy(data_item['image'])
            mask = self._tensor_to_numpy(data_item['mask'])
        elif 'volume' in data_item:
            # Extract middle slice from 3D volume
            image = self._tensor_to_numpy(data_item['volume'])
            mask = self._tensor_to_numpy(data_item['mask'])
            if image.ndim == 3:
                mid_idx = image.shape[0] // 2
                image = image[mid_idx]
                if mask is not None:
                    mask = mask[mid_idx]
        else:
            raise ValueError("Invalid data format")
            
        # Normalize arrays
        image_norm = self._normalize_array(image)
        mask_norm = self._normalize_array(mask) if mask is not None else None
        
        # Set up the plot
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Plot raw image
        axes[0].imshow(image_norm, cmap=self.cmap)
        axes[0].set_title("Raw Image")
        axes[0].axis('off')
        
        # Plot mask
        if mask_norm is not None:
            axes[1].imshow(mask_norm, cmap='binary')
            axes[1].set_title("Mask")
            axes[1].axis('off')
        else:
            axes[1].set_visible(False)
        
        # Plot overlay
        if mask_norm is not None:
            axes[2].imshow(image_norm, cmap=self.cmap)
            # Create a more visible mask overlay with red color
            mask_overlay = np.ma.masked_where(mask_norm < 0.5, mask_norm)  # Increase threshold to make only strong mask regions visible
            axes[2].imshow(mask_overlay, cmap='autumn', alpha=0.7, interpolation='none')  # Use 'autumn' colormap which is more visible
            axes[2].set_title("Overlay")
            axes[2].axis('off')
        else:
            axes[2].set_visible(False)
            
        # Set metadata in the figure title
        metadata = data_item.get('metadata', {})
        if title:
            fig.suptitle(title)
        else:
            sample_id = metadata.get('sample_id', 'Unknown')
            slice_num = metadata.get('slice_num', 'Unknown')
            class_name = metadata.get('class_name', 'Unknown')
            fig.suptitle(f"Sample: {sample_id}, Slice: {slice_num}, Class: {class_name}")
            
        plt.tight_layout()
        
        # Save the figure if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        elif show:
            # Auto-generate a filename if save_path is not provided
            if metadata:
                sample_id = metadata.get('sample_id', 'unknown')
                slice_num = metadata.get('slice_num', '0')
                auto_save_path = os.path.join(self.output_dir, f"{sample_id}_slice_{slice_num}.png")
                plt.savefig(auto_save_path, dpi=300, bbox_inches='tight')
                
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes

    def visualize_volume(self, data_item, save_path=None, show=True, title=None, 
                         axis='z', frames=10, interval=200, colorbar=True):
        """
        Visualize a 3D volume from the dataset as an animation.
        
        Args:
            data_item (dict): Data item from the dataloader (__getitem__ output).
            save_path (str, optional): Path to save the animation (as .gif or .mp4).
            show (bool): Whether to display the animation.
            title (str, optional): Title for the animation.
            axis (str): Axis to slice along ('x', 'y', or 'z').
            frames (int): Number of frames to show (0 for all slices).
            interval (int): Interval between frames in ms.
            colorbar (bool): Whether to show a colorbar.
            
        Returns:
            tuple: Figure, axes, and animation objects.
        """
        # Extract data from data_item
        if 'volume' not in data_item:
            raise ValueError("Input data does not contain a volume")
            
        volume = self._tensor_to_numpy(data_item['volume'])
        mask = self._tensor_to_numpy(data_item['mask'])
        
        # Normalize arrays
        volume_norm = self._normalize_array(volume)
        mask_norm = self._normalize_array(mask) if mask is not None else None
        
        # Determine slicing axis
        if axis == 'z':
            n_slices = volume_norm.shape[0]
            get_slice = lambda i: (volume_norm[i], mask_norm[i] if mask_norm is not None else None)
        elif axis == 'y':
            n_slices = volume_norm.shape[1]
            get_slice = lambda i: (volume_norm[:, i, :], mask_norm[:, i, :] if mask_norm is not None else None)
        elif axis == 'x':
            n_slices = volume_norm.shape[2]
            get_slice = lambda i: (volume_norm[:, :, i], mask_norm[:, :, i] if mask_norm is not None else None)
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'.")
        
        # Determine frames to show
        if frames <= 0 or frames > n_slices:
            frames = n_slices
            
        step = max(1, n_slices // frames)
        slice_indices = range(0, n_slices, step)[:frames]
        
        # Set up the figure
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Initialize plots
        img_plots = []
        mask_plots = []
        overlay_plots = []
        
        # Raw image
        img_slice, _ = get_slice(slice_indices[0])
        img_plot = axes[0].imshow(img_slice, cmap=self.cmap)
        axes[0].set_title("Raw Volume")
        axes[0].axis('off')
        img_plots.append(img_plot)
        
        # Add colorbar
        if colorbar:
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(img_plot, cax=cax)
        
        # Mask
        if mask_norm is not None:
            _, mask_slice = get_slice(slice_indices[0])
            mask_plot = axes[1].imshow(mask_slice, cmap='binary')
            axes[1].set_title("Mask")
            axes[1].axis('off')
            mask_plots.append(mask_plot)
            
            if colorbar:
                divider = make_axes_locatable(axes[1])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(mask_plot, cax=cax)
        else:
            axes[1].set_visible(False)
        
        # Overlay
        if mask_norm is not None:
            img_slice, mask_slice = get_slice(slice_indices[0])
            overlay_img = axes[2].imshow(img_slice, cmap=self.cmap)
            masked_data = np.ma.masked_where(mask_slice < 0.5, mask_slice)  # Increased threshold
            overlay_mask = axes[2].imshow(masked_data, cmap='autumn', alpha=0.7, interpolation='none')  # Using autumn colormap
            axes[2].set_title("Overlay")
            axes[2].axis('off')
            overlay_plots.append(overlay_img)
            overlay_plots.append(overlay_mask)
        else:
            axes[2].set_visible(False)
            
        # Set metadata in the figure title
        metadata = data_item.get('metadata', {})
        if title:
            main_title = title
        else:
            sample_id = metadata.get('sample_id', 'Unknown')
            class_name = metadata.get('class_name', 'Unknown')
            main_title = f"Sample: {sample_id}, Class: {class_name}"
            
        plt.tight_layout()
        
        # Update function for animation
        def update(i):
            slice_idx = slice_indices[i]
            img_slice, mask_slice = get_slice(slice_idx)
            
            # Update raw image
            img_plots[0].set_array(img_slice)
            
            # Update mask
            if mask_norm is not None and mask_plots:
                mask_plots[0].set_array(mask_slice)
            
            # Update overlay
            if mask_norm is not None and overlay_plots:
                overlay_plots[0].set_array(img_slice)
                masked_data = np.ma.masked_where(mask_slice < 0.5, mask_slice)
                overlay_plots[1].set_array(masked_data)
                
            # Update title with slice number
            fig.suptitle(f"{main_title} - Slice {slice_idx}/{n_slices-1} ({axis}-axis)")
            
            return img_plots + mask_plots + overlay_plots
            
        # Create animation
        ani = animation.FuncAnimation(fig, update, frames=len(slice_indices), 
                                      interval=interval, blit=False)
        # Save animation if requested
        if save_path:
            try:
                # Ensure .gif extension
                if not save_path.endswith('.gif'):
                    save_path += '.gif'
                ani.save(save_path, writer='pillow', fps=1000/interval)
            except Exception as e:
                print(f"Failed to save animation: {e}")
        elif show:
            # Auto-generate a filename if save_path is not provided
            if metadata:
                sample_id = metadata.get('sample_id', 'unknown')
                auto_save_path = os.path.join(self.output_dir, f"{sample_id}_volume_{axis}_axis.gif")
                try:
                    ani.save(auto_save_path, writer='pillow', fps=1000/interval)
                except Exception as e:
                    print(f"Failed to save animation: {e}")
                
        if show:
            plt.show()
        else:
            plt.close(fig)
            
        return fig, axes, ani

    def visualize_batch(self, batch, max_samples=4, save_dir=None, show=True):
        """
        Visualize a batch of samples.
        
        Args:
            batch (dict): Batch of samples from dataloader.
            max_samples (int): Maximum number of samples to visualize.
            save_dir (str, optional): Directory to save visualizations.
            show (bool): Whether to display the visualizations.
            
        Returns:
            list: List of figure objects.
        """
        figures = []
        
        # Determine if we have volumes or 2D images
        if 'volume' in batch:
            is_volume = True
            data_key = 'volume'
            mask_key = 'mask'
        elif 'image' in batch:
            is_volume = False
            data_key = 'image'
            mask_key = 'mask'
        else:
            raise ValueError("Batch does not contain recognizable data format")
            
        # Get batch size - with the custom collate_fn, these are now lists
        batch_size = len(batch[data_key])
            
        # Limit samples to visualize
        n_samples = min(batch_size, max_samples)
        
        # Create save directory if needed
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        # Visualize each sample
        for i in range(n_samples):
            # Extract sample data
            sample = {}
            
            # With custom_collate_fn, everything is already a list
            for key in batch.keys():
                if key == 'metadata':
                    # Handle metadata dictionary special case
                    sample[key] = {k: batch[key][k][i] for k in batch[key]}
                else:
                    # For other keys, just get the i-th element from the list
                    sample[key] = batch[key][i]
            
            # Generate save path if needed
            if save_dir:
                metadata = sample.get('metadata', {})
                sample_id = metadata.get('sample_id', f'sample_{i}')
                if is_volume:
                    save_path = os.path.join(save_dir, f"{sample_id}_volume.png")
                else:
                    slice_num = metadata.get('slice_num', 0)
                    save_path = os.path.join(save_dir, f"{sample_id}_slice_{slice_num}.png")
            else:
                save_path = None
                
            # Call appropriate visualization method
            if is_volume:
                # For volumes, just show the middle slice for batch visualization
                fig, axes = self.visualize_slice(sample, save_path=save_path, show=show)
                figures.append(fig)
            else:
                fig, axes = self.visualize_slice(sample, save_path=save_path, show=show)
                figures.append(fig)
                
        return figures 