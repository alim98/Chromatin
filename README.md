# Chromatin Analysis Project

This project provides tools for analyzing and classifying 3D nuclei samples using deep learning.

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- Git (for cloning this repository)

### Creating the Environment
1. Clone this repository:
   ```bash
   git clone https://github.com/alim98/Chromatin
   cd Chromatin
   ```

2. Create the conda environment from the YAML file:
   ```bash
   conda env create -f environment.yml
   ```

3. Activate the environment:
   ```bash
   conda activate chromatin
   ```

## Project Structure
- `data/`: Contains nuclei sample data
- `dataloader/`: Data loading utilities
- `model/`: Neural network implementation
- `scripts/`: Utility scripts for data processing and visualization
- `results/`: Output directory for analysis results and visualizations

## Usage

### Fine-tuning VGG3D Model
```bash
python scripts/finetune_vgg3d.py --data_dir data/nuclei_sample_1a_v1 --batch_size 4 --epochs 20
```

### Visualizing Nuclei Samples
```bash
python scripts/visualize_example.py --mode 2d --num_samples 5
```

### Creating Nuclei Index
```bash
python scripts/create_nuclei_index.py --data_dir data/nuclei_sample_1a_v1
```

### Working with 3D Meshes
This project supports generating triangle meshes from nuclei mask volumes using the marching cubes algorithm.

#### Visualizing Meshes
```bash
python scripts/visualize_mesh.py --sample_id 1 --smooth 2
```

#### Comparing Point Clouds and Meshes
To visualize both point cloud and mesh representations side by side:
```bash
python scripts/visualize_mesh.py --sample_id 1 --compare
```

#### Using Meshes in the Dataloader
You can switch between point clouds and meshes in your code:
```python
from dataloader.mesh_dataloader import get_mesh_dataloader

# For mesh data
mesh_loader = get_mesh_dataloader(
    root_dir="data/nuclei_sample_1a_v1",
    class_csv_path="chromatin_classes_and_samples.csv",
    use_mesh=True,  # Set to True for meshes, False for point clouds
    smoothing_iterations=1,  # Control mesh smoothness
    cache_dir="data/pointclouds_cache"
)

# Access the mesh data
for batch in mesh_loader:
    vertices = batch['vertices']  # Shape: [batch_size, max_vertices, 3]
    faces = batch['faces']        # Shape: [batch_size, max_faces, 3]
    vertex_masks = batch['vertex_masks']  # Masks for valid vertices
    face_masks = batch['face_masks']      # Masks for valid faces
    break
```

Meshes provide a more complete surface representation compared to point clouds, which can be beneficial for visualization and analysis of complex chromatin structures.
