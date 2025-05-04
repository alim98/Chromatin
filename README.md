# Chromatin Analysis Project

This project provides tools for analyzing and classifying 3D nuclei samples using deep learning.

## Environment Setup

### Prerequisites
- Anaconda or Miniconda installed ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- Git (for cloning this repository)

### Creating the Environment
1. Clone this repository:
   ```bash
   git clone <repository-url>
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

## License
[Your License Here] 