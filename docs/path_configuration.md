# Path Configuration

This document details the path configuration system implemented for the Chromatin project.

## Table of Contents
- [Overview](#overview)
- [Configuration File](#configuration-file)
- [Usage in Code](#usage-in-code)
- [Benefits](#benefits)

## Overview

The Chromatin project uses a centralized path configuration system, where all file and directory paths are defined in a single `config.py` file. This approach ensures consistency across different modules and simplifies project maintenance.

## Configuration File

The configuration file `config.py` contains definitions for all important paths used throughout the project:

```python
import os

# Project paths
DATA_ROOT = os.path.join("data", "nuclei_sample_1a_v1")
CLASS_CSV_PATH = os.path.join("data", "chromatin_classes_and_samples.csv")
RESULTS_DIR = "results"
ANALYSIS_OUTPUT_DIR = os.path.join(RESULTS_DIR, "analysis_output")
VISUALIZATION_OUTPUT_DIR = os.path.join(RESULTS_DIR, "visualizations")

# Create directories if they don't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_OUTPUT_DIR, exist_ok=True)
```

### Key Paths

- **DATA_ROOT**: Root directory for nuclei dataset
- **CLASS_CSV_PATH**: Path to CSV file containing chromatin class information
- **RESULTS_DIR**: Main directory for storing results
- **ANALYSIS_OUTPUT_DIR**: Directory for analysis outputs (plots, statistics)
- **VISUALIZATION_OUTPUT_DIR**: Directory for visualization outputs (images, GIFs)

### Auto-creation of Directories

The configuration file automatically creates necessary directories if they don't exist, ensuring that the project structure is consistent and ready for use.

## Usage in Code

Throughout the codebase, modules import the configuration to access path information:

```python
import config

# Use paths from config
output_dir = config.ANALYSIS_OUTPUT_DIR
csv_path = config.CLASS_CSV_PATH
```

This approach is used in:
- Analysis scripts
- Visualization modules
- Dataloader code
- Command-line interfaces

## Benefits

The centralized path configuration approach provides several benefits:

1. **Single Point of Change**: Paths can be updated in one place
2. **Consistency**: All modules use the same paths
3. **Environment Independence**: Easier to adapt to different environments
4. **Automatic Directory Creation**: Ensures required directories exist
5. **Cleaner Code**: Removes hardcoded paths from scripts

## Implementation

The path configuration was implemented with these steps:

1. Created a central `config.py` file with all path definitions
2. Added automatic directory creation
3. Modified scripts to import and use paths from config
4. Removed hardcoded paths from throughout the codebase

## Example Integration

### Before (Hardcoded Paths):

```python
output_dir = os.path.join(os.path.dirname(__file__), '../../analysis_output')
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
```

### After (Using Config):

```python
import config

output_dir = config.ANALYSIS_OUTPUT_DIR
os.makedirs(output_dir, exist_ok=True)
plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300)
``` 