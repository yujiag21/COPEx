# Constrained Bayesian Experimental Design via Online Planning

A unified multi-step experiment runner for Bayesian Experimental Design tasks with constraints.

## Supported Tasks

| Task Name | Description | Config File |
|-----------|-------------|-------------|
| `AL_benchmark` | Active Learning Benchmark | `config_al_benchmark` |
| `Location_budgeted` | Location Finding | `config_location_finding` |
| `CES` | Choice Experiment Simulation | `config_ces` |

## Installation

### 1. Create a virtual environment

```bash
# Using conda
conda create -n bed python=3.10
conda activate bed

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/macOS
```

### 2. Install PyTorch

Install PyTorch with CUDA support (recommended for GPU acceleration)

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Running Experiments

```bash
# For Location Finding (budgeted) task
python main.py --config-name=config_location_finding

# For AL benchmark task
python main.py --config-name=config_al_benchmark

# For CES task
python main.py --config-name=config_ces
```

### Overriding Configuration Parameters

You can override any configuration parameter from the command line using Hydra syntax:

```bash
# Change trial number
python main.py --config-name=config_location_finding experiment.trial=5

# Change budget
python main.py --config-name=config_location_finding experiment.budget=10.0

# Change optimizer method
python main.py --config-name=config_location_finding optimizer.method=slsqp

# Multiple overrides
python main.py --config-name=config_al_benchmark experiment.trial=0 experiment.n_step=20 warmstart.n_random=10
```

### Configuration Structure

```
config/
├── config_al_benchmark.yaml    # AL benchmark main config
├── config_ces.yaml             # CES main config
├── config_location_finding.yaml # Location finding main config
├── experiment/                 # Experiment parameters
├── optimizer/                  # Optimizer settings (lbfgsb, slsqp, random)
├── paths/                      # Model paths
├── task/                       # Task-specific settings
└── warmstart/                  # Warmstart parameters
```

## Output

Results are saved in the `results/` directory with the following structure:

```
results/<problem_name>/<save_folder>/<group_name>/
├── X/              # Design points
├── Y/              # Observations
├── cost_X/         # Cost values (if applicable)
├── value_X/        # Objective values
├── eig_X/          # EIG/RMSE values
├── time_info/      # Timing information
├── plots/          # Visualization plots
├── theta/          # True parameters (for non-AL tasks)
└── params.json     # Configuration snapshot
```

## Requirements

- Python >= 3.8
- PyTorch (with CUDA recommended)
- See `requirements.txt` for full dependency list

