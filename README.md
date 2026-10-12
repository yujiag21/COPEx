# Constrained Bayesian Experimental Design via Online Planning

A unified multi-step experiment runner for Bayesian Experimental Design tasks with constraints.

## Supported Tasks

| Task Name | Description | Config File |
|-----------|-------------|-------------|
| `AL_benchmark` | Active learning on benchmark functions with design-dependent costs | `config_al_benchmark` |
| `AL_benchmark` (budget mode) | Budget-constrained active learning | `config_al_budget` |
| `Location_budgeted` | Location Finding | `config_location_finding` |
| `CES` | Constant Elasticity of Substitution | `config_ces` |

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
# For Location Finding task
python main.py --config-name=config_location_finding

# For AL benchmark task
python main.py --config-name=config_al_benchmark

# For CES task
python main.py --config-name=config_ces

# For budget-constrained AL task
python main.py --config-name=config_al_budget

# Budget-constrained AL with the rough terrain cost
python main.py --config-name=config_al_budget experiment.cost_mode=rough_terrain experiment.budget=4 optimizer.soft_constraint_weight=3
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

## Hyperparameters

| Hyperparameter | Meaning |
|---|---|
| `experiment.trial` | Trial index; selects the ground-truth parameters (or test-function instance) of the run |
| `experiment.n_step` | Maximum number of design steps |
| `experiment.budget` | Total cost budget; the run stops once the accumulated cost exceeds it. In `budget` mode it also constrains planning |
| `experiment.cost_mode` | Cost of a design: `hazard_center` (Gaussian peak at the center of the design space), `rough_terrain` (multi-frequency terrain, 2D, `l2_dist` / `l1_dist` (L2 / L1 distance from the previous design), or `~` (no cost) |
| `experiment.constraint_mode` | Transition constraint between consecutive designs: `linf` (L∞ distance), `l2_dist` (L2 distance), or `None` |
| `experiment.step_size` | Maximum distance between consecutive designs under `constraint_mode` (`null`: no transition constraint) |
| `experiment.lookahead_n_fantasies` | Planning horizon H: number of future steps in the scenario tree (0 = myopic) |
| `experiment.n_fantasies` | Branching factor m: number of fantasized outcomes per decision node |
| `experiment.discount_factor` | Discount γ applied to the utility of future steps in the tree |
| `experiment.n_y` | Number of Monte Carlo samples used to estimate the utility of a design |
| `experiment.fantasized_with_model` | `true`: fantasize outcomes by sampling the predictive distribution; `false`: reuse the outcome samples drawn by the utility estimator |
| `experiment.verbose` | Plot the state every `verbose` steps |
| `warmstart.n_random` | Number of randomly initialized scenario trees |
| `warmstart.n_optimal` | Number of scenario trees initialized by the pre-trained design policy |
| `warmstart.top_k_starts` | Number of initial trees (with the highest utility) that are optimized; the root design of the best optimized tree is executed |
| `acquisition_function` | Utility of a design: `ACEInfoGain` (EIG), `EntropyDifferenceEIG` (EPIG), or `UncertaintySampling` (predictive standard deviation), all in `acquisition_functions.eig_acquisition` |
| `optimizer.method` | `slsqp` (default), `lbfgsb`, or `design_only` (follow the design policy without planning) |
| `optimizer.optimize_mode` | `hard`: transition constraint as inequality constraints; `soft`: transition constraint plus the utility divided by `1 + soft_constraint_weight · cost`; `budget`: total cost of the tree ≤ remaining budget, plus the transition constraint if `budget_transition_constraint` is set |
| `optimizer.soft_constraint_weight` | Weight `w` of the cost normalization `utility / (1 + w · cost of the tree)` (`soft` mode, and `budget` mode with `budget_cost_normalization: true`) |
| `optimizer.budget_cost_normalization` | `budget` mode: maximize the cost-normalized utility instead of the utility |
| `optimizer.budget_transition_constraint` | `budget` mode: also enforce the transition constraint (`experiment.step_size`) at every node of the tree (default `true`) |
| `experiment.adaptive_horizon` | `budget` mode: plan only as many future steps as the remaining budget can afford |
| `warmstart.feasible_random` | Random initial trees are random walks / straight lines within `step_size` of their parent node (feasible under the transition constraint) |
| `optimizer.lower_bound`, `optimizer.upper_bound` | Bounds of the design space in every dimension |
| `optimizer.maxiter` | Maximum SLSQP iterations per tree |
| `optimizer.fd_eps` | Finite-difference step size of the optimizer |
| `optimizer.bounds_eps` | Margin by which the design bounds are shrunk, to keep designs off the boundary |
| `optimizer.boundary_penalty_weight` | Weight of the log-barrier penalty near the design bounds |
| `optimizer.trust_region_radius` | If > 0, restricts the optimization to a box of this radius around the initial tree |
| `optimizer.n_parallel_jobs` | Number of threads used to optimize the trees (default 1, i.e. one after the other; add with `+optimizer.n_parallel_jobs=...`) |
| `task.benchmark_name`, `task.dim_x` | Test function and input dimension of the active learning tasks |
| `paths.model_path`, `paths.design_model_path` | Pre-trained inference model and design policy |
| `paths.save_folder`, `wandb.group` | Result sub-folder and run group name |
| `device`, `seed`, `fix_seed` | Compute device (`cuda` falls back to `cpu`), random seed, and whether to fix it |

## Running Experiment Batches

The scripts in `scripts/` run each (setting, trial) combination as a separate `main.py` process and need no
cluster scheduler; they can be called from any directory. Use `--jobs N` to run N trials in parallel,
`--skip-existing` to resume an interrupted run, and `--dry-run` to print the commands. Arguments after `--`
are passed to `main.py` as Hydra overrides. Logs are written to `logs/<experiment>/<group>/trial_<t>.log`.
wandb logging is used when `WANDB_API_KEY` is set and disabled otherwise.

| Experiment | Command |
|---|---|
| Location finding | `python scripts/run_location_finding.py` |
| CES | `python scripts/run_ces.py` |
| Cost-aware AL | `python scripts/run_al_benchmark.py [--cost-mode rough_terrain] [--acquisition uncertainty]` |
| Budget-constrained AL | `python scripts/run_budget_al.py [--cost-mode rough_terrain] [--acquisition uncertainty]` |

Pre-trained models (`ALINE/model/<task>/`):

| Task | Inference model | Design policy |
|---|---|---|
| Location finding | `location_finding/inference/inference_model.tar` | `location_finding/design/design_model.tar` |
| CES | `ces/inference/inference_model.pth` | `ces/design/design_model.tar` |
| Active learning (2D) | `AL/inference/inference_model.tar` | `AL/design/design_model.tar` |

The bundled AL models are 2D.

**Reproducibility.** Runs are deterministic on CPU for a fixed trial: every objective evaluation of a tree uses the
same base noise, and the trees of a step are optimized one after the other. Runs with
`+optimizer.n_parallel_jobs` > 1 (trees optimized in parallel threads) and GPU runs are not bit-for-bit repeatable.

## Configuration Structure

```
config/
├── config_al_benchmark.yaml    # AL benchmark main config
├── config_al_budget.yaml       # Budget-constrained AL main config
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

