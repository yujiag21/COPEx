#!/usr/bin/env python3
"""Run the cost-aware active learning experiments (transition constraint, cost-normalized utility, no budget).

Results: results/al_<dim_x>/<save_folder>/<benchmark>/<group>/
Logs:    logs/al/<group>/trial_<t>.log
"""
import itertools
import os

import torch

from _runner import REPO_ROOT, add_common_args, execute, fmt, make_parser, make_run, parse_args, parse_trials

EXAMPLES = """examples:
  # Hazard Center cost, H in {0, 1, 2}, benchmarks goldstein_price / branin / ackley2d, trials 100-199
  python scripts/run_al_benchmark.py

  # Rough Terrain cost, H=0 and H=1, 10 trials, 4 in parallel
  python scripts/run_al_benchmark.py --cost-mode rough_terrain --lookahead 0 1 --trials 0-9 --jobs 4
"""

# Cost weight w of the cost-normalized utility per cost landscape
COST_WEIGHTS = {"hazard_center": 1.0, "rough_terrain": 3.0}

ACQUISITION_FUNCTIONS = {
    "epig": "acquisition_functions.eig_acquisition.EntropyDifferenceEIG",
    "uncertainty": "acquisition_functions.eig_acquisition.UncertaintySampling",
}


def hazard_center_cost(X: torch.Tensor, prev_x: torch.Tensor = None) -> torch.Tensor:
    """Hazard Center: c_min + A * exp(-||x||^2 / (2 * sigma^2)), designs near the center are the most expensive."""
    c_min, A, sigma = 0.05, 1.0, 1.5
    r2 = (X ** 2).sum(dim=-1)
    return c_min + A * torch.exp(-r2 / (2 * sigma ** 2))


def rough_terrain_cost(X: torch.Tensor, prev_x: torch.Tensor = None) -> torch.Tensor:
    """Rough Terrain (2D designs): softplus(sum_i a_i * sin(k_i * x_1 + phi_i) * cos(k_i * x_2 + psi_i)) + 1e-3."""
    a = (1.0, 0.5, 0.25)
    k = (1.0, 2.0, 4.0)
    phi = (0.3, 1.1, 2.2)
    psi = (1.5, 0.7, 2.8)
    if X.shape[-1] != 2:
        raise ValueError(f"Expected X shape (..., 2), got {tuple(X.shape)}")
    x, y = X[..., 0], X[..., 1]
    cost = torch.zeros_like(x)
    for ai, ki, phii, psii in zip(a, k, phi, psi):
        cost = cost + ai * torch.sin(ki * x + phii) * torch.cos(ki * y + psii)
    return torch.nn.functional.softplus(cost) + 1e-3


def main():
    parser = make_parser("Run cost-aware AL benchmark experiments.", EXAMPLES)
    parser.add_argument("--benchmarks", nargs="+", default=["goldstein_price", "branin", "ackley2d"])
    parser.add_argument("--lookahead", nargs="+", type=int, default=[0, 1, 2],
                        help="Planning horizons H (experiment.lookahead_n_fantasies)")
    parser.add_argument("--n-fantasies", type=int, default=1, help="Branching factor m")
    parser.add_argument("--cost-mode", choices=sorted(COST_WEIGHTS), default="hazard_center")
    parser.add_argument("--constraint-mode", choices=["linf", "l2_dist"], default="linf")
    parser.add_argument("--acquisition", choices=sorted(ACQUISITION_FUNCTIONS), default="epig")
    parser.add_argument("--n-step", type=int, default=51, help="Number of steps")
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--n-optimal", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.8)
    parser.add_argument("--n-y", type=int, default=20)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float,
                        help="w of utility / (1 + w * cost) (default: 1 hazard_center, 3 rough_terrain)")
    add_common_args(parser, default_trials="100-199")
    args = parse_args(parser)

    cost_weight = args.cost_weight if args.cost_weight is not None else COST_WEIGHTS[args.cost_mode]
    save_folder = f"AL_{args.cost_mode}"
    acq_tag = "_uncertain" if args.acquisition == "uncertainty" else ""

    runs = []
    for benchmark, lookahead in itertools.product(args.benchmarks, args.lookahead):
        group = (f"bm{benchmark}{acq_tag}_{args.n_random}_opt{args.n_optimal}"
                 f"_COST_{args.cost_mode}_CONSTRAINT_{args.constraint_mode}_scw{fmt(cost_weight)}"
                 f"_lf{lookahead}_nf{args.n_fantasies}_disc{fmt(args.discount)}_k{args.top_k}"
                 f"_step{fmt(args.step_size)}_nstep{args.n_step}_ny{args.n_y}")
        result_dir = os.path.join(REPO_ROOT, "results", "al_2", save_folder, benchmark, group)
        log_dir = os.path.join(REPO_ROOT, "logs", "al", group)
        for trial in parse_trials(args.trials):
            overrides = {
                "experiment.trial": trial,
                "experiment.lookahead_n_fantasies": lookahead,
                "experiment.n_fantasies": args.n_fantasies,
                "experiment.discount_factor": args.discount,
                "experiment.n_y": args.n_y,
                "experiment.step_size": args.step_size,
                "experiment.cost_mode": args.cost_mode,
                "experiment.constraint_mode": args.constraint_mode,
                "experiment.n_step": args.n_step,
                "warmstart.n_random": args.n_random,
                "warmstart.n_optimal": args.n_optimal,
                "warmstart.top_k_starts": args.top_k,
                "optimizer.optimize_mode": "soft",
                "optimizer.soft_constraint_weight": cost_weight,
                "experiment.cost_function": f"run_al_benchmark:{args.cost_mode}_cost",
                "task.benchmark_name": benchmark,
                "paths.save_path": group,
                "paths.save_folder": save_folder,
                "wandb.group": group,
                "acquisition_function": ACQUISITION_FUNCTIONS[args.acquisition],
            }
            runs.append(make_run(args, "config_al_benchmark", overrides, result_dir, log_dir, trial, group))
    execute(runs, args)


if __name__ == "__main__":
    main()
