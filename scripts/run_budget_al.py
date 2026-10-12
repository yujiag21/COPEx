#!/usr/bin/env python3
"""Run the budget-constrained active learning experiments.

Results: results/al_budget_<dim_x>/<save_folder>/<benchmark>/<group>/
Logs:    logs/al_budget/<group>/trial_<t>.log
"""
import argparse
import itertools
import os

import torch

from _runner import REPO_ROOT, add_common_args, execute, fmt, make_parser, make_run, parse_args, parse_trials

EXAMPLES = """examples:
  # Hazard Center cost, budgets 2 and 4, H in {0, 1, 3}, benchmarks goldstein_price / branin / ackley2d, trials 100-199
  python scripts/run_budget_al.py

  # Rough Terrain cost, budget 4, H=0 and H=3, 10 trials, 4 in parallel
  python scripts/run_budget_al.py --cost-mode rough_terrain --budget 4 --lookahead 0 3 --trials 0-9 --jobs 4

  # Plan with the budget only (no transition constraint) and without the cost normalization
  python scripts/run_budget_al.py --no-transition-constraint --cost-weight 0
"""

# Budgets (small / large) and cost weight w per cost landscape
COST_PRESETS = {"hazard_center": {"budgets": [2.0, 4.0], "cost_weight": 1.0},
                "rough_terrain": {"budgets": [4.0, 8.0], "cost_weight": 3.0}}

ACQUISITION_FUNCTIONS = {
    "eig": "acquisition_functions.eig_acquisition.EntropyDifferenceEIG",
    "uncertainty": "acquisition_functions.eig_acquisition.UncertaintySampling",
}


def hazard_center_cost(X: torch.Tensor, prev_x: torch.Tensor = None) -> torch.Tensor:
    """Hazard Center: c_min + A * exp(-||x||^2 / (2 * sigma^2)), designs near the center are the most expensive."""
    c_min, A, sigma = 0.05, 1.0, 1.5
    r2 = (X ** 2).sum(dim=-1)
    return c_min + A * torch.exp(-r2 / (2 * sigma ** 2))


def rough_terrain_cost(X: torch.Tensor, prev_x: torch.Tensor = None) -> torch.Tensor:
    """Rough Terrain (2D designs): softplus(sum_i a_i * sin(k_i * x_1 + phi_i) * cos(k_i * x_2 + psi_i)) + 1e-3."""
    a = (4.0, 2.0, 1.0)
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
    parser = make_parser("Run budget-constrained AL benchmark experiments.", EXAMPLES)
    parser.add_argument("--benchmarks", nargs="+", default=["goldstein_price", "branin", "ackley2d"])
    parser.add_argument("--lookahead", nargs="+", type=int, default=[0, 1, 3],
                        help="Planning horizons H (experiment.lookahead_n_fantasies)")
    parser.add_argument("--n-fantasies", type=int, default=1)
    parser.add_argument("--cost-mode", choices=sorted(COST_PRESETS), default="hazard_center")
    parser.add_argument("--constraint-mode", choices=["linf", "l2_dist"], default="linf")
    parser.add_argument("--acquisition", choices=sorted(ACQUISITION_FUNCTIONS), default="uncertainty")
    parser.add_argument("--budget", nargs="+", type=float, help="Total budgets (default: those of the cost preset)")
    parser.add_argument("--n-step", type=int, default=200, help="Maximum number of steps")
    parser.add_argument("--n-random", type=int, default=4)
    parser.add_argument("--n-optimal", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--discount", type=float, default=0.8)
    parser.add_argument("--n-y", type=int, default=30)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--cost-weight", type=float,
                        help="w of utility / (1 + w * cost), 0 disables it (default: 1 hazard_center, 3 rough_terrain)")
    parser.add_argument("--adaptive-horizon", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--feasible-warmstart", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-transition-constraint", action="store_true", help="Plan with the budget constraint only")
    add_common_args(parser, default_trials="100-199")
    args = parse_args(parser)

    preset = COST_PRESETS[args.cost_mode]
    budgets = args.budget or preset["budgets"]
    n_step = args.n_step
    cost_weight = args.cost_weight if args.cost_weight is not None else preset["cost_weight"]
    save_folder = f"AL_{args.cost_mode}"
    acq_tag = "_uncertain" if args.acquisition == "uncertainty" else ""

    runs = []
    for benchmark, budget, lookahead in itertools.product(args.benchmarks, budgets, args.lookahead):
        group = (f"bm{benchmark}{acq_tag}_budget{fmt(budget)}_{args.n_random}_opt{args.n_optimal}"
                 f"_COST_{args.cost_mode}_CONSTRAINT_{args.constraint_mode}_scw{fmt(cost_weight)}"
                 f"_lf{lookahead}_nf{args.n_fantasies}_disc{fmt(args.discount)}_k{args.top_k}"
                 f"_step{fmt(args.step_size)}_nstep{n_step}"
                 f"{'_ah' if args.adaptive_horizon else ''}{'_fw' if args.feasible_warmstart else ''}"
                 f"{'_notc' if args.no_transition_constraint else ''}")
        result_dir = os.path.join(REPO_ROOT, "results", "al_budget_2", save_folder, benchmark, group)
        log_dir = os.path.join(REPO_ROOT, "logs", "al_budget", group)
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
                "experiment.n_step": n_step,
                "experiment.budget": budget,
                "warmstart.n_random": args.n_random,
                "warmstart.n_optimal": args.n_optimal,
                "warmstart.top_k_starts": args.top_k,
                "optimizer.optimize_mode": "budget",
                "optimizer.soft_constraint_weight": cost_weight,
                "optimizer.budget_cost_normalization": cost_weight > 0,
                "experiment.cost_function": f"run_budget_al:{args.cost_mode}_cost",
                "task.benchmark_name": benchmark,
                "paths.save_path": group,
                "paths.save_folder": save_folder,
                "wandb.group": group,
                "acquisition_function": ACQUISITION_FUNCTIONS[args.acquisition],
                "experiment.adaptive_horizon": args.adaptive_horizon,
                "warmstart.feasible_random": args.feasible_warmstart,
                "optimizer.budget_transition_constraint": not args.no_transition_constraint,
            }
            runs.append(make_run(args, "config_al_budget", overrides, result_dir, log_dir, trial, group))
    execute(runs, args)


if __name__ == "__main__":
    main()
