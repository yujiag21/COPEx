#!/usr/bin/env python3
"""Run the CES experiments with a global budget constraint.

Results: results/ces/<save_folder>/<group>/
Logs:    logs/ces/<group>/trial_<t>.log
"""
import os

from _runner import REPO_ROOT, add_common_args, execute, fmt, make_parser, make_run, parse_args, parse_trials

EXAMPLES = """examples:
  # All default horizons and budgets, trials 100-199, 4 in parallel
  python scripts/run_ces.py --jobs 4

  # Only H=1 with budget 100, 10 trials
  python scripts/run_ces.py --lookahead 1 --budget 100 --trials 100-109
"""


def main():
    parser = make_parser("Run CES experiments with a global budget.", EXAMPLES)
    parser.add_argument("--lookahead", nargs="+", type=int, default=[0, 1, 3],
                        help="Planning horizons H (experiment.lookahead_n_fantasies)")
    parser.add_argument("--budget", nargs="+", type=float, default=[100.0, 150.0], help="Total budgets B_total")
    parser.add_argument("--n-fantasies", type=int, default=1, help="Branching factor m")
    parser.add_argument("--discount", type=float, default=0.8)
    parser.add_argument("--n-y", type=int, default=40)
    parser.add_argument("--n-step", type=int, default=10, help="Maximum number of steps")
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--n-optimal", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--save-folder", default="ces")
    add_common_args(parser, default_trials="100-199")
    args = parse_args(parser)

    runs = []
    for lookahead in args.lookahead:
        for budget in args.budget:
            group = (f"ces_lf{lookahead}_budget{fmt(budget)}_rand{args.n_random}_opt{args.n_optimal}"
                     f"_k{args.top_k}_ny{args.n_y}_disc{fmt(args.discount)}_nf{args.n_fantasies}")
            result_dir = os.path.join(REPO_ROOT, "results", "ces", args.save_folder, group)
            log_dir = os.path.join(REPO_ROOT, "logs", "ces", group)
            for trial in parse_trials(args.trials):
                overrides = {
                    "experiment.trial": trial,
                    "experiment.lookahead_n_fantasies": lookahead,
                    "experiment.n_fantasies": args.n_fantasies,
                    "experiment.budget": budget,
                    "experiment.n_step": args.n_step,
                    "experiment.discount_factor": args.discount,
                    "experiment.n_y": args.n_y,
                    "warmstart.n_random": args.n_random,
                    "warmstart.n_optimal": args.n_optimal,
                    "warmstart.top_k_starts": args.top_k,
                    "paths.save_path": group,
                    "paths.save_folder": args.save_folder,
                    "wandb.group": group,
                }
                runs.append(make_run(args, "config_ces", overrides, result_dir, log_dir, trial, group))
    execute(runs, args)


if __name__ == "__main__":
    main()
