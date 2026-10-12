#!/usr/bin/env python3
"""Run the location finding experiments with transition constraints.

Results: results/location_finding/<save_folder>/<group>/
Logs:    logs/location_finding/<group>/trial_<t>.log
"""
import os

from _runner import REPO_ROOT, add_common_args, execute, fmt, make_parser, make_run, parse_args, parse_trials

EXAMPLES = """examples:
  # All default horizons and constraint levels, trials 0-99, 8 in parallel
  python scripts/run_location_finding.py --jobs 8

  # Only H=1 at delta=0.1, 10 trials
  python scripts/run_location_finding.py --lookahead 1 --step-size 0.1 --trials 0-9
"""


def main():
    parser = make_parser("Run location finding experiments.", EXAMPLES)
    parser.add_argument("--lookahead", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5],
                        help="Planning horizons H (experiment.lookahead_n_fantasies)")
    parser.add_argument("--step-size", nargs="+", type=float, default=[0.05, 0.1, 0.2],
                        help="Constraint levels delta (maximum L-inf distance between consecutive designs)")
    parser.add_argument("--n-fantasies", type=int, default=2, help="Branching factor m")
    parser.add_argument("--discount", type=float, default=0.8)
    parser.add_argument("--n-y", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=0)
    parser.add_argument("--n-optimal", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--save-folder", default="location_finding")
    add_common_args(parser, default_trials="0-99")
    args = parse_args(parser)

    runs = []
    for lookahead in args.lookahead:
        for step_size in args.step_size:
            group = (f"lf{lookahead}_nf{args.n_fantasies}_rand{args.n_random}_opt{args.n_optimal}"
                     f"_disc{fmt(args.discount)}_k{args.top_k}_step{fmt(step_size)}_ny{args.n_y}")
            result_dir = os.path.join(REPO_ROOT, "results", "location_finding", args.save_folder, group)
            log_dir = os.path.join(REPO_ROOT, "logs", "location_finding", group)
            for trial in parse_trials(args.trials):
                overrides = {
                    "experiment.trial": trial,
                    "experiment.lookahead_n_fantasies": lookahead,
                    "experiment.n_fantasies": args.n_fantasies,
                    "experiment.step_size": step_size,
                    "experiment.discount_factor": args.discount,
                    "experiment.n_y": args.n_y,
                    "warmstart.n_random": args.n_random,
                    "warmstart.n_optimal": args.n_optimal,
                    "warmstart.top_k_starts": args.top_k,
                    "paths.save_path": group,
                    "paths.save_folder": args.save_folder,
                    "wandb.group": group,
                }
                runs.append(make_run(args, "config_location_finding", overrides, result_dir, log_dir, trial, group))
    execute(runs, args)


if __name__ == "__main__":
    main()
