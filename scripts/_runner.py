"""Shared helpers of the local runners: every trial is a separate main.py process, --jobs run in parallel."""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_trials(spec):
    """Parse '0-9', '3', or '0-4,10,12-13' into a sorted list of trial indices."""
    trials = set()
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-")
            trials.update(range(int(start), int(end) + 1))
        else:
            trials.add(int(part))
    return sorted(trials)


def fmt(value):
    """Format a number for directory names (0.10 -> 0.1, 100.0 -> 100)."""
    return f"{value:g}" if isinstance(value, float) else str(value)


def make_parser(description, examples):
    return argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter,
                                   epilog=examples + "\nArguments after `--` are passed to main.py as Hydra overrides.")


def add_common_args(parser, default_trials):
    parser.add_argument("--trials", default=default_trials, help="e.g. '0-9' or '0-4,10'")
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Default: cuda if available")
    parser.add_argument("--jobs", type=int, default=1, help="Number of trials to run in parallel")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip trials whose results are already saved")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to run main.py")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands without running them")


def parse_args(parser):
    """Parse arguments; everything after `--` is returned as a dict of Hydra overrides."""
    argv = sys.argv[1:]
    extra = []
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1:]
    args = parser.parse_args(argv)
    args.extra_overrides = {}
    for item in extra:
        key, sep, value = item.partition("=")
        if not sep:
            parser.error(f"Invalid Hydra override after `--`: {item!r} (expected key=value)")
        args.extra_overrides[key] = value
    return args


def make_run(args, config_name, overrides, result_dir, log_dir, trial, name):
    """Build one run: the main.py command, the file that marks success, and the log file."""
    overrides = dict(overrides)
    if args.device is not None:
        overrides["device"] = args.device
    overrides.update(args.extra_overrides)
    cmd = [args.python, "-u", "main.py", f"--config-name={config_name}"]
    cmd += [f"{k}={v}" for k, v in overrides.items()]
    return {
        "name": f"{name} trial {trial}",
        "cmd": cmd,
        "result_file": os.path.join(result_dir, "eig_X", f"eig_X_{trial}.txt"),
        "log_file": os.path.join(log_dir, f"trial_{trial}.log"),
    }


def _run_one(run, env):
    os.makedirs(os.path.dirname(run["log_file"]), exist_ok=True)
    start = time.time()
    with open(run["log_file"], "w") as log:
        proc = subprocess.run(run["cmd"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    # Success = results written (main.py can fail after saving, e.g. when plotting)
    ok = os.path.exists(run["result_file"])
    return ok, proc.returncode, time.time() - start


def execute(runs, args):
    env = os.environ.copy()
    if "WANDB_API_KEY" not in env and "WANDB_MODE" not in env:
        env["WANDB_MODE"] = "disabled"

    if args.skip_existing:
        n_before = len(runs)
        runs = [r for r in runs if not os.path.exists(r["result_file"])]
        print(f"Skipping {n_before - len(runs)} trials with existing results.")

    if args.dry_run:
        for run in runs:
            print(" ".join(run["cmd"]))
        return

    print(f"Running {len(runs)} trials with {args.jobs} parallel job(s)")
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(_run_one, run, env): run for run in runs}
        for i, future in enumerate(as_completed(futures), 1):
            run = futures[future]
            ok, returncode, elapsed = future.result()
            if not ok:
                status = f"FAILED (exit {returncode}, see {run['log_file']})"
                failed.append(run)
            elif returncode != 0:
                status = f"ok (results saved, but exit {returncode}; see {run['log_file']})"
            else:
                status = "ok"
            print(f"[{i}/{len(runs)}] {run['name']}: {status} [{elapsed:.0f}s]", flush=True)

    print(f"Done: {len(runs) - len(failed)} succeeded, {len(failed)} failed.")
    if failed:
        sys.exit(1)
