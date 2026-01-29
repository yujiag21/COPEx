"""
Unified multi-step experiment runner for different tasks.

Supported task names (from task.name in ALINE/config/task/):
    - AL_benchmark
    - Location_budgeted
    - CES

Usage:
    # For AL benchmark task
    python main.py --config-name=config_al_benchmark
    
    # For Location Finding (budgeted) task  
    python main.py --config-name=config
    
    # For CES task
    python main.py --config-name=config_ces
"""
import os
import json
import sys
import time
from typing import List, Tuple, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import hydra
import torch
import torch.distributions as dist
import wandb
import matplotlib.pyplot as plt
import psutil
from omegaconf import DictConfig, OmegaConf

from ALINE.utils.eval import compute_EIG_from_history, compute_rmse
from acquisition_functions.multi_step_lookahead import (
    MultiStepLookaheadEIG, 
    get_X_from_multi_step_tree_input_representation
)
from algorithm.optimize import maximize_objective_slsqp, maximize_objective_lbfgsb
from algorithm.utils import (
    plot_objective_values,
    plot_theta_2d_logprob_distributions,
    get_objective_cost_constraint_function,
    load_model_and_config,
    evaluate_obj_at_X,
    fit_model,
    plot_start_and_optimized_trees,
    AttrDict,
    StepVisualizer,
    compute_total_cost,
)
from algorithm.warmstart_multistep import (
    warmstart_multistep,
    # warmstart_multistep_al,
    # warmstart_multistep_ces,
)
torch.set_default_dtype(torch.float32)


script_dir = os.path.dirname(os.path.realpath(__file__))
if script_dir not in sys.path:
    sys.path.append(script_dir)


def build_results_folder(problem_name: str, group_name: str, project_path: str, save_folder= None, benchmark_name=None) -> str:
    """Unified construction of results path for reuse in different places."""
    if benchmark_name is not None:
        return os.path.join(project_path, "results", problem_name, save_folder, benchmark_name, group_name)
    else:
        return os.path.join(project_path, "results", problem_name, save_folder, group_name)



def save_params(cfg: DictConfig, params_path: str):
    """Write complete configuration to disk for later experiment traceability."""
    try:
        with open(params_path, "w", encoding="utf-8") as f:
            json.dump(OmegaConf.to_container(cfg, resolve=True), f, indent=2)
    except Exception as exc:
        print(f"save params failed: {exc}")


def get_problem_name(task_name: str, cfg: DictConfig) -> str:
    """Get problem name based on task name."""
    if task_name == 'AL_benchmark':
        return f"al_{cfg.task.dim_x}"
    elif task_name == 'AL_data':
        return f"al_data_{cfg.task.dim_x}"
    elif task_name == 'CES':
        return f"ces"
    else:  # Location_budgeted, CES, etc.
        return f"location_finding"


def get_n_steps(cfg: DictConfig) -> int:
    """Get number of steps from config."""
    if hasattr(cfg.experiment, 'n_step'):
        return cfg.experiment.n_step
    # Default if n_step not specified
    return 30


def get_dim_theta(task_name: str, task) -> int:
    """Get dim_theta based on task name."""
    if task_name == 'AL_benchmark':
        return task.n_target_theta
    elif task_name == 'CES':
        return task.n_theta
    else:  # Location_budgeted and other location_finding variants
        return task.n_target_theta


def optimize_single_tree(
    idx: int,
    x: int,
    start_x: np.ndarray,
    optimize_method: str,
    n_random: int,
    model,
    acquisition_function,
    cfg_optimizer: Dict,
    task,
    step_size: float,
    lower_bound: float,
    upper_bound: float,
    lookahead_budget: Optional[float],
    cost_function,
    constraint_function,
    objective_value: float,
) -> Dict[str, Any]:
    """
    Optimize a single tree starting point.
    
    Returns a dict with optimization results for later aggregation.
    """
    import time
    
    # Determine source of this starting point: random or design
    current_source = 'random' if x < n_random else 'design'
    
    # Time this tree optimization
    tree_opt_start = time.time()
    
    # Generate seed before any CUDA operations to avoid threading issues
    seed = int(torch.randint(0, 10_000, (1,)).item())
    
    if optimize_method == 'lbfgsb':
        x_opt, obj_opt = maximize_objective_lbfgsb(
            model=model,
            acquisition_function=acquisition_function,
            start_x=start_x,
            maxiter=cfg_optimizer['maxiter'],
            seed=seed,
            bounds_eps=cfg_optimizer['bounds_eps'],
            boundary_penalty_weight=cfg_optimizer['boundary_penalty_weight'],
            trust_region_radius=cfg_optimizer['trust_region_radius'],
            fd_eps=cfg_optimizer['fd_eps'],
            task=task,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
        )
    elif optimize_method == 'slsqp':
        soft_constraint_weight = cfg_optimizer.get('soft_constraint_weight') if cost_function is not None else None
        x_opt, obj_opt = maximize_objective_slsqp(
            model=model,
            acquisition_function=acquisition_function,
            start_x=start_x,
            maxiter=cfg_optimizer['maxiter'],
            seed=seed,
            bounds_eps=cfg_optimizer['bounds_eps'],
            boundary_penalty_weight=cfg_optimizer['boundary_penalty_weight'],
            trust_region_radius=cfg_optimizer['trust_region_radius'],
            fd_eps=cfg_optimizer['fd_eps'],
            task=task,
            step_size=step_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            optimize_mode=cfg_optimizer['optimize_mode'],
            soft_constraint_weight=soft_constraint_weight,
            budget=lookahead_budget,
            cost_function=cost_function,
            constraint_function=constraint_function,
        )
    else:
        raise ValueError(f"Unknown optimize_method: {optimize_method}")
    
    tree_opt_time = time.time() - tree_opt_start
    
    return {
        'idx': idx,
        'x': x,
        'x_opt': x_opt,
        'obj_opt': obj_opt,
        'start_x': start_x,
        'current_source': current_source,
        'tree_opt_time': tree_opt_time,
        'objective_value': objective_value,
    }


def main(cfg: DictConfig):
    """
    Main function - unified runner for different tasks
    
    Configuration structure:
        cfg.experiment.*    - Experiment & algorithm parameters
        cfg.warmstart.*     - Warmstart parameters
        cfg.optimizer.*     - Optimizer parameters
        cfg.paths.*         - Model paths
        cfg.wandb.*         - WandB configuration
    
    Supported task names (from task.name in ALINE/config/task/):
        - AL_benchmark
        - Location_budgeted
        - CES
    """
    # Extract commonly used variables
    n_random = cfg.warmstart.n_random
    n_optimal = cfg.warmstart.n_optimal
    verbose = cfg.experiment.verbose
    trial = cfg.experiment.trial

    # Load models
    model_cfg, task, aline_model, design_model = load_model_and_config(
        cfg.paths.model_path, cfg, #cfg.paths.config_name
        cfg.paths.design_model_path, cfg.paths.design_config_name)
    
    # Get task name from task object
    task_name = task.name
    print(f"Task name: {task_name}")
    
    # Get problem name based on task_name
    problem_name = get_problem_name(task_name, cfg)
    
    # Switch models to eval mode
    try:
        design_model.eval()
    except Exception:
        pass
    try:
        aline_model.eval()
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    n_steps = get_n_steps(cfg)
    
    # WandB setup
    wandb_run = None
    wandb_api_key = os.environ.get("WANDB_API_KEY", cfg.wandb.api_key)
    try:
        wandb.login(key=wandb_api_key, relogin=True)
    except Exception as exc:
        print(f"wandb login failed: {exc}")
    
    default_group = (
        f"step{cfg.experiment.step_size}"
        f"_lf{cfg.experiment.lookahead_n_fantasies}"
        f"_nf{cfg.experiment.n_fantasies}"
        f"_rand{cfg.warmstart.n_random}"
        f"_opt{cfg.warmstart.n_optimal}"
        f"_disc{cfg.experiment.discount_factor}"
        f"_k{cfg.warmstart.top_k_starts}"
    )
    if task_name == 'AL_benchmark' or task_name == 'AL_data':
        # Add optional parameters to group name if they are not None
        benchmark_name = getattr(cfg.task, 'benchmark_name', 'al_data')
        optimize_mode = getattr(cfg.optimizer, 'optimize_mode', None)
        soft_constraint_weight = getattr(cfg.optimizer, 'soft_constraint_weight', None)
        
        if benchmark_name is not None:
            default_group += f"_bm{benchmark_name}"
            task.benchmark_name = benchmark_name
        if optimize_mode is not None:
            default_group += f"_cm{optimize_mode}"
        if soft_constraint_weight is not None:
            default_group += f"_scw{soft_constraint_weight}"
        
    group_name = cfg.wandb.group if cfg.wandb.group else default_group

    try:
        wandb_run = wandb.init(
            project=cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            group=group_name,
            name=f"trial_{trial}",
            reinit=True,
            settings=wandb.Settings(disable_git=True)
        )
    except Exception as exc:
        print(f"wandb init failed: {exc}")

    
    # Sample batch based on task name
    theta = None
    target_x = None
    target_y = None
    
    if task_name == 'AL_benchmark':
        task.seed = trial
        _, batch = task.sample_batch(1, 1)
        target_x = batch.target_x.to(torch.get_default_dtype())
        target_y = batch.target_y.to(torch.get_default_dtype())
    elif task_name == 'AL_data':
        task.seed = trial
        theta, batch = task.sample_batch(1, 1)
        task.set_theta(theta)
        target_x = batch.target_x.to(torch.get_default_dtype())
        target_y = batch.target_y.to(torch.get_default_dtype())
    elif task_name == 'Location_budgeted':
        theta, batch = task.sample_batch(1, 20, trial)
        task.set_theta(theta)
        theta = theta.squeeze(0)
    else:  # CES and other tasks
        all_theta, all_batch = task.sample_batch(200, 1)  # batch_size=200, n_context=1
        theta = all_theta[trial:trial+1].squeeze(0)  # [1, 5]
        task.set_theta(theta)
        batch = AttrDict()
        for key in all_batch.keys():
            if key in ['target_theta', 'n_target_theta']:
                batch[key] = all_batch[key]
            else:
                batch[key] = all_batch[key][trial:trial+1]
        batch = AttrDict(batch)


    cost_mode = cfg.experiment.get("cost_mode", "quadratic")
    constraint_mode = cfg.experiment.get("constraint_mode", "linf")
    use_prev_x = cost_mode in ["l2_dist", "l1_dist"]
    objective_function, cost_function, constraint_function = get_objective_cost_constraint_function(seed=trial, task=task, cost_mode=cost_mode, constraint_mode=constraint_mode)

    # Results save directory
    project_path = script_dir

    if task_name == 'AL_benchmark' or task_name == 'AL_data':
        results_folder = build_results_folder(problem_name, group_name, project_path, cfg.paths.save_folder, benchmark_name)
    else:
        results_folder = build_results_folder(problem_name, group_name, project_path, cfg.paths.save_folder)

    # Common subdirectories for results directory
    if cost_function is not None:
        results_subdirs = ["X", "Y", "cost_X", "value_X", "eig_X", "time_info", "plots", "theta"]
    else:
        results_subdirs = ["X", "Y", "value_X", "eig_X", "time_info", "plots", "theta"]

    os.makedirs(results_folder, exist_ok=True)
    for sub in results_subdirs:
        os.makedirs(os.path.join(results_folder, sub), exist_ok=True)

    plots_prefix = os.path.join(results_folder, "plots", f"trial_{trial}")
    params_path = os.path.join(results_folder, "params.json")
    save_params(cfg, params_path)
    
    # Save theta for non-AL tasks
    if theta is not None:
        np.savetxt(os.path.join(results_folder, "theta", f"theta_{trial}.txt"), 
                   task.theta.squeeze(0).cpu().numpy())
        print('theta:', task.theta.cpu().squeeze(0).numpy())
    
    print("Results saved in", results_folder)

    k = max(1, cfg.warmstart.top_k_starts)
    print('k:', k)


    # Initialize new_x based on task name
    if task_name == 'AL_benchmark' or task_name == 'AL_data':
        new_x = batch.context_x[:, 0].to(torch.get_default_dtype())  # [1, D]
    elif task_name == 'Location_budgeted':
        new_x = batch.context_x[:, :1].squeeze(0).to(torch.get_default_dtype())  # [1, D]
    else:  # CES and other tasks
        new_x = batch.context_x.squeeze(0).to(torch.get_default_dtype())  # [1, D]

    eig_X = []
    value_X = []
    time_X = []
    
    # Get warmstart function for this task
    # warmstart_func = get_warmstart_function(task_name)

    warmstart_func = warmstart_multistep
    
    # Unified visualization setup for all tasks
    visualize = task_name in ['AL_benchmark', 'Location_budgeted', 'AL_data']
    visualizer = None
    if visualize:
        visualizer = StepVisualizer(
            task_name=task_name,
            n_steps=n_steps,
            cols=5,
            figsize_per_subplot=(3.2, 2.8),
            verbose=verbose,
        )
    print(cfg.experiment.budget)
    total_budget = cfg.experiment.budget
    cumulative_cost = 0
    # step = 0
    for step in range(n_steps):
    # while total_budget > 0 and step < n_steps:
        # step += 1
        lookahead_n_fantasies = cfg.experiment.lookahead_n_fantasies if cfg.experiment.lookahead_n_fantasies<n_steps-step-1 else n_steps-step-1
        print("lookahead_n_fantasies: ", lookahead_n_fantasies)
        num_fantasies = [cfg.experiment.n_fantasies for _ in range(lookahead_n_fantasies + 1)]
        algo_params = {
            "lookahead_n_fantasies": [cfg.experiment.n_fantasies for _ in range(lookahead_n_fantasies)],
            "refill_until_lower_bound_is_reached": True, 
            "soft_plus_transform_budget": False
        }
        
        # Evaluate objective at X
        # new_y, cost_new_x = evaluate_obj_at_X(
        #     X=new_x, objective_function=objective_function, cost_function=cost_function)
        new_y = objective_function(new_x)
        # if cost_function is not None:
        #     cost_new_x = cost_function(new_x)

        # Accumulate X and Y
        if step == 0:
            X = new_x.to(torch.get_default_dtype())
            Y = new_y.to(torch.get_default_dtype())
        else:
            if task_name == 'AL_benchmark' or task_name == 'AL_data':
                X = torch.cat([X, new_x.reshape(1, *X.shape[1:]).to(torch.get_default_dtype())], 0)
                Y = torch.cat([Y, new_y.reshape(1, *Y.shape[1:]).to(torch.get_default_dtype())], 0)
            else:
                X = torch.cat([X, new_x.to(torch.get_default_dtype())], 0)
                Y = torch.cat([Y, new_y.to(torch.get_default_dtype())], 0)

        # Calculate cost using selected cost_function
        if cost_function is not None:
            if step == 0:
                # Use .view(-1) instead of .squeeze() to ensure cost_X is at least 1-dimensional
                cost_X = torch.zeros_like(Y) if use_prev_x else cost_function(new_x).view(-1)
            else:
                # prev_x = X[-2]  # Previous point (X[-1] is the newly added new_x)
                step_cost = cost_function(new_x, prev_x=X[-2]).view(-1) if use_prev_x else cost_function(new_x).view(-1)
                cost_X = torch.cat([cost_X, step_cost.view(*Y[-1:].shape)], dim=0)

        X = X.detach().to(torch.get_default_dtype())
        Y = Y.detach().to(torch.get_default_dtype())
        if cost_function is not None:
            cost_X = cost_X.detach().to(torch.get_default_dtype())
            total_budget = total_budget - cost_X[-1].item()
            cumulative_cost = cumulative_cost + cost_X[-1].item()
        print('cumulative_cost:', cumulative_cost)
        # Start timing for this step
        step_start_time = time.time()

        # Fit model with appropriate parameters
        dim_theta = get_dim_theta(task_name, task)
        # if task_name == 'AL_benchmark' or task_name == 'AL_data':
        model = fit_model(
            X=X, Y=Y, #cost_X=cost_X,
            training_mode="objective",
            noiseless_obs=True,
            dim_theta=dim_theta,
            ALINE_model=aline_model,
            target_x=target_x
        )

        # Create acquisition function
        acq_kwargs = {
            "model": model,
            "task": task,
            "batch_size": 1,
            "lookahead_batch_sizes": [1 for i in algo_params.get("lookahead_n_fantasies") if i > 0],
            "num_fantasies": num_fantasies,
            "discount_factor": cfg.experiment.discount_factor,
            "last_X": X[-1:, :],
            "n_y": cfg.experiment.n_y,
            "fantasized_with_model": cfg.experiment.fantasized_with_model,
        }
        
        # Add valfunc for AL benchmark if specified
        if hasattr(cfg, 'acquisition_function'):
            acq_kwargs["valfunc"] = hydra.utils.get_class(cfg.acquisition_function)
        
        acquisition_function = MultiStepLookaheadEIG(**acq_kwargs)
        
        # Get posterior and compute metrics
        if task_name == 'AL_benchmark' or task_name == 'AL_data':
            posterior_data_0 = model.posterior_data_0()
            rmse = compute_rmse(
                target_y, 
                posterior_data_0['mixture_means'],
                posterior_data_0['mixture_stds'],
                posterior_data_0['mixture_weights']
            )
            rmse_mean = rmse.mean(dim=0)
            rmse_std = rmse.std(dim=0, unbiased=False)
            print('step:', step)
            mean_np = rmse_mean.detach().cpu().numpy()
            print('RMSE:', mean_np)
            eig_X.append(mean_np)
            
            # WandB logging for AL benchmark
            mean_arr = np.atleast_1d(np.asarray(mean_np))
            std_arr = np.atleast_1d(rmse_std.detach().cpu().numpy())
            wandb.log({
                "step": step,
                "RMSE_X_mean": float(mean_arr.mean()),
                "RMSE_X_values": mean_arr.tolist(),
                "RMSE_X_std": std_arr.tolist(),
                "cumulative_cost": cumulative_cost,
            })
        else:
            posterior_theta_0 = model.posterior_theta_0()
            pce_losses, nmc_losses = compute_EIG_from_history(
                task, theta.unsqueeze(0), X.unsqueeze(0), Y.unsqueeze(-1).unsqueeze(0),
                batch_size=1, stepwise=False)
            eig = pce_losses
            mean = eig.mean(dim=0)
            std = eig.std(dim=0, unbiased=False)
            print('step:', step)
            mean_np = mean.detach().cpu().numpy()
            print('eig:', mean_np)
            eig_X.append(mean_np)
            
            # WandB logging
            mean_arr = np.atleast_1d(np.asarray(mean_np))
            std_arr = np.atleast_1d(std.detach().cpu().numpy())
            wandb.log({
                "step": step,
                "eig_X_mean": float(mean_arr.mean()),
                "eig_X_values": mean_arr.tolist(),
                "eig_X_std": std_arr.tolist(),
                "cumulative_cost": cumulative_cost,
            })

        if cost_function is not None and total_budget < 0:
            print('total_budget is less than 0')
            break
        print('total_budget:', total_budget)
        best_x_opt = None
        best_obj_opt = None
        best_opt_tree = None
        success_any = False
        start_points = []
        opt_points = []
        best_opt_vec = None
        objective = None
        new_xs = None
        obj_list = None
        
        # Track optimization time and source info for each tree
        tree_opt_times = []  # Time for each tree optimization
        tree_opt_sources = []  # Source for each tree: 'random' or 'design'
        best_opt_source = None  # Source of the best optimized tree

        optimize_method = cfg.optimizer.method
        # Get optimizer bounds
        lower_bound = getattr(cfg.optimizer, 'lower_bound', 0.0)
        upper_bound = getattr(cfg.optimizer, 'upper_bound', 1.0)

        design_start_time = time.time()
        if optimize_method in ['lbfgsb', 'slsqp']:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Call warmstart function based on task name
            warmstart_start_time = time.time()
            new_xs = warmstart_func(X, Y, task, num_fantasies, model, design_model,
                                       n_random, n_optimal, sample='random_and_design', step_size=None, lower_bound=lower_bound,upper_bound=upper_bound,target_x=target_x)
            warmstart_time = time.time() - warmstart_start_time
            new_xs = get_X_from_multi_step_tree_input_representation(new_xs)
            eig_calculation_start_time = time.time()
            
            # Memory monitoring before forward pass
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            gpu_mem_before = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_mem_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_mem_reserved_before = torch.cuda.memory_reserved() / 1024 / 1024  # MB
            
            with torch.no_grad():
                objective, obj_list = acquisition_function.forward(new_xs)
                objective = objective.view(-1).detach()
            
            # Memory monitoring after forward pass
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_info = {
                "cpu_mem_before_mb": mem_before,
                "cpu_mem_after_mb": mem_after,
                "cpu_mem_delta_mb": mem_after - mem_before,
            }
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_mem_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
                gpu_mem_reserved_after = torch.cuda.memory_reserved() / 1024 / 1024  # MB
                gpu_mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
                memory_info.update({
                    "gpu_mem_allocated_before_mb": gpu_mem_before,
                    "gpu_mem_allocated_after_mb": gpu_mem_after,
                    "gpu_mem_allocated_delta_mb": gpu_mem_after - gpu_mem_before,
                    "gpu_mem_reserved_before_mb": gpu_mem_reserved_before,
                    "gpu_mem_reserved_after_mb": gpu_mem_reserved_after,
                    "gpu_mem_peak_allocated_mb": gpu_mem_peak,
                })
                print(f"[Memory] acquisition_function.forward():")
                print(f"  CPU Memory: {mem_before:.2f} MB -> {mem_after:.2f} MB (delta: {mem_after - mem_before:.2f} MB)")
                print(f"  GPU Allocated: {gpu_mem_before:.2f} MB -> {gpu_mem_after:.2f} MB (delta: {gpu_mem_after - gpu_mem_before:.2f} MB)")
                print(f"  GPU Reserved: {gpu_mem_reserved_before:.2f} MB -> {gpu_mem_reserved_after:.2f} MB")
                print(f"  GPU Peak Allocated: {gpu_mem_peak:.2f} MB")
                torch.cuda.reset_peak_memory_stats()  # Reset for next measurement
            else:
                print(f"[Memory] acquisition_function.forward():")
                print(f"  CPU Memory: {mem_before:.2f} MB -> {mem_after:.2f} MB (delta: {mem_after - mem_before:.2f} MB)")

            print(f"Warmstart time: {warmstart_time:.4f} seconds")


            eig_calculation_time = time.time() - eig_calculation_start_time
            print(f"Eig calculation time: {eig_calculation_time:.4f} seconds")
            obj_list = get_X_from_multi_step_tree_input_representation([i.unsqueeze(-1) for i in obj_list])
            top_vals, top_idx = torch.topk(objective, k=k, largest=True)

            if cfg.optimizer.optimize_mode == 'budget':
                lookahead_budget = total_budget
            else:
                lookahead_budget = None
            
            # Prepare optimizer config dict for parallel execution
            cfg_optimizer_dict = {
                'maxiter': cfg.optimizer.maxiter,
                'bounds_eps': cfg.optimizer.bounds_eps,
                'boundary_penalty_weight': cfg.optimizer.boundary_penalty_weight,
                'trust_region_radius': cfg.optimizer.trust_region_radius,
                'fd_eps': cfg.optimizer.fd_eps,
                'optimize_mode': cfg.optimizer.optimize_mode,
                'soft_constraint_weight': getattr(cfg.optimizer, 'soft_constraint_weight', None),
            }
            
            # Prepare tasks for parallel execution
            parallel_tasks = []
            for idx, x in enumerate(top_idx):
                start_x = new_xs[x:x + 1, ...].detach().cpu().numpy()
                parallel_tasks.append({
                    'idx': idx,
                    'x': int(x.item()),
                    'start_x': start_x,
                    'objective_value': objective[x].item(),
                })
            
            # Get number of parallel jobs from config or default to k
            n_jobs = getattr(cfg.optimizer, 'n_parallel_jobs', k)
            
            # Execute optimization in parallel using ThreadPoolExecutor
            # Threading is used because PyTorch models/tensors share memory across threads
            print(f"Starting parallel optimization with {n_jobs} threads for {len(parallel_tasks)} trees...")
            parallel_start_time = time.time()
            
            results = []
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all tasks
                future_to_task = {
                    executor.submit(
                        optimize_single_tree,
                        idx=task_item['idx'],
                        x=task_item['x'],
                        start_x=task_item['start_x'],
                        optimize_method=optimize_method,
                        n_random=n_random,
                        model=model,
                        acquisition_function=acquisition_function,
                        cfg_optimizer=cfg_optimizer_dict,
                        task=task,
                        step_size=cfg.experiment.step_size,
                        lower_bound=lower_bound,
                        upper_bound=upper_bound,
                        lookahead_budget=lookahead_budget,
                        cost_function=cost_function,
                        constraint_function=constraint_function,
                        objective_value=task_item['objective_value'],
                    ): task_item
                    for task_item in parallel_tasks
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_task):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        print(f"Optimization task failed with error: {e}")
                        raise
            
            parallel_total_time = time.time() - parallel_start_time
            print(f"Parallel optimization completed in {parallel_total_time:.4f} seconds")
            
            # Process results from parallel execution
            for result in results:
                idx = result['idx']
                x = result['x']
                x_opt = result['x_opt']
                obj_opt = result['obj_opt']
                start_x = result['start_x']
                current_source = result['current_source']
                tree_opt_time = result['tree_opt_time']
                objective_value = result['objective_value']
                
                # Record tree optimization time and source
                tree_opt_times.append(tree_opt_time)
                tree_opt_sources.append(current_source)
                
                success_any = True
                
                start_np = np.asarray(start_x)
                start_np = start_np.reshape(1, start_np.shape[-2], start_np.shape[-1])
                opt_np = np.asarray(x_opt).reshape(start_x.shape)
                start_points.append(start_np)
                opt_points.append(opt_np)

                if (best_obj_opt is None) or (obj_opt > best_obj_opt):
                    best_obj_opt = obj_opt
                    best_opt_tree = np.asarray(x_opt).reshape(start_x.shape)
                    best_x_opt = best_opt_tree[0, 0]
                    best_opt_source = current_source  # Track source of best tree
                
                print(f"{optimize_method} start idx {x} ({start_x.flatten()},{objective_value}): optimum x: {x_opt}, objective: {obj_opt}")

            if success_any and best_x_opt is not None:
                if task_name == 'Location_budgeted' and theta is not None:
                    print(f"{optimize_method} best optimum x: {best_x_opt}, best_opt_source: {best_opt_source}, objective: {best_obj_opt}\n")                
                       
                        # (theta: {np.array(theta[0].cpu())}," {np.linalg.norm(np.array(best_x_opt) - np.array(theta[0].cpu()))}
                else:
                    print(f"{optimize_method} best optimum x: {best_x_opt}, best_opt_source: {best_opt_source}, objective: {best_obj_opt}\n")
                new_x = torch.tensor(best_x_opt, dtype=torch.get_default_dtype()).unsqueeze(0)
                value_X.append(best_obj_opt)
            else:
                print(f"{optimize_method} no success for step {step}")

            print('lookahead_n_fantasies:', lookahead_n_fantasies, 'eig', obj_list[:, 0].max().item())

            # Visualization for Location_budgeted using unified visualizer (two figures)
            if task_name == 'Location_budgeted' and visualizer is not None:
                # Figure 1: Theta posterior distribution
                visualizer.plot_step_location_theta(
                    step=step,
                    posterior_theta_0=posterior_theta_0,
                    true_theta=theta,
                    xs_all=X,
                    # best_opt_x=best_x_opt,
                )
                
                # Figure 2: Objective values scatter plot
                start_pts_np = np.array(start_points).squeeze(1) if len(start_points) > 0 else None
                opt_pts_np = np.array(opt_points).squeeze(1) if len(opt_points) > 0 else None
                
                # Plot objective for first lookahead step (index 0)
                if new_xs is not None and obj_list is not None:
                    visualizer.plot_step_location_objective(
                        step=step,
                        new_xs=new_xs[:, 0, :],  # First step candidates
                        obj_values=obj_list[:, 0],  # First step objective values
                        true_theta=theta,
                        best_opt_x=best_opt_tree[0, 0] if best_opt_tree is not None else None,
                        start_points=start_pts_np[:, 0, :] if start_pts_np is not None else None,
                        opt_points=opt_pts_np[:, 0, :] if opt_pts_np is not None else None,
                    )

        if optimize_method == 'design_only' or (success_any and best_x_opt is None):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            new_xs = warmstart_func(X, Y, task, [1], model, design_model,
                        n_random=0, n_optimal=1, sample='design', step_size=cfg.experiment.step_size,design_query_samples=200,lower_bound=lower_bound,upper_bound=upper_bound,target_x=target_x)
            new_xs = get_X_from_multi_step_tree_input_representation(new_xs) #[1, 1, D]
            # with torch.no_grad():
            #     objective, obj_list = acquisition_function.forward(new_xs)
            #     objective = objective.detach()
            # objective = 0
            new_x = new_xs[..., 0, 0, :].to(torch.get_default_dtype())
            new_x = new_x.reshape(-1, new_x.size(-1))
            print("use policy new_xs", new_xs)
            value_X.append(0)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if optimize_method == 'None':
            
            new_xs = warmstart_func(X, Y, task, num_fantasies, model, design_model,
                                    n_random=20, n_optimal=0, sample='random', 
                                    step_size=cfg.experiment.step_size,design_query_samples=200,lower_bound=lower_bound,upper_bound=upper_bound,target_x=target_x)

            new_xs = get_X_from_multi_step_tree_input_representation(new_xs)
            with torch.no_grad():
                objective, obj_list = acquisition_function.forward(new_xs)
                objective = objective.detach().view(-1)
            top_vals, top_idx = torch.topk(objective, k=1, largest=True)
            new_x = new_xs[top_idx, :].to(torch.get_default_dtype())
            print("use policy new_xs", new_x)
            value_X.append(top_vals.item())

        if optimize_method == 'random':
            xs = task.sample_data(1, 1)
            new_x = xs[:, 0, :].to(torch.get_default_dtype())

        
        
        # Visualization for AL benchmark using unified visualizer
        if (task_name == 'AL_benchmark' or task_name == 'AL_data') and visualizer is not None:
            if cfg.task.dim_x == 1:
                visualizer.plot_step_al_benchmark(
                    step=step,
                    target_x=target_x,
                    target_y=target_y,
                    posterior_data_0=posterior_data_0,
                    context_X=X,
                    context_Y=Y,
                    new_x=new_x,
                    objective=objective,
                    new_xs=new_xs,
                )
            else:
                # Use 2D visualization for dim_x >= 2
                visualizer.plot_step_al_benchmark_2d(
                    step=step,
                    target_x=target_x,
                    target_y=target_y,
                    posterior_data_0=posterior_data_0,
                    context_X=X,
                    context_Y=Y,
                    new_x=new_x,
                    objective=objective,
                    new_xs=new_xs,
                )

        print(f"next x is {new_x}. \n")
        
        # Compute and print cumulative distance between consecutive points
        # if X.shape[0] > 1:
        #     X_np = X.cpu().numpy()
        #     consecutive_dists = np.linalg.norm(np.diff(X_np, axis=0), axis=-1)
        #     cumulative_dist = np.sum(consecutive_dists)
        #     print(f"Cumulative cost (sum of consecutive point distances): {cumulative_dist:.4f}")
        #     print(f"Individual cost: {consecutive_dists.flatten()}")
        
        # End timing for this step
        step_time = time.time() - step_start_time
        design_time = time.time() - design_start_time
        time_X.append(step_time)
        print(f"Step {step} time: {step_time:.4f} seconds")
        print(f"Design time: {design_time:.4f} seconds")
        
        # Save results
        np.savetxt(os.path.join(results_folder, "X", f"X_{trial}.txt"), X.cpu().numpy())
        np.savetxt(os.path.join(results_folder, "Y", f"Y_{trial}.txt"), Y.cpu().numpy())
        if cost_function is not None:
            np.savetxt(os.path.join(results_folder, "cost_X", f"cost_X_{trial}.txt"), cost_X.cpu().numpy())
        np.savetxt(os.path.join(results_folder, "value_X", f"value_X_{trial}.txt"), np.atleast_1d(value_X))
        np.savetxt(os.path.join(results_folder, "eig_X", f"eig_X_{trial}.txt"), np.atleast_1d(eig_X))
        
        # Save detailed time info to time_info folder
        time_info_folder = os.path.join(results_folder, "time_info")
        np.savetxt(os.path.join(time_info_folder, f"total_time_{trial}.txt"), np.atleast_1d(time_X))
        
        # Save step-specific time info as JSON for detailed analysis
        device_info = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        # Use memory_info if available (from lbfgsb/slsqp branch), otherwise set to None
        step_memory_info = memory_info if 'memory_info' in dir() else None
        warmstart_time = warmstart_time if 'warmstart_time' in dir() else None
        eig_calculation_time = eig_calculation_time if 'eig_calculation_time' in dir() else None
        step_time_info = {
            "device": device_info,
            "step": step,
            "total_step_time": step_time,  # time for optimization with lookahead in one step in total (+ model fitting time)
            "tree_opt_times": tree_opt_times,  # time for optimization with lookahead in one step for each tree
            "tree_opt_sources": tree_opt_sources,
            "best_opt_source": best_opt_source,
            "n_random": n_random,
            "n_optimal": n_optimal,
            "warmstart_time": warmstart_time,
            "eig_calculation_time": eig_calculation_time,
            "design_time": design_time, # time for optimization with lookahead/time for policy design (optimize_method == 'design_only'/random)
            "memory_info": step_memory_info,  # Memory usage during acquisition_function.forward()
        }
        # Append step info to a JSON file
        time_info_json_path = os.path.join(time_info_folder, f"step_details_{trial}.json")
        try:
            if os.path.exists(time_info_json_path):
                with open(time_info_json_path, "r") as f:
                    all_step_info = json.load(f)
            else:
                all_step_info = []
            # Update or append current step info
            step_exists = False
            for i, info in enumerate(all_step_info):
                if info.get("step") == step:
                    all_step_info[i] = step_time_info
                    step_exists = True
                    break
            if not step_exists:
                all_step_info.append(step_time_info)
            with open(time_info_json_path, "w") as f:
                json.dump(all_step_info, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save step time info: {e}")

        # Clean up to avoid OOM
        try:
            del objective
            del new_xs
        except Exception:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Results saved in", results_folder)

    # Save and show unified visualization figure
    if visualizer is not None:
        if task_name == 'AL_benchmark' or task_name == 'AL_data':
            fig_dir = f"results/figures/AL/{benchmark_name}"
            os.makedirs(fig_dir, exist_ok=True)
            save_path = f"{fig_dir}/{trial}.pdf"
        elif task_name == 'Location_budgeted':
            fig_dir = f"results/figures/Location/{problem_name}"
            os.makedirs(fig_dir, exist_ok=True)
            save_path = f"{fig_dir}/{trial}.pdf"
        else:
            save_path = None
        
        visualizer.save_and_show(save_path=save_path, show=True)

    if wandb_run is not None:
        wandb_run.log({"results_folder": results_folder})
        wandb_run.finish()


@hydra.main(version_base=None, config_path="config", config_name="config_location_finding")
def hydra_main(cfg: DictConfig):
    """
    Hydra entry point - supports different config files.
    Task name is obtained from task.name (defined in config/task/*.yaml)
    
    Usage:
        python main.py --config-name=config_location_finding       # Location_budgeted
        python main.py --config-name=config_al_benchmark # AL_benchmark
        python main.py --config-name=config_ces          # CES
    """
    print("=" * 60)
    print("Configuration parameters:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    main(cfg)

if __name__ == "__main__":
    hydra_main()

