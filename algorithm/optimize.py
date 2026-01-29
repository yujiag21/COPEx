from scipy.optimize import minimize, Bounds
import torch
import numpy as np
from scipy.optimize import minimize
from algorithm.utils import compute_total_cost, compute_budget_cost


def make_slsqp_ineq_constraint(acquisition_function, start_x, h=None, constraint_function=None):
    """Create inequality constraint function for SLSQP optimizer.
    
    Args:
        acquisition_function: Acquisition function with get_multi_step_tree_input_representation
        start_x: Original start_x shape for reshaping
        h: Step size constraint value
        norm_type: 'l2' for ||X_i - X_{i-1}||_2 ≤ h (L2 norm)
                   'linf' for |X_i - X_{i-1}| ≤ h (element-wise, L-infinity style)
    
    Returns:
        List of constraint dicts compatible with scipy.optimize.minimize.
    """
    def slsqp_ineq_vec(x_np: np.ndarray) -> np.ndarray:
        # Note: get_multi_step_tree_input_representation requires batch dim
        # Determine target device from last_X
        last_X = acquisition_function.last_X
        if isinstance(last_X, np.ndarray):
            last_X = torch.from_numpy(last_X)
        target_device = last_X.device if isinstance(last_X, torch.Tensor) else torch.device('cpu')
        
        # Create tensor on the same device as last_X
        x_tensor = torch.tensor(x_np, dtype=torch.get_default_dtype(), device=target_device)
        X_list = acquisition_function.get_multi_step_tree_input_representation(
            x_tensor.view(start_x.shape)
        )
        if len(X_list) == 0:
            # No layers at all, return always-positive placeholder
            return np.array([1.0], dtype=np.float64)

        ineq_parts = []

        X0 = X_list[0]  # shape: [batch=1, q0, d]

        # Compatible shape: allow last_X to be [q0, d] or [1, q0, d]
        if last_X.ndim == 2:
            last_X = last_X.unsqueeze(0)  # -> [1, q0, d]
        
        # Ensure last_X is on the same device as X0
        last_X = last_X.to(X0.device)
        assert last_X.shape == X0.shape, f"last_X shape {last_X.shape} != X0 {X0.shape}"

        # X0 vs last_X constraint
        # diff0 = X0 - last_X  # [batch, q0, d]
        if constraint_function is not None:
            constraints = constraint_function(X0, prev_x=last_X, h=h) # [batch, q0, d]
            for constraint in constraints:
                ineq_parts.append(constraint.reshape(-1))

        # Subsequent layers (if any)
        for i in range(1, len(X_list)):
            Xi = X_list[i]      # [f_i, ..., f_1, batch, q_i, d]
            Xim1 = X_list[i-1]  # [f_{i-1},..., f_1, batch, q_{i-1}, d]

            # Broadcast parent to Xi's shape directly
            parent = Xim1.unsqueeze(0)  # [1, f_{i-1}, ..., batch, q_{i-1}, d]
            parent_broadcast = parent.expand(Xi.shape)  # broadcast to Xi's shape

            if constraint_function is not None:
                constraints = constraint_function(Xi, prev_x=parent_broadcast, h=h) # [f_i, ..., f_1, batch, q_i, d]
                for constraint in constraints:
                    ineq_parts.append(constraint.reshape(-1))

        ineq_all = torch.cat(ineq_parts).to(dtype=torch.float64, device='cpu').numpy()
        return ineq_all


    return [{'type': 'ineq', 'fun': slsqp_ineq_vec}]


def make_slsqp_budget_constraint(acquisition_function, start_x, total_budget, cost_function=None):
    """Create budget constraint: total L2 distance <= total_budget.
    
    Constraint: total_budget - sum(L2_distances) >= 0
    
    Args:
        acquisition_function: Acquisition function with get_multi_step_tree_input_representation
        start_x: Original start_x shape for reshaping
        total_budget: Total budget for all lookahead steps (step_size * (lookahead_n_fantasies + 1))
        
    Returns:
        List of constraint dicts compatible with scipy.optimize.minimize.
    """
    def slsqp_budget_ineq(x_np: np.ndarray) -> np.ndarray:
        # Determine target device from last_X
        last_X = acquisition_function.last_X
        if isinstance(last_X, torch.Tensor):
            target_device = last_X.device
        else:
            target_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        x_tensor = torch.tensor(x_np, dtype=torch.get_default_dtype(), device=target_device)
        total_dist = compute_total_cost(acquisition_function, x_tensor, start_x, cost_function=cost_function)
        # total_budget - total_dist >= 0
        return np.array([total_budget - total_dist.detach().cpu().item()], dtype=np.float64)
    
    return [{'type': 'ineq', 'fun': slsqp_budget_ineq}]


# SLSQP
def maximize_objective_slsqp(model, acquisition_function, start_x, num_y_samples=200, num_theta_samples=500,
                              maxiter=100, seed=12345,
                              bounds_eps=0.0,
                              boundary_penalty_weight=0.0,
                              trust_region_radius=0.0,
                              fd_eps=None, task=None, step_size=1, use_analytical_grad=True,
                              lower_bound=0.0, upper_bound=1.0,
                              optimize_mode='hard', soft_constraint_weight=1.0,
                              budget=0, step_norm_type='linf', cost_function=None, constraint_function=None):
    """Maximize objective within [lower_bound, upper_bound]^d using SLSQP, return optimal x and objective.

    Args:
        model: Fitted model providing posterior and aline_objective_model.
        acquisition_function: Acquisition function providing compute_theta_log_probs.
        x0 (array-like): Initial point, shape [d], within [lower_bound, upper_bound]^d.
        num_y_samples (int): Number of y samples.
        num_theta_samples (int): Number of theta samples.
        maxiter (int): Maximum iterations.
        seed (int): Random seed for deterministic optimization.
        bounds_eps (float): Absolute amount to shrink bounds, avoid sticking to boundary.
        boundary_penalty_weight (float): Weight for boundary log-barrier penalty.
        trust_region_radius (float): If >0, set local box constraint centered at x0.
        fd_eps (float|None): SciPy finite difference step size.
        lower_bound (float): Lower bound of optimization space, default 0.0.
        upper_bound (float): Upper bound of optimization space, default 1.0.
        optimize_mode (str): 'hard' - use SLSQP inequality constraints
                               'soft' - use soft constraint obj / (1 + L2_distance)
                               'budget' - use total budget constraint: sum(L2_distances) <= budget
                               'none' - no step size constraint
        soft_constraint_weight (float): weight for soft constraint: obj / (1 + weight * dist)
        budget: budget constraint value
        step_norm_type (str): 'l2' for ||X_i - X_{i-1}||_2 <= h (L2 norm)
                              'linf' for |X_i - X_{i-1}| <= h (element-wise, L-infinity style)

    Returns:
        (x_opt, obj_opt): x_opt is optimal point (np.ndarray), obj_opt is optimal objective (float).
    """
    if minimize is None:
        raise ImportError("SciPy not installed, cannot use SLSQP. Please install scipy first.")
    x0 = start_x
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    dim = x0.shape[0]
    dim_theta = task.n_target_theta
    # Base bounds [lower_bound + eps, upper_bound - eps]
    lower_base = float(lower_bound) + float(bounds_eps)
    upper_base = float(upper_bound) - float(bounds_eps)
    # If trust region radius is set, intersect with base bounds
    if trust_region_radius and trust_region_radius > 0.0:
        lower = np.maximum(lower_base, x0 - trust_region_radius)
        upper = np.minimum(upper_base, x0 + trust_region_radius)
    else:
        lower = np.full(dim, lower_base, dtype=np.float64)
        upper = np.full(dim, upper_base, dtype=np.float64)
    # Ensure lower bound is strictly less than upper bound
    upper = np.maximum(upper, lower + 1e-9)
    bounds = [(float(l), float(u)) for l, u in zip(lower, upper)]

    options = {'maxiter': maxiter}
    if fd_eps is not None:
        options['eps'] = fd_eps

    # Only convert step_size to float if it's not None (needed for constraint_function mode)
    h = float(step_size) if step_size is not None else None

    # Determine device - try multiple sources
    # Priority: acquisition_function.task.theta > task.theta > model params > CUDA > CPU
    opt_device = None
    
    # Try to get device from acquisition_function's task
    if hasattr(acquisition_function, 'task'):
        acq_task = acquisition_function.task
        if hasattr(acq_task, 'theta') and acq_task.theta is not None and isinstance(acq_task.theta, torch.Tensor):
            opt_device = acq_task.theta.device
    
    # Fallback to direct task
    if opt_device is None and hasattr(task, 'theta') and task.theta is not None and isinstance(task.theta, torch.Tensor):
        opt_device = task.theta.device
    
    # Fallback to model parameters
    if opt_device is None and hasattr(model, 'aline_objective_model') and model.aline_objective_model is not None:
        try:
            opt_device = next(model.aline_objective_model.parameters()).device
        except StopIteration:
            pass
    
    # Final fallback: CUDA if available
    if opt_device is None:
        opt_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    

    def f_neg_with_grad(x_np: np.ndarray):

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        x_leaf = torch.tensor(
            x_np,
            dtype=torch.get_default_dtype(),
            device=opt_device,
            requires_grad=True,
        )
        x_t = x_leaf.view(start_x.shape)

        obj, obj_list = acquisition_function.forward(x_t)
        
        # Soft constraint: obj / (1 + L2_distance)
        # if optimize_mode == 'soft':
        if cost_function is not None and optimize_mode == 'soft':
            dist = compute_total_cost(acquisition_function, x_t, start_x, cost_function=cost_function)
            obj = obj / (1 + soft_constraint_weight * dist)

        if boundary_penalty_weight and boundary_penalty_weight > 0.0:
            eps_val = max(bounds_eps, 1e-12)
            barrier_tensor = -torch.sum(
                torch.log(torch.clamp(x_leaf.flatten() - eps_val, min=1e-12)) +
                torch.log(torch.clamp(1.0 - eps_val - x_leaf.flatten(), min=1e-12))
            )
            obj = obj - boundary_penalty_weight * barrier_tensor

        obj.backward()
        grad = x_leaf.grad.view(-1).detach().cpu().numpy().astype(np.float64)

        return -obj.detach().cpu().item(), -grad

    # Choose constraints based on optimize_mode
    # if optimize_mode == 'hard':

    if optimize_mode == 'budget':
        total_budget = budget
        constraints = make_slsqp_budget_constraint(acquisition_function, start_x, total_budget, cost_function=cost_function)
    elif constraint_function is not None:
        constraints = make_slsqp_ineq_constraint(acquisition_function, start_x, h=h,
                                                     constraint_function=constraint_function)
    else:
        constraints = []  # 'soft' or 'none' mode: no hard constraints

    # if use_analytical_grad:
    res = minimize(f_neg_with_grad, x0=x0, method='SLSQP',
                       bounds=bounds, constraints=constraints, jac=True, options=options)
    # else:
    #     res = minimize(f_neg, x0=x0, method='SLSQP', constraints=constraints, bounds=bounds, options=options)

    x_opt = res.x
    obj_opt = -res.fun
    return x_opt, float(obj_opt)

# L - BFGS - B
def maximize_objective_lbfgsb(model, acquisition_function, start_x, num_y_samples=200, num_theta_samples=500,
                              maxiter=100, seed=12345,
                              bounds_eps=0.0,
                              boundary_penalty_weight=0.0,
                              trust_region_radius=0.0,
                              fd_eps=None, task=None, use_analytical_grad = True,
                              lower_bound=0.0, upper_bound=1.0):
    """Maximize objective within [lower_bound, upper_bound]^d using L-BFGS-B, return optimal x and objective.

    Args:
        model: Fitted model providing posterior and aline_objective_model.
        acquisition_function: Acquisition function providing compute_theta_log_probs.
        x0 (array-like): Initial point, shape [d], within [lower_bound, upper_bound]^d.
        num_y_samples (int): Number of y samples.
        num_theta_samples (int): Number of theta samples.
        maxiter (int): Maximum iterations.
        seed (int): Random seed for deterministic optimization.
        bounds_eps (float): Absolute amount to shrink bounds, avoid sticking to boundary.
        boundary_penalty_weight (float): Weight for boundary log-barrier penalty.
        trust_region_radius (float): If >0, set local box constraint centered at x0.
        fd_eps (float|None): SciPy L-BFGS-B finite difference step size.
        lower_bound (float): Lower bound of optimization space, default 0.0.
        upper_bound (float): Upper bound of optimization space, default 1.0.

    Returns:
        (x_opt, obj_opt): x_opt is optimal point (np.ndarray), obj_opt is optimal objective (float).
    """
    if minimize is None:
        raise ImportError("SciPy not installed, cannot use L-BFGS-B. Please install scipy first.")
    x0 = start_x
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    dim = x0.shape[0]
    try:    
        dim_theta = task.n_target_theta
    except:
        dim_theta = task.n_theta
    # Base bounds [lower_bound + eps, upper_bound - eps]
    lower_base = float(lower_bound) + float(bounds_eps)
    upper_base = float(upper_bound) - float(bounds_eps)
    # If trust region radius is set, intersect with base bounds
    if trust_region_radius and trust_region_radius > 0.0:
        lower = np.maximum(lower_base, x0 - trust_region_radius)
        upper = np.minimum(upper_base, x0 + trust_region_radius)
    else:
        lower = np.full(dim, lower_base, dtype=np.float64)
        upper = np.full(dim, upper_base, dtype=np.float64)
    # Ensure lower bound is strictly less than upper bound
    upper = np.maximum(upper, lower + 1e-9)
    bounds = [(float(l), float(u)) for l, u in zip(lower, upper)]

    # Determine device - try multiple sources
    # Priority: acquisition_function.task.theta > task.theta > model params > CUDA > CPU
    opt_device = None
    
    # Try to get device from acquisition_function's task
    if hasattr(acquisition_function, 'task'):
        acq_task = acquisition_function.task
        if hasattr(acq_task, 'theta') and acq_task.theta is not None and isinstance(acq_task.theta, torch.Tensor):
            opt_device = acq_task.theta.device
    
    # Fallback to direct task
    if opt_device is None and hasattr(task, 'theta') and task.theta is not None and isinstance(task.theta, torch.Tensor):
        opt_device = task.theta.device
    
    # Fallback to model parameters
    if opt_device is None and hasattr(model, 'aline_objective_model') and model.aline_objective_model is not None:
        try:
            opt_device = next(model.aline_objective_model.parameters()).device
        except StopIteration:
            pass
    
    # Final fallback: CUDA if available
    if opt_device is None:
        opt_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def f_neg(x_np: np.ndarray) -> float:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        x_t = torch.tensor(x_np, dtype=torch.get_default_dtype(), device=opt_device)
        x_t = x_t.view(x0.shape)

        obj = acquisition_function.forward(x_t)
        obj = obj.detach().cpu().item()
        if boundary_penalty_weight and boundary_penalty_weight > 0.0:
            eps_val = max(bounds_eps, 1e-12)
            barrier = -np.sum(
                np.log(np.clip(x_np - eps_val, 1e-12, None)) +
                np.log(np.clip(1.0 - eps_val - x_np, 1e-12, None))
            )
            obj = obj - boundary_penalty_weight * barrier
        return -obj

    def f_neg_with_grad(x_np: np.ndarray):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        x_leaf = torch.tensor(
            x_np,
            dtype=torch.get_default_dtype(),
            device=opt_device,
            requires_grad=True,
        )
        x_t = x_leaf.view(start_x.shape)

        obj, obj_list = acquisition_function.forward(x_t)

        if boundary_penalty_weight and boundary_penalty_weight > 0.0:
            eps_val = max(bounds_eps, 1e-12)
            barrier_tensor = -torch.sum(
                torch.log(torch.clamp(x_leaf.flatten() - eps_val, min=1e-12)) +
                torch.log(torch.clamp(1.0 - eps_val - x_leaf.flatten(), min=1e-12))
            )
            obj = obj - boundary_penalty_weight * barrier_tensor

        obj.backward()
        grad = x_leaf.grad.view(-1).detach().cpu().numpy().astype(np.float64)

        return -obj.detach().cpu().item(), -grad

    options = {'maxiter': maxiter}
    if fd_eps is not None:
        options['eps'] = fd_eps

    if use_analytical_grad:
        res = minimize(f_neg_with_grad, x0=x0, method='L-BFGS-B', bounds=bounds, options=options, jac=True)
    else:
        res = minimize(f_neg, x0=x0, method='L-BFGS-B', bounds=bounds, options=options)

    x_opt = res.x
    obj_opt = -res.fun
    return x_opt, float(obj_opt)
