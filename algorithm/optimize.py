import functools
import threading

from scipy.optimize import minimize, Bounds
import torch
import numpy as np
from scipy.optimize import minimize
from algorithm.utils import compute_total_cost, compute_budget_cost

_RNG_LOCK = threading.RLock()


def _with_fixed_noise(seed):
    """Decorator: evaluate the objective with torch's global generator seeded with `seed`, atomically across threads."""
    def decorate(fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            with _RNG_LOCK:
                torch.manual_seed(seed)
                return fn(*args, **kwargs)
        return wrapped
    return decorate


def _tree_tensor(acquisition_function, x_np):
    """Flat numpy vector -> tensor on the device of acquisition_function.last_X."""
    last_X = acquisition_function.last_X
    if isinstance(last_X, np.ndarray):
        last_X = torch.from_numpy(last_X)
    device = last_X.device if isinstance(last_X, torch.Tensor) else torch.device('cpu')
    return torch.tensor(x_np, dtype=torch.get_default_dtype(), device=device)


def _constraint_dict(fun_torch, acquisition_function):
    """SLSQP constraint fun >= 0 with an autograd Jacobian."""
    def fun(x_np):
        with torch.no_grad():
            return fun_torch(_tree_tensor(acquisition_function, x_np)).to(torch.float64).cpu().numpy()

    def jac(x_np):
        x_t = _tree_tensor(acquisition_function, x_np)
        jacobian = torch.autograd.functional.jacobian(fun_torch, x_t, vectorize=True)
        return jacobian.reshape(-1, x_t.numel()).to(torch.float64).cpu().numpy()

    return {'type': 'ineq', 'fun': fun, 'jac': jac}


def make_slsqp_ineq_constraint(acquisition_function, start_x, h=None, constraint_function=None):
    """Transition constraints constraint_function(node, prev_x=parent, h) >= 0 for every node of the tree
    (the root's parent is acquisition_function.last_X)."""
    def ineq_torch(x_tensor):
        X_list = acquisition_function.get_multi_step_tree_input_representation(x_tensor.view(start_x.shape))
        if len(X_list) == 0:
            return torch.ones(1, dtype=x_tensor.dtype, device=x_tensor.device)
        last_X = acquisition_function.last_X
        if isinstance(last_X, np.ndarray):
            last_X = torch.from_numpy(last_X)
        parents = [last_X.to(X_list[0]).reshape(1, -1, X_list[0].shape[-1]).expand(X_list[0].shape)]
        parents += [X_list[i - 1].unsqueeze(0).expand(X_list[i].shape) for i in range(1, len(X_list))]
        parts = []
        for Xi, parent in zip(X_list, parents):
            constraints = constraint_function(Xi, prev_x=parent, h=h)
            if isinstance(constraints, torch.Tensor):
                constraints = (constraints,)
            parts += [c.reshape(-1) for c in constraints]
        return torch.cat(parts)

    return [_constraint_dict(ineq_torch, acquisition_function)]


def make_slsqp_budget_constraint(acquisition_function, start_x, total_budget, cost_function=None):
    """Budget constraint: the undiscounted cost of every root-to-leaf path of the tree <= total_budget."""
    def budget_torch(x_tensor):
        path_cost = compute_budget_cost(acquisition_function, x_tensor, start_x, cost_function=cost_function)
        return total_budget - path_cost

    return [_constraint_dict(budget_torch, acquisition_function)]


def transition_box(acquisition_function, dim, h, lower, upper):
    """Bounds of the root design under the L-inf transition constraint."""
    last_X = acquisition_function.last_X
    last_x = (last_X.detach().cpu().numpy() if isinstance(last_X, torch.Tensor) else np.asarray(last_X)).reshape(-1)[:dim]
    lower, upper = lower.copy(), upper.copy()
    lower[:dim] = np.maximum(lower[:dim], last_x - h)
    upper[:dim] = np.minimum(upper[:dim], last_x + h)
    return lower, upper


def max_constraint_violation(constraints, x):
    """Largest violation of the SLSQP inequality constraints (0 if x is feasible)."""
    return max([0.0] + [-float(np.min(np.atleast_1d(c['fun'](x)))) for c in constraints])


def log_barrier(x, lower_bound, upper_bound, bounds_eps):
    """Log barrier of the designs normalized to [0, 1] by the bounds (torch tensor or numpy array)."""
    scale = float(upper_bound) - float(lower_bound)
    eps = max(bounds_eps, 1e-12) / scale
    u = (x - float(lower_bound)) / scale
    if isinstance(u, torch.Tensor):
        return -torch.sum(torch.log(torch.clamp(u - eps, min=1e-12)) + torch.log(torch.clamp(1.0 - eps - u, min=1e-12)))
    return -np.sum(np.log(np.clip(u - eps, 1e-12, None)) + np.log(np.clip(1.0 - eps - u, 1e-12, None)))


# SLSQP
def maximize_objective_slsqp(model, acquisition_function, start_x, num_y_samples=200, num_theta_samples=500,
                              maxiter=100, seed=12345,
                              bounds_eps=0.0,
                              boundary_penalty_weight=0.0,
                              trust_region_radius=0.0,
                              fd_eps=None, task=None, step_size=1, use_analytical_grad=True,
                              lower_bound=0.0, upper_bound=1.0,
                              optimize_mode='hard', soft_constraint_weight=1.0,
                              budget=0, step_norm_type='linf', cost_function=None, constraint_function=None,
                              budget_transition_constraint=False, feasibility_tol=1e-4):
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
        budget_transition_constraint (bool): in 'budget' mode, also enforce the transition constraint
        feasibility_tol (float): largest accepted constraint violation

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
    h = float(step_size) if step_size is not None else None
    enforce_transition = constraint_function is not None and h is not None and (
        optimize_mode != 'budget' or budget_transition_constraint)
    # The root's L-inf transition constraint is expressed through the bounds
    dim_x = start_x.shape[-1]
    if enforce_transition and getattr(constraint_function, '__name__', '') == 'linf_constraint_function':
        lower, upper = transition_box(acquisition_function, dim_x, h, lower, upper)
    # Ensure lower bound is strictly less than upper bound
    upper = np.maximum(upper, lower + 1e-9)
    bounds = [(float(l), float(u)) for l, u in zip(lower, upper)]
    x0 = np.clip(x0, lower, upper)

    options = {'maxiter': maxiter}
    if fd_eps is not None:
        options['eps'] = fd_eps

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
    

    @_with_fixed_noise(seed)
    def f_neg_with_grad(x_np: np.ndarray):
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
        
        # Cost-normalized utility obj / (1 + w * cost of the tree)
        if cost_function is not None and optimize_mode in ('soft', 'budget') and soft_constraint_weight:
            dist = compute_total_cost(acquisition_function, x_t, start_x, cost_function=cost_function)
            obj = obj / (1 + soft_constraint_weight * dist)

        if boundary_penalty_weight and boundary_penalty_weight > 0.0:
            obj = obj - boundary_penalty_weight * log_barrier(x_leaf.flatten(), lower_bound, upper_bound, bounds_eps)

        obj.backward()
        grad = x_leaf.grad.view(-1).detach().cpu().numpy().astype(np.float64)

        return -obj.detach().cpu().item(), -grad

    constraints = []
    if optimize_mode == 'budget':
        constraints += make_slsqp_budget_constraint(acquisition_function, start_x, budget, cost_function=cost_function)
    if enforce_transition:
        constraints += make_slsqp_ineq_constraint(acquisition_function, start_x, h=h,
                                                  constraint_function=constraint_function)

    res = minimize(f_neg_with_grad, x0=x0, method='SLSQP',
                   bounds=bounds, constraints=constraints, jac=True, options=options)

    x_opt, obj_opt = res.x, -res.fun
    start_obj = -f_neg_with_grad(x0)[0]
    violation = max_constraint_violation(constraints, x_opt)
    # If SLSQP stops at an infeasible or worse point, keep the (feasible) start tree
    if violation > feasibility_tol or obj_opt < start_obj:
        start_violation = max_constraint_violation(constraints, x0)
        if start_violation <= max(feasibility_tol, violation):
            print(f"SLSQP: result rejected (violation {violation:.2e}, obj {obj_opt:.5f} vs start {start_obj:.5f}); "
                  f"keeping the start tree ({res.message})")
            x_opt, obj_opt, violation = x0, start_obj, start_violation
    print(f"SLSQP: success={res.success} nit={res.nit} start_obj={start_obj:.5f} obj={obj_opt:.5f} "
          f"max_violation={violation:.2e} msg={res.message}")
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

    @_with_fixed_noise(seed)
    def f_neg(x_np: np.ndarray) -> float:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        x_t = torch.tensor(x_np, dtype=torch.get_default_dtype(), device=opt_device)
        x_t = x_t.view(x0.shape)

        obj = acquisition_function.forward(x_t)
        obj = obj.detach().cpu().item()
        if boundary_penalty_weight and boundary_penalty_weight > 0.0:
            obj = obj - boundary_penalty_weight * log_barrier(x_np, lower_bound, upper_bound, bounds_eps)
        return -obj

    @_with_fixed_noise(seed)
    def f_neg_with_grad(x_np: np.ndarray):
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
            obj = obj - boundary_penalty_weight * log_barrier(x_leaf.flatten(), lower_bound, upper_bound, bounds_eps)

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
