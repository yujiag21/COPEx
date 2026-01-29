import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from ALINE.utils import set_seed

from ALINE.inference_model import BaseTransformer
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from typing import Callable, Dict, Optional, List, Tuple
import os
import torch.nn.functional as F
from models.amortized_model import AmortizedBEDModel
from omegaconf import DictConfig

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch
from glob import glob

# Optional: better style
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 12

plt.rcParams.update({
    'font.family': 'times',
    'font.size': 14.0,
    'lines.linewidth': 2,
    'lines.antialiased': True,
    'axes.facecolor': 'fdfdfd',
    'axes.edgecolor': '777777',
    'axes.linewidth': 1,
    'axes.titlesize': 'medium',
    'axes.labelsize': 'medium',
    'axes.axisbelow': True,
    'xtick.major.size': 0,  # major tick size in points
    'xtick.minor.size': 0,  # minor tick size in points
    'xtick.major.pad': 6,  # distance to major tick label in points
    'xtick.minor.pad': 6,  # distance to the minor tick label in points
    'xtick.color': '333333',  # color of the tick labels
    'xtick.labelsize': 'medium',  # fontsize of the tick labels
    'xtick.direction': 'in',  # direction: in or out
    'ytick.major.size': 0,  # major tick size in points
    'ytick.minor.size': 0,  # minor tick size in points
    'ytick.major.pad': 6,  # distance to major tick label in points
    'ytick.minor.pad': 6,  # distance to the minor tick label in points
    'ytick.color': '333333',  # color of the tick labels
    'ytick.labelsize': 'medium',  # fontsize of the tick labels
    'ytick.direction': 'in',  # direction: in or out
    'axes.grid': False,
    'grid.alpha': 0.3,
    'grid.linewidth': 1,
    'legend.fancybox': True,
    'legend.fontsize': 'Small',
    'figure.figsize': (2.5, 2.5),
    'figure.facecolor': '1.0',
    'figure.edgecolor': '0.5',
    'hatch.linewidth': 0.1,
    'text.usetex': True})

tol_colors = {
    'blue':   '#4477AA',
    'green':  '#228833',
    'red':    '#EE6677',
    'purple': '#AA3377',
    'yellow': '#CCBB44',
    'cyan':   '#66CCEE',
    'grey':   '#BBBBBB',
    'orange': '#FF9933',
}


def compute_total_cost(acquisition_function, x_tensor, start_x, cost_function=None):
    """Compute total L2 norm distance between adjacent layers in multi-step tree.
    
    For each point in the tree, computes L2 distance to its parent node along the
    design dimension (d), then sums all distances.
    
    Args:
        acquisition_function: Acquisition function with get_multi_step_tree_input_representation
        x_tensor: Current x tensor (same dtype/device as model)
        start_x: Original start_x shape for reshaping
        
    Returns:
        torch.Tensor: Total L2 distance (scalar), supports gradient backprop
    """
    X_list = acquisition_function.get_multi_step_tree_input_representation(
        x_tensor.view(start_x.shape)
    )
    
    if len(X_list) == 0:
        # No layers at all, return 0 distance
        return torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)
    
    # total_dist = torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)
    
    # Layer 0 vs last_X (always compute if X_list has at least 1 layer)
    X0 = X_list[0]  # shape: [batch=1, q0, d]
    last_X = acquisition_function.last_X
    if isinstance(last_X, np.ndarray):
        last_X = torch.from_numpy(last_X)
    last_X = last_X.to(dtype=x_tensor.dtype, device=x_tensor.device)
    
    # Compatible shape: allow last_X to be [q0, d] or [1, q0, d]
    if last_X.ndim == 2:
        last_X = last_X.unsqueeze(0)  # -> [1, q0, d]

    dist0_per_point = cost_function(X0, prev_x=last_X)
    # diff0 = X0 - last_X  # [batch, q0, d]
    # # Compute L2 norm along d dimension for each point, then sum
    # dist0_per_point = torch.norm(diff0, dim=-1)  # [batch, q0]
    total_dist = dist0_per_point
    weight=1
    # Subsequent layers (if any)
    for i in range(1, len(X_list)):
        Xi = X_list[i]      # [f_i, ..., f_1, batch, q_i, d]
        Xim1 = X_list[i-1]  # [f_{i-1},..., f_1, batch, q_{i-1}, d]
        
        # Weight by inverse of total nodes at this layer (average distance per node)
        # weight = 1 / int(np.prod(list(Xi.shape[:-2])))
        weight= weight*acquisition_function.discount_factor
        
        # Broadcast parent to Xi's shape directly
        # Xi: [f_i, f_{i-1}, ..., f_1, batch, q_i, d]
        # Xim1: [f_{i-1}, ..., f_1, batch, q_{i-1}, d]
        # Need to add f_i dim at front and match q dimension
        # Assume q_{i-1} = q_i = 1 (single candidate per step)
        parent = Xim1.unsqueeze(0)  # [1, f_{i-1}, ..., batch, q_{i-1}, d]
        parent_broadcast = parent.expand(Xi.shape)  # broadcast to Xi's shape
        
        # diff = Xi - parent_broadcast  # [f_i, ..., f_1, batch, q_i, d]
        # Compute L2 norm along d dimension for each point, then weighted sum
        # dist_per_point = torch.norm(diff, dim=-1)  # [f_i, ..., f_1, batch, q_i]
        # if cost_function is not None:
        cost_per_point = cost_function(Xi, prev_x=parent_broadcast)  # [f_i, ..., f_1, batch, q_i]
        # else:
            # cost_per_point = torch.norm(diff, dim=-1)  # [f_i, ..., f_1, batch, q_i]
        total_dist = total_dist + weight * cost_per_point
    
    batch_shape = X0.shape[:-1]
    total_dist = total_dist.view(-1, *batch_shape).mean(dim=0)
    return total_dist


def compute_budget_cost(acquisition_function, x_tensor, start_x, cost_function=None):
    """Compute total L2 norm distance between adjacent layers in multi-step tree.

    For each point in the tree, computes L2 distance to its parent node along the
    design dimension (d), then sums all distances.

    Args:
        acquisition_function: Acquisition function with get_multi_step_tree_input_representation
        x_tensor: Current x tensor (same dtype/device as model)
        start_x: Original start_x shape for reshaping

    Returns:
        torch.Tensor: Total L2 distance (scalar), supports gradient backprop
    """
    X_list = acquisition_function.get_multi_step_tree_input_representation(
        x_tensor.view(start_x.shape)
    )

    if len(X_list) == 0:
        # No layers at all, return 0 distance
        return torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)

    total_dist = torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)

    # Layer 0 vs last_X (always compute if X_list has at least 1 layer)
    X0 = X_list[0]  # shape: [batch=1, q0, d]
    last_X = acquisition_function.last_X
    if isinstance(last_X, np.ndarray):
        last_X = torch.from_numpy(last_X)
    last_X = last_X.to(dtype=x_tensor.dtype, device=x_tensor.device)

    # Compatible shape: allow last_X to be [q0, d] or [1, q0, d]
    if last_X.ndim == 2:
        last_X = last_X.unsqueeze(0)  # -> [1, q0, d]

    dist0_per_point = cost_function(X0, prev_x=last_X)
    # diff0 = X0 - last_X  # [batch, q0, d]
    # # Compute L2 norm along d dimension for each point, then sum
    # dist0_per_point = torch.norm(diff0, dim=-1)  # [batch, q0]
    total_dist = total_dist + dist0_per_point.sum()
    # Subsequent layers (if any)
    for i in range(1, len(X_list)):
        Xi = X_list[i]  # [f_i, ..., f_1, batch, q_i, d]
        Xim1 = X_list[i - 1]  # [f_{i-1},..., f_1, batch, q_{i-1}, d]

        # Weight by inverse of total nodes at this layer (average distance per node)
        # weight = 1 / int(np.prod(list(Xi.shape[:-2])))
        # weight = weight * weight_factor

        # Broadcast parent to Xi's shape directly
        # Xi: [f_i, f_{i-1}, ..., f_1, batch, q_i, d]
        # Xim1: [f_{i-1}, ..., f_1, batch, q_{i-1}, d]
        # Need to add f_i dim at front and match q dimension
        # Assume q_{i-1} = q_i = 1 (single candidate per step)
        parent = Xim1.unsqueeze(0)  # [1, f_{i-1}, ..., batch, q_{i-1}, d]
        parent_broadcast = parent.expand(Xi.shape)  # broadcast to Xi's shape

        # diff = Xi - parent_broadcast  # [f_i, ..., f_1, batch, q_i, d]
        # Compute L2 norm along d dimension for each point, then weighted sum
        # dist_per_point = torch.norm(diff, dim=-1)  # [f_i, ..., f_1, batch, q_i]
        # if cost_function is not None:
        cost_per_point = cost_function(Xi, prev_x=parent_broadcast)  # [f_i, ..., f_1, batch, q_i]
        # else:
        # cost_per_point = torch.norm(diff, dim=-1)  # [f_i, ..., f_1, batch, q_i]
        total_dist = total_dist + cost_per_point.sum()

    return total_dist


def compute_total_distance_l2(acquisition_function, x_tensor, start_x):
    """Compute total L2 norm distance between adjacent layers in multi-step tree.
    
    Args:
        acquisition_function: Acquisition function with get_multi_step_tree_input_representation
        x_tensor: Current x tensor (same dtype/device as model)
        start_x: Original start_x shape for reshaping
        
    Returns:
        torch.Tensor: Total L2 distance (scalar), supports gradient backprop
    """
    X_list = acquisition_function.get_multi_step_tree_input_representation(
        x_tensor.view(start_x.shape)
    )
    
    if len(X_list) == 0:
        # No layers at all, return 0 distance
        return torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)
    
    total_dist = torch.tensor(0.0, dtype=x_tensor.dtype, device=x_tensor.device)
    
    # Layer 0 vs last_X (always compute if X_list has at least 1 layer)
    X0 = X_list[0]  # shape: [batch=1, q0, d]
    last_X = acquisition_function.last_X
    if isinstance(last_X, np.ndarray):
        last_X = torch.from_numpy(last_X)
    last_X = last_X.to(dtype=x_tensor.dtype, device=x_tensor.device)
    
    # Compatible shape: allow last_X to be [q0, d] or [1, q0, d]
    if last_X.ndim == 2:
        last_X = last_X.unsqueeze(0)  # -> [1, q0, d]
    
    diff0 = X0 - last_X
    total_dist = total_dist + torch.norm(diff0)  # L2 norm
    
    # Subsequent layers (if any)
    for i in range(1, len(X_list)):
        Xi = X_list[i]      # [f_i, ..., f_1, 1, q_i, d]
        Xim1 = X_list[i-1]  # [f_{i-1},..., f_1, 1, q_{i-1}, d]
        
        # Broadcast parent layer to child layer
        fi = Xi.shape[0]
        qi = Xi.shape[-2]
        parent = Xim1.unsqueeze(0).unsqueeze(-2)
        reps = [fi] + [1] * (parent.ndim - 2) + [qi, 1]
        parent_broadcast = parent.repeat(*reps)
        
        diff = Xi - parent_broadcast
        total_dist = total_dist + torch.norm(diff)  # L2 norm
    
    return total_dist




def load_model_and_config(model_path: str, cfg: DictConfig, design_model_path: str = None, design_config_name: str = None):
    """Load model and config

    Args:
        model_path (str): Model file path
        config_name (str): Config file name

    Returns:
        tuple: (cfg, experiment, model)
    """

    print(f"cfg: {cfg}")

    print("Config info:")
    print(OmegaConf.to_yaml(cfg))

    # Set device
    if not torch.cuda.is_available():
        cfg.device = "cpu"
    torch.set_default_device(cfg.device)
    if cfg.device == "cuda":
        torch.set_default_dtype(torch.float32)

    # Set random seed
    if cfg.fix_seed:
        set_seed(cfg.seed)

    # Create task
    experiment = instantiate(cfg.task)

    # Create model
    embedder = instantiate(cfg.embedder)
    encoder = instantiate(cfg.encoder)
    head = instantiate(cfg.head)
    model = BaseTransformer(embedder, encoder, head)

    # Load model weights
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=cfg.device, weights_only=False)
        if type(state_dict) is dict:
            state_dict = state_dict["model"]
        model.load_state_dict(state_dict)
        print(f"Successfully loaded model: {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model.eval()
    if design_model_path is not None:
        # Clear existing Hydra instance to avoid conflicts
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize(version_base=None, config_path="../ALINE/config"):
            cfg = compose(config_name=design_config_name)

        # Set device
        if not torch.cuda.is_available():
            cfg.device = "cpu"
        torch.set_default_device(cfg.device)
        if cfg.device == "cuda":
            torch.set_default_dtype(torch.float32)

        # Set random seed
        if cfg.fix_seed:
            set_seed(cfg.seed)

        # Create model
        embedder = instantiate(cfg.embedder)
        encoder = instantiate(cfg.encoder)
        head = instantiate(cfg.head)
        design_model = BaseTransformer(embedder, encoder, head)

        # Load model weights
        if os.path.exists(design_model_path):
            state_dict = torch.load(design_model_path, map_location=cfg.device, weights_only=False)
            if type(state_dict) is dict:
                state_dict = state_dict["model"]
            design_model.load_state_dict(state_dict)
            print(f"Successfully loaded design model: {design_model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {design_model_path}")
        design_model.eval()
    else:
        design_model = None
    return cfg, experiment, model, design_model

class StepVisualizer:
    """Unified visualization manager for multi-step experiments.
    
    Manages subplot layout and provides consistent plotting interface
    for different task types (AL_benchmark, Location_budgeted, etc.).
    
    For Location_budgeted, manages TWO figures:
        - fig_theta: theta posterior distribution heatmaps
        - fig_obj: objective value scatter plots
    
    Usage:
        viz = StepVisualizer(task_name='Location_budgeted', n_steps=30, cols=5)
        for step in range(n_steps):
            # ... experiment code ...
            viz.plot_step(step, ...)
        viz.save_and_show('output.pdf')
    """
    
    def __init__(
        self, 
        task_name: str, 
        n_steps: int, 
        cols: int = 5,
        figsize_per_subplot: Tuple[float, float] = (3.2, 2.8),
        verbose: int = 1,
    ):
        """Initialize the visualizer.
        
        Args:
            task_name: Name of the task ('AL_benchmark', 'Location_budgeted', etc.)
            n_steps: Total number of steps in the experiment
            cols: Number of columns in the subplot grid
            figsize_per_subplot: (width, height) per subplot
            verbose: Plotting frequency (plot every `verbose` steps)
        """
        self.task_name = task_name
        self.n_steps = n_steps
        self.cols = cols
        self.verbose = verbose
        
        # Calculate layout
        self.n_plots = (n_steps + verbose - 1) // verbose
        self.rows = (self.n_plots + cols - 1) // cols
        
        # Create figure(s)
        fig_width = figsize_per_subplot[0] * cols
        fig_height = figsize_per_subplot[1] * self.rows
        
        if task_name == 'Location_budgeted':
            # Two separate figures for Location_budgeted
            self.fig_theta = plt.figure(figsize=(fig_width, fig_height))
            self.fig_obj = plt.figure(figsize=(fig_width, fig_height))
            self._plot_idx_theta = 0
            self._plot_idx_obj = 0
            self.fig = self.fig_theta  # For compatibility
        else:
            self.fig = plt.figure(figsize=(fig_width, fig_height))
            self.fig_theta = None
            self.fig_obj = None
        
        self._plot_idx = 0  # Track which subplot we're on
        
    def _get_next_ax(self, step: int, fig_type: str = 'main') -> Optional[plt.Axes]:
        """Get the next subplot axis if this step should be plotted.
        
        Args:
            step: Current step index
            fig_type: 'main', 'theta', or 'obj' for Location_budgeted
        """
        if (step + 1) % self.verbose != 0:
            return None
        
        if fig_type == 'theta' and self.fig_theta is not None:
            self._plot_idx_theta += 1
            return self.fig_theta.add_subplot(self.rows, self.cols, self._plot_idx_theta)
        elif fig_type == 'obj' and self.fig_obj is not None:
            self._plot_idx_obj += 1
            return self.fig_obj.add_subplot(self.rows, self.cols, self._plot_idx_obj)
        else:
            self._plot_idx += 1
            return self.fig.add_subplot(self.rows, self.cols, self._plot_idx)
    
    def plot_step_al_benchmark(
        self, 
        step: int,
        target_x: Tensor,
        target_y: Tensor,
        posterior_data_0: dict,
        context_X: Tensor,
        context_Y: Tensor,
        new_x: Tensor,
        objective: Optional[Tensor] = None,
        new_xs: Optional[Tensor] = None,
    ) -> Optional[plt.Axes]:
        """Plot a single step for AL_benchmark task.
        
        Args:
            step: Current step index
            target_x: Target x values for ground truth
            target_y: Target y values for ground truth
            posterior_data_0: Model posterior containing mixture parameters
            context_X: Context x points
            context_Y: Context y values
            new_x: Next query point
            objective: Optional EPIG values for candidate points
            new_xs: Optional candidate points corresponding to objective values
        
        Returns:
            The matplotlib Axes object, or None if step not plotted
        """
        ax = self._get_next_ax(step)
        if ax is None:
            return None
            
        x_values = target_x.detach().cpu()
        y_values = target_y.detach().cpu()
        
        means = posterior_data_0['mixture_means'][0].detach().cpu()
        stds = posterior_data_0['mixture_stds'][0].detach().cpu()
        weights = posterior_data_0['mixture_weights'][0].detach().cpu()
        
        all_x = x_values.flatten().numpy()
        
        weighted_means = np.sum(weights.numpy() * means.numpy(), axis=-1)
        weighted_variance = np.sum(weights.numpy() * (stds.numpy() ** 2 +
                                   (means.numpy() - weighted_means[:, None]) ** 2), axis=-1)
        weighted_stds = np.sqrt(weighted_variance)
        
        all_means = weighted_means
        all_lower = weighted_means - 2 * weighted_stds
        all_upper = weighted_means + 2 * weighted_stds
        
        sort_indices = np.argsort(all_x)
        all_x = all_x[sort_indices]
        all_means = all_means[sort_indices]
        all_lower = all_lower[sort_indices]
        all_upper = all_upper[sort_indices]
        
        all_gt = y_values.numpy().reshape(-1)[sort_indices]
        
        ax.plot(all_x, all_means, 'C0', label='Prediction')
        ax.fill_between(all_x, all_lower, all_upper, color='b', alpha=0.2)
        ax.plot(all_x, all_gt, 'C3', label='Ground Truth')
        ax.scatter(all_x, all_gt, color='black', s=10, label='Targets')
        
        context_x_np = context_X.detach().cpu().numpy()
        context_y_np = context_Y.detach().cpu().numpy()
        ax.scatter(context_x_np, context_y_np, color='C2', s=30, marker='o', label='Context')
        
        next_x_plot = new_x.detach().cpu().numpy()
        ax.axvline(x=next_x_plot.flatten()[0], color="r", linestyle="--", linewidth=1.5, label="Next Query")
        
        # Plot EPIG values on secondary axis if available
        if objective is not None and new_xs is not None:
            try:
                ax2 = ax.twinx()
                epig_x = new_xs[:, 0, 0].detach().cpu().numpy()
                epig_vals = objective.detach().cpu().numpy()
                
                epig_sort_idx = np.argsort(epig_x)
                epig_x = epig_x[epig_sort_idx]
                epig_vals = epig_vals[epig_sort_idx]
                
                ax2.plot(epig_x, epig_vals, 'C4--', alpha=0.7, label='EPIG')
                ax2.scatter(epig_x, epig_vals, color='C4', s=15, alpha=0.7)
                
                if self._plot_idx % self.cols == 0 or step == self.n_steps - 1:
                    ax2.set_ylabel('EPIG', fontsize=10, color='C4')
                ax2.tick_params(axis='y', labelcolor='C4')
                
                if self._plot_idx == 1:
                    ax2.legend(loc='lower right', fontsize='x-small')
            except Exception:
                pass
        
        ax.set_title(f'Step {step + 1}', fontsize=12)
        
        if self._plot_idx > (self.rows - 1) * self.cols:
            ax.set_xlabel('x', fontsize=10)
        if (self._plot_idx - 1) % self.cols == 0:
            ax.set_ylabel('y', fontsize=10)
        if self._plot_idx == 1:
            ax.legend(loc='upper right', fontsize='x-small')
            
        return ax

    def plot_step_al_benchmark_2d(
        self,
        step: int,
        target_x: Tensor,
        target_y: Tensor,
        posterior_data_0: dict,
        context_X: Tensor,
        context_Y: Tensor,
        new_x: Tensor,
        objective: Optional[Tensor] = None,
        new_xs: Optional[Tensor] = None,
        grid_size: int = 50,
    ) -> Optional[plt.Axes]:
        """Plot a single step for AL_benchmark task with 2D input.
        
        Uses contour plot for ground truth and prediction heatmap overlay.
        
        Args:
            step: Current step index
            target_x: Target x values for ground truth, shape [N, 2]
            target_y: Target y values for ground truth, shape [N, 1] or [N]
            posterior_data_0: Model posterior containing mixture parameters
            context_X: Context x points, shape [M, 2]
            context_Y: Context y values, shape [M, 1] or [M]
            new_x: Next query point, shape [1, 2] or [2]
            objective: Optional EPIG values for candidate points
            new_xs: Optional candidate points corresponding to objective values
            grid_size: Resolution for interpolation grid
        
        Returns:
            The matplotlib Axes object, or None if step not plotted
        """
        ax = self._get_next_ax(step)
        if ax is None:
            return None
        
        # Extract data and handle batch dimension
        x_values = target_x.detach().cpu().numpy()
        y_values = target_y.detach().cpu().numpy()
        
        # Remove batch dimension if present: [1, N, 2] -> [N, 2]
        if x_values.ndim == 3:
            x_values = x_values[0]
        if y_values.ndim == 3:
            y_values = y_values[0]
        y_values = y_values.flatten()  # [N]
        
        means = posterior_data_0['mixture_means'][0].detach().cpu().numpy()  # [N, C]
        stds = posterior_data_0['mixture_stds'][0].detach().cpu().numpy()  # [N, C]
        weights = posterior_data_0['mixture_weights'][0].detach().cpu().numpy()  # [N, C]
        
        # Compute weighted mean predictions
        weighted_means = np.sum(weights * means, axis=-1)  # [N]
        
        # Get coordinate ranges (x_values is now [N, 2])
        x1_min, x1_max = x_values[:, 0].min(), x_values[:, 0].max()
        x2_min, x2_max = x_values[:, 1].min(), x_values[:, 1].max()
        
        # Create interpolation grid
        x1_grid = np.linspace(x1_min, x1_max, grid_size)
        x2_grid = np.linspace(x2_min, x2_max, grid_size)
        X1, X2 = np.meshgrid(x1_grid, x2_grid)
        
        # Interpolate ground truth and predictions onto grid
        from scipy.interpolate import griddata
        
        Z_gt = griddata(x_values, y_values, (X1, X2), method='cubic', fill_value=np.nan)
        Z_pred = griddata(x_values, weighted_means, (X1, X2), method='cubic', fill_value=np.nan)
        
        # Plot ground truth as contour lines
        contour_gt = ax.contour(X1, X2, Z_gt, levels=15, colors='black', alpha=0.4, linewidths=0.5)
        
        # Plot prediction as filled contour (heatmap)
        contour_pred = ax.contourf(X1, X2, Z_pred, levels=30, cmap='viridis', alpha=0.8)
        
        # Plot context points
        context_x_np = context_X.detach().cpu().numpy()
        context_y_np = context_Y.detach().cpu().numpy().flatten()
        scatter = ax.scatter(
            context_x_np[:, 0], context_x_np[:, 1], 
            c=context_y_np, cmap='coolwarm', 
            s=50, edgecolors='white', linewidth=1.0, 
            zorder=10, label='Context'
        )
        
        # Plot next query point
        next_x_np = new_x.detach().cpu().numpy().flatten()
        ax.scatter(
            next_x_np[0], next_x_np[1], 
            marker='*', s=200, color='red', 
            edgecolors='black', linewidth=1.0,
            zorder=15, label='Next Query'
        )
        
        # Title and labels
        ax.set_title(f'Step {step + 1}', fontsize=16)
        
        if self._plot_idx > (self.rows - 1) * self.cols:
            ax.set_xlabel('$x_1$', fontsize=14)
        if (self._plot_idx - 1) % self.cols == 0:
            ax.set_ylabel('$x_2$', fontsize=14)
        if self._plot_idx == 1:
            ax.legend(loc='lower right', fontsize='medium')
        
        ax.set_xlim(x1_min, x1_max)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_ylim(x2_min, x2_max)
        # ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_aspect('equal', adjustable='box')
        
        return ax
    
    def plot_step_location_theta(
        self,
        step: int,
        posterior_theta_0,
        true_theta: Tensor,
        xs_all: Tensor,
        best_opt_x: Optional[np.ndarray] = None,
    ) -> Optional[plt.Axes]:
        """Plot theta posterior distribution for Location_budgeted task.
        
        Shows theta posterior distribution as heatmap with history points.
        
        Args:
            step: Current step index
            posterior_theta_0: Model posterior for theta
            true_theta: Ground truth theta value
            xs_all: All observed X points so far
            best_opt_x: Optional best optimized point
        
        Returns:
            The matplotlib Axes object, or None if step not plotted
        """
        ax = self._get_next_ax(step, fig_type='theta')
        if ax is None:
            return None
        
        grid_size = 80  # Reduced for subplot performance
        x_grid = np.linspace(0.0, 1.0, grid_size)
        y_grid = np.linspace(0.0, 1.0, grid_size)
        two_pi = 2.0 * np.pi
        eps = 1e-12
        
        with torch.no_grad():
            theta_means = posterior_theta_0.mixture_means[:, -2:, :].squeeze(-1).cpu().numpy()
            theta_stds = np.clip(
                posterior_theta_0.mixture_stds[:, -2:, :].squeeze(-1).cpu().numpy(), 1e-6, None)
            theta_weights = posterior_theta_0.mixture_weights[:, -2:, :].squeeze(-1).cpu().numpy()
            
            if isinstance(true_theta, torch.Tensor):
                theta_true = true_theta.detach().cpu().numpy()
            else:
                theta_true = np.array(true_theta)
            if theta_true.ndim == 1:
                theta_true = theta_true.reshape(1, -1)
        
        means_i = theta_means[0]  # [2, C]
        stds_i = theta_stds[0]
        weights_i = theta_weights[0]
        w_i = np.mean(weights_i, axis=0)
        
        mixture = np.zeros((grid_size, grid_size), dtype=np.float64)
        for k in range(means_i.shape[1]):
            mu_x, mu_y = means_i[0, k], means_i[1, k]
            sx, sy = stds_i[0, k], stds_i[1, k]
            gx = (1.0 / (np.sqrt(two_pi) * sx)) * np.exp(-0.5 * ((x_grid - mu_x) / sx) ** 2)
            gy = (1.0 / (np.sqrt(two_pi) * sy)) * np.exp(-0.5 * ((y_grid - mu_y) / sy) ** 2)
            comp = np.outer(gy, gx)
            mixture += w_i[k] * comp
        
        log_mix = np.log(mixture + eps)
        ax.imshow(log_mix, extent=[0, 1, 0, 1], origin='lower', cmap='magma', aspect='equal')
        
        # Weighted mean
        mu_x_w = float(np.sum(w_i * means_i[0]))
        mu_y_w = float(np.sum(w_i * means_i[1]))
        ax.plot(mu_x_w, mu_y_w, marker='x', color='white', markersize=6, linewidth=0, label='Mean')
        
        # True theta
        ax.scatter(theta_true[0, 0], theta_true[0, 1], marker='*', s=80, color='cyan', 
                   edgecolors='black', linewidth=0.5, label='True θ')
        
        # History points
        if xs_all is not None:
            if isinstance(xs_all, torch.Tensor):
                xs_hist = xs_all.detach().cpu().numpy()
            else:
                xs_hist = np.array(xs_all)
            xs_hist = xs_hist.reshape(-1, xs_hist.shape[-1])
            order_vals = np.arange(xs_hist.shape[0])
            ax.scatter(xs_hist[:, 0], xs_hist[:, 1], c=order_vals, cmap='viridis_r', s=20,
                       edgecolors='white', linewidths=0.2, alpha=0.9, zorder=12)
            # Label last point
            if len(xs_hist) > 0:
                ax.scatter(xs_hist[-1, 0], xs_hist[-1, 1], marker='D', s=40, color='lime',
                           edgecolors='black', linewidth=0.5, zorder=13, label='Latest')
        
        # Best optimized point
        if best_opt_x is not None:
            opt_np = np.array(best_opt_x).flatten()
            if len(opt_np) >= 2:
                ax.scatter(opt_np[0], opt_np[1], marker='D', s=50, color='lime',
                           edgecolors='black', linewidth=0.8, zorder=14, label='Opt')
        
        ax.set_title(f'Step {step + 1}', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if self._plot_idx_theta > (self.rows - 1) * self.cols:
            ax.set_xlabel('θ₁', fontsize=9)
        if (self._plot_idx_theta - 1) % self.cols == 0:
            ax.set_ylabel('θ₂', fontsize=9)
        if self._plot_idx_theta == 1:
            ax.legend(loc='upper right', fontsize='x-small')
        ax.grid(False)
        return ax
    
    def plot_step_location_objective(
        self,
        step: int,
        new_xs: Tensor,
        obj_values: Tensor,
        true_theta: Tensor,
        best_opt_x: Optional[np.ndarray] = None,
        start_points: Optional[np.ndarray] = None,
        opt_points: Optional[np.ndarray] = None,
    ) -> Optional[plt.Axes]:
        """Plot objective values for Location_budgeted task.
        
        Shows objective values as scatter plot colored by value.
        
        Args:
            step: Current step index
            new_xs: Candidate points [N, 2]
            obj_values: Objective values for candidates [N]
            true_theta: Ground truth theta value
            best_opt_x: Optional best optimized point
            start_points: Optional optimization start points
            opt_points: Optional optimized points
        
        Returns:
            The matplotlib Axes object, or None if step not plotted
        """
        ax = self._get_next_ax(step, fig_type='obj')
        if ax is None:
            return None
        
        # Convert to numpy
        if isinstance(new_xs, torch.Tensor):
            if len(new_xs.shape) == 3:
                xs_np = new_xs.reshape(new_xs.shape[-3], new_xs.shape[-1]).detach().cpu().numpy()
            else:
                xs_np = new_xs.detach().cpu().numpy()
        else:
            xs_np = np.array(new_xs)
        
        if isinstance(obj_values, torch.Tensor):
            obj_np = obj_values.detach().cpu().numpy()
        else:
            obj_np = np.array(obj_values)
        
        if isinstance(true_theta, torch.Tensor):
            theta_np = true_theta.detach().cpu().numpy()
        else:
            theta_np = np.array(true_theta)
        if theta_np.ndim > 1:
            theta_np = theta_np.squeeze()
        
        # Scatter plot with objective values as color
        scatter = ax.scatter(xs_np[:, 0], xs_np[:, 1], c=obj_np,
                            cmap='viridis', s=25, alpha=0.7, edgecolors='black', linewidth=0.3)
        
        # True theta
        ax.scatter(theta_np[0], theta_np[1], marker='*', s=100, color='red',
                   edgecolors='white', linewidth=1, label='True θ', zorder=10)
        
        # Max objective point
        max_idx = np.argmax(obj_np)
        ax.scatter(xs_np[max_idx, 0], xs_np[max_idx, 1], marker='x', s=80,
                   color='white', linewidth=2, label='Max Obj', zorder=9)
        
        # Best optimized point
        if best_opt_x is not None:
            opt_np_single = np.array(best_opt_x).flatten()
            if len(opt_np_single) >= 2:
                ax.scatter(opt_np_single[0], opt_np_single[1], marker='D', s=60, color='lime',
                           edgecolors='black', linewidth=0.5, label='L-BFGS-B Opt', zorder=11)
        
        # Draw optimization trajectories if available
        if start_points is not None and opt_points is not None:
            for s_pt, o_pt in zip(start_points, opt_points):
                s_pt = np.array(s_pt).flatten()
                o_pt = np.array(o_pt).flatten()
                if len(s_pt) >= 2 and len(o_pt) >= 2:
                    ax.annotate('', xy=(o_pt[0], o_pt[1]), xytext=(s_pt[0], s_pt[1]),
                               arrowprops=dict(arrowstyle='->', color='yellow', lw=0.8, alpha=0.6))
        
        ax.set_title(f'Step {step + 1}', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        if self._plot_idx_obj > (self.rows - 1) * self.cols:
            ax.set_xlabel('X₁', fontsize=9)
        if (self._plot_idx_obj - 1) % self.cols == 0:
            ax.set_ylabel('X₂', fontsize=9)
        if self._plot_idx_obj == 1:
            ax.legend(loc='upper right', fontsize='x-small')
        ax.grid(True, alpha=0.2)
        
        return ax
    
    def save_and_show(self, save_path: Optional[str] = None, show: bool = True):
        """Finalize, save and optionally display the figure(s).
        
        For Location_budgeted, saves TWO figures:
            - {save_path}_theta.pdf: theta posterior distributions
            - {save_path}_objective.pdf: objective value plots
        
        Args:
            save_path: Path to save the figure (e.g., 'output.pdf')
                       For Location_budgeted, suffix will be added before extension.
            show: Whether to display the figure
        """
        if self.task_name == 'Location_budgeted' and self.fig_theta is not None:
            # Handle two figures for Location_budgeted
            self.fig_theta.tight_layout()
            self.fig_obj.tight_layout()
            
            if save_path:
                base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
                ext = '.' + save_path.rsplit('.', 1)[1] if '.' in save_path else '.pdf'
                
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                
                theta_path = f"{base_path}_theta{ext}"
                obj_path = f"{base_path}_objective{ext}"
                
                self.fig_theta.savefig(theta_path, dpi=300, bbox_inches='tight')
                self.fig_obj.savefig(obj_path, dpi=300, bbox_inches='tight')
                print(f"Theta figure saved to {theta_path}")
                print(f"Objective figure saved to {obj_path}")
            
            if show:
                plt.show()
            
            plt.close(self.fig_theta)
            plt.close(self.fig_obj)
        else:
            # Single figure for other tasks
            self.fig.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
                self.fig.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Figure saved to {save_path}")
            
            if show:
                plt.show()
            
            plt.close(self.fig)
try:
    print("CWD:", os.getcwd())
except Exception:
    pass

class AttrDict(dict):
    """Simplified attribute dictionary"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def get_objective_cost_constraint_function(seed: int, task, cost_mode: str = "quadratic", constraint_mode: str = "linf"):
    """
    Get objective and cost functions.
    
    Args:
        seed: Random seed
        task: Task object
        cost_mode: Cost function mode, one of:
            - "quadratic": α * sum(x_i^2) for all dimensions
            - "l2_dist": L2 distance from previous point (requires prev_x)
    
    Returns:
        [objective_function, cost_function]
    """

    def objective_function(X: Tensor) -> Tensor:
        return task.forward(X).squeeze(0)
    
    def cost_bowl_gaussian(X: Tensor, prev_x: Tensor = None) -> Tensor:
        """
        Bowl-shaped Gaussian cost function.
        
        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Not used for bowl-shaped Gaussian cost, kept for API consistency
        
        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        alpha = 0.05
        hills=(
            (4.5,  1.5,  1.0, 0.9),
            (-2.0, -1.0, -1.8, 0.8),
            (4.5,  -1.5,  1.5, 0.8),
            (-2.0, 1.0, 1.8, 0.9),
        )# (A, mux, muy, sigma)  A>0 hill, A<0 hole
        if X.shape[-1] != 2:
            raise ValueError(f"Expected X shape (..., 2), got {tuple(X.shape)}")

        x = X[..., 0]
        y = X[..., 1]

        # base bowl
        cost = alpha * (X ** 2).sum(dim=-1) + 2
        # r2 = (X ** 2).sum(dim=-1)
        # cost = c_min + A * torch.exp(-r2 / (2 * sigma**2))

        # add Gaussian hills/holes
        for A, mux, muy, sigma in hills:
            r2 = (x - mux) ** 2 + (y - muy) ** 2
            cost = cost + A * torch.exp(-r2 / (2 * sigma ** 2))

        return cost


    def terrain_cost_function(X: Tensor, prev_x: Tensor = None) -> Tensor:
        """
        Quadratic cost: α * sum(x_i^2) for all dimensions.
        
        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Not used for quadratic cost, kept for API consistency
        
        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        # alpha = 0.1  # scaling factor cost 0-3.2
        # cost_X = alpha * (X ** 2).sum(dim=-1)
        
        # r2 = (X ** 2).sum(dim=-1)
        # cost_X = c_min + A * torch.exp(-r2 / (2 * sigma**2))
        # return cost_X

        """
        X: (..., 2) where X[...,0]=x, X[...,1]=y
        returns: (...) cost
        """
        a = (1.0, 0.5, 0.25)
        k = (1.0, 2.0, 4.0)
        phi = (0.3, 1.1, 2.2)
        psi = (1.5, 0.7, 2.8)
        eps = 1e-3
        if X.shape[-1] != 2:
            raise ValueError(f"Expected X shape (..., 2), got {tuple(X.shape)}")

        x = X[..., 0]
        y = X[..., 1]

        cost = torch.zeros_like(x)
        for ai, ki, phii, psii in zip(a, k, phi, psi):
            cost = cost + ai * torch.sin(ki * x + phii) * torch.cos(ki * y + psii)

        cost = torch.nn.functional.softplus(cost) + 1e-3

        return cost

    def quadratic_cost_function(X: Tensor, prev_x: Tensor = None, A: float = 1.00) -> Tensor:
        """
        Quadratic cost: α * sum(x_i^2) for all dimensions.

        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Not used for quadratic cost, kept for API consistency

        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        # alpha = 0.5  # scaling factor cost 0-3.2
        # cost_X = alpha * (X ** 2).sum(dim=-1)

        r2 = (X ** 2).sum(dim=-1)
        c_min = 0.05
        sigma = 1.5
        cost_X = c_min + A * torch.exp(-r2 / (2 * sigma**2))
        return cost_X



    def l2_dist_cost_function(X: Tensor, prev_x: Tensor = None) -> Tensor:
        """
        L2 distance cost from previous point.
        
        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Previous point tensor, will be broadcast to match X's batch dims
                    Can be shape [dim], [1, dim], or [..., dim]
        
        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        if prev_x is None:
            # If no previous point, return zero cost
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        
        # Ensure prev_x can broadcast with X
        # If prev_x has fewer dimensions, expand it
        while prev_x.dim() < X.dim():
            prev_x = prev_x.unsqueeze(0)
        
        # Calculate L2 distance along the last dimension
        cost_X = torch.norm(X - prev_x, p=2, dim=-1)
        return cost_X
    
    def l2_dist_constraint_function(X: Tensor, prev_x: Tensor = None, h: float = 1.0) -> Tensor:
        """
        L2 distance cost from previous point.
        
        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Previous point tensor, will be broadcast to match X's batch dims
                    Can be shape [dim], [1, dim], or [..., dim]
        
        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        if prev_x is None:
            # If no previous point, return zero cost
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        
        # Ensure prev_x can broadcast with X
        # If prev_x has fewer dimensions, expand it
        while prev_x.dim() < X.dim():
            prev_x = prev_x.unsqueeze(0)
        
        # Calculate L2 distance along the last dimension
        cost_X = torch.norm(X - prev_x, p=2, dim=-1)
        return h-cost_X

    def linf_constraint_function(X: Tensor, prev_x: Tensor = None, h: float = 1.0) -> Tensor:
        """
        L-infinity distance cost from previous point.
        
        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Previous point tensor, will be broadcast to match X's batch dims
                    Can be shape [dim], [1, dim], or [..., dim]
        
        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        if prev_x is None:
            # If no previous point, return zero cost
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        
        # Ensure prev_x can broadcast with X
        # If prev_x has fewer dimensions, expand it
        while prev_x.dim() < X.dim():
            prev_x = prev_x.unsqueeze(0)
        
        # Calculate L2 distance along the last dimension
        cost_X = X - prev_x
        return h-cost_X, h+cost_X

    def l1_dist_cost_function(X: Tensor, prev_x: Tensor = None) -> Tensor:
        """
        L1 distance cost function: sum of absolute differences across all dimensions.
        (Also known as L1 distance / Manhattan distance)
        
        Args:
            X: Input tensor of shape [..., dim] (supports arbitrary batch dimensions)
            prev_x: Previous point tensor, will be broadcast to match X's batch dims
                    Can be shape [dim], [1, dim], or [..., dim]
        
        Returns:
            cost_X: Tensor of shape [...] (batch dimensions preserved)
        """
        if prev_x is None:
            # If no previous point, return zero cost
            return torch.zeros(X.shape[:-1], device=X.device, dtype=X.dtype)
        
        # Ensure prev_x can broadcast with X
        # If prev_x has fewer dimensions, expand it
        while prev_x.dim() < X.dim():
            prev_x = prev_x.unsqueeze(0)
        
        # Calculate sum of absolute differences along the last dimension
        cost_X = torch.sum(torch.abs(X - prev_x), dim=-1)
        return cost_X

    # Select cost function based on cost_mode
    if cost_mode == "terrain":
        cost_function = terrain_cost_function
    elif cost_mode == "l2_dist":
        cost_function = l2_dist_cost_function
    elif cost_mode =="quadratic":
        cost_function = quadratic_cost_function
    elif cost_mode == "bowls":
        cost_function = cost_bowl_gaussian
    elif cost_mode == "l1_dist":
        cost_function = l1_dist_cost_function
    else:
        cost_function = None
    

    if constraint_mode == "linf":
        constraint_function = linf_constraint_function
    elif constraint_mode == "l2_dist":
        constraint_function = l2_dist_constraint_function
    else:
        constraint_function = None

    return [objective_function, cost_function, constraint_function]


def evaluate_obj_at_X(
        X: Tensor,
        objective_function: Optional[Callable],
        cost_function: Optional[Callable],
        # objective_cost_function: Optional[Callable],
) -> Tensor:
    # if (objective_cost_function is None) and (objective_function is None or cost_function is None):
    #     raise RuntimeError(
    #         "Both the objective and cost functions must be passed as inputs.")
    if objective_function is None or cost_function is None:
        raise RuntimeError(
            "Both the objective and cost functions must be passed as inputs.")

    # if objective_cost_function is not None:
    #     objective_X, cost_X = objective_cost_function(X)
    # else:
    objective_X = objective_function(X)
    cost_X = cost_function(X)
    # assert objective_X.shape == cost_X.shape, f"shape of objective_X and cost_X don't match, objective_X:{objective_X.shape},cost_X.shape:{cost_X.shape} "
    return objective_X, cost_X


def plot_theta_2d_logprob_distributions(posterior_out, true_theta, n_context: int = None, n_samples: int = 1,
                                        grid_size: int = 120, save_path: str = None, xs_all=None, annotate_indices: bool = True):
    """Visualize theta 2D mixture distribution: show log probability with color.

    Assumes each mixture component is conditionally independent across two coordinate dimensions,
    uses (mu, sigma) for each dimension and cross-dimension averaged weights to approximate joint distribution.

    Args:
        posterior_out: [1,n_total_posteriors, n_components], model output mixture Gaussian parameters
        n_samples (int): Number of samples per visualization
        n_contexts (list): List of contexts to display
        grid_size (int): Grid resolution (per axis)
        save_path (str): Save path prefix
    """
    x_grid = np.linspace(0.0, 1.0, grid_size)
    y_grid = np.linspace(0.0, 1.0, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    XY_stack = np.stack([X, Y], axis=-1)  # [G, G, 2]
    two_pi = 2.0 * np.pi
    eps = 1e-12


    with torch.no_grad():
        theta_means = posterior_out.mixture_means[:, -2:, :].squeeze(-1).cpu().numpy()
        theta_stds = np.clip(
            posterior_out.mixture_stds[:, -2:, :].squeeze(-1).cpu().numpy(), 1e-6, None)
        theta_weights = posterior_out.mixture_weights[:, -2:, :].squeeze(-1).cpu().numpy()
        if isinstance(true_theta, torch.Tensor):
            theta_true = true_theta.detach().cpu().numpy()
        else:
            theta_true = np.array(true_theta)

        if theta_true.ndim == 1:
            theta_true = theta_true.reshape(1, -1)
    fig, axes = plt.subplots(1, n_samples, figsize=(10 * n_samples, 8))
    if n_samples == 1:
        axes = [axes]

    for i in range(n_samples):
        ax = axes[i]
        means_i = theta_means[i]
        stds_i = theta_stds[i]
        weights_i = theta_weights[i]
        w_i = np.mean(weights_i, axis=0)
        mixture = np.zeros((grid_size, grid_size), dtype=np.float64)
        for k in range(means_i.shape[1]):
            mu_x, mu_y = means_i[0, k], means_i[1, k]
            sx, sy = stds_i[0, k], stds_i[1, k]
            gx = (1.0 / (np.sqrt(two_pi) * sx)) * np.exp(-0.5 * ((x_grid - mu_x) / sx) ** 2)
            gy = (1.0 / (np.sqrt(two_pi) * sy)) * np.exp(-0.5 * ((y_grid - mu_y) / sy) ** 2)
            comp = np.outer(gy, gx)
            mixture += w_i[k] * comp
        log_mix = np.log(mixture + eps)
        im = ax.imshow(log_mix, extent=[0, 1, 0, 1], origin='lower', cmap='magma', aspect='equal')
        ax.set_xlabel('Theta X')
        ax.set_ylabel('Theta Y')
        ax.set_title(f'Sample {i + 1} (n_ctx={n_context})')
        mu_x_w = float(np.sum(w_i * means_i[0]))
        mu_y_w = float(np.sum(w_i * means_i[1]))
        ax.plot(mu_x_w, mu_y_w, marker='x', color='white', markersize=8, linewidth=0, label='Weighted mean')
        ax.scatter(theta_true[i, 0], theta_true[i, 1], marker='*', s=120, color='cyan', edgecolors='black',
                   linewidth=0.8, label='True')
        if xs_all is not None:
            if isinstance(xs_all, torch.Tensor):
                xs_hist = xs_all.detach().cpu().numpy()
            else:
                xs_hist = np.array(xs_all)
            xs_hist = xs_hist.reshape(-1, xs_hist.shape[-1])
            order_vals = np.arange(xs_hist.shape[0])
            pts = ax.scatter(xs_hist[:, 0], xs_hist[:, 1], c=order_vals, cmap='viridis_r', s=36,
                             edgecolors='white', linewidths=0.25, alpha=0.95, zorder=12, label='X history')
            if annotate_indices:
                for j, (xj, yj) in enumerate(xs_hist):
                    ax.text(xj, yj, str(j + 1), fontsize=6, color='white', ha='left', va='bottom', zorder=13)
            if i == 0:
                cbar_pts = plt.colorbar(pts, ax=ax, fraction=0.046, pad=0.08)
                cbar_pts.set_label('Order')
        if i == n_samples - 1:
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('log probability')
        if i == 0:
            ax.legend(loc='upper right')
        ax.grid(False)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path + f'_theta_2d_logprob_{n_context}.png', dpi=300, bbox_inches='tight')
        plt.show()


def plot_objective_values_optimization(new_xs, objective_values, true_theta, context_step, save_path=None, opt_x=None, start_xs=None, opt_xs=None):
    """Plot objective values as scatter plot with new_xs as coordinates
    
    Args:
        new_xs (torch.Tensor): Coordinate points, shape [N, 2]
        objective_values (torch.Tensor): Objective values, shape [N]
        true_theta (torch.Tensor): True theta value, shape [2] or [1, 2]
        context_step (int): Current context step
        save_path (str, optional): Save path prefix
        opt_x (np.ndarray or torch.Tensor, optional): L-BFGS-B optimal x, shape [2]
    """
    if isinstance(new_xs, torch.Tensor):
        if len(new_xs.shape)==3:
            xs_np = new_xs.reshape(new_xs.shape[-3],new_xs.shape[-1]).detach().cpu().numpy()
        else:
            xs_np = new_xs.detach().cpu().numpy()
    else:
        xs_np = np.array(new_xs)
    
    if isinstance(objective_values, torch.Tensor):
        obj_np = objective_values.detach().cpu().numpy()
    else:
        obj_np = np.array(objective_values)
    
    if isinstance(true_theta, torch.Tensor):
        theta_np = true_theta.detach().cpu().numpy()
    else:
        theta_np = np.array(true_theta)
    
    if theta_np.ndim > 1:
        theta_np = theta_np.squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(xs_np[:, 0], xs_np[:, 1], c=obj_np,
                        cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Objective Value', fontsize=12)
    
    ax.scatter(theta_np[0], theta_np[1], marker='*', s=200, color='red', 
              edgecolors='white', linewidth=2, label='True Theta', zorder=10)
    
    max_idx = np.argmax(obj_np)
    ax.scatter(xs_np[max_idx, 0], xs_np[max_idx, 1], marker='x', s=200, 
              color='white', linewidth=3, label='Max Objective', zorder=9)

    for i, (xi, yi) in enumerate(xs_np[:, :2]):
        ax.text(xi, yi, str(i + 1), fontsize=8, color='black',
                ha='left', va='bottom', zorder=12,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.15'))

    if opt_x is not None:
        if isinstance(opt_x, torch.Tensor):
            opt_np_single = opt_x.detach().cpu().numpy()
        else:
            opt_np_single = np.array(opt_x)
        ax.scatter(opt_np_single[0], opt_np_single[1], marker='D', s=120, color='lime',
                   edgecolors='black', linewidth=0.8, label='L-BFGS-B Optimum', zorder=11)

    if (start_xs is not None) and (opt_xs is not None):
        def _to_xy_np(arr):
            if isinstance(arr, torch.Tensor):
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = np.array(arr)
            arr_np = np.asarray(arr_np)
            arr_np = arr_np.reshape(-1, arr_np.shape[-1])
            arr_np = arr_np[:, :2]
            return arr_np

        starts_np = _to_xy_np(start_xs)
        opts_np = _to_xy_np(opt_xs)
        k_pairs = min(len(starts_np), len(opts_np))
        if k_pairs > 0:
            ax.scatter(starts_np[:k_pairs, 0], starts_np[:k_pairs, 1], marker='o', s=60, color='orange',
                       edgecolors='black', linewidth=0.5, label='LBFGS Start', zorder=12)
            ax.scatter(opts_np[:k_pairs, 0], opts_np[:k_pairs, 1], marker='s', s=70, color='lime',
                       edgecolors='black', linewidth=0.6, label='LBFGS Optimum (all)', zorder=13)
            for i in range(k_pairs):
                label = 'Optimization path' if i == 0 else None
                ax.plot([starts_np[i, 0], opts_np[i, 0]], [starts_np[i, 1], opts_np[i, 1]],
                        color='white', linewidth=1.2, alpha=0.9, zorder=11, label=label)
                ax.text(starts_np[i, 0], starts_np[i, 1], f'S{i+1}', fontsize=9, color='black',
                        ha='right', va='bottom', zorder=14,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.15'))
                ax.text(opts_np[i, 0], opts_np[i, 1], f'O{i+1}', fontsize=9, color='black',
                        ha='left', va='bottom', zorder=14,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.15'))
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Objective Values at Sampled Points (Context Step {context_step})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}_objective_values_step_{context_step}.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_objective_values(new_xs, objective_values, true_theta, context_step, save_path=None, opt_x=None, start_xs=None, opt_xs=None):
    """Plot objective values as scatter plot with new_xs as coordinates
    
    Args:
        new_xs (torch.Tensor): Coordinate points, shape [N, 2]
        objective_values (torch.Tensor): Objective values, shape [N]
        true_theta (torch.Tensor): True theta value, shape [2] or [1, 2]
        context_step (string): Current context step
        save_path (str, optional): Save path prefix
        opt_x (np.ndarray or torch.Tensor, optional): L-BFGS-B optimal x, shape [2]
    """
    if isinstance(new_xs, torch.Tensor):
        if len(new_xs.shape)==3:
            xs_np = new_xs.reshape(new_xs.shape[-3],new_xs.shape[-1]).detach().cpu().numpy()
        else:
            xs_np = new_xs.detach().cpu().numpy()
    else:
        xs_np = np.array(new_xs)
    
    if isinstance(objective_values, torch.Tensor):
        obj_np = objective_values.detach().cpu().numpy()
    else:
        obj_np = np.array(objective_values)
    
    if isinstance(true_theta, torch.Tensor):
        theta_np = true_theta.detach().cpu().numpy()
    else:
        theta_np = np.array(true_theta)
    
    if theta_np.ndim > 1:
        theta_np = theta_np.squeeze()
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    scatter = ax.scatter(xs_np[:, 0], xs_np[:, 1], c=obj_np,
                        cmap='viridis', s=60, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Objective Value', fontsize=12)
    
    ax.scatter(theta_np[0], theta_np[1], marker='*', s=200, color='red', 
              edgecolors='white', linewidth=2, label='True Theta', zorder=10)
    
    max_idx = np.argmax(obj_np)
    ax.scatter(xs_np[max_idx, 0], xs_np[max_idx, 1], marker='x', s=200, 
              color='white', linewidth=3, label='Max Objective', zorder=9)

    for i, (xi, yi) in enumerate(xs_np[:, :2]):
        ax.text(xi, yi, str(i + 1), fontsize=8, color='black',
                ha='left', va='bottom', zorder=12,
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.15'))

    if opt_x is not None:
        if isinstance(opt_x, torch.Tensor):
            opt_np_single = opt_x.detach().cpu().numpy()
        else:
            opt_np_single = np.array(opt_x)
        ax.scatter(opt_np_single[0], opt_np_single[1], marker='D', s=120, color='lime',
                   edgecolors='black', linewidth=0.8, label='L-BFGS-B Optimum', zorder=11)

    if (start_xs is not None) and (opt_xs is not None):
        def _to_xy_np(arr):
            if isinstance(arr, torch.Tensor):
                arr_np = arr.detach().cpu().numpy()
            else:
                arr_np = np.array(arr)
            arr_np = np.asarray(arr_np)
            arr_np = arr_np.reshape(-1, arr_np.shape[-1])
            arr_np = arr_np[:, :2]
            return arr_np

        starts_np = _to_xy_np(start_xs)
        opts_np = _to_xy_np(opt_xs)
        k_pairs = min(len(starts_np), len(opts_np))
        if k_pairs > 0:
            ax.scatter(starts_np[:k_pairs, 0], starts_np[:k_pairs, 1], marker='o', s=60, color='orange',
                       edgecolors='black', linewidth=0.5, label='LBFGS Start', zorder=12)
            ax.scatter(opts_np[:k_pairs, 0], opts_np[:k_pairs, 1], marker='s', s=70, color='lime',
                       edgecolors='black', linewidth=0.6, label='LBFGS Optimum (all)', zorder=13)
            for i in range(k_pairs):
                label = 'Optimization path' if i == 0 else None
                ax.plot([starts_np[i, 0], opts_np[i, 0]], [starts_np[i, 1], opts_np[i, 1]],
                        color='white', linewidth=1.2, alpha=0.9, zorder=11, label=label)
                ax.text(starts_np[i, 0], starts_np[i, 1], f'S{i+1}', fontsize=9, color='black',
                        ha='right', va='bottom', zorder=14,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.15'))
                ax.text(opts_np[i, 0], opts_np[i, 1], f'O{i+1}', fontsize=9, color='black',
                        ha='left', va='bottom', zorder=14,
                        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.15'))
    
    ax.set_xlabel('X Coordinate', fontsize=12)
    ax.set_ylabel('Y Coordinate', fontsize=12)
    ax.set_title(f'Objective Values at Sampled Points (Context Step {context_step})', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(f'{save_path}_objective_values_step_{context_step}.png', 
                   dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_start_and_optimized_trees(start_points, opt_points, k, context_step, true_theta=None, save_path=None):
    """Plot initial tree vs optimized tree for each starting point.

    Args:
        start_points: Shape [K, tree_size, D], or compatible torch.Tensor/np.ndarray
        opt_points:   Shape [K, tree_size, D], or compatible torch.Tensor/np.ndarray
        k:            Maximum number of starting points to plot
        context_step: Current step number (for title/filename)
        true_theta:   True theta (length>=2, supports torch.Tensor/np.ndarray)
        save_path:    Save path prefix; if provided, saves PNG file
    """
    def _to_np(arr):
        if isinstance(arr, torch.Tensor):
            arr_np = arr.detach().cpu().numpy()
        else:
            arr_np = np.array(arr)
        arr_np = np.asarray(arr_np)
        return arr_np

    if start_points is None or opt_points is None:
        return
    sp_all = _to_np(start_points)
    op_all = _to_np(opt_points)
    if sp_all.ndim != 3 or op_all.ndim != 3:
        return

    num_starts = int(min(k, sp_all.shape[0], op_all.shape[0]))
    theta_np = None
    if true_theta is not None:
        if isinstance(true_theta, torch.Tensor):
            theta_np = true_theta.detach().cpu().numpy()
        else:
            theta_np = np.array(true_theta)
        theta_np = np.asarray(theta_np).reshape(-1)
        if theta_np.size >= 2:
            theta_np = theta_np[:2]
    for s in range(num_starts):
        sp = sp_all[s]
        op = op_all[s]
        if sp.ndim != 2 or op.ndim != 2 or sp.shape[-1] < 2 or op.shape[-1] < 2:
            continue

        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.scatter(sp[:, 0], sp[:, 1], c='tab:blue', s=40, alpha=0.8,
                   edgecolors='black', linewidth=0.4, label='Start tree', zorder=10)
        ax.scatter(op[:, 0], op[:, 1], c='tab:orange', s=40, alpha=0.8,
                   edgecolors='black', linewidth=0.4, label='Optimized tree', zorder=11)
        tsize = min(sp.shape[0], op.shape[0])
        for j in range(tsize):
            # path line
            ax.plot([sp[j, 0], op[j, 0]], [sp[j, 1], op[j, 1]],
                    color='red', linewidth=1.4, alpha=0.9, zorder=11,
                    label='Optimization path' if j == 0 else None)
            ax.text(sp[j, 0], sp[j, 1], f'S{j+1}', fontsize=9, color='black',
                    ha='right', va='bottom', zorder=12,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.15'))
            ax.text(op[j, 0], op[j, 1], f'O{j+1}', fontsize=9, color='black',
                    ha='left', va='bottom', zorder=12,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.15'))

        if theta_np is not None and theta_np.size == 2:
            ax.scatter(theta_np[0], theta_np[1], marker='*', s=200, color='red',
                       edgecolors='white', linewidth=2, label='True Theta', zorder=15)

        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'Optimization Trees (Context Step {context_step}, Start {s})', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(f'{save_path}_start_{s}_trees_step_{context_step}.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)


def fit_model(
        X: Tensor,
        Y: Tensor,
        training_mode: str,
        noiseless_obs: bool = False,
        ALINE_model=None,
        dim_theta=2,
        target_x=None,
):
    obj_Y = Y.unsqueeze(-1) if Y.ndim == 1 else Y

    bed_model = AmortizedBEDModel(
        train_X=X,
        train_Y=obj_Y,
        aline_model=ALINE_model,
        dim_theta=dim_theta,
        target_x=target_x
    )
    return bed_model
