import torch
import numpy as np
import torch.distributions as dist
from attrdictionary import AttrDict
from tasks.base_task import Task

class GPTask(Task):
    """
    Gaussian process data generation for active learning with support for isotropic/anisotropic kernels

    Args:
        name (str): Name of the task
        dim_x (int): Dimension of input
        dim_y (int): Dimension of output
        embedding_type (str): Mode of the experiment: "data", "theta", or "mix"
        n_context_init (int): Number of initial context points
        n_query_init (int): Number of initial query points
        n_target_theta (int): Number of parameters: [lengthscale, variance]
        n_target_data (int): Number of target points (for data and mix modes)
        design_scale: Scale of the design space
        noise_scale: Noise scale
        p_iso: Probability of isotropic kernel
        kernel_weights: Weights for different kernel types
        lengthscale_lower: Lower bound for lengthscale
        lengthscale_upper: Upper bound for lengthscale
    """

    def __init__(
            self,
            name: str = "AL_mix",
            dim_x: int = 1,  # dimension of input
            dim_y: int = 1,  # dimension of output
            embedding_type="data",  # mode of the experiment: "data", "theta", or "mix"
            n_context_init: int = 5,  # number of initial context points
            n_query_init: int = 10,  # number of initial query points
            n_target_theta: int = 2,  # number of parameters: [lengthscale, variance]
            n_target_data: int = 5,  # number of target points (for data and mix modes)
            design_scale = None,  # scale of the design space
            noise_scale: float = 0.01,  # noise scale
            p_iso: float = 0.5,  # probability of isotropic kernel
            kernel_weights = None,
            lengthscale_lower: float = 0.1,
            lengthscale_upper: float = 2.0,
            **kwargs
    ) -> None:
        super().__init__(dim_x=dim_x, dim_y=dim_y)

        self.name = name
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.n_context_init = n_context_init
        self.n_query_init = n_query_init
        self.n_target_theta = n_target_theta
        self.n_target_data = n_target_data

        self.embedding_type = embedding_type
        self.jitter = 1e-5
        self.p_iso = p_iso  # probability of isotropic kernel
        self.kernel_weights = kernel_weights if kernel_weights is not None else [1/3, 0, 1/3, 1/3]
        self.kernel_types = ["rbf", "matern12", "matern32", "matern52"]

        if self.embedding_type in ["mix", "theta"]:
            if self.n_target_theta != dim_x + 1:
                raise ValueError("n_target_theta must be equal to dim_x + 1 for theta or mix embedding type")
        else:
            self.n_target_theta = 0

        # Define priors for GP hyperparameters
        # Lengthscale prior (now for each dimension)
        base_ls_factor = torch.sqrt(torch.tensor(dim_x, dtype=torch.float))
        self.lengthscale_lower = lengthscale_lower * base_ls_factor
        self.lengthscale_upper = lengthscale_upper * base_ls_factor

        # Variance
        self.scale_lower = 0.1
        self.scale_upper = 1.0

        self.noise_scale = noise_scale

        self.design_scale = torch.tensor(design_scale) if design_scale is not None else torch.tensor(5.0)
        
        # Storage for observed data to maintain GP function consistency
        self._observed_x = None  # [N, dim_x] - observed input locations
        self._observed_f = None  # [N] - observed latent function values (noise-free)
        self._observed_y = None  # [N] - observed noisy values (for returning same value on re-query)
        self._kernel_type = None  # kernel type for this GP instance
    
    def theta_to_unconstrained(self, theta: torch.Tensor) -> torch.Tensor:
        return theta

    def unconstrained_to_theta(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def set_theta(self, theta):
        self.theta = theta
    
    def reset_observations(self):
        """Reset stored observations - call this at the start of each new AL trial"""
        self._observed_x = None
        self._observed_f = None
        self._observed_y = None
        self._kernel_type = None
    
    def _initialize_kernel_type(self, batch_size=1):
        """Initialize and store kernel type for consistency"""
        if self._kernel_type is None:
            self._kernel_type = self.sample_kernel_type(batch_size)[0]
        return self._kernel_type

    @torch.no_grad()
    def sample_theta(self, batch_size, seed=None):
        """Sample hyperparameters from the prior, supporting both isotropic and anisotropic kernels
        
        Args:
            batch_size (int): Number of batches to generate
            seed (int, optional): Random seed for reproducibility. If None, uses self.seed if available.
        """
        # Set random seed for reproducibility based on trial
        if seed is None and hasattr(self, 'seed') and self.seed is not None:
            seed = self.seed
        if seed is not None:
            torch.manual_seed(seed)
            
        # Sample per-dimension lengthscales
        lengthscale_range = self.lengthscale_upper - self.lengthscale_lower
        length_scales = self.lengthscale_lower + lengthscale_range * torch.rand(batch_size, self.dim_x)

        # Determine if each batch uses isotropic kernel (same lengthscale for all dimensions)
        is_iso = torch.bernoulli(torch.ones(batch_size) * self.p_iso).bool()

        # If isotropic, set all lengthscales to be the same as the first dimension's lengthscale
        length_scales[is_iso] = length_scales[is_iso, 0].unsqueeze(1)

        # Sample output scale
        output_scale = self.scale_lower + (self.scale_upper - self.scale_lower) * torch.rand(batch_size)

        # Stack all parameters: [lengthscales, variance]
        theta = torch.cat([
            length_scales,  # [B, D]
            output_scale.unsqueeze(1),  # [B, 1]
        ], dim=1)

        return theta.unsqueeze(2)  # [B, D+1, 1]

    @torch.no_grad()
    def sample_data(self, batch_size, n_data):
        """
        Sample input points from the domain

        Args:
            batch_size (int): Number of batches to generate
            n_data (int): Number of data points to generate per batch

        Returns:
            Input points [batch_size, n_data, dim_x]
        """
        # Simple uniform random sampling in the design space
        return torch.rand(batch_size, n_data, self.dim_x) * 2 * self.design_scale - self.design_scale
    

    @torch.no_grad()
    def sample_data_sobol(self, batch_size, n_data, scramble=True):
        """
        Generate points using the Sobol sequence within the design space.

        Args:
            batch_size (int): Number of batches to generate.
            n_data (int): Number of data points per batch.
            scramble (bool, optional): Whether to scramble the sequence.

        Returns:
            Tensor: Generated points in the design space, shape [batch_size, n_data, dim_x].
        """
        device = self.design_scale.device

        # Define lower and upper bounds for the design space
        lb = -self.design_scale * torch.ones(self.dim_x, device=device)
        ub = self.design_scale * torch.ones(self.dim_x, device=device)

        # Initialize result tensor
        points = torch.zeros(batch_size, n_data, self.dim_x, device=device)

        # Generate points for each batch
        for b in range(batch_size):
            # Create a Sobol engine for this batch
            soboleng = torch.quasirandom.SobolEngine(dimension=self.dim_x, scramble=scramble)

            # Draw points and scale to the design space
            batch_points = soboleng.draw(n_data)
            # Move batch_points to the correct device before operations
            batch_points = batch_points.to(device)
            batch_points = batch_points * (ub - lb) + lb

            # Store in result tensor
            points[b] = batch_points

            # Optional: Apply random permutation to introduce randomness while preserving uniformity
            if scramble:
                for d in range(self.dim_x):
                    perm_idx = torch.randperm(n_data, device=device)
                    points[b, :, d] = points[b, perm_idx, d]

        return points

    def to_design_space(self, xi):
        """
        Convert normalized design to actual input domain

        Args:
            xi (torch.Tensor): Normalized design points [batch_size, n_data, dim_x]

        Returns:
            Design points [batch_size, n_data, dim_x]
        """
        return xi * self.design_scale

    def normalise_outcomes(self, y):
        """
        Normalize outcomes if needed

        Args:
            y (torch.Tensor): Outcomes [batch_size, n_data, dim_y]

        Returns:
            Normalized outcomes [batch_size, n_data, dim_y]
        """

        return y
    

    @torch.no_grad()
    def rbf_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute RBF kernel between x1 and x2 with support for both isotropic and anisotropic kernels

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x] (can be all same for isotropic)
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting: [N, 1, dim_x] - [1, M, dim_x]
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate squared distance with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Sum across feature dimensions for RBF kernel
        sq_dist = weighted_sq_diff.sum(dim=-1)  # [N, M]

        # Apply RBF kernel formula
        kernel = scale * torch.exp(-0.5 * sq_dist)

        return kernel  # [N, M]

    @torch.no_grad()
    def matern12_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 1/2 kernel (exponential kernel) between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 1/2 kernel formula
        kernel = scale * torch.exp(-dist)

        return kernel  # [N, M]

    @torch.no_grad()
    def matern32_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 3/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 3/2 kernel formula
        sqrt3 = torch.sqrt(torch.tensor(3.0, device=x1.device))
        kernel = scale * (1 + sqrt3 * dist) * torch.exp(-sqrt3 * dist)

        return kernel  # [N, M]

    @torch.no_grad()
    def matern52_kernel(self, x1, x2, lengthscales, scale):
        """
        Compute Matérn 5/2 kernel between x1 and x2

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)

        Returns:
            kernel matrix [N, M]
        """
        # Reshape for broadcasting
        x1_expanded = x1.unsqueeze(1)  # [N, 1, dim_x]
        x2_expanded = x2.unsqueeze(0)  # [1, M, dim_x]

        # Calculate weighted distances with per-dimension lengthscales
        sq_diff = (x1_expanded - x2_expanded) ** 2  # [N, M, dim_x]
        ls_squared = (lengthscales ** 2).view(1, 1, -1)  # [1, 1, dim_x]
        weighted_sq_diff = sq_diff / ls_squared  # [N, M, dim_x]

        # Compute Euclidean distance
        dist = torch.sqrt(weighted_sq_diff.sum(dim=-1))  # [N, M]

        # Apply Matérn 5/2 kernel formula
        sqrt5 = torch.sqrt(torch.tensor(5.0, device=x1.device))
        kernel = scale * (1 + sqrt5 * dist + (5.0 / 3.0) * dist ** 2) * torch.exp(-sqrt5 * dist)

        return kernel  # [N, M]

    def compute_kernel_matrix(self, x1, x2, lengthscales, scale, kernel_type):
        """
        Compute kernel matrix based on the specified kernel type

        Args:
            x1: first input [N, dim_x]
            x2: second input [M, dim_x]
            lengthscales: lengthscale parameters [dim_x]
            scale: variance parameter (scalar)
            kernel_type: one of "rbf", "matern12", "matern32", "matern52"

        Returns:
            kernel matrix [N, M]
        """
        if kernel_type == "rbf":
            return self.rbf_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern12":
            return self.matern12_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern32":
            return self.matern32_kernel(x1, x2, lengthscales, scale)
        elif kernel_type == "matern52":
            return self.matern52_kernel(x1, x2, lengthscales, scale)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")

    def sample_kernel_type(self, batch_size):
        """
        Sample kernel types for each batch element based on weights

        Args:
            batch_size: Number of batches to generate kernel types for

        Returns:
            List of kernel types, one for each batch element
        """
        # Convert weights to PyTorch tensor and normalize
        weights = torch.tensor(self.kernel_weights, dtype=torch.float)
        weights = weights / weights.sum()

        # Sample indices based on weights
        indices = torch.multinomial(weights, batch_size, replacement=True)

        # Map indices to kernel types
        kernel_types = [self.kernel_types[idx] for idx in indices]

        return kernel_types

    def generate_gp_data(self, x, theta):
        """
        Generate GP data for all points at once using proper correlation.
        Used for initial batch sampling (not for sequential forward calls).

        Args:
            x: input locations [B, N, dim_x]
            theta: GP hyperparameters [B, dim_x+1, 1]

        Returns:
            GP function values with noise [B, N, 1]
        """
        batch_size, n_points, _ = x.shape

        # Extract parameters:
        # First dim_x elements are lengthscales, then variance, then noise
        lengthscales = theta[:, :self.dim_x, 0]  # [B, dim_x]
        scale = theta[:, self.dim_x, 0]  # [B]

        # Sample kernel types for each batch element
        kernel_types = self.sample_kernel_type(batch_size)

        # Result tensor
        samples = torch.zeros(batch_size, n_points, 1)

        # Process each batch independently
        for b in range(batch_size):
            # Compute kernel matrix with batch-specific lengthscales and kernel type
            K = self.compute_kernel_matrix(
                x[b], x[b], lengthscales[b], scale[b], kernel_types[b]
            )  # [N, N]

            # Add jitter for numerical stability
            K = K + self.jitter * torch.eye(n_points)

            # Sample from multivariate normal
            try:
                # Try Cholesky decomposition method (most efficient)
                L = torch.linalg.cholesky(K)
                z = torch.randn(n_points)
                f = torch.matmul(L, z)
            except:
                # Alternative: use MultivariateNormal directly
                mvn = dist.MultivariateNormal(
                    loc=torch.zeros(n_points),
                    covariance_matrix=K
                )
                f = mvn.sample()

            # Add observation noise
            samples[b, :, 0] = f + self.noise_scale * torch.randn(n_points)

        return samples
    
    def generate_gp_data_consistent(self, x_new, theta):
        """
        Generate GP data for new points while maintaining consistency with previous observations.
        Uses GP conditional distribution to ensure the same underlying function.
        
        If a point has been observed before (within tolerance), returns the SAME stored y value.

        Args:
            x_new: new input locations [N_new, dim_x] (single batch assumed)
            theta: GP hyperparameters [1, dim_x+1, 1]

        Returns:
            Function values with noise [N_new, 1]
        """
        device = x_new.device
        n_new = x_new.shape[0]
        
        # Extract parameters
        lengthscales = theta[0, :self.dim_x, 0]  # [dim_x]
        scale = theta[0, self.dim_x, 0]  # scalar
        
        # Initialize kernel type if not set
        kernel_type = self._initialize_kernel_type()
        
        if self._observed_x is None or self._observed_x.shape[0] == 0:
            # No previous observations - sample from prior
            K_new = self.compute_kernel_matrix(
                x_new, x_new, lengthscales, scale, kernel_type
            )
            K_new = K_new + self.jitter * torch.eye(n_new, device=device)
            
            try:
                L = torch.linalg.cholesky(K_new)
                z = torch.randn(n_new, device=device)
                f_new = torch.matmul(L, z)
            except:
                mvn = dist.MultivariateNormal(
                    loc=torch.zeros(n_new, device=device),
                    covariance_matrix=K_new
                )
                f_new = mvn.sample()
            
            # Add observation noise
            y_new = f_new + self.noise_scale * torch.randn(n_new, device=device)
            
            # Store both f (noise-free) and y (with noise)
            self._observed_x = x_new.clone()
            self._observed_f = f_new.clone()
            self._observed_y = y_new.clone()
            
            return y_new.unsqueeze(-1)  # [n_new, 1]
        
        # Check if points already exist in observations
        x_obs = self._observed_x.to(device)
        f_obs = self._observed_f.to(device)
        y_obs = self._observed_y.to(device)
        n_obs = x_obs.shape[0]
        
        # Find existing points (within tolerance)
        tolerance = 1e-6
        y_new = torch.zeros(n_new, device=device)
        is_existing = torch.zeros(n_new, dtype=torch.bool, device=device)
        
        for i in range(n_new):
            # Check distance to all observed points
            dists = torch.norm(x_obs - x_new[i:i+1], dim=-1)
            min_dist, min_idx = dists.min(dim=0)
            if min_dist < tolerance:
                # Point already observed - return SAME stored y value (no new noise!)
                y_new[i] = y_obs[min_idx]
                is_existing[i] = True
        
        # Get indices of truly new points
        new_indices = (~is_existing).nonzero(as_tuple=True)[0]
        
        if len(new_indices) > 0:
            # Sample new points using conditional distribution
            x_truly_new = x_new[new_indices]  # [n_truly_new, dim_x]
            n_truly_new = x_truly_new.shape[0]
            
            # Compute kernel matrices
            K_obs_obs = self.compute_kernel_matrix(
                x_obs, x_obs, lengthscales, scale, kernel_type
            )  # [n_obs, n_obs]
            K_obs_obs = K_obs_obs + self.jitter * torch.eye(n_obs, device=device)
            
            K_new_obs = self.compute_kernel_matrix(
                x_truly_new, x_obs, lengthscales, scale, kernel_type
            )  # [n_truly_new, n_obs]
            
            K_new_new = self.compute_kernel_matrix(
                x_truly_new, x_truly_new, lengthscales, scale, kernel_type
            )  # [n_truly_new, n_truly_new]
            
            # Compute conditional mean and covariance
            try:
                L_obs = torch.linalg.cholesky(K_obs_obs)
                alpha = torch.linalg.solve_triangular(L_obs, f_obs, upper=False)
                alpha = torch.linalg.solve_triangular(L_obs.T, alpha, upper=True)
                mu_cond = K_new_obs @ alpha
                
                v = torch.linalg.solve_triangular(L_obs, K_new_obs.T, upper=False)
                cov_cond = K_new_new - v.T @ v
            except:
                # Fallback using pseudo-inverse
                K_obs_inv = torch.linalg.pinv(K_obs_obs)
                mu_cond = K_new_obs @ K_obs_inv @ f_obs
                cov_cond = K_new_new - K_new_obs @ K_obs_inv @ K_new_obs.T
            
            # Ensure positive definiteness
            cov_cond = cov_cond + self.jitter * torch.eye(n_truly_new, device=device)
            
            # Sample from conditional distribution
            try:
                L_cond = torch.linalg.cholesky(cov_cond)
                z = torch.randn(n_truly_new, device=device)
                f_truly_new = mu_cond + L_cond @ z
            except:
                # If Cholesky fails, just use the mean
                f_truly_new = mu_cond
            
            # Add observation noise for truly new points
            y_truly_new = f_truly_new + self.noise_scale * torch.randn(n_truly_new, device=device)
            
            # Put sampled values into y_new
            y_new[new_indices] = y_truly_new
            
            # Append truly new observations to storage
            self._observed_x = torch.cat([self._observed_x, x_truly_new.clone()], dim=0)
            self._observed_f = torch.cat([self._observed_f, f_truly_new.clone()], dim=0)
            self._observed_y = torch.cat([self._observed_y, y_truly_new.clone()], dim=0)
        
        return y_new.unsqueeze(-1)  # [n_new, 1]

    def forward(self, xi, theta=None):
        """
        Generate observations from a GP while maintaining function consistency.
        
        Uses GP conditional distribution to ensure all observations come from
        the same underlying function throughout an active learning trial.

        Args:
            xi: normalized input locations [..., dim_x] where ... can be any batch dims
            theta: GP hyperparameters with compatible batch dims, if None uses self.theta

        Returns:
            noisy observations [..., 1]
        """
        # Use self.theta if theta not provided
        if theta is None:
            theta = self.theta
        
        # Convert to actual input domain
        x = self.to_design_space(xi)
        dim_x = x.shape[-1]

        # Flatten to 2D for consistent processing
        original_shape = x.shape[:-1]
        x_flat = x.reshape(-1, dim_x)  # [N_total, dim_x]
        
        # Use consistent GP sampling
        y = self.generate_gp_data_consistent(x_flat, theta)  # [N_total, 1]
        
        # Reshape back to original shape
        return y.reshape(*original_shape, 1)

    def sample_batch(self, batch_size, n_context_init=1, seed=None):
        """
        Sample a batch of data based on the mode.
        
        Also initializes the observation storage for maintaining GP function
        consistency during active learning.

        Args:
            batch_size (int): Number of batches to generate
            n_context_init (int): Number of initial context points
            seed (int, optional): Random seed for reproducibility. If None, uses self.seed if available.

        Returns:
            Batch of data [batch_size, n_context_init, n_query_init, n_target_data]
        """
        # Reset observations for new trial
        self.reset_observations()
        
        # Set random seed for reproducibility based on trial
        if seed is None and hasattr(self, 'seed') and self.seed is not None:
            seed = self.seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # Initialize the dictionary to return
        batch = AttrDict()

        # Sample hyperparameters (seed already set, so sample_theta will use it)
        theta = self.sample_theta(batch_size, seed=None)  # Don't reset seed again
        if n_context_init is None:
            n_context_init = self.n_context_init

        # Sample input points based on the mode
        if self.embedding_type == "theta":
            # Only context and query points
            n_total = n_context_init + self.n_query_init
            x = self.sample_data(batch_size, n_total)

            # Generate all observations at once (more efficient)
            y = self.generate_gp_data(x, theta)

            # Split into context and query
            batch.context_x = x[:, :n_context_init]
            batch.context_y = y[:, :n_context_init]
            batch.query_x = x[:, n_context_init:]
            batch.query_y = y[:, n_context_init:]

            # For theta mode, target is the hyperparameters
            batch.target_all = batch.target_theta = theta
            batch.target_x = None
            batch.target_y = None

        elif self.embedding_type == "data":
            # Context, query, and target points
            n_total = n_context_init + self.n_query_init + self.n_target_data
            x = self.sample_data(batch_size, n_total)

            # Generate all observations at once
            y = self.generate_gp_data(x, theta)
            
            # Initialize observation storage with ALL sampled points
            # This ensures forward() calls use the same underlying GP function
            # Store in design space coordinates
            x_design = self.to_design_space(x[0])  # [n_total, dim_x]
            self._kernel_type = self.sample_kernel_type(1)[0]
            self._observed_x = x_design.clone()
            # Store both f (noise-free approximation) and y (with noise)
            # Note: y already contains noise, we use it as-is for consistency
            self._observed_y = y[0, :, 0].clone()
            # For f, we approximate by using y (small noise anyway)
            self._observed_f = y[0, :, 0].clone()

            # Split into context, query, and target
            batch.context_x = x[:, :n_context_init]
            batch.context_y = y[:, :n_context_init]
            batch.query_x = x[:, n_context_init:n_context_init + self.n_query_init]
            batch.query_y = y[:, n_context_init:n_context_init + self.n_query_init]
            batch.target_x = x[:, n_context_init + self.n_query_init:]
            batch.target_y = y[:, n_context_init + self.n_query_init:]

            # For data mode, target is the target data
            batch.target_all = batch.target_y
            batch.target_theta = None  # No target theta in data mode

        else:  # "mix" mode
            # Context, query, and target points
            n_total = n_context_init + self.n_query_init + self.n_target_data
            x = self.sample_data(batch_size, n_total)

            # Generate all observations at once
            y = self.generate_gp_data(x, theta)
            
            # Initialize observation storage with ALL sampled points
            x_design = self.to_design_space(x[0])  # [n_total, dim_x]
            self._kernel_type = self.sample_kernel_type(1)[0]
            self._observed_x = x_design.clone()
            self._observed_y = y[0, :, 0].clone()
            self._observed_f = y[0, :, 0].clone()

            # Split into context, query, and target
            batch.context_x = x[:, :n_context_init]
            batch.context_y = y[:, :n_context_init]
            batch.query_x = x[:, n_context_init:n_context_init + self.n_query_init]
            batch.query_y = y[:, n_context_init:n_context_init + self.n_query_init]
            batch.target_x = x[:, n_context_init + self.n_query_init:]
            batch.target_y = y[:, n_context_init + self.n_query_init:]

            # For mix mode, both data and theta are targets
            batch.target_theta = theta

            # Combine target data and theta for target_all
            batch.target_all = torch.cat([batch.target_y, batch.target_theta], dim=1)

        batch.n_target_theta = self.n_target_theta

        return theta,batch

    def __str__(self) -> str:
        info = self.__dict__.copy()
        del_keys = []

        for key in info.keys():
            if key[0] == "_":
                del_keys.append(key)

        for key in del_keys:
            del info[key]
        return f"Active learning task with GP prior data({', '.join('{}={}'.format(key, val) for key, val in info.items())})"



if __name__ == "__main__":
    # Import visualization libraries
    import matplotlib.pyplot as plt
    import argparse
    from utils.plot_config import apply_style
    apply_style(use_tex=False)

    parser = argparse.ArgumentParser(description='GPTask demonstration or offline data generation')
    parser.add_argument('--dim_x', type=int, default=3, help='Input dimension')
    parser.add_argument('--embedding_type', type=str, default='data', help='Embedding type: data, theta, or mix')
    args = parser.parse_args()

    # Create the GP task
    task = GPTask(
        dim_x=args.dim_x,
        embedding_type=args.embedding_type,
        n_context_init=1,
        n_query_init=500,
        n_target_data=200,
        noise_scale=0.01,
        design_scale=5
    )

    batch = task.sample_batch(20)

    plt.figure(figsize=(15, 10))

    # Only visualize 1D GPs
    if args.dim_x == 1:
        for i in range(20):
            # Create subplot
            plt.subplot(5, 4, i + 1)

            # Plot the sampled points
            # plt.scatter(batch.context_x[i], batch.context_y[i], c='blue', label='Context' if i == 0 else "", s=10)

            query_x_i = batch.query_x[i].squeeze().numpy()  # Squeeze to 1D numpy array
            query_y_i = batch.query_y[i].squeeze().numpy()  # Squeeze to 1D numpy array

            sorted_indices = np.argsort(query_x_i)
            query_x_sorted = query_x_i[sorted_indices]
            query_y_sorted = query_y_i[sorted_indices]
            plt.plot(query_x_sorted, query_y_sorted, c='C0', label='Context' if i == 0 else "")

            # plt.plot(batch.query_x[i], batch.query_y[i], c='red', label='Query' if i == 0 else "", s=10)
            plt.ylim(-3, 3)
            # if hasattr(batch, 'target_x'):
            #     plt.scatter(batch.target_x[i], batch.target_y[i], c='green', label='Target' if i == 0 else "", s=10)

            # Add title with hyperparameters
            lengthscale = batch.target_theta[i, 0, 0].item()
            variance = batch.target_theta[i, 1, 0].item()
            plt.title(f'Sample {i + 1}: ls = {lengthscale:.2f}, scale = {variance:.2f}')
            # plt.text(-5, 2, f'lengthscale={lengthscale:.2f}', fontsize=10)
            # plt.text(-5, 1, f'scale={variance:.2f}', fontsize=10)

            # if i == 0:
            #     plt.legend()

            plt.grid(False)

        plt.tight_layout()
        plt.savefig("../figures/gp_samples.pdf", dpi=300)
        plt.show()
