import torch
import torch.nn as nn
from typing import Optional, Union, Any, Dict, List, Tuple
from torch import Tensor
import numpy as np
import torch.nn.functional as F
import torch.distributions as dist
from models.mixture_posterior import GaussianMixturePosterior

# Simplified AttrDict to avoid external dependencies
class AttrDict(dict):
    """Simplified attribute dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class AmortizedBEDModel():
    """Amortized Bayesian Experimental Design model using neural network for posterior inference.
    
    This model wraps a pre-trained neural network (e.g., ALINE) to provide amortized inference
    for Bayesian experimental design tasks. It follows the SingleTaskGP design pattern for
    compatibility with BoTorch interfaces.
    """
    
    def __init__(
        self,
        train_X: Tensor,
        train_Y: Tensor,
        target_x: Optional[Tensor] = None,
        aline_model: Optional[nn.Module] = None,
        dim_theta: int = 2,
        **kwargs
    ):
        """
        Args:
            train_X: A `batch_shape x n x d` tensor of training features.
            train_Y: A `batch_shape x n x 1` tensor of training observations (objective only).
            target_x: Target x points for prediction
            aline_model: Pre-trained neural network model for amortized inference
            dim_theta: Dimension of latent variable theta
        """
        # Validate and process input data (following SingleTaskGP)
        self.dim_theta = dim_theta
        theta_loc = torch.zeros((1, dim_theta))  # K=1        # low
        theta_cov = torch.ones((1, dim_theta))  # scale: high - low
        self.theta_prior = dist.Uniform(
            theta_loc, theta_cov
        )

        # Set dimension information (following SingleTaskGP)
        self._set_dimensions(train_X=train_X, train_Y=train_Y)
        
        # First call parent class initialization
        super().__init__()
        
        self.aline_model = aline_model
        
        # Store training data (following SingleTaskGP)
        if not isinstance(train_X, (tuple, list)):
            self.train_inputs = (train_X,)
            self.target_x = target_x
        self.train_targets = train_Y.squeeze(-1) if train_Y.shape[-1] == 1 else train_Y
        self.target_x = target_x    #[1, n_target_x,dimx]
        

    def rsample_from_mixture(self, prediction, num_samples: int) -> Tensor:
        """Exact sampling from mixture Gaussian (preserving multimodal characteristics), returns shape [batch_size, total_posteriors, num_samples, 1]"""

        # Get mixture Gaussian parameters
        means = prediction.mixture_means  # [batch_size, total_posteriors, n_components]
        stds = prediction.mixture_stds  # [batch_size, total_posteriors, n_components]
        weights = prediction.mixture_weights  # [batch_size, total_posteriors, n_components]

        *batch_size, total_posteriors, n_components = means.shape

        # Expand to sample dimension for vectorization
        # Target shape: [B, P, S, K]
        means_expanded = means.unsqueeze(-3).expand(*batch_size, num_samples, total_posteriors, n_components)
        stds_expanded = stds.unsqueeze(-3).expand(*batch_size, num_samples, total_posteriors, n_components)
        logits_expanded = torch.log(weights + 1e-10).unsqueeze(-3).expand(*batch_size, num_samples, total_posteriors,
                                                                          n_components)
        # Gumbel-Softmax differentiable component selection
        gumbels = -torch.empty_like(logits_expanded).exponential_().log()
        component_probs = F.softmax(logits_expanded + gumbels, dim=-1)  # [B, P, S, K]

        # Reparameterized sampling from each component
        noise = torch.randn_like(means_expanded)
        component_samples = means_expanded + stds_expanded * noise  # [B, P, S, K]

        # Component weighted to get 1D sample for each point
        samples = (component_probs * component_samples).sum(dim=-1)  # [B, P, S]

        # Add event dimension as required
        samples = samples.view(*means.shape[:-2], num_samples, -1)  # [B, P, 1]

        return samples

        
    def _set_dimensions(self, train_X: Tensor, train_Y: Tensor):
        """Set model dimensions (following SingleTaskGP's _set_dimensions)"""
        self._num_outputs = 1  # Only objective
        self._input_batch_shape = train_X[0].shape[:-2]
        self._aug_batch_shape = train_Y.shape[:-2] 
        self._input_dim = train_X[0].shape[-1]

    
    @property 
    def num_outputs(self) -> int:
        """Return number of outputs (following SingleTaskGP)"""
        return self._num_outputs
        
    @property
    def batch_shape(self) -> torch.Size:
        """Return batch shape (following SingleTaskGP)"""
        return self._aug_batch_shape
        


    def sample_theta(self, batch_size):
        """ Sample latent variable from the prior

        Args:
            batch_size (int, tuple, or list):

        """
        if isinstance(batch_size, int):
            shape = [batch_size]  # Convert int to list
        elif isinstance(batch_size, tuple):
            shape = list(batch_size)  # Convert tuple to list

        theta = self.theta_prior.sample(shape)
        return theta
        
    def build_eig_data(self, X: Tensor, y: Tensor) -> AttrDict:
        """Convert input to data format for EIG computation.
        
        Build data for theta posterior prediction after adding current x and each sample y to context.
        
        Args:
            X: [...(batch_size), num_points(1), input_dim]
            y: [...(batch_size), num_points(1), num_y]
            
        Returns:
            data: AttrDict with context_x, context_y, target_x
        """
        # Parse X shape to [B, P, D]
        original_shape = X.shape
        if X.ndim == 3:
            batch_size, num_points, input_dim = X.shape
        elif X.ndim == 2:
            num_points, input_dim = X.shape
            batch_size = 1
            X = X.unsqueeze(0)
        elif X.ndim > 3:
            batch_size = int(np.prod(list(X.shape[:-2])))
            num_points, input_dim = X.shape[-2], X.shape[-1]
            X = X.reshape(batch_size, num_points, input_dim)
        else:
            raise ValueError(f"Invalid X shape {original_shape}. Expected at least 2 dimensions.")

        # Parse y, expected shape [B, P(=1), S]
        if y.ndim < 3:
            raise ValueError(f"Invalid y shape {y.shape}. Expected [batch_size, num_points, num_y].")
        if int(np.prod(list(y.shape[:-2])))!= batch_size:
            raise ValueError(f"Mismatched batch size between X ({batch_size}) and y ({int(np.prod(list(y.shape[:-2])))}).")
        if y.shape[-2] != num_points:
            raise ValueError(f"Mismatched num_points between X ({num_points}) and y ({y.shape[-2]}).")
        num_y = y.shape[-1]
        # Reshape y to [B, P, S], aligned with flattened X
        y = y.reshape(batch_size, num_points, num_y)

        data = AttrDict()

        # Standardize training context to [B, n_ctx, D] and [B, n_ctx, 1]
        if hasattr(self, 'train_inputs') and self.train_inputs is not None:
            train_X = self.train_inputs[0]
            train_Y = self.train_targets

            # Unified handling of train_X, supporting multi-dimensional batch: flatten leading batch dimensions
            if train_X.ndim < 2:
                raise ValueError(f"Invalid train_X shape {train_X.shape}. Expected at least 2D.")
            n_ctx = train_X.shape[-2]
            tx_input_dim = train_X.shape[-1]
            if tx_input_dim != input_dim:
                raise ValueError(
                    f"Mismatched input_dim between X ({input_dim}) and train_X ({tx_input_dim})."
                )

            if train_X.ndim == 2:
                # [n_ctx, D] -> [1, n_ctx, D]
                B_tx = 1
                train_X = train_X.to(X.device).unsqueeze(0)
            else:
                # [..., n_ctx, D] -> [B_tx, n_ctx, D]
                B_tx = int(np.prod(list(train_X.shape[:-2])))
                train_X = train_X.to(X.device).reshape(B_tx, n_ctx, tx_input_dim)

            # Unified handling of train_Y, supporting multi-dimensional batch and whether it has trailing output dimension
            if train_Y.ndim == 1:
                # [n_ctx] -> [1, n_ctx, 1]
                B_ty = 1
                if train_Y.shape[0] != n_ctx:
                    raise ValueError(
                        f"Mismatched n_ctx between train_Y ({train_Y.shape[0]}) and train_X ({n_ctx})."
                    )
                train_Y = train_Y.to(X.device).unsqueeze(0).unsqueeze(-1)
            else:
                if train_Y.shape[-1] == 1:
                    # [..., n_ctx, 1]
                    n_ctx_y = train_Y.shape[-2]
                    B_ty = int(np.prod(list(train_Y.shape[:-2])))
                    train_Y = train_Y.to(X.device).view(B_ty, n_ctx_y, 1)
                else:
                    # [..., n_ctx] (no trailing output dim)
                    n_ctx_y = train_Y.shape[-1]
                    B_ty = int(np.prod(list(train_Y.shape[:-1])))
                    train_Y = train_Y.to(X.device).view(B_ty, n_ctx_y, 1)

                if train_Y.shape[-2] != n_ctx:
                    raise ValueError(
                        f"Mismatched n_ctx between train_Y ({train_Y.shape[-2]}) and train_X ({n_ctx})."
                    )

            # Align train_X / train_Y batch flattened dimensions with X's batch_size (support expansion from 1)
            def _maybe_expand_to_batch(tensor, current_B, target_B):
                if current_B == target_B:
                    return tensor
                if current_B == 1:
                    return tensor.expand(target_B, *tensor.shape[1:])
                raise ValueError(
                    f"Mismatched batch size between context ({current_B}) and X ({target_B})."
                )

            train_X = _maybe_expand_to_batch(train_X, B_tx, batch_size)
            train_Y = _maybe_expand_to_batch(train_Y, B_ty, batch_size)
        else:
            n_ctx = 0
            train_X = torch.zeros(batch_size, 0, input_dim, device=X.device, dtype=X.dtype)
            train_Y = torch.zeros(batch_size, 0, 1, device=X.device, dtype=X.dtype)

        # Select target design point x: take the last point in X's point dimension
        if num_points < 1:
            raise ValueError("X must contain at least one point to form (context ⊕ x).")
        x_target = X[:, -1:, :]  # [B, 1, D]

        # Get corresponding y samples (aligned with x_target): y[:, -1, s]
        y_at_x = y[:, -1, :].view(batch_size, num_y, 1)  # [B, S, 1]

        # Repeat training context for each y sample, and concatenate with (x, y_s)
        ctx_x_rep = train_X.unsqueeze(1).expand(batch_size, num_y, n_ctx, input_dim)  # [B, S, n_ctx, D]
        ctx_y_rep = train_Y.unsqueeze(1).expand(batch_size, num_y, n_ctx, 1)          # [B, S, n_ctx, 1]
        x_rep = x_target.unsqueeze(1).expand(batch_size, num_y, 1, input_dim)         # [B, S, 1, D]
        y_rep = y_at_x.unsqueeze(2)                                                   # [B, S, 1, 1]

        context_x = torch.cat([ctx_x_rep, x_rep], dim=2).reshape(
            batch_size * num_y, n_ctx + 1, input_dim
        )
        context_y = torch.cat([ctx_y_rep, y_rep], dim=2).reshape(
            batch_size * num_y, n_ctx + 1, 1
        )

        # Set target_x to empty, as it's not needed for theta prediction
        if self.target_x is not None:
            target_x = self.target_x.expand(batch_size * num_y, -1, -1)
        else:
            target_x = torch.zeros(batch_size * num_y, 0, input_dim, device=X.device, dtype=X.dtype)

        data.context_x = context_x
        data.context_y = context_y
        data.target_x = target_x

        return data

    def posterior_theta(self, X: Tensor, y: Tensor) -> GaussianMixturePosterior:
        """Get theta posterior under multiple y values
        
        Args:
            X: [...(batch_size), num_points(1), input_dim]
            y: [...(batch_size), num_points(1), num_y]
            
        Returns:
            AttrDict with mixture_means, mixture_stds, mixture_weights
        """
        X = X.to(self.aline_model.encoder.device)
        y = y.to(self.aline_model.encoder.device)
        data = self.build_eig_data(X, y)
        
        posterior = self.aline_model(data, predict=True)
        posterior = self.slice_posterior(posterior['posterior_out'], theta=True)

        num_y = y.shape[-1]
        batch_size = X.shape[:-2]
        sliced_posterior = {}

        for key, tensor in posterior.items():
            sliced_posterior[key] = tensor.view(*batch_size, num_y, *tensor.shape[-2:])

        return AttrDict(sliced_posterior)
    
    def posterior_data(self, X: Tensor, y: Tensor) -> GaussianMixturePosterior:
        """Get data posterior under multiple y values
        
        Args:
            X: [...(batch_size), num_points(1), input_dim]
            y: [...(batch_size), num_points(1), num_y]
            
        Returns:
            AttrDict with mixture_means, mixture_stds, mixture_weights
        """
        X = X.to(self.aline_model.encoder.device)
        y = y.to(self.aline_model.encoder.device)
        data = self.build_eig_data(X, y)
        
        posterior = self.aline_model(data, predict=True)
        posterior = self.slice_posterior(posterior['posterior_out'], theta=False)

        num_y = y.shape[-1]
        batch_size = X.shape[:-2]
        sliced_posterior = {}

        for key, tensor in posterior.items():
            sliced_posterior[key] = tensor.view(*batch_size, num_y, *tensor.shape[-2:])

        return AttrDict(sliced_posterior)

    def build_context_only_data(self) -> AttrDict:
        """Construct input data based only on current context, for θ posterior (no x and y).
        
        Returns:
            data: AttrDict with context_x, context_y, target_x
        """
        data = AttrDict()
        num_y = 1

        # Device and dtype inference
        if hasattr(self, 'train_inputs') and self.train_inputs is not None:
            device = self.train_inputs[0].device
            dtype = self.train_inputs[0].dtype
        else:
            device = torch.device('cpu')
            dtype = torch.get_default_dtype()

        input_dim = getattr(self, '_input_dim', None)
        if input_dim is None and hasattr(self, 'train_inputs') and self.train_inputs is not None:
            input_dim = self.train_inputs[0].shape[-1]
        if input_dim is None:
            raise ValueError("Unable to infer input_dim for context-only EIG data.")

        # Standardize training context to [B, n_ctx, D] and [B, n_ctx, 1]
        if hasattr(self, 'train_inputs') and self.train_inputs is not None:
            train_X = self.train_inputs[0]
            train_Y = self.train_targets

            if train_X.ndim < 2:
                raise ValueError(f"Invalid train_X shape {train_X.shape}. Expected at least 2D.")
            n_ctx = train_X.shape[-2]

            if train_X.ndim == 2:
                B_ctx = 1
                train_X = train_X.to(device).unsqueeze(0)
            else:
                B_ctx = int(np.prod(list(train_X.shape[:-2])))
                train_X = train_X.to(device).reshape(B_ctx, n_ctx, input_dim)

            if train_Y.ndim == 1:
                if train_Y.shape[0] != n_ctx:
                    raise ValueError(
                        f"Mismatched n_ctx between train_Y ({train_Y.shape[0]}) and train_X ({n_ctx})."
                    )
                B_ty = 1
                train_Y = train_Y.to(device).unsqueeze(0).unsqueeze(-1)
            else:
                if train_Y.shape[-1] == 1:
                    n_ctx_y = train_Y.shape[-2]
                    B_ty = int(np.prod(list(train_Y.shape[:-2])))
                    train_Y = train_Y.to(device).reshape(B_ty, n_ctx_y, 1)
                else:
                    n_ctx_y = train_Y.shape[-1]
                    B_ty = int(np.prod(list(train_Y.shape[:-1])))
                    train_Y = train_Y.to(device).reshape(B_ty, n_ctx_y, 1)

                if train_Y.shape[-2] != n_ctx:
                    raise ValueError(
                        f"Mismatched n_ctx between train_Y ({train_Y.shape[-2]}) and train_X ({n_ctx})."
                    )

            # Align batch dimensions
            if B_ctx == B_ty:
                B = B_ctx
            elif B_ctx == 1:
                B = B_ty
                train_X = train_X.expand(B, n_ctx, input_dim)
            elif B_ty == 1:
                B = B_ctx
                train_Y = train_Y.expand(B, n_ctx, 1)
            else:
                raise ValueError(
                    f"Mismatched batch size between context X ({B_ctx}) and Y ({B_ty})."
                )
        else:
            # Empty context
            B = 1
            n_ctx = 0
            train_X = torch.zeros(B, 0, input_dim, device=device, dtype=dtype)
            train_Y = torch.zeros(B, 0, 1, device=device, dtype=dtype)

        # Replicate context along num_y, and flatten to [B*num_y, n_ctx, D]
        ctx_x_rep = train_X.unsqueeze(1).expand(B, num_y, n_ctx, input_dim)
        ctx_y_rep = train_Y.unsqueeze(1).expand(B, num_y, n_ctx, 1)
        context_x = ctx_x_rep.reshape(B * num_y, n_ctx, input_dim)
        context_y = ctx_y_rep.reshape(B * num_y, n_ctx, 1)

        if self.target_x is not None:
            target_x = self.target_x.expand(B * num_y, -1, -1)
        else:
            target_x = torch.zeros(B * num_y, 0, input_dim, device=device, dtype=dtype)

        data.context_x = context_x
        data.context_y = context_y
        data.target_x = target_x

        return data
    
    def posterior_theta_0(self) -> GaussianMixturePosterior:
        """Compute θ posterior based only on current context (no x and y).
        
        Returns:
            AttrDict with mixture parameters, shapes [B, num_y, num_theta, num_mixtures].
        """
        num_y = self.train_inputs[0].shape[0]
        B = self.train_inputs[0].shape[:-2]
        
        data = self.build_context_only_data()
        posterior = self.aline_model(data, predict=True)
        posterior = self.slice_posterior(posterior['posterior_out'], theta=True)

        if len(B) > 0:
            reshaped = {}
            for key, tensor in posterior.items():
                reshaped[key] = tensor.view(*B, 1, *tensor.shape[-2:])
            return AttrDict(reshaped)
        else:
            return posterior

    def posterior_data_0(self) -> GaussianMixturePosterior:
        """Compute data posterior based only on current context (no x and y).
        
        Returns:
            AttrDict with mixture parameters, shapes [B, num_y, num_theta, num_mixtures].
        """
        num_y = self.train_inputs[0].shape[0]
        B = self.train_inputs[0].shape[:-2]
        
        data = self.build_context_only_data()
        posterior = self.aline_model(data, predict=True)
        posterior = self.slice_posterior(posterior['posterior_out'], theta=False)

        if len(B) > 0:
            reshaped = {}
            for key, tensor in posterior.items():
                reshaped[key] = tensor.view(*B, 1, *tensor.shape[-2:])
            return AttrDict(reshaped)
        else:
            return posterior


    def slice_posterior(self, posterior, theta=True):
        """Perform slicing operation on each tensor in posterior.
        
        Args:
            posterior: AttrDict containing mixture_means, mixture_stds, mixture_weights
            theta: If True, slice theta dimensions; if False, slice data dimensions
            
        Returns:
            Sliced AttrDict
        """
        dim_theta = self.dim_theta

        # Handle dim_theta = 0: no slicing, return complete posterior directly
        if dim_theta == 0:
            return posterior if isinstance(posterior, AttrDict) else AttrDict(dict(posterior))

        sliced_posterior = {}

        for key, tensor in posterior.items():
            if theta:
                # Slice each tensor: [..., -dim_theta:, :]
                sliced_tensor = tensor[..., -dim_theta:, :]
            else:
                sliced_tensor = tensor[..., 0:-dim_theta, :]
            sliced_posterior[key] = sliced_tensor

        return AttrDict(sliced_posterior)
        
    def posterior(
        self,
        X: Tensor,
        output_indices: Optional[List[int]] = None,
        observation_noise: bool = False,
        posterior_transform=None,
        **kwargs: Any,
    ) -> GaussianMixturePosterior:
        """Get posterior distribution (following SingleTaskGP's posterior method)
        
        Args:
            X: [..., num_points, input_dim] prediction points
        """
        # Save original X shape information
        original_X_shape = X.shape
        
        # Build data format and predict
        data = self._build_data(X)
        if self.aline_model is not None:
            prediction = self.aline_model(data, predict=True)
        else:
            raise ValueError("Neural network model is not provided.")

        prediction = prediction['posterior_out']
        prediction = self.slice_posterior(prediction, theta=False)
        if X.ndim > 3:
            for key, tensor in prediction.items():
                prediction[key] = tensor.reshape(*original_X_shape[:-2], *tensor.shape[-2:])

        gmm_posterior = GaussianMixturePosterior(
            gmm_prediction=prediction,
            X_input=X,
            original_X_shape=original_X_shape
        )
        
        # Apply posterior transform
        if posterior_transform is not None:
            gmm_posterior = posterior_transform(gmm_posterior)
        
        return gmm_posterior

    def _build_data(self, X: Tensor) -> AttrDict:
        """Convert input to data format for prediction.
        
        Args:
            X: [..., num_points, input_dim]
        """
        original_shape = X.shape
        try:
            if X.ndim > 3:
                batch_size = int(np.prod(list(original_shape[:-2])))
                num_points, input_dim = original_shape[-2], original_shape[-1]
            elif X.ndim == 3:
                batch_size = 1
                num_points, input_dim = original_shape[-3], original_shape[-1]
            elif X.ndim == 2:
                batch_size = 1
                num_points, input_dim = original_shape[-2], original_shape[-1]
        except IndexError:
            raise ValueError(f"Invalid input shape {original_shape}. Expected at least 3 dimensions.")
        data = AttrDict()

        if hasattr(self, 'train_inputs') and self.train_inputs is not None:
            train_X = self.train_inputs[0]
            train_Y = self.train_targets
            n_ctx = train_X.shape[-2] if self.train_inputs else 0
            batch_ctx = int(np.prod(list(train_X.shape[:-2]))) if train_X.ndim > 2 else 1
            assert batch_ctx == batch_size, f"Expected train_X to have batch size {batch_size}, got {batch_ctx}"

            data.context_x = train_X.to(X.device).reshape(batch_ctx, n_ctx, input_dim)
            data.context_y = train_Y.reshape(batch_ctx, n_ctx, 1)
        else:
            # Empty context
            data.context_x = torch.zeros(1, 0, input_dim, device=X.device, dtype=X.dtype)
            data.context_y = torch.zeros(1, 0, 1, device=X.device, dtype=X.dtype)

        data.target_x = X.reshape(batch_size, num_points, input_dim)

        return data

    def get_fantasy_model(self, inputs: Tensor, targets: Tensor, **kwargs) -> "AmortizedBEDModel":
        """Create fantasy model (implementing GPyTorch core interface)"""
        noise = kwargs.get("noise")
        
        # Validate tensor parameters
        if targets.shape[-1] != 1:
            raise ValueError(f"AmortizedBEDModel expects single output, got {targets.shape[-1]} outputs")
        
        if noise is not None:
            kwargs.update({"noise": noise})
            
        # Create new training data
        if len(inputs.shape[:-2]) > 0:
            # Batch case: need to broadcast original training data
            batch_shape = inputs.shape[:-2]
            original_train_X = self.train_inputs[0]
            original_train_Y = self.train_targets
            
            # Broadcast original training data to batch dimension
            expanded_train_X = original_train_X.expand(*batch_shape, -1, -1)
            if original_train_Y.ndim == 1:
                expanded_train_Y = original_train_Y.expand(*batch_shape, -1)
            else:
                expanded_train_Y = original_train_Y.expand(*batch_shape, -1, -1)
            
            # Concatenate original data and new observations
            new_train_X = torch.cat([expanded_train_X, inputs], dim=-2)
            if targets.ndim == len(inputs.shape):
                targets_squeezed = targets.squeeze(-1)
                new_train_Y = torch.cat([expanded_train_Y, targets_squeezed], dim=-1)
            else:
                raise ValueError(f"Unexpected targets shape: {targets.shape} vs inputs shape: {inputs.shape}")
        else:
            # Non-batch case: concatenate directly
            new_train_X = torch.cat([self.train_inputs[0], inputs], dim=-2)
            targets_squeezed = targets.squeeze(-1) if targets.ndim > 1 else targets
            new_train_Y = torch.cat([self.train_targets, targets_squeezed], dim=-1)
        
        if new_train_Y.ndim == len(new_train_X.shape[:-1]):
            final_train_Y = new_train_Y.unsqueeze(-1)
        else:
            final_train_Y = new_train_Y
            
        fantasy_model = AmortizedBEDModel(
            train_X=new_train_X,
            train_Y=final_train_Y,
            aline_model=self.aline_model,
            dim_theta=self.dim_theta,
            target_x=self.target_x
        )
        
        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[:-1]
        fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]
        
        return fantasy_model
        
    def condition_on_observations(self, X: Tensor, Y: Tensor, **kwargs) -> "AmortizedBEDModel":
        """Update model based on new observations (via get_fantasy_model)
        
        Args:
            X: [num_x, 1, dim_x]
            Y: [1, num_x, dim_y]
        """
        if self._num_outputs > 1:
            inputs, targets, noise = multioutput_to_batch_mode_transform(
                train_X=X, train_Y=Y, num_outputs=self._num_outputs, train_Yvar=None
            )
            targets = targets.unsqueeze(-1)
        else:
            inputs = X
            targets = Y

        if not isinstance(inputs, list):
            inputs = [inputs]
        model_batch_shape = self.train_inputs[0].shape[:-2]
        inputs = [i.unsqueeze(-1) if i.ndimension() == 1 else i for i in inputs]
        target_batch_shape = targets.shape[:-1]
        input_batch_shape = inputs[0].shape[:-2]
        tbdim, ibdim = len(target_batch_shape), len(input_batch_shape)

        if not (tbdim == ibdim + 1 or tbdim == ibdim):
            raise RuntimeError(
                f"Unsupported batch shapes: The target batch shape ({target_batch_shape}) must have either the "
                f"same dimension as or one more dimension than the input batch shape ({input_batch_shape})"
            )

        err_msg = (
            f"Model batch shape ({model_batch_shape}) and target batch shape "
            f"({target_batch_shape}) are not broadcastable."
        )

        if len(model_batch_shape) > len(input_batch_shape):
            input_batch_shape = model_batch_shape
        if len(model_batch_shape) > len(target_batch_shape):
            target_batch_shape = model_batch_shape

        train_inputs = [tin.expand(input_batch_shape + tin.shape[-2:]) for tin in self.train_inputs]
        train_targets = self.train_targets.expand(target_batch_shape + self.train_targets.shape[-1:])

        full_inputs = [
            torch.cat([train_input, input.expand(input_batch_shape + input.shape[-2:])], dim=-2)
            for train_input, input in zip(train_inputs, inputs)
        ]
        full_targets = torch.cat([train_targets, targets.expand(target_batch_shape + targets.shape[-1:])], dim=-1)

        if tbdim == ibdim + 1:
            train_inputs = [fi.expand(target_batch_shape + fi.shape[-2:]) for fi in full_inputs]
        else:
            train_inputs = full_inputs
        train_targets = full_targets
        
        fantasy_model = AmortizedBEDModel(
            train_X=train_inputs[0],
            train_Y=train_targets,
            aline_model=self.aline_model,
            dim_theta=self.dim_theta,
            target_x=self.target_x
        )

        fantasy_model._input_batch_shape = fantasy_model.train_targets.shape[
                                           : (-1 if self._num_outputs == 1 else -2)
                                           ]
        fantasy_model._aug_batch_shape = fantasy_model.train_targets.shape[:-1]

        return fantasy_model
        
    def subset_output(self, idcs: List[int]) -> "AmortizedBEDModel":
        """Subset output (following SingleTaskGP's subset_output)"""
        if idcs == [0]:
            return AmortizedBEDModel(
                train_X=self.train_inputs[0],
                train_Y=self.train_targets.unsqueeze(-1) if self.train_targets.ndim == 1 else self.train_targets,
                aline_model=self.aline_model,
            )
        else:
            raise ValueError("AmortizedBEDModel only supports output index [0]")

    def train(self, mode=True):
        """Set training mode (following SingleTaskGP)"""
        super().train(mode)
        if self.aline_model is not None:
            self.aline_model.train(mode)
        return self
    
    def eval(self):
        """Set evaluation mode (following SingleTaskGP)"""
        super().eval()
        if self.aline_model is not None:
            self.aline_model.eval()
        return self
    
    def to(self, *args, **kwargs):
        """Move model to specified device/dtype (following SingleTaskGP)"""
        super().to(*args, **kwargs)
        if self.aline_model is not None:
            self.aline_model.to(*args, **kwargs)
        return self


def multioutput_to_batch_mode_transform(
    train_X: Tensor,
    train_Y: Tensor,
    num_outputs: int,
    train_Yvar: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    r"""Transforms training inputs for a multi-output model.

    Args:
        train_X: A `n x d` or `input_batch_shape x n x d` (batch mode) tensor of training features.
        train_Y: A `n x m` or `target_batch_shape x n x m` (batch mode) tensor of training observations.
        num_outputs: number of outputs
        train_Yvar: A `n x m` or `target_batch_shape x n x m` tensor of observed measurement noise.

    Returns:
        3-element tuple containing transformed tensors.
    """
    # make train_Y `batch_shape x m x n`
    train_Y = train_Y.transpose(-1, -2)
    # expand train_X to `batch_shape x m x n x d`
    train_X = train_X.unsqueeze(-3).expand(
        train_X.shape[:-2] + torch.Size([num_outputs]) + train_X.shape[-2:]
    )
    if train_Yvar is not None:
        # make train_Yvar `batch_shape x m x n`
        train_Yvar = train_Yvar.transpose(-1, -2)
    return train_X, train_Y, train_Yvar


# Backward compatibility alias
ALINEObjectiveModel = AmortizedBEDModel

