import torch
import torch.nn as nn
from typing import Optional, Union, Any, Dict, List
from torch import Tensor
import torch.nn.functional as F


# Simplified AttrDict to avoid external dependencies
class AttrDict(dict):
    """Simplified attribute dictionary"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class GaussianMixturePosterior():
    """Gaussian Mixture Model posterior distribution wrapper for Bayesian Experimental Design.
    
    This class wraps GMM prediction results and provides interfaces compatible with BoTorch posteriors.
    """

    def __init__(self, gmm_prediction: AttrDict, X_input: Optional[Tensor] = None,
                 original_X_shape: Optional[torch.Size] = None):
        """
        Args:
            gmm_prediction: GMM prediction results containing mixture_means, mixture_stds, mixture_weights
            X_input: Original input tensor, used to preserve gradient information
            original_X_shape: Original input X shape, used to ensure consistent output shape
        """
        self.gmm_prediction = gmm_prediction
        self.X_input = X_input
        self.original_X_shape = original_X_shape

        # Cache mean and variance calculation results
        self._cached_mean = None
        self._cached_variance = None

        # Set BoTorch required attributes (following single output posterior)
        self._base_sample_shape = X_input.shape[:-1] + torch.Size([1])
        self._event_shape = torch.Size([1])
        self._batch_shape = X_input.shape[:-1]
        self.dim_y = 1

    @property
    def mixture_means(self) -> Tensor:
        return getattr(self.gmm_prediction, 'mixture_means')

    @property
    def mixture_stds(self) -> Tensor:
        return getattr(self.gmm_prediction, 'mixture_stds')

    @property
    def mixture_weights(self) -> Tensor:
        return getattr(self.gmm_prediction, 'mixture_weights')

    def _compute_mean_from_mixture(self):
        """Compute mean from mixture Gaussian parameters (objective only) - using notebook version algorithm"""
        prediction = self.gmm_prediction

        # Check if mixture distribution parameters exist
        if (hasattr(prediction, 'mixture_means') and prediction.mixture_means is not None and
                hasattr(prediction, 'mixture_weights') and prediction.mixture_weights is not None):

            try:
                # Get prediction data shape information
                if prediction.mixture_means.ndim == 3:  # [1, n_points, n_components]
                    # Compute mean for all test points (notebook version method)
                    mixture_means = prediction.mixture_means  # [1, n_points, n_components]
                    mixture_weights = prediction.mixture_weights  # [1, n_points, n_components]

                    # Compute mixture distribution mean: μ = Σ w_i * μ_i
                    computed_means = torch.sum(mixture_weights * mixture_means, dim=-1)  # [1, n_points]
                    computed_means = computed_means.squeeze()  # [n_points]

                    # If only one point, ensure correct shape
                    if computed_means.ndim == 0:
                        computed_means = computed_means.unsqueeze(0)

                    # Return correct shape: [n_points, 1]
                    return computed_means.unsqueeze(-1)

                else:
                    # Handle other shape cases (backward compatibility)
                    if prediction.mixture_means.ndim == 4:  # [1, 2, num_points, 1]
                        means = prediction.mixture_means[0, 0, -1, :]  # [components]
                        weights = prediction.mixture_weights[0, 0, -1, :]  # [components]
                    else:
                        means = prediction.mixture_means[0, 0, :]  # [components]
                        weights = prediction.mixture_weights[0, 0, :]  # [components]

                    # Compute mixture distribution mean
                    computed_mean = torch.sum(weights * means)

                    # Return correct shape: [1, 1]
                    return computed_mean.unsqueeze(0).unsqueeze(-1)

            except (IndexError, RuntimeError) as e:
                print(f"Mean calculation failed: {e}, using default zero mean")
                # If shape mismatch, use default zero mean
        else:
            # Default zero mean
            return torch.zeros(1, 1)



    def _extract_mean_variance(self):
        """Extract objective mean and variance from GMM prediction results"""
        prediction = self.gmm_prediction

        # Prioritize computing mean and variance from mixture Gaussian parameters (consistent with notebook version)
        if (hasattr(prediction, 'mixture_means') and prediction.mixture_means is not None and
                hasattr(prediction, 'mixture_weights') and prediction.mixture_weights is not None):
            # Compute mean from mixture Gaussian parameters
            mean = self._compute_mean_from_mixture()

            # Compute variance from mixture Gaussian - now using notebook version method
            variance = self._compute_variance_from_mixture()

        # Ensure variance shape matches mean
        return mean, variance

    def _compute_variance_from_mixture(self):
        """Compute variance from mixture Gaussian parameters (objective only) - using notebook version algorithm"""
        prediction = self.gmm_prediction

        # Check if mixture distribution parameters exist
        if (hasattr(prediction, 'mixture_means') and prediction.mixture_means is not None and
                hasattr(prediction, 'mixture_stds') and prediction.mixture_stds is not None and
                hasattr(prediction, 'mixture_weights') and prediction.mixture_weights is not None):

            try:
                # Get prediction data shape information
                if prediction.mixture_means.ndim == 3:  # [1, n_points, n_components]
                    # Compute variance for all test points (notebook version method)
                    mixture_means = prediction.mixture_means  # [1, n_points, n_components]
                    mixture_stds = prediction.mixture_stds  # [1, n_points, n_components]
                    mixture_weights = prediction.mixture_weights  # [1, n_points, n_components]

                    # Compute mixture distribution variance: Var(X) = Σ w_i * σ_i^2 + Σ w_i * μ_i^2 - (Σ w_i * μ_i)^2
                    mixture_variances = mixture_stds ** 2  # Component variances

                    weighted_variances = torch.sum(mixture_weights * mixture_variances, dim=-1)
                    weighted_means_squared = torch.sum(mixture_weights * (mixture_means ** 2), dim=-1)
                    overall_mean_squared = (torch.sum(mixture_weights * mixture_means, dim=-1)) ** 2

                    variance_from_mixture = weighted_variances + weighted_means_squared - overall_mean_squared
                    variance_from_mixture = variance_from_mixture.squeeze()  # [n_points]

                    # Ensure variance is positive
                    variance_from_mixture = torch.clamp(variance_from_mixture, min=1e-6)

                    # If only one point, ensure correct shape
                    if variance_from_mixture.ndim == 0:
                        variance_from_mixture = variance_from_mixture.unsqueeze(0)

                    # Return correct shape: [n_points, 1]
                    return variance_from_mixture.unsqueeze(-1)

                else:
                    # Handle other shape cases (backward compatibility)
                    if prediction.mixture_means.ndim == 4:  # [1, 2, num_points, 1]
                        means = prediction.mixture_means[0, 0, -1, :]  # [components]
                        stds = prediction.mixture_stds[0, 0, -1, :]  # [components]
                        weights = prediction.mixture_weights[0, 0, -1, :]  # [components]
                    else:
                        means = prediction.mixture_means[0, 0, :]  # [components]
                        stds = prediction.mixture_stds[0, 0, :]  # [components]
                        weights = prediction.mixture_weights[0, 0, :]  # [components]

                    # Use notebook version variance calculation formula
                    mixture_variances = stds ** 2
                    weighted_variances = torch.sum(weights * mixture_variances)
                    weighted_means_squared = torch.sum(weights * (means ** 2))
                    overall_mean_squared = (torch.sum(weights * means)) ** 2

                    variance = weighted_variances + weighted_means_squared - overall_mean_squared

                    # Ensure variance is positive
                    variance = torch.clamp(variance, min=1e-6)

                    # Return correct shape: [1, 1]
                    return variance.unsqueeze(0).unsqueeze(-1)

            except (IndexError, RuntimeError) as e:
                print(f"Variance calculation failed: {e}")
                # If shape mismatch, use default unit variance
        else:
            # Default unit variance
            print(f"Variance calculation failed")

    @property
    def base_sample_shape(self):
        """BoTorch required attribute"""
        return self._base_sample_shape

    @property
    def batch_shape(self):
        """BoTorch required attribute"""
        return self._batch_shape

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @property
    def dtype(self) -> torch.dtype:
        return self.mean.dtype

    @property
    def event_shape(self) -> torch.Size:
        return self._event_shape

    @property
    def mean(self) -> Tensor:
        """Return objective prediction mean"""
        if self._cached_mean is not None:
            return self._cached_mean
        else:
            mean, _ = self._extract_mean_variance()
            return mean.view(*self.original_X_shape[:-1], 1)

    @property
    def variance(self) -> Tensor:
        """Return objective prediction variance"""
        if self._cached_variance is not None:
            return self._cached_variance
        else:
            _, variance = self._extract_mean_variance()
            return variance.view(*self.original_X_shape[:-1], 1)

    def rsample(self, sample_shape: torch.Size = torch.Size(), base_samples: Optional[Tensor] = None,
                prediction=None) -> Tensor:
        """Unified reparameterized sampling method

        Args:
            sample_shape: Sample shape
            base_samples: Base samples (optional)
        """
        if sample_shape:
            num_samples = sample_shape.numel()
        else:
            num_samples = 1
        prediction = self.gmm_prediction

        # Original exact mixture sampling logic (preserving multimodal characteristics)
        if (hasattr(prediction, 'mixture_means') and prediction.mixture_means is not None and
                hasattr(prediction, 'mixture_stds') and prediction.mixture_stds is not None and
                hasattr(prediction, 'mixture_weights') and prediction.mixture_weights is not None):
            samples = self.rsample_from_mixture(prediction, num_samples)

            if base_samples is not None:
                samples = samples.unsqueeze(0)
        return samples


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
        samples = samples.view(num_samples, *self.X_input.shape[:-1])
        return samples


# Backward compatibility alias
ALINEObjectivePosterior = GaussianMixturePosterior

