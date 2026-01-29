#!/usr/bin/env python3
"""Expected Information Gain (EIG) acquisition functions for Bayesian Experimental Design."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Union
from torch import Tensor
import torch
from torch.distributions import Normal, Categorical
import math
import sys
import os
# Add project root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mixture_posterior import GaussianMixturePosterior

def compute_ll(value: torch.Tensor, means: torch.Tensor, stds: torch.Tensor,
               weights: torch.Tensor) -> torch.Tensor:
    """
    Compute per-dimension log density of Gaussian Mixture Model (GMM) at given value (dynamic batch version).

    Convention:
    - value: [*batch_dims_theta, num_y, num_theta, dim_theta]
    - means/stds/weights: [*batch_dims, num_y, dim_theta, num_components] (or broadcastable to this shape)

    weighted_log_probs = (1/(num_y*num_theta)) * Σ log p(value_i | means_j, stds_j, weights_j) for all i in num_theta, j in num_y

    Returns:
    - ll_per_theta: [*batch_dims_theta, num_theta]
    """
    # Ensure all tensors are on the same device
    device = value.device
    means = means.to(device)
    stds = stds.to(device)
    weights = weights.to(device)
    
    *batch_dims_theta, num_y, num_theta, dim_theta = value.shape
    *batch_dims, _, _, num_components = means.shape

    num_batch_dims = len(batch_dims) - len(batch_dims_theta)

    for _ in range(num_batch_dims):
        value = value.unsqueeze(0)

    value = value.unsqueeze(-1)

    target_shape = (*batch_dims_theta, num_y, num_theta, dim_theta, 1)
    value_expanded = value.expand(target_shape)
    target_shape = (*batch_dims_theta, num_y, num_theta, dim_theta, num_components)

    means_expanded = means.unsqueeze(-3).expand(target_shape)
    stds_expanded = stds.unsqueeze(-3).expand(target_shape)
    weights_expanded = weights.unsqueeze(-3).expand(target_shape)

    components_expanded = Normal(means_expanded, stds_expanded,
                                 validate_args=False)

    log_probs = components_expanded.log_prob(value_expanded)

    weighted_log_probs = log_probs + torch.log(weights_expanded)

    ll_per_theta = (torch.logsumexp(weighted_log_probs, dim=-1))
    return ll_per_theta


class EntropyDifferenceEIG():
    r"""Single-step EIG acquisition using entropy difference form.
    
    Computes EIG as: H[p(y_tar|D)] - E_{y~p(y|x,D)}[H[p(y_tar|D∪{(x,y)})]]
    
    This is the "entropy difference form" of information gain, which estimates
    the expected reduction in entropy of the target posterior after observing
    a new data point.
    """

    def __init__(
        self,
        model,
        posterior_transform: Optional[Any] = None, task=None,
        budget: Optional[Union[float, Tensor]] = None,
        L=32, Ntheta0=20, Ny = 20, eps=1e-12,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Amortized BED model providing posterior inference
            task: Task providing likelihood/log-likelihood interface
            L: Number of contrastive samples
            Ntheta0: Number of theta samples for entropy estimation
            Ny: Number of MC samples for y
        """
        self._budget = budget
        self.L = L  # number of contrastive theta samples
        self.Ntheta0 = Ntheta0  # number of theta0 samples
        self.Ny = Ny  # number of y samples per theta0
        self.eps = eps
        self.model = model
        self.task = task

    def forward(self, X):
        """
        Computes the Expected Predictive Information Gain (EPIG) for given inputs X.
        
        EPIG_φ(x|D) = H(p^φ(y_tar|D)) - E_{y~p^φ(y|x,D)}[H(p^φ(y_tar|D∪{(x,y)}))]
        
        Args:
            X: [*B, Nx, x_dim] design points
            
        Returns:
            EPIG values: Tensor [..., Nx, 1]
        """
        Ntheta0 = self.Ntheta0
        Ny = self.Ny
        model = self.model

        # Unify X shape to [*B, Nx, 1, Dx]
        if len(X.shape) < 3:
            root = True
            new_xs = X.unsqueeze(-2)
        else:
            new_xs = X
            root = False

        *B, Nx, _, Dx = new_xs.shape
        original_device = new_xs.device
        
        # Determine device: prefer CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(model, 'device'):
            device = model.device
        else:
            try:
                device = next(model.aline_model.parameters()).device
            except (StopIteration, AttributeError):
                device = original_device
        
        new_xs = new_xs.to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # (1) H[p(θ|D)]: MC entropy estimation for current θ posterior (context only)
        prior_data = model.posterior_data_0()
        data_prior_samples = model.rsample_from_mixture(prior_data, num_samples=Ntheta0).to(device)
        log_p_data = self.compute_theta_log_probs(prior_data, data_prior_samples).sum(dim=-1).to(device)
        H_prior = (-log_p_data).mean(dim=-1, keepdim=True)

        # (2) E_{y~p(y|x,D)}[ H[p(θ|D,x,y)] ]
        # 2.1 Sample y ~ p(y|x,D)
        y_posterior = model.posterior(new_xs)
        y_samples_raw = y_posterior.rsample(sample_shape=torch.Size([Ny])).to(device)
        y_samples = torch.movedim(y_samples_raw, 0, -1).to(device)

        # 2.2 Construct q(data|y) and estimate its entropy
        q_data_given_y = model.posterior_data(new_xs, y_samples)

        # MC entropy estimation for θ posterior under each y sample
        data_post_samples = model.rsample_from_mixture(q_data_given_y, num_samples=Ntheta0).to(device)
        log_q_data = self.compute_theta_log_probs(q_data_given_y, data_post_samples).sum(dim=-1).to(device)
        H_post_given_y = (-log_q_data).mean(dim=-1)
        E_H_post = H_post_given_y.mean(dim=-1, keepdim=True)

        # (3) Information Gain
        H_prior_squeezed = H_prior.squeeze(-1) if H_prior.dim() > len(B) + 2 else H_prior
        H_prior_x = H_prior_squeezed.expand(*B, Nx, 1)
        EPIG = (H_prior_x - E_H_post).to(original_device)
        return EPIG, y_samples_raw

    def compute_theta_log_probs(
        self,
        posterior_theta: GaussianMixturePosterior,
        theta_samples: torch.Tensor,
    ) -> torch.Tensor:
        """Return per-dimension log prob of theta_samples under the given GMM (posterior_theta)."""
        return compute_ll(
            theta_samples,
            posterior_theta.mixture_means,
            posterior_theta.mixture_stds,
            posterior_theta.mixture_weights,
        )


class ACEInfoGain():
    r"""Single-step EIG acquisition using Amortized Contrastive Estimation (ACE) lower bound.
    
    Monte Carlo estimation of ACE lower bound:
        I_ACE = E_{θ0~p, y~p(.|θ0,ξ), θs~q}[ log p(y|θ0,ξ)
                - log (1/(L+1) * Σ_{l=0}^L  p(θ_l)p(y|θ_l,ξ)/q(θ_l|y)) ]
    """

    def __init__(
        self,
        model,
        posterior_transform: Optional[Any] = None, task=None,
        budget: Optional[Union[float, Tensor]] = None,
        L=32, Ntheta0=20, Ny = 1, eps=1e-12,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            model: Amortized BED model providing posterior inference
            task: Task providing likelihood/log-likelihood interface
            L: Number of contrastive samples
            Ntheta0: Number of theta0 samples
            Ny: Number of MC samples for y
        """
        self._budget = budget
        self.L = L  # number of contrastive theta samples
        self.Ntheta0 = Ntheta0  # number of theta0 samples
        self.Ny = Ny  # number of y samples per theta0
        self.eps = eps
        self.model = model
        self.task = task

    def forward(self, X):
        """
        Monte Carlo estimation of ACE lower bound.

        Args:
            X: [*B, Nx, 1, x_dim] design points

        Returns:
            I_ACE estimate, shape [..., Nx, 1]
        """
        Ntheta0 = self.Ntheta0
        task = self.task
        L = self.L
        model = self.model
        
        if len(X.shape) < 4:
            root = True
            new_xs = X.unsqueeze(0).unsqueeze(0)
        else:
            new_xs = X
            root = False
            
        *B, Nx, _, Dx = new_xs.shape
        
        # Determine device
        if hasattr(task, 'theta') and task.theta is not None and isinstance(task.theta, torch.Tensor):
            device = task.theta.device
        else:
            device = new_xs.device
        new_xs = new_xs.to(device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # (1) Prior p(θ)
        prior_theta = model.posterior_theta_0()
        
        # (2) Sample θ0 and sample y under it
        theta0 = model.rsample_from_mixture(prior_theta, num_samples=Ntheta0)
        Dtheta = theta0.shape[-1]
        B_theta0 = torch.prod(torch.tensor(theta0.shape[:-2]))
        xi_for_y = new_xs.expand(*B, Nx, Ntheta0, Dx)
        if B_theta0 == torch.prod(torch.tensor(new_xs.shape[:-2])):
            theta0_for_y = theta0.unsqueeze(-4).reshape(*B, Nx, Ntheta0, 1, Dtheta)
        else:
            theta0_for_y = theta0.unsqueeze(-2).unsqueeze(-4).expand(*B, Nx, Ntheta0, 1, Dtheta)

        theta0_for_y_phys = task.unconstrained_to_theta(theta0_for_y)
        y_samples = task.forward(xi_for_y, theta0_for_y_phys).to(device)

        # (3) Variational posterior q_φ(θ|y)
        y_for_q = y_samples.squeeze(-1).unsqueeze(-2).to(device)
        q_theta_given_y = model.posterior_theta(new_xs, y_for_q)

        # (4) Contrastive samples θ1:L ~ q_φ(θ|y), concatenate with θ0
        theta_contrast = model.rsample_from_mixture(prediction=q_theta_given_y, num_samples=L)
        thetas_all = torch.cat([theta0_for_y, theta_contrast], dim=-2).to(device)
        
        # (5) Construct denominator weights log w_ℓ
        log_p_theta_l = self.compute_theta_log_probs(prior_theta, thetas_all).sum(dim=-1).to(device)
        log_q_theta_l = self.compute_theta_log_probs(q_theta_given_y, thetas_all).sum(dim=-1).to(device)
        xi_for_all = new_xs.unsqueeze(-3).expand(*B, Nx, Ntheta0, L + 1, Dx)
        y_for_all = y_samples.unsqueeze(-2).expand(*B, Nx, Ntheta0, L + 1, 1)
        thetas_all_phys = task.unconstrained_to_theta(thetas_all)
        
        log_p_y_given_theta_l = task.log_likelihood(y_for_all, xi_for_all, thetas_all_phys).squeeze(-1)

        log_w = log_p_theta_l + log_p_y_given_theta_l - log_q_theta_l
        log_den = torch.logsumexp(log_w, dim=-1) - math.log(log_w.size(-1))

        # (6) First term log p(y|θ0,ξ)
        log_p_y_given_theta0 = task.log_likelihood(y_samples, xi_for_y, theta0_for_y_phys.squeeze(-2)).squeeze(-1)

        # (7) ACE single sample and average over y
        ace_per_y = log_p_y_given_theta0 - log_den
        I_ACE = ace_per_y.mean(dim=-1, keepdim=True)

        if root:
            y_samples = y_samples.squeeze(0).squeeze(0)
        y_samples_rearranged = torch.movedim(y_samples, -2, 0)
        return I_ACE.view(X.shape[:-1]).to(device), y_samples_rearranged

    def compute_theta_log_probs(self, posterior_theta: GaussianMixturePosterior, theta_samples: torch.Tensor) -> torch.Tensor:
        """
        Computes the log-likelihood of the posterior distribution.
        """
        ll = compute_ll(theta_samples, posterior_theta.mixture_means, posterior_theta.mixture_stds, posterior_theta.mixture_weights)
        return ll


class LinearMCObjective():
    r"""Linear objective constructed from a weight tensor.

    For input `samples` and `mc_obj = LinearMCObjective(weights)`, this produces
    `mc_obj(samples) = sum_{i} weights[i] * samples[..., i]`

    Example:
        Example for a model with two outcomes:

        >>> weights = torch.tensor([0.75, 0.25])
        >>> linear_objective = LinearMCObjective(weights)
        >>> samples = sampler(posterior)
        >>> objective = linear_objective(samples)
    """

    def __init__(self, weights: Tensor) -> None:
        r"""
        Args:
            weights: A one-dimensional tensor with `m` elements representing the
                linear weights on the outputs.
        """
        super().__init__()
        if weights.dim() != 1:
            raise ValueError("weights must be a one-dimensional tensor.")
        self.register_buffer("weights", weights)

    def forward(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Evaluate the linear objective on the samples.

        Args:
            samples: A `sample_shape x batch_shape x q x m`-dim tensors of
                samples from a model posterior.
            X: A `batch_shape x q x d`-dim tensor of inputs. Relevant only if
                the objective depends on the inputs explicitly.

        Returns:
            A `sample_shape x batch_shape x q`-dim tensor of objective values.
        """
        if samples.shape[-1] != self.weights.shape[-1]:
            raise RuntimeError("Output shape of samples not equal to that of weights")
        return torch.einsum("...m, m", [samples, self.weights])


# Backward compatibility aliases
BudgetedEIGAcquisition = ACEInfoGain
BudgetedDataEIGAcquisition = EntropyDifferenceEIG

