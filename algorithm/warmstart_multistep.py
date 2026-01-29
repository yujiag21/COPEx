import random
import torch.distributions as dist
from typing import Any, Dict, List
import torch
from torch import Tensor
from torch.distributions import Beta
from algorithm.utils import AttrDict
import numpy as np


def warmstart_multistep(X, Y, task, num_fantasies, model, design_model, n_random=50, n_optimal=5,
                        sample='random_and_design', step_size=None, design_query_samples=200,lower_bound=0,upper_bound=1,target_x=None):
    train_X = X
    train_Y = Y
    new_xs = []
    # print(num_fantasies)
    num_fantasies = num_fantasies if len(num_fantasies) > 0 else [1]

    new_xs_random = []
    if (sample == 'random' or sample == 'random_and_design') and n_random > 0:
        for i, num_fantasy in enumerate(num_fantasies):
            new_xs = []
            for _ in range(num_fantasy ** i):  # 1, num_fantasy, num_fantasy^2, ...
                with torch.no_grad():
                    xs = task.sample_data(batch_size=n_random, n_data=1).unsqueeze(0)  # [1, n_random, 1, dim]
                new_xs.append(xs)  # [batch_size, num_points,1, dim]
            new_xs = torch.cat(new_xs, -2).reshape(*num_fantasies[:i], n_random, 1, -1)
            new_xs_random.append(new_xs)
    new_xs_design = []
    if (sample == 'design' or sample == 'random_and_design') and n_optimal > 0:
        for i, num_fantasy in enumerate(num_fantasies):
            batch_size = train_X.shape[:-2] if train_X.ndim > 2 else torch.Size([1])
            batch = build_aline_data(train_X, train_Y, design_query_samples, task, step_size=step_size,lower_bound=lower_bound,upper_bound=upper_bound,target_x=target_x)
            # print("batch", batch)
            with torch.no_grad():
                outs = design_model.forward(batch)  # idx: [B, 1], log_prob: [B], zt: [B, n_query]
            design_out = outs.design_out

            zt = design_out.zt.view(*batch_size, -1)
            if i == 0:
                # k = n_optimal
                top_k_probs, top_k_indices = torch.topk(zt, k=n_optimal, dim=-1)
            else:
                top_k_probs, top_k_indices = torch.topk(zt, k=1, dim=-1)
            # If you need log probabilities
            # log_probs = torch.log(top_k_probs)
            input_dim = batch.query_x.shape[-1]
            # Reshape query_x to match zt shape for gather operation
            # batch.query_x: [prod(batch_size), design_query_samples, input_dim]
            # zt: [*batch_size, design_query_samples]
            # We need query_x to be [*batch_size, design_query_samples, input_dim]
            query_x_reshaped = batch.query_x.view(*batch_size, design_query_samples, input_dim)
            # top_k_indices: [*batch_size, k] where k = n_optimal (i=0) or 1 (i>0)
            # Expand indices for gather: [*batch_size, k, input_dim]
            gather_indices = top_k_indices.unsqueeze(-1).expand(*top_k_indices.shape, input_dim)
            next_x = torch.gather(query_x_reshaped, -2, gather_indices)  # [*batch_size, k, input_dim]
            next_x = next_x.view(*[num_fantasy for j in range(i)], n_optimal, 1,
                                 -1)  # [n_optimal,1, dim]// [num_fantasy,.., n_optimal,1, dim]
            with torch.no_grad():
                posterior = model.posterior(next_x)
                Y_fantasized = posterior.rsample(sample_shape=torch.Size([num_fantasy]))  # [B,num_x,dim_y]
                model = model.condition_on_observations(X=next_x, Y=Y_fantasized)  # [B, num_x,1,dim_x] [B,num_x,dim_y]
            # if i==0:
            train_X = model.train_inputs[0]  # [B,1,dim]
            train_Y = model.train_targets  # [B,1]
            new_xs_design.append(next_x)
            # batch_initial_conditions = torch.cat([batch_initial_conditions, next_xs], -2)
        # new_xs_design = torch.cat(new_xs, -2).reshape(n_optimal, lookahead_n_fantasies + 1, -1)
    
    # New mode: design_one_branch
    # Step 1: Use the same logic as sample='design' to generate all trees with design_model
    # Step 2: Replace all branches except the first one ([0], [0,0], ...) with random samples

    if sample == 'random_and_design' and n_random > 0 and n_optimal > 0:
        batch_initial_conditions = [torch.cat([xs_random, xs_design], dim=-3) for xs_random, xs_design in
                                    zip(new_xs_random, new_xs_design)]
    # elif sample == 'design_one_branch':
    #     batch_initial_conditions = new_xs_one_branch
    elif sample == 'design' and n_optimal > 0:
        batch_initial_conditions = new_xs_design
    elif sample == 'random' and n_random > 0:
        batch_initial_conditions = new_xs_random
    else:
        batch_initial_conditions = new_xs_random if new_xs_random else new_xs_design
    # batch_initial_conditions = new_xs.reshape(n_random + n_optimal, lookahead_n_fantasies + 1, -1)
    return batch_initial_conditions


def warmstart_multistep_al(X, Y, task, num_fantasies, model, design_model, n_random=50, n_optimal=5,
                        sample='random_and_design', step_size=None):
    num_fantasies = num_fantasies if len(num_fantasies) > 0 else [1]
    new_xs_random = []
    if (sample == 'random' or sample == 'random_and_design') and n_random > 0:
        for i, num_fantasy in enumerate(num_fantasies):
            new_xs = []
            for _ in range(num_fantasy ** i):  # 1, num_fantasy, num_fantasy^2, ...
                with torch.no_grad():
                    xs = task.sample_data(1, n_random).unsqueeze(-2)   # [1, n_random, 1, dim]
                    new_xs.append(xs)  # [batch_size, num_points,1, dim]
            new_xs = torch.cat(new_xs, -1).reshape(*num_fantasies[:i], n_random, 1, -1)
            new_xs_random.append(new_xs)
    batch_initial_conditions = new_xs_random
    return batch_initial_conditions

def warmstart_multistep_ces(X, Y, task, num_fantasies, model, design_model, n_random=50, n_optimal=5,
                        sample='random_and_design', step_size=None):
    train_X = X
    train_Y = Y
    new_xs = []
    # print(num_fantasies)
    num_fantasies = num_fantasies if len(num_fantasies) > 0 else [1]

    new_xs_random = []
    if (sample == 'random' or sample == 'random_and_design') and n_random > 0:
        for i, num_fantasy in enumerate(num_fantasies):
            new_xs = []
            for _ in range(num_fantasy ** i):  # 1, num_fantasy, num_fantasy^2, ...
                with torch.no_grad():
                    xs = task.sample_data(1, n_random).unsqueeze(-2)  # [1, n_random, 1, dim]
                    new_xs.append(xs)  # [batch_size, num_points,1, dim]
            new_xs = torch.cat(new_xs, -2).reshape(*num_fantasies[:i], n_random, 1, -1)
            new_xs_random.append(new_xs)
    new_xs_design = []
    if (sample == 'design' or sample == 'random_and_design') and n_optimal > 0:
        for i, num_fantasy in enumerate(num_fantasies):
            batch_size = train_X.shape[:-2] if train_X.ndim > 2 else torch.Size([1])
            batch = build_aline_data(train_X, train_Y, DESIGN_QUERY_SAMPLES, task, step_size=step_size)
            with torch.no_grad():
                outs = design_model.forward(batch)  # idx: [B, 1], log_prob: [B], zt: [B, n_query]
            design_out = outs.design_out

            zt = design_out.zt.view(*batch_size, -1)
            if i == 0:
                # k = n_optimal
                top_k_probs, top_k_indices = torch.topk(zt, k=n_optimal, dim=-1)
            else:
                top_k_probs, top_k_indices = torch.topk(zt, k=1, dim=-1)
            # If you need log probabilities
            # log_probs = torch.log(top_k_probs)
            input_dim = batch.query_x.shape[-1]
            next_x = torch.gather(batch.query_x.view(*zt.shape, -1), -2,
                                  top_k_indices.unsqueeze(2).expand(*[-1] * (top_k_indices.ndim),
                                                                    input_dim))  # top_k_indices:[k,1], batch.query_x: [k,2000,dim]
            next_x = next_x.view(*[num_fantasy for i in range(i)], n_optimal, 1,
                                 -1)  # [n_optimal,1, dim]// [num_fantasy,.., n_optimal,1, dim]
            with torch.no_grad():
                posterior = model.posterior(next_x)
                Y_fantasized = posterior.rsample(sample_shape=torch.Size([num_fantasy]))  # [B,num_x,dim_y]
                model = model.condition_on_observations(X=next_x, Y=Y_fantasized)  # [B, num_x,1,dim_x] [B,num_x,dim_y]
            # if i==0:
            train_X = model.train_inputs[0]  # [B,1,dim]
            train_Y = model.train_targets  # [B,1]

            new_xs_design.append(next_x)
            # batch_initial_conditions = torch.cat([batch_initial_conditions, next_xs], -2)
        # new_xs_design = torch.cat(new_xs, -2).reshape(n_optimal, lookahead_n_fantasies + 1, -1)
    if sample == 'random_and_design' and n_random > 0 and n_optimal > 0:
        batch_initial_conditions = [torch.cat([xs_random, xs_design], dim=-3) for xs_random, xs_design in
                                    zip(new_xs_random, new_xs_design)]
    elif sample == 'design' or n_optimal > 0:
        batch_initial_conditions = new_xs_design
    else:
        batch_initial_conditions = new_xs_random
    # batch_initial_conditions = new_xs.reshape(n_random + n_optimal, lookahead_n_fantasies + 1, -1)
    return batch_initial_conditions


def build_aline_data(train_X, train_Y, num_samples, task, step_size=None,lower_bound=0,upper_bound=1,target_x=None):
    """Convert input to ALINE data format"""
    original_shape = train_X.shape
    try:
        if train_X.ndim > 2:
            batch_size = torch.Size([int(np.prod(list(original_shape[:-2])))])
            num_points, input_dim = original_shape[-2], original_shape[-1]
        else:
            batch_size = torch.Size([1])
            num_points, input_dim = original_shape[-2], original_shape[-1]
    except IndexError:
        raise ValueError(f"Invalid input shape {original_shape}. Expected at least 3 dimensions.")
    data = AttrDict()

    # Context set (training data) - only use objective data
    if train_X is not None and train_Y is not None:
        data.context_x = train_X.reshape(*batch_size, num_points, input_dim)
        data.context_y = train_Y.reshape(*batch_size, num_points, 1)
    else:
        # Empty context
        data.context_x = torch.zeros(1, 0, input_dim, device=train_X.device, dtype=train_X.dtype)
        data.context_y = torch.zeros(1, 0, 1, device=train_X.device, dtype=train_X.dtype)

    if step_size is not None:
        last_point = train_X[..., -1:, :].reshape(1, input_dim)
        data_sampler = dist.Uniform(
            torch.clamp(last_point - step_size, min=lower_bound, max=upper_bound),
            torch.clamp(last_point + step_size, min=lower_bound, max=upper_bound)
        )
        X = data_sampler.sample([int(np.prod(batch_size)) if len(batch_size) > 0 else 1, num_samples])[..., 0, :]
    else:
        data_sampler = None
        # X = task.sample_data(int(np.prod(batch_size)) if len(batch_size) > 0 else 1, num_samples, data_sampler)
        X = task.sample_data(int(np.prod(batch_size)) if len(batch_size) > 0 else 1, num_samples)


    data.query_x = X.reshape(*batch_size, num_samples, input_dim)
    
    # Get n_target_theta from task (for mix/theta embedding modes)
    n_target_theta = getattr(task, 'n_target_theta', 0)
    
    if target_x is not None:
        n_target_x = target_x.shape[-2]
        data.target_x = target_x.expand(*batch_size, n_target_x, input_dim)
    else:
        n_target_x = 0
        data.target_x = torch.zeros(*batch_size, 0, input_dim, device=train_X.device, dtype=train_X.dtype)
    
    # target_all.shape[-2] must match the total number of target embeddings in the embedder
    # For mix/theta modes: embedder concatenates target_x embeddings + theta_tokens
    # So target_all.shape[-2] = n_target_x + n_target_theta
    n_target_total = n_target_x + n_target_theta
    data.target_all = torch.zeros(*batch_size, n_target_total, 0, device=train_X.device, dtype=train_X.dtype)
    
    return data

