"""Multi-step lookahead acquisition functions for Bayesian Experimental Design."""

from __future__ import annotations
from typing import List, Optional, Tuple, Type
from acquisition_functions.eig_acquisition import ACEInfoGain
import numpy as np
import torch
from torch import Size, Tensor


class MultiStepLookaheadEIG():
    """Multi-step lookahead acquisition function for Expected Information Gain.
    
    This class implements a tree-based multi-step lookahead strategy for
    Bayesian experimental design, where future observations are fantasized
    to evaluate the long-term value of design choices.
    """

    def __init__(self, model, batch_size, lookahead_batch_sizes, task, num_fantasies, discount_factor, n_y=100, last_X=None, fantasized_with_model=True, valfunc=None):
        """
        Args:
            model: Amortized BED model for posterior inference
            batch_size: Number of candidates at the first step
            lookahead_batch_sizes: List of batch sizes for lookahead steps
            task: Task providing likelihood interface
            num_fantasies: List of fantasy counts for each step
            discount_factor: Discount factor for future values
            n_y: Number of y samples for EIG estimation
            last_X: Last observed X point
            fantasized_with_model: Whether to use model for fantasizing
            valfunc: Value function class (default: ACEInfoGain)
        """
        self.model = model
        self.batch_size = batch_size
        if lookahead_batch_sizes == [0]:
            batch_sizes = [batch_size]
        else:
            batch_sizes = [batch_size] + lookahead_batch_sizes
        self.num_fantasies = num_fantasies
        self.n_y = n_y
        self.valfunc = valfunc if valfunc is not None else ACEInfoGain
        self.task = task
        self._valfunc_cls = [self.valfunc for _ in batch_sizes]
        self.batch_sizes = lookahead_batch_sizes
        self._num_auxiliary = np.dot(self.batch_sizes, np.cumprod(self.num_fantasies[1:])).item() if len(self.num_fantasies[1:]) > 0 else 0
        self.discount_factor = discount_factor
        self.last_X = last_X
        self.fantasized_with_model = fantasized_with_model

    def input_transform(self, Xs: List[Tensor], last_X: Tensor = None) -> List[Tensor]:
        if last_X is None:
            last_X = self.last_X
        else:
            last_X = last_X

        for i in range(len(Xs)):
            Xs[i] = last_X + Xs[i]
            last_X = Xs[i]
        return Xs

    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate MultiStepLookaheadEIG on the candidate set X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            The acquisition value for each batch as a tensor of shape `batch_shape`.
        """
        Xs = self.get_multi_step_tree_input_representation(X)

        return self._step(
            model=self.model,
            Xs=Xs,
            valfunc_cls=self._valfunc_cls,
            running_val=None,
            task=self.task,
            discount_factor=1,
            num_fantasies=[self.num_fantasies[0]] + self.num_fantasies if len(self.num_fantasies) > 0 else [1],
            stage_val_list=[],
        )

    def get_split_shapes(self, X: Tensor) -> Tuple[Size, List[Size], List[int]]:
        r"""Get the split shapes from X.

        Args:
            X: A `batch_shape x q_aug x d`-dim tensor including fantasy points.

        Returns:
            A 3-tuple `(batch_shape, shapes, sizes)`, where
            `shape[i] = f_i x .... x f_1 x batch_shape x q_i x d` and
            `size[i] = f_i * ... f_1 * q_i`.
        """
        batch_shape, (q_aug, d) = X.shape[:-2], X.shape[-2:]
        q = q_aug - self._num_auxiliary
        batch_size = [q] + self.batch_sizes
        # X_i needs to have shape f_i x .... x f_1 x batch_shape x q_i x d
        if len(self.num_fantasies) == 0:
            shapes = [
                torch.Size([*batch_shape, q, d])
            ]
        else:
            shapes = [
                torch.Size(self.num_fantasies[:i][::-1] + [*batch_shape, q_i, d])
                for i, q_i in enumerate(batch_size)
            ]
        # Each X_i in the split X has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        sizes = [s[: (-2 - len(batch_shape))].numel() * s[-2] for s in shapes]
        return batch_shape, shapes, sizes

    def get_multi_step_tree_input_representation(self, X: Tensor) -> List[Tensor]:
        r"""Get the multi-step tree representation of X.

        Args:
            X: A `batch_shape x q' x d`-dim Tensor with `q'` design points for each
                batch, where `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`. Here `q_i`
                is the number of candidates jointly considered in look-ahead step
                `i`, and `f_i` is respective number of fantasies.

        Returns:
            A list `[X_j, ..., X_k]` of tensors, where `X_i` has shape
            `f_i x .... x f_1 x batch_shape x q_i x d`.

        """
        batch_shape, shapes, sizes = self.get_split_shapes(X=X)
        # Each X_i in Xsplit has shape batch_shape x qtilde x d with
        # qtilde = f_i * ... * f_1 * q_i
        Xsplit = torch.split(X, sizes, dim=-2)
        # now reshape (need to permute batch_shape and qtilde dimensions for i > 0)
        perm = [-2] + list(range(len(batch_shape))) + [-1]
        X0 = Xsplit[0].reshape(shapes[0])
        Xother = [
            X.permute(*perm).reshape(shape) for X, shape in zip(Xsplit[1:], shapes[1:])
        ]
        return [X0] + Xother

    def _step(
        self,
        model,
        Xs: List[Tensor],
        valfunc_cls: List[Optional[Type]],
        running_val: Optional[Tensor] = None,
        step_index: int = 0,
        task=None,
        discount_factor=1,
        num_fantasies=None,
        stage_val_list=None
    ) -> Tensor:
        X = Xs[0]
        valfunc_cl = valfunc_cls[0]
        stage_val_func = valfunc_cl(model=model, task=task, Ntheta0=self.n_y)
        stage_val, y_samples = stage_val_func.forward(X=X)
        
        if stage_val is not None:
            stage_val_list = stage_val_list + [stage_val]
            running_val = stage_val if running_val is None else running_val + discount_factor * stage_val
            
        if len(Xs) == 1:
            batch_shape = running_val.shape[step_index:]
            return running_val.view(-1, *batch_shape).mean(dim=0), stage_val_list

        # construct fantasy model (with batch shape f_{j+1} x ... x f_1 x batch_shape)
        prop_grads = step_index > 0  # need to propagate gradients for steps > 0
        if self.fantasized_with_model:
            posterior = model.posterior(X)
            Y_fantasized = posterior.rsample(sample_shape=torch.Size([num_fantasies[1]]))
        else:
            Y_fantasized = y_samples[:num_fantasies[0]]
        fantasy_model = model.condition_on_observations(X=X, Y=Y_fantasized)

        return self._step(
            model=fantasy_model,
            Xs=Xs[1:],
            valfunc_cls=valfunc_cls[1:],
            running_val=running_val,
            step_index=step_index + 1,
            task=task,
            discount_factor=discount_factor * self.discount_factor,
            num_fantasies=num_fantasies[1:],
            stage_val_list=stage_val_list
        )


def reduce_to_last_dim_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Average tensor of shape [..., k] over all dimensions except the last,
    output shape = [k]
    """
    if x.ndim == 1:
        return x

    dims_to_reduce = tuple(range(x.ndim - 1))
    return x.mean(dim=dims_to_reduce)


def get_X_from_multi_step_tree_input_representation(Xs: List[Tensor]) -> Tensor:
    r"""Inverse of `get_multi_step_tree_input_representation`.

    Args:
        Xs: List of tensors `[X_0, X_1, ..., X_k]`, where `X_0` has shape
            `batch_shape x q_0 x d` and for `i>0`, `X_i` has shape
            `f_i x ... x f_1 x batch_shape x q_i x d`.

    Returns:
        A tensor `X` of shape `batch_shape x q' x d` with
        `q' = q_0 + f_1 q_1 + f_2 f_1 q_2 + ...`.
    """
    X = Xs[0]
    for Xi in Xs[1:]:
        X = torch.cat([X, Xi.flatten(0, -4).squeeze(-2).transpose(0, 1)], dim=1)
    return X


# Backward compatibility alias
qMultiStepLookaheadEIG = MultiStepLookaheadEIG
