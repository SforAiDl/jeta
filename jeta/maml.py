from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import core
from sympy import Float, Integer


def maml_adapt(
    params: core.FrozenDict[str, Any],
    apply_fn: Callable[[core.FrozenDict[str, Any], jnp.ndarray], jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    support_set: Tuple[jnp.ndarray, jnp.ndarray],
    maml_lr: Float,
    fas: Integer,
) -> core.FrozenDict[str, Any]:

    """Adapts with respect to the support set using the MAML algorithm.

    Paper: https://arxiv.org/abs/1703.03400

    Args:
        params: The parameters of the model.
        apply_fn: A function that applies the model to a batch of data.
        loss_fn: A function that computes the loss of a batch of data.
        support_set: A tuple of (x_train, y_train).
        maml_lr : Inner learning rate.
        fas: Fast adaption step.
    Returns:
        adapted_params: adapted parameters
    """

    theta = params["params"]
    mutable_params = [key for key in params if key != "params"]

    def loss(theta, batch):
        x_train, y_train = batch
        logits, new_mutable_param_values = apply_fn(
            params.copy({"params": theta}), x_train, train=True, mutable=mutable_params
        )
        return loss_fn(logits, y_train), new_mutable_param_values

    for _ in range(fas):
        grads, new_mutable_param_values = jax.grad(loss, has_aux=True)(
            theta, support_set
        )
        params = params.copy(new_mutable_param_values)
        theta = jax.tree_util.tree_map(lambda t, g: t - maml_lr * g, theta, grads)

    return theta
