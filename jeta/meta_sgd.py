from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
from flax import core

# from loss import mse


def meta_sgd_adapt(
    params: core.FrozenDict[str, Any],
    apply_fn: Callable[[core.FrozenDict[str, Any], jnp.ndarray], jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    support_set: Tuple[jnp.ndarray, jnp.ndarray],
) -> core.FrozenDict[str, Any]:

    """Adapts with respect to the support set using the Meta-SGD algorithm.

    Paper: https://arxiv.org/abs/1707.09835

    Args:
        params: The parameters of the model.
        apply_fn: A function that applies the model to a batch of data.
        loss_fn: A function that computes the loss of a batch of data.
        support_set: A tuple of (x_train, y_train).

    Returns:
        adapted_params: adapted parameters
    """

    theta, alpha = params["params"].pop("alpha")
    mutable_params = [key for key in params if key != "params"]
    # alpha = params['params']['alpha']
    # Remove 'alpha' from the the parameter list
    # params = params.unfreeze()
    # params['params'].pop('alpha')
    # params = core.frozen_dict.freeze(params)
    # alpha = params["alpha"]

    def loss(theta, batch):
        x_train, y_train = batch
        logits, new_mutable_param_values = apply_fn(
            params.copy({"params": theta}), x_train, train=True, mutable=mutable_params
        )
        return loss_fn(logits, y_train), new_mutable_param_values

    fas = 1  # Fast adaptation steps. TODO: take this parameter as an argument

    for _ in range(fas):
        grads, new_mutable_param_values = jax.grad(loss, has_aux=True)(
            theta, support_set
        )
        params = params.copy(new_mutable_param_values)
        theta = jax.tree_util.tree_map(lambda t, g, a: t - a * g, theta, grads, alpha)

    return theta


def meta_sgd_init(model, init_key, arr):
    """Initializes the parameters of the model.

    The default parameters initilized by flax don't convege for
    optimization based meta learning algorithms.
    Hence they are scaled to match a normal distribution with mean 0 and std 0.01.

    In addition, this function adds a new parameter 'alpha' to the model.
    This parameter is of the same shape as the weights (and biases) of the model
    and initialised to 0.01. This parameter is used to scale the gradients

    Args:
        model (nn.Module): model whose parameters are to be initialised
        init_key (random.PRNGKey): PRNG Key used for initialisation
        arr (jnp.ndarray): a random array used to initialize the parameters

    Returns:
        Parameters: A frozen dict of model parameters.
    """

    # EPSILON = 1e-8
    params = model.init(init_key, arr).unfreeze()
    alpha = jax.tree_util.tree_map(lambda t: jnp.ones_like(t) * 0.01, params["params"])
    # params = jax.tree_util.tree_map(
    #     lambda p: 0.01 * (p - p.mean()) / (p.std() + EPSILON), params
    # )
    params["params"]["alpha"] = alpha
    params = core.frozen_dict.freeze(params)
    return params
