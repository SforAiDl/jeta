from typing import Any, Callable, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from flax import core
from flax.training import train_state
from sympy import Float, Integer


def anil_adapt(
    params: core.FrozenDict[str, Any],
    apply_fn: Callable[[core.FrozenDict[str, Any], jnp.ndarray], jnp.ndarray],
    loss_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    support_set: Tuple[jnp.ndarray, jnp.ndarray],
    anil_lr: Float,
    fas: Integer,
) -> core.FrozenDict[str, Any]:

    """Adapts with respect to the support set using the ANIL algorithm.
    Paper: https://arxiv.org/abs/1909.09157
    Args:
        params: The parameters of the model.
        apply_fn: A function that applies the model to a batch of data.
        loss_fn: A function that computes the loss of a batch of data.
        support_set: A tuple of (x_train, y_train).
        anil_lr : Inner learning rate.
        fas: Fast adaption step.
    Returns:
        adapted_params: adapted parameters
    """

    mutable_params = [key for key in params if key != "params"]

    def flattened_traversal(fn):
        """Returns function that is called with `(path, param)` instead of pytree."""

        def mask(tree):
            flat = flax.traverse_util.flatten_dict(tree)
            return flax.traverse_util.unflatten_dict(
                {k: fn(k, v) for k, v in flat.items()}
            )

        return mask

    # Freezes all but the last layer.
    label_fn = flattened_traversal(
        lambda path, _: "sgd"
        if path[-2] == str(list(params["params"].keys())[-1])
        else "none"
    )

    tx = optax.multi_transform(
        {"sgd": optax.sgd(anil_lr), "none": optax.set_to_zero()}, label_fn
    )
    state = train_state.TrainState.create(
        apply_fn=apply_fn, params=(flax.core.frozen_dict.unfreeze(params)), tx=tx
    )

    def loss(theta, batch):
        x_train, y_train = batch
        logits, new_mutable_param_values = state.apply_fn(
            params.copy({"params": theta["params"]}), x_train, mutable=mutable_params
        )

        return loss_fn(logits, y_train), new_mutable_param_values

    for _ in range(fas):
        grads, new_mutable_param_values = jax.grad(loss, has_aux=True)(
            state.params, support_set
        )
        state = state.apply_gradients(grads=grads)
        print(list(state.params["params"].keys()))
        temp_params = state.params
        temp_params = flax.core.frozen_dict.freeze(temp_params)
        temp_params = temp_params.copy(new_mutable_param_values)
        state.params["batch_stats"] = flax.core.frozen_dict.unfreeze(
            temp_params["batch_stats"]
        )

    return flax.core.frozen_dict.freeze(state.params["params"])
