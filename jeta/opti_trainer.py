from typing import Callable, Tuple

import jax
from flax import struct
from flax.training import train_state
from jax.numpy import ndarray


class MetaTrainState(train_state.TrainState):
    adapt_fn: Callable = struct.field(pytree_node=False)
    loss_fn: Callable = struct.field(pytree_node=False)


class OptiTrainer:
    @staticmethod
    def create(params, apply_fn, adapt_fn, loss_fn, tx) -> MetaTrainState:
        """Creates a new MetaTrainState object which is the default object used for training.

        Args:
            params (flax.core.FrozenDict[str, Any]): Parameters of the model.
            apply_fn ((params, x) -> y): Function that applies the model to a batch of data.
            adapt_fn ((params, apply_fn, loss_fn, support_set) -> adapted_params): Specific meta learning function
            which adapts to the support set.
            loss_fn ((logits, targets) -> loss): Loss Function.
            tx (Optax Optimizer): Optax optimizer.

        Returns:
            MetaTrainState: Initialized MetaTrainState object.
        """

        return MetaTrainState.create(
            params=params, apply_fn=apply_fn, adapt_fn=adapt_fn, loss_fn=loss_fn, tx=tx
        )

    @staticmethod
    @jax.jit
    def meta_train_step(state: MetaTrainState, tasks) -> Tuple[MetaTrainState, ndarray]:
        """Performs a single meta-training step on a batch of tasks.

        The fuctions first adapts to the support set and then evaluates it's perfomance
        on the query set.

        Args:
            state (MetaTrainState): Contains information regarding the current state.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be trained on.

        Returns:
            Tuple[MetaTrainState, jnp.ndarray]: (Next_State, Loss).
        """

        def batch_meta_train_loss(params, apply_fn, adapt_fn, loss_fn, tasks):
            loss = jax.vmap(OptiTrainer.meta_loss, in_axes=(None, None, None, None, 0))(
                params, apply_fn, adapt_fn, loss_fn, tasks
            )
            return loss.mean()

        loss, grads = jax.value_and_grad(batch_meta_train_loss)(
            state.params, state.apply_fn, state.adapt_fn, state.loss_fn, tasks
        )
        state = state.apply_gradients(grads=grads)
        return state, loss

    @staticmethod
    @jax.jit
    def meta_test_step(state: MetaTrainState, tasks) -> ndarray:
        """Performs a single meta-testing step on a batch of tasks.

        The function first adapts to the support set and then evaluates it's perfomance
        on the query set.

        Args:
            state (MetaTrainState): Contains information regarding the current state.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be trained on.

        Returns:
            jnp.ndarray: Loss.

        """
        params = state.params
        apply_fn = state.apply_fn
        loss_fn = state.loss_fn
        adapt_fn = state.adapt_fn
        loss = jax.vmap(OptiTrainer.meta_loss, in_axes=(None, None, None, None, 0))(
            params, apply_fn, adapt_fn, loss_fn, tasks
        )
        return loss.mean()

    @staticmethod
    def meta_loss(params, apply_fn, adapt_fn, loss_fn, task) -> ndarray:
        """Calculates the Meta Loss of a task

        Args:
            params (flax.core.FrozenDict[str, Any]): Parameters of the model.
            apply_fn ((params, x) -> y): Function that applies the model to a batch of data.
            adapt_fn ((params, apply_fn, loss_fn, support_set) -> adapted_params): Specific meta learning function
            which adapts to the support set.
            loss_fn ((logits, targets) -> loss): Loss Function.
            tx (Optax Optimizer): Optax optimizer.

        Returns:
            jnp.ndarray: Loss of the task.
        """
        support_set, query_set = task

        # Adaptation step
        theta = adapt_fn(params, apply_fn, loss_fn, support_set)

        # Evaluation step
        x_train, y_train = query_set
        logits = apply_fn({"params": theta}, x_train)
        return loss_fn(logits, y_train).mean()
