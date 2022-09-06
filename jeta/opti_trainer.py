from functools import partial
from typing import Callable, List, Tuple

import jax
import logger
import optax
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

        state = MetaTrainState.create(
            params=params, apply_fn=apply_fn, adapt_fn=adapt_fn, loss_fn=loss_fn, tx=tx
        )
        return state.replace(opt_state=tx.init(params["params"]))

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def meta_train_step(
        state: MetaTrainState,
        tasks,
        metrics: List[Callable[[ndarray, ndarray], ndarray]] = [],
        logger_type: str = "tensorboard",
    ) -> Tuple[MetaTrainState, ndarray, List[ndarray]]:
        """Performs a single meta-training step on a batch of tasks.

        The fuctions first adapts to the support set and then evaluates it's perfomance
        on the query set.

        Args:
            state (MetaTrainState): Contains information regarding the current state.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be trained on.
            metrics (List[(ndarray, ndarray) -> ndarray]): List of metrics to be evaluated on the query set.

        Returns:
            Tuple[MetaTrainState, jnp.ndarray, List[jnp.ndarray]]: (Next_State, Loss, metrics).
        """

        params = state.params
        theta = params["params"]

        def batch_meta_train_loss(theta, apply_fn, adapt_fn, loss_fn, tasks):
            loss, metrics_value = jax.vmap(
                OptiTrainer.meta_loss, in_axes=(None, None, None, None, 0, None)
            )(
                params.copy({"params": theta}),
                apply_fn,
                adapt_fn,
                loss_fn,
                tasks,
                metrics,
            )
            return loss.mean(), [metric.mean() for metric in metrics_value]

        (loss, metrics_value), grads = jax.value_and_grad(
            batch_meta_train_loss, has_aux=True
        )(theta, state.apply_fn, state.adapt_fn, state.loss_fn, tasks)
        # state = state.apply_gradients(grads=grads)

        # if state.step == 0: # Initialize optimizer
        #     state = state.replace(opt_state=state.tx.init(state.params["params"]))

        updates, new_opt_state = state.tx.update(
            grads, state.opt_state, state.params["params"]
        )
        new_params = optax.apply_updates(state.params["params"], updates)
        params = state.params.copy({"params": new_params})

        state = state.replace(
            step=state.step + 1, params=params, opt_state=new_opt_state
        )
        if logger_type == "tensorboard":
            logger.TensorboardLogger.__enter__(state.step)
            logger.TensorboardLogger.scalar("loss", loss, state.step)
            for i, metric in enumerate(metrics):
                logger.TensorboardLogger.scalar(
                    metric.__name__, metrics_value[i], state.step
                )
        if logger_type == "wandb":
            logger.WandbLogger.__enter__(state.step)
            logger.WandbLogger.scalar("loss", loss, state.step)
            for i, metric in enumerate(metrics):
                logger.WandbLogger.scalar(metric.__name__, metrics_value[i], state.step)

        return state, loss, metrics_value

    @staticmethod
    @partial(jax.jit, static_argnums=(2,))
    def meta_test_step(
        state: MetaTrainState,
        tasks,
        metrics: List[Callable[[ndarray, ndarray], ndarray]] = [],
    ) -> Tuple[ndarray, List[ndarray]]:
        """Performs a single meta-testing step on a batch of tasks.

        The function first adapts to the support set and then evaluates it's perfomance
        on the query set.

        Args:
            state (MetaTrainState): Contains information regarding the current state.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be evaluated on.
            metrics (List[(ndarray, ndarray) -> ndarray]): List of metrics to be evaluated on the query set.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]: (Loss, metrics).
        """

        params = state.params
        apply_fn = state.apply_fn
        loss_fn = state.loss_fn
        adapt_fn = state.adapt_fn
        loss, metrics_value = jax.vmap(
            OptiTrainer.meta_loss, in_axes=(None, None, None, None, 0, None)
        )(params, apply_fn, adapt_fn, loss_fn, tasks, metrics)
        return loss.mean(), [metric.mean() for metric in metrics_value]

    @staticmethod
    def meta_loss(
        params, apply_fn, adapt_fn, loss_fn, task, metrics
    ) -> Tuple[ndarray, List[ndarray]]:
        """Calculates the Meta Loss of a task

        Args:
            params (flax.core.FrozenDict[str, Any]): Parameters of the model.
            apply_fn ((params, x) -> y): Function that applies the model to a batch of data.
            adapt_fn ((params, apply_fn, loss_fn, support_set) -> adapted_params): Specific meta learning function
            which adapts to the support set.
            loss_fn ((logits, targets) -> loss): Loss Function.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be trained on

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]: (Loss, metrics).
        """
        support_set, query_set = task

        # Adaptation step
        theta = adapt_fn(params, apply_fn, loss_fn, support_set)

        # Evaluation step
        x_train, y_train = query_set
        logits = apply_fn(params.copy({"params": theta}), x_train, train=False)

        # Calculate metrics
        metrics_value = [metric(logits, y_train) for metric in metrics]

        return loss_fn(logits, y_train), metrics_value
