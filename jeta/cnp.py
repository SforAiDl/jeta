from functools import partial
from typing import Callable, List, Tuple

import jax
from flax.training.train_state import TrainState
from jax.numpy import ndarray


class ModelTrainer:
    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def meta_train_step(
        encoder_state: TrainState,
        decoder_state: TrainState,
        tasks,
        loss_fn: Callable[[ndarray, ndarray], ndarray],
        metrics: List[Callable[[ndarray, ndarray], ndarray]] = [],
    ) -> Tuple[TrainState, TrainState, ndarray, List[ndarray]]:
        """Performs a single meta-training step on a batch of tasks.

        The fuctions first adapts to the support set and then evaluates it's perfomance
        on the query set.

        Args:
            encoder_state (TrainState): Contains information regarding the current encoder state.
            decoder_state (TrainState): Contains information regarding the current decoder state.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be evaluated on.
            loss_fn ((logits, targets) -> loss): Loss Function.
            metrics (Tuple[(ndarray, ndarray) -> ndarray]): Tuple of metrics to be evaluated on the query set.

        Returns:
            Tuple[TrainState, TrainState, jnp.ndarray, List[jnp.ndarray]]: (Next_Encoder_State, Next_Decoder_State, Loss, metrics).
        """

        def batch_meta_train_loss(encoder_params, decoder_params):
            batch_encoder_state = encoder_state.replace(params=encoder_params)
            batch_decoder_state = decoder_state.replace(params=decoder_params)

            loss, metrics_value = jax.vmap(
                ModelTrainer.meta_loss,
                in_axes=(None, None, None, 0, None, None),
            )(batch_encoder_state, batch_decoder_state, loss_fn, tasks, metrics, True)
            return loss.mean(), [metric.mean() for metric in metrics_value]

        (loss, metrics_value), (encoder_grads, decoder_grads) = jax.value_and_grad(
            batch_meta_train_loss, argnums=(0, 1), has_aux=True
        )(encoder_state.params, decoder_state.params)

        encoder_state = encoder_state.apply_gradients(grads=encoder_grads)
        decoder_state = decoder_state.apply_gradients(grads=decoder_grads)

        return encoder_state, decoder_state, loss, metrics_value

    @staticmethod
    @partial(jax.jit, static_argnums=(3, 4))
    def meta_test_step(
        encoder_state: TrainState,
        decoder_state: TrainState,
        tasks,
        loss_fn: Callable[[ndarray, ndarray], ndarray],
        metrics: List[Callable[[ndarray, ndarray], ndarray]] = [],
    ) -> Tuple[ndarray, List[ndarray]]:
        """Performs a single meta-testing step on a batch of tasks.

        The function first adapts to the support set and then evaluates it's perfomance
        on the query set.

        Args:
            encoder_state (TrainState): Contains information regarding the current encoder state.
            decoder_state (TrainState): Contains information regarding the current decoder state.
            tasks ((x_train, y_train), (x_test, y_test)): Batch of tasks to be evaluated on.
            loss_fn ((logits, targets) -> loss): Loss Function.
            metrics (Tuple[(ndarray, ndarray) -> ndarray]): Tuple of metrics to be evaluated on the query set.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]: (Loss, metrics).
        """

        loss, metrics_value = jax.vmap(
            ModelTrainer.meta_loss,
            in_axes=(None, None, None, 0, None, None),
        )(encoder_state, decoder_state, loss_fn, tasks, metrics, False)
        return loss.mean(), [metric.mean() for metric in metrics_value]

    @staticmethod
    def meta_loss(
        encoder_state: TrainState,
        decoder_state: TrainState,
        loss_fn,
        task,
        metrics,
        train,
    ) -> Tuple[ndarray, List[ndarray]]:
        """Calculates the Meta Loss of a task

        Args:
            encoder_state (TrainState): Contains information regarding the current encoder state.
            decoder_state (TrainState): Contains information regarding the current decoder state.
            loss_fn ((logits, targets) -> loss): Loss Function.
            task ((x_train, y_train), (x_test, y_test)): Task to be evaluated on.
            metrics (Tuple[(ndarray, ndarray) -> ndarray]): Tuple of metrics to be evaluated on the query set.
            train (bool): Whether the encoder/decoder functions are in train/test mode.

        Returns:
            Tuple[jnp.ndarray, List[jnp.ndarray]]: (Loss, metrics).
        """
        support_set, query_set = task

        # Adaptation step
        r = encoder_state.apply_fn(encoder_state.params, *support_set, train=train)

        # Evaluation step
        x_test, y_test = query_set
        logits = decoder_state.apply_fn(decoder_state.params, r, x_test, train=train)

        # Calculate metrics
        metrics_value = [metric(logits, y_test) for metric in metrics]

        return loss_fn(logits, y_test), metrics_value
