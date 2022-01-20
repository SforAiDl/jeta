import jax
import jax.numpy as jnp

from loss import task_loss

@jax.jit
def train_step(optimizer, batch, fit_task):
    """Perform the training step used for each epoch.

    Parameters
    ----------
    optimizer: flax.optim.Optimizer
        An optimizer object.
    batch: tuple
        A batch of tasks: (X_adap, y_adap, X_eval, y_eval)
    fit_task: A callable that fits on a given task and returns the updated model.

    Returns
    -------
    optimizer: flax.optim.Optimizer
        The updated optimizer object.
    loss: float
        Loss for an epoch.

    """
    model = optimizer.target

    @jax.jit
    def loss(model, train_x, train_y, val_x, val_y):
        """Calculates loss on the query set after fitting `model` to the support set.

        Parameters
        ----------
        model: flax.nn.Model
        train_x, train_y: Support set for a single task.
        val_x, val_y: Query set for a single task.

        Returns
        -------
        loss: float
            Task loss.

        """
        train_batch = (train_x, train_y)
        val_batch = (val_x, val_y)
        # Fast adaptation (inner loop)
        # TODO: Allow *args/**kwargs for `fit_task`.
        updated_model = fit_task(model, train_batch)
        # Evaluate on query set to get the loss on the query set.
        loss = task_loss(updated_model, val_batch)
        return loss

    @jax.jit
    def loss_fn(model):
        """
        Parameters
        ----------
        model: flax.nn.Model

        Returns
        -------
        Mean of task losses.

        """
        train_x, train_y, val_x, val_y = batch
        task_losses = jax.vmap(partial(loss, model))(  # Apply the `loss` function to each task. `vmap` is used to apply it on all tasks.
            train_x, train_y, val_x, val_y
        )
        return jnp.mean(task_losses)

    loss, grad = jax.value_and_grad(loss_fn)(model)
    # `apply_gradient` returns a new `Optimizer` instance with the updated target and optimizer state.
    optimizer = optimizer.apply_gradient(grad)

    return optimizer, loss
