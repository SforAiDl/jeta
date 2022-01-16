import jax
from flax import optim

from max.loss import task_loss


@jax.jit
def maml_fit_task(model, batch):
    """Performs fast adaptation on support set.

    Parameters
    ----------
    model: flax.nn.Model
        A Flax model object.
    batch: tuple
        Batch of data of a task: (X, y). This is the support set.
    inner_optimizer_def: flax.optim.Optimizer
        Optimizer to use for fast adaptation steps.

    Returns
    -------
    updated_model: flax.nn.Model
        After applying the inner adaptation step.

    """
    maml_lr = 0.001  # TODO: Allow this as an argument.
    fas = 5
    inner_optimizer_def = optim.GradientDescent(learning_rate=maml_lr)

    model_grad = jax.grad(task_loss)(model, batch)
    # Create a new optimizer for the given `model`.
    # This is analogous to creating a learner for each task using `.clone()`.
    inner_opt = inner_optimizer_def.create(model)
    for _ in range(fas):  # Do fast adaptation for `fas` number of times.
        inner_opt = inner_opt.apply_gradient(
            model_grad
        )  # Analogous to `learner.adapt` step from `learn2learn`.
    return (
        inner_opt.target
    )  # return the updated model stored as an attribute of the updated optimizer.
