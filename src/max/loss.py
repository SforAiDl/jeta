import jax
import jax.numpy as jnp
import optax

@jax.jit
def task_loss(model, batch):
    """Calculates loss for a single classification task."""
    x, labels = batch
    logits = model(x)

    return jnp.mean(
        optax.softmax_cross_entropy(
            logits=logits, labels=jax.nn.one_hot(labels, num_classes=ways)
        )
    )  # Use MSE, for example, for regression.
