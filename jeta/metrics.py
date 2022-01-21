import jax
import jax.numpy as jnp


@jax.jit
def task_accuracy(logits, labels):
    """Calculates classification accuracy for a task."""
    return jnp.mean(jnp.argmax(logits, -1) == labels)
