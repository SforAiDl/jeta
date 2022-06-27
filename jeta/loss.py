import jax.numpy as jnp

def mse(logits, targets) -> jnp.ndarray:
    return jnp.mean(jnp.square(logits - targets))