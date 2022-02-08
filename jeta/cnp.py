import numpy as np
from typing import Sequence

import jax
from jax import random, numpy as jnp

from flax import optim
from flax import linen as nn

from models import MLP


class Encoder(nn.Module):
  features: Sequence[int]

  def setup(self):
    self.mlp1 = MLP(features=self.features)  # features = [...]

  def __call__(self, context_x, context_y, features):
    encoder_input = jnp.concatenate((context_x, context_y), axis=-1)
    batch_size, set_size, filter_size = encoder_input.shape
    key1, key2 = random.split(random.PRNGKey(0))
    x_input = jnp.reshape(encoder_input, (batch_size*set_size, -1))
    x_dummy = random.normal(key1, x_input.shape) # Dummy input
    params = self.mlp1.init(key2, x_dummy)

    out = self.mlp1.apply(params, x_input)
    out = out.reshape(batch_size, set_size, -1)
    representation = jnp.mean(out, axis=1)
    return representation

class Decoder(nn.Module):
  features: Sequence[int]

  def setup(self):
    self.mlp2 = MLP(features=self.features)  # features = [...]

  def __call__(self, representation, target_x, features):
    batch_size, set_size, d = target_x.shape

    representation = jnp.repeat(
        jnp.expand_dims(representation, axis=1),
        set_size, axis=1
    )
    input = jnp.concatenate((representation, target_x), axis=-1)

    key1, key2 = random.split(random.PRNGKey(0))
    x_input = jnp.reshape(input, (batch_size*set_size, -1))
    x_dummy = random.normal(key1, x_input.shape) # Dummy input
    params = self.mlp2.init(key2, x_dummy)

    out = self.mlp2.apply(params, x_input)
    out = out.reshape(batch_size, set_size, -1)
    mu, log_sigma = jnp.split(out, 2, axis=-1)
    sigma = 0.1 + 0.9 * jax.nn.softplus(log_sigma)

    key = random.PRNGKey(100)
    # random.multivariate_normal(key, mean=mu, cov=sigma) ?
    dist = mu + sigma * random.normal(10000,)  # TODO: Check this
    return dist, mu, sigma

class CNP(nn.Module):
  encoder_features: Sequence[int]
  decoder_features: Sequence[int]

  def setup(self):
    self.encoder = Encoder(self.encoder_features)
    self.decoder = Decoder(self.decoder_features)

  def __call__(self, encoder_sizes, decoder_sizes, query, target_y=None):
    (context_x, context_y), target_x = query

    context_x = context_x.numpy()
    context_y = context_y.numpy()
    target_x = target_x.numpy()
    if target_y is not None:
      target_y = target_y.numpy()

    representation = self.encoder(context_x, context_y, features=encoder_sizes)
    dist, mu, sigma = self.decoder(representation, target_x, features=decoder_sizes)

    log_p = None if target_y is None else jnp.log(target_y)
    return log_p, mu, sigma
