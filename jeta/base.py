from abc import ABC
from typing import Callable

import jax.numpy as jnp
from jax import random


class JetaBase(ABC):
    def __init__(
        self, batch_size: int = 32, alpha: float = 0.001, fas: int = 1, **kwargs
    ):

        self.batch_size = batch_size
        self.alpha = alpha
        self.fas = fas
        self.losses = []
        self.seed = kwargs["seed"] if "seed" in kwargs else None
        self.first_order = (
            kwargs["first_order"] if "first_order" in kwargs else None
        )  # for MAML and FOMAML
        if self.seed is not None:
            self.key = random.PRNGKey(self.seed)

    def train(self):
        pass

    def loss(self, theta, batch):
        pass

    def test(self):
        pass
