from typing import Sequence

from flax import linen as nn


class MLP(nn.Module):
    """
    Taken from https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html

    """
    features: Sequence[int]

    @nn.compact
    def __call__(self, inputs):
      x = inputs
      for i, feat in enumerate(self.features):
          x = nn.Dense(feat, name=f'layers_{i}')(x)
          if i != len(self.features) - 1:
            x = nn.relu(x)
      return x
