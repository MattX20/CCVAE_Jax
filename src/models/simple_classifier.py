from typing import Type

from flax import linen as nn


class SimpleClassifier(nn.Module):
    """
        SimpleClassifier is a baseline classification model to compare agains self-supervised methods.
        It returns a num_class vector from a (batch_size, ...) input.
    """
    encoder: Type[nn.Module]
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, ...)
        x = self.encoder()(x)
        x = nn.Dense(features=self.num_classes)(x)
        return x