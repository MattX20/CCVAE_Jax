from typing import Type, Tuple

import jax.numpy as jnp
from jax import random

from flax import linen as nn

import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist


