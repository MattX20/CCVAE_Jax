from typing import Type, Tuple

import jax.numpy as jnp
from jax import random

from flax import linen as nn

import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist


class M2FirstEncoder(nn.Module):
    encoder_class: Type[nn.Module]
    num_classes: int
    latent_dim: int

    def setup(self):
        self.encoder = self.encoder_class()
        self.fc_h = nn.Dense(features=self.latent_dim)
        self.fc_y = nn.Dense(features=self.num_classes)
    
    def __call__(self, x):
        x = self.encoder(x)
        h = self.fc_h(x)
        logy = self.fc_y(x)
        h = nn.relu(h)
        y = nn.softmax(logy)

        return h, y

class M2SecondEncoder(nn.Module):
    latent_dim: int

    @nn.compact
    def __call__(self, h, y_one_hot):
        hy = jnp.concatenate([h, y_one_hot], axis=-1)
        loc = nn.Dense(features=self.latent_dim)(hy)
        log_scale = nn.Dense(features=self.latent_dim)(hy)
        scale = jnp.exp(log_scale)

        return loc, scale

class M2Decoder(nn.Module):
    decoder_class: Type[nn.Module]

    def setup(self):
        self.decoder = self.decoder_class()
        self.fc = nn.Dense(features=self.decoder.input_dim)
    
    def __call__(self, z, y_one_hot):
        zy = jnp.concatenate([z, y_one_hot], axis=-1)
        zy = self.fc(zy)
        reconstructed_x = self.decoder(zy)
        return reconstructed_x


class M2VAE:

    def __init__(self,
                 encoder_class: Type[nn.Module],
                 decoder_class: Type[nn.Module],
                 num_classes: int,
                 latent_dim: int,
                 img_shape: Tuple[int, int, int]):
        
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_shape = img_shape
    
    def model(self, xs, ys=None):
        batch_size = xs.shape[0]

        decoder = flax_module(
            "decoder", 
            M2Decoder(self.decoder_class), 
            z=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        ) 

        # with numpyro.plate("data", batch_size):

        prior_loc = jnp.zeros((batch_size, self.latent_dim))
        prior_scale = jnp.ones((batch_size, self.latent_dim))
        zs = numpyro.sample("z", dist.Normal(prior_loc, prior_scale))

        alpha_prior = jnp.ones((batch_size, self.num_classes)) / self.num_classes
        ys = numpyro.sample("y", dist.Categorical(alpha_prior), obs=ys)

        if ys is not None:
            with numpyro.handlers.scale(scale=100.):
                numpyro.sample("y_aux", dist.Categorical(alpha_prior), obs=ys)
        
        y_one_hot = jnp.eye(self.num_classes)[ys]
        loc = decoder(zs, y_one_hot)
        numpyro.sample("x", dist.Bernoulli(loc), obs=xs)

        return loc

    def guide(self, xs, ys=None):
        batch_size = xs.shape[0]

        encoder1 = flax_module(
            "encoder1",
            M2FirstEncoder(self.encoder_class, self.num_classes, self.latent_dim),
            x=jnp.ones((1,) + self.img_shape)
        )

        encoder2 = flax_module(
            "encoder2",
            M2SecondEncoder(self.latent_dim),
            h=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        )

        # with numpyro.plate("data", batch_size):

        h, y_prob = encoder1(xs)
        if ys is None:
            ys = numpyro.sample("y", dist.Categorical(y_prob))
        y_one_hot = jnp.eye(self.num_classes)[ys]
        loc, scale = encoder2(h, y_one_hot)
        numpyro.sample("z", dist.Normal(loc, scale))