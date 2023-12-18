from typing import Type, Tuple

import jax.numpy as jnp
from jax import random

from flax import linen as nn

import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist


class CCVAEEncoder(nn.Module):
    """
        CCVAEEncoder takes as input an element of the dataset, and returns a loc and a scale
        to sample the latent z.
    """
    encoder_class: Type[nn.Module]
    latent_dim: int

    def setup(self):
        self.encoder = self.encoder_class()
        self.fc_loc = nn.Dense(features=self.latent_dim)
        self.fc_scale = nn.Dense(features=self.latent_dim)
    
    def __call__(self, x):
        x = self.encoder(x)
        
        loc = self.fc_loc(x)
        log_scale = self.fc_scale(x)
        scale = jnp.exp(log_scale)

        return loc, scale

class CCVAEDecoder(nn.Module):
    """
        CCVAEDecoder takes a latent variable and return a reconstructed image.
    """
    decoder_class: Type[nn.Module]

    def setup(self):
        self.decoder = self.decoder_class()
        self.fc = nn.Dense(features=self.decoder.input_dim)
    
    def __call__(self, z):
        z = self.fc(z)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

class CondPrior(nn.Module):
    num_classes: int

    def setup(self):
        self.diag_loc_true = self.param('diag_loc_true', nn.initializers.ones, (self.num_classes,)) * 2
        self.diag_loc_false = self.param('diag_loc_false', nn.initializers.ones, (self.num_classes,)) * -2
        self.diag_scale_true = self.param('diag_scale_true', nn.initializers.ones, (self.num_classes,))
        self.diag_scale_false = self.param('diag_scale_false', nn.initializers.ones, (self.num_classes,))

    def __call__(self, y):
        loc = y * self.diag_loc_true + (1 - y) * self.diag_loc_false
        scale = y * self.diag_scale_true + (1 - y) * self.diag_scale_false
        return loc, nn.activation.softplus(scale)

class Classifier(nn.Module):
    dim: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.dim,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.dim,))

    def __call__(self, x):
        return x * self.weight + self.bias   


class CCVAE:
    """
        Implementation of the CCVAE model.
    """

    def __init__(self,
                 encoder_class: Type[nn.Module],
                 decoder_class: Type[nn.Module],
                 num_classes: int,
                 latent_dim: int,
                 img_shape: Tuple[int, int, int],
                 scale_factor: float,
                 distribution: str,
                 multiclass: bool = False):
        
        assert latent_dim > num_classes, "The dimention of the latent must be greater than the number of classes"

        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.num_classes = num_classes
        
        self.latent_dim = latent_dim
        self.latent_class = num_classes
        self.latent_style = latent_dim - num_classes

        self.img_shape = img_shape
        self.scale_factor = scale_factor
        self.distribution = distribution

        self.multiclass = multiclass
    
    def model_supervised(self, xs, ys):
        batch_size = xs.shape[0]

        cond_prior_class = flax_module(
            "cond_prior_class", 
            CondPrior(self.latent_class), 
            y=jnp.ones((1, self.latent_class))
        ) 

        decoder = flax_module(
            "decoder", 
            CCVAEDecoder(self.decoder_class), 
            z=jnp.ones((1, self.latent_dim))
        )

        with numpyro.plate("data", batch_size):
            if self.multiclass:
                alpha_prior = jnp.ones((batch_size, self.num_classes)) / 2
                y_one_hot = numpyro.sample("y", dist.Bernoulli(alpha_prior).to_event(1), obs=ys)
            else :
                alpha_prior = jnp.ones((batch_size, self.num_classes)) / self.num_classes
                ys_p = numpyro.sample("y", dist.Categorical(alpha_prior), obs=ys)
                y_one_hot = jnp.eye(self.num_classes)[ys_p]

            z_class_prior_loc, z_class_prior_scale = cond_prior_class(y_one_hot)
            z_class = numpyro.sample("z_class", 
                                     dist.Normal(z_class_prior_loc, z_class_prior_scale).to_event(1)
            )

            z_style_prior_loc = jnp.zeros((batch_size, self.latent_style))
            z_style_prior_scale = jnp.ones((batch_size, self.latent_style))
            z_style = numpyro.sample("z_style",
                                     dist.Normal(z_style_prior_loc, z_style_prior_scale).to_event(1)
            )

            z = jnp.concatenate([z_class, z_style], axis=-1)

            loc = decoder(z)
            numpyro.deterministic("loc", loc)

            if self.distribution == "bernoulli":
                numpyro.sample("x", dist.Bernoulli(loc).to_event(3), obs=xs)
            elif self.distribution == "laplace":
                numpyro.sample("x", dist.Laplace(loc).to_event(3), obs=xs)
    
    def guide_supervised(self, xs, ys):
        