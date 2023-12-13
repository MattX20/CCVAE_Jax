from typing import Type

import jax.numpy as jnp
from jax import random
from flax import linen as nn


class M2AboveLayers(nn.Module):
    """
        M2AboveLayers returns the mean and log variance (latent_dim vectors), and logits (num_classes vector)
        from an input_dim vector.
    """
    input_dim: int
    latent_dim: int
    num_classes: int

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, input_dim)
        mu = nn.Dense(features=self.latent_dim)(x)
        logvar = nn.Dense(features=self.latent_dim)(x)
        logy = nn.Dense(features=self.num_classes)(x)
        return mu, logvar, logy

class M2BelowLayers(nn.Module):
    """
        M2BelowLayers returns a output_dim vector from a latent_dim vector.
    """
    latent_dim: int
    output_dim: int

    @nn.compact
    def __call__(self, x):
        # Input x is expected to be of shape (batch_size, latent_dim)
        x = nn.Dense(features=self.output_dim)(x)
        return x
    

class M2VAE(nn.Module):
    """
        M2VAE is the implementation of the M2 model from 'Semi-supervised learning with deep generative models.' by Kingma et al.
        Given an encoder class, decoder class, a latent dim and a number of classes, returns a similar shape reconstructed input,
        mu a latent_dim vector (mean of latent), logvar a latent_dim vector (log variance of latent), and logy a num_classes vector
        (logits for classification).
    """
    encoder: nn.Module
    decoder: nn.Module
    above_layers: nn.Module
    below_layers: nn.Module
    latent_dim: int
    num_classes: int

    def __call__(self, x, rng_key):
        # Input x is expected to be of shape (batch_size, ...), and rng_key a jax random key

        # Encoding
        x = self.encoder(x)
        mu, logvar, logy = self.above_layers(x)

        # Reparameterization trick
        std = jnp.exp(0.5 * logvar)
        epsilon = random.normal(rng_key, mu.shape)
        z = mu + std * epsilon  

        # Decoding
        y = nn.softmax(logy)
        zy = jnp.concatenate([z, y], axis=-1)
        zy = self.below_layers(zy)
        reconstructed_x = self.decoder(zy)
        
        return reconstructed_x, mu, logvar, logy


def instanciate_MVAE(encoder_class: Type[nn.Module], decoder_class: Type[nn.Module], latent_dim: int, num_classes: int):
    encoder = encoder_class()
    decoder = decoder_class()
    above_layers = M2AboveLayers(input_dim=encoder.output_dim, 
                                            latent_dim=latent_dim, 
                                            num_classes=num_classes)
    below_layers = M2BelowLayers(latent_dim=latent_dim + num_classes,
                                        output_dim=decoder.input_dim)

    return M2VAE(encoder, decoder, above_layers, below_layers, latent_dim, num_classes)