import numpyro
import numpyro.distributions as dist
from numpyro.infer import ELBO
import jax.numpy as jnp
from jax import random
from jax import vmap
from numpyro.infer.util import get_importance_trace


def binary_cross_entropy_loss(reconstructed_x, x):
    """
        binary_cross_entropy_loss between x and reconstructed_x.
        x and reconstructed_x are expected to be of shape (batch_size,) + img_shape.
    """
    epsilon = 1e-10  # To prevent log(0)
    reconstructed_x = jnp.clip(reconstructed_x, epsilon, 1 - epsilon)
    bce_loss = -(x * jnp.log(reconstructed_x) + (1 - x) * jnp.log(1 - reconstructed_x))
    return jnp.mean(bce_loss)

class TraceCCVAE_ELBO(ELBO):
    def loss(self, rng_key, model, guide, *args, **kwargs):
        def single_particle_elbo(rng_key):
            model_trace, guide_trace = get_importance_trace(
                rng_key, model, guide, args, kwargs, 1
            )

            log_vi = model_trace.nodes['recon']['log_prob']
            log_vi += model_trace.nodes['z_class']['log_prob']
            log_vi += model_trace.nodes['z_style']['log_prob']
            log_vi -= guide_trace.nodes['z_class']['log_prob']
            log_vi -= guide_trace.nodes['z_style']['log_prob']
            log_vi -= guide_trace.nodes['y']['log_prob']
            
            lpy = model_trace.nodes['y']['log_prob']

            w = jnp.exp(guide_trace.nodes['y']['log_prob'] - lpy)

            elbo = (w * log_vi + lpy).sum()
            return elbo

        elbo_particles = vmap(single_particle_elbo)(random.split(rng_key, self.num_particles))
        loss = -jnp.mean(elbo_particles)

        return loss