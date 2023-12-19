from numpyro.handlers import seed
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

class CCVAE_ELBO(ELBO):
    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        # Careful, slow if num particules too big (due to lqyx computation)
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)

            model_trace, guide_trace = get_importance_trace(
                seeded_model, seeded_guide, args, kwargs, params
            )

            lqy_x = model_trace["lqy_x"]["value"]

            log_vi = model_trace['x']['log_prob']
            log_vi += model_trace['z_class']['log_prob']
            log_vi += model_trace['z_style']['log_prob']
            
            log_vi -= guide_trace['z_class']['log_prob']
            log_vi -= guide_trace['z_style']['log_prob']
            log_vi -= guide_trace['y']['log_prob']
            
            lpy = model_trace['y']['log_prob']

            w = jnp.exp(guide_trace['y']['log_prob'] - lqy_x)

            elbo = (w * log_vi + lpy).sum()
            return elbo, lqy_x

        rng_keys = random.split(rng_key, self.num_particles)
        elbo_particles, lqy_x_particles = vmap(single_particle_elbo)(rng_keys)
        loss = -jnp.mean(elbo_particles) - jnp.sum(lqy_x_particles) / self.num_particles

        return loss