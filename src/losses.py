import jax
import jax.numpy as jnp


def binary_cross_entropy_loss(reconstructed_x, x):
    """
        binary_cross_entropy_loss between x and reconstructed_x.
        x and reconstructed_x are expected to be of shape (batch_size,) + img_shape.
    """
    epsilon = 1e-10  # To prevent log(0)
    reconstructed_x = jnp.clip(reconstructed_x, epsilon, 1 - epsilon)
    bce_loss = -(x * jnp.log(reconstructed_x) + (1 - x) * jnp.log(1 - reconstructed_x))
    return jnp.mean(bce_loss)

def gaussian_kl(mu, logvar):
    """
        KL divergence from a diagonal Gaussian to the standard Gaussian.
        mu and logvar are both expected to be of shape (batch_size, latent_dim). 
        mu is the mean of the gaussian, and logvar its log variance.
    """
    return -0.5 * jnp.mean(1 + logvar - jnp.square(mu) - jnp.exp(logvar))

def cross_entropy_loss(logits, labels):
    """
        Cross-entropy loss between the logits and the corresponding labels.
        logits is expected to be of shape (batch_size, num_classes).
        labels is a integer array of shape (batch_size, 1), where all its values are
        comprised in [0..num_classes-1].
    """
    return -jnp.mean(jnp.take_along_axis(jax.nn.log_softmax(logits), labels, axis=1))