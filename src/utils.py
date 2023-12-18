import jax.numpy as jnp


def compute_accuracy(logits, labels):
    """
        Computes the accuracy between the labels and the given logits.
        labels are expected to be of shape (batch_size, 1).
        logits are expected to be of shape (batch_size, num_classes).
    """
    preds = jnp.argmax(logits, axis=1)
    return jnp.mean(preds == labels)