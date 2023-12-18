import numpy as np


def jax_supervised_collate_fn(batch):
    """
        collate function to be used with torch dataloader to get jax
        compatible inputs, supervised case.
    """
    inputs, labels = zip(*batch)

    inputs_np = np.array([np.array(x.numpy()) for x in inputs])
    labels_np = np.array(labels)

    inputs_np_reshaped = np.transpose(inputs_np, (0, 2, 3, 1))

    return inputs_np_reshaped, labels_np

def jax_unsupervised_collate_fn(batch):
    """
        collate function to be used with torch dataloader to get jax
        compatible inputs, unsupervised case.
    """
    inputs, _ = zip(*batch)

    inputs_np = np.array([np.array(x.numpy()) for x in inputs])

    inputs_np_reshaped = np.transpose(inputs_np, (0, 2, 3, 1))

    return inputs_np_reshaped