import numpy as np

import torch
from torchvision.transforms import v2

# Base transform, works for MNIST and CIFAR10 (not for CELEBA, images need to be reshaped to 64x64)
default_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True)
])

def jax_collate_fn(batch, img_shape):
    """
        collate function to be used with torch dataloader to get jax
        compatible inputs.
    """
    inputs, labels = zip(*batch)

    inputs_np = np.array([np.array(x.numpy()) for x in inputs])
    labels_np = np.array(labels)

    inputs_np_reshaped = inputs_np.reshape((-1,) + img_shape)
    labels_np_reshaped = labels_np.reshape((-1, 1))

    return inputs_np_reshaped, labels_np_reshaped

def jax_semi_supervised_collate_fn(batch, img_shape):
    """
        collate function to be used with torch dataloader to get jax
        compatible inputs when semi-supervised learning is used.

        Returns labeled and unlabeled data separately. Assume unlabeled data
        has -1 label.
    """
    inputs, labels = zip(*batch)

    inputs_np = np.array([np.array(x.numpy()) for x in inputs])
    labels_np = np.array(labels)

    inputs_np_reshaped = inputs_np.reshape((-1,) + img_shape)
    
    labeled_mask = labels_np >= 0
    unlabeled_mask = labels_np < 0

    labeled_inputs = inputs_np_reshaped[labeled_mask]
    labeled_labels = labels_np[labeled_mask]
    labeled_labels_reshaped = labeled_labels.reshape((-1, 1))

    unlabeled_inputs = inputs_np_reshaped[unlabeled_mask]

    return labeled_inputs, labeled_labels_reshaped, unlabeled_inputs