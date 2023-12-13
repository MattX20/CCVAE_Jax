import jax
import jax.numpy as jnp
from jax import random
from jax import device_put
import numpy as np
from tqdm import tqdm
import flax
from flax.training import train_state

import pickle

from jax import grad, jit, value_and_grad
import optax

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from src.data_transforms import default_transform, jax_collate_fn, jax_semi_supervised_collate_fn
from src.constants import CIFAR10_IMG_SHAPE, MNIST_IMG_SHAPE
from src.models.encoder_decoder import CIFAR10Encoder, CIFAR10Decoder, MNISTEncoder, MNISTDecoder
from src.models.M2VAE import instanciate_MVAE
from src.losses import binary_cross_entropy_loss, gaussian_kl, cross_entropy_loss
from src.utils import compute_accuracy


from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

# Load dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)

# Split the training set into training and validation sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

print(int(0.1 * train_size), test_size)

# Mask 80% of the training labels
mask = np.random.rand(len(train_dataset)) < 0.9
for idx in np.where(mask)[0]:
    train_dataset.dataset.targets[train_dataset.indices[idx]] = -1  # Use -1 or any other placeholder for unknown labels

# Create data loaders
train_loader = DataLoader(
    train_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=6, 
    collate_fn=lambda batch : jax_semi_supervised_collate_fn(batch, CIFAR10_IMG_SHAPE)
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=6, 
    collate_fn=lambda batch : jax_collate_fn(batch, CIFAR10_IMG_SHAPE)
)

# Initialize your model and optimizer
model = instanciate_MVAE(encoder_class=CIFAR10Encoder, decoder_class=CIFAR10Decoder, latent_dim=50, num_classes=10)
optimizer = optax.adam(learning_rate=1e-4)

# Initialize parameters
params = model.init(random.PRNGKey(0), jnp.ones((1,) + CIFAR10_IMG_SHAPE), random.PRNGKey(0))
optimizer = optax.adam(1e-3)

key = random.PRNGKey(42)

@jit
def train_step(state, batch, rng_key):

    def loss_fn(params):
        # Unpack the batches
        labeled_inputs, labeled_labels, unlabeled_inputs = batch

        # Initialize losses
        recon_loss_labeled, kl_loss_labeled, ce_loss = 0, 0, 0
        recon_loss_unlabeled, kl_loss_unlabeled = 0, 0

        # Check and process labeled data
        if labeled_inputs.size > 0:
            reconstructed_labeled, mu_labeled, logvar_labeled, logy_labeled = model.apply(params, labeled_inputs, rng_key)
            recon_loss_labeled = binary_cross_entropy_loss(reconstructed_labeled, labeled_inputs)
            kl_loss_labeled = gaussian_kl(mu_labeled, logvar_labeled)
            ce_loss = 100 *  cross_entropy_loss(logy_labeled, labeled_labels)

        # Check and process unlabeled data
        if unlabeled_inputs.size > 0:
            reconstructed_unlabeled, mu_unlabeled, logvar_unlabeled, _ = model.apply(params, unlabeled_inputs, rng_key)
            recon_loss_unlabeled = binary_cross_entropy_loss(reconstructed_unlabeled, unlabeled_inputs)
            kl_loss_unlabeled = gaussian_kl(mu_unlabeled, logvar_unlabeled)

        # Calculate total loss
        total_recon_loss = recon_loss_labeled + recon_loss_unlabeled
        total_kl_loss = kl_loss_labeled + kl_loss_unlabeled
        total_loss = total_recon_loss + total_kl_loss + ce_loss
        return total_loss

    grad_fn = value_and_grad(loss_fn, has_aux=False)
    total_loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss

# Initialize the training state
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)



# Training loop
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    for batch in train_loader: 
        batch = device_put(batch)
        key, subkey = random.split(key)
        state, total_loss = train_step(state, batch, subkey)

    # Evaluation
    accuracies = []
    for batch in test_loader:
        # Move batch to GPU
        batch = jax.device_put(batch)
        key, subkey = random.split(key)
        
        _, _, _, logits = model.apply(state.params, batch[0], subkey)
        accuracies.append(compute_accuracy(logits, batch[1]))

    print(f"Epoch {epoch}, Loss: {total_loss}, Test Accuracy: {np.mean(accuracies)}")

# Save parameters
with open('model_weights.pkl', 'wb') as file:
    pickle.dump(state.params, file)