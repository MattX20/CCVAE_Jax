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
from src.models.encoder_decoder import CIFAR10Encoder, CIFAR10Decoder, MNISTEncoder, MNISTDecoder
from src.models.M2VAE import instanciate_MVAE
from src.losses import binary_cross_entropy_loss, gaussian_kl, cross_entropy_loss
from src.utils import compute_accuracy
from src.data_loading.loaders import get_data_loaders

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)


seed = 42

# Data loading
img_shape, loader_dict = get_data_loaders(dataset_name="MNIST", 
                                          p_test=0.2, 
                                          p_val=0.1, 
                                          p_supervised=0.2, 
                                          batch_size=2, 
                                          num_workers=0, 
                                          seed=seed)

# Initialize your model and optimizer
model = instanciate_MVAE(encoder_class=MNISTEncoder, decoder_class=MNISTDecoder, latent_dim=10, num_classes=10)
optimizer = optax.adam(learning_rate=1e-4)

# Initialize parameters
params = model.init(random.PRNGKey(0), jnp.ones((1,) + img_shape), random.PRNGKey(0))
optimizer = optax.adam(1e-3)

key = random.PRNGKey(seed)

@jit
def train_step_supervised(state, batch, rng_key):

    def loss_fn(params):
        # Unpack the batch
        X, y = batch

        # Process the batch
        reconstructed_X, mu, logvar, logy = model.apply(params, X, rng_key)
        
        # Get losses
        recon_loss = binary_cross_entropy_loss(reconstructed_X, X)
        kl_loss = gaussian_kl(mu, logvar)
        ce_loss = cross_entropy_loss(logy, y)

        total_loss = recon_loss + kl_loss + ce_loss

        return total_loss

    grad_fn = value_and_grad(loss_fn, has_aux=False)
    total_loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss

@jit
def train_step_unsupervised(state, batch, rng_key):

    def loss_fn(params):
        # Unpack the batch
        X = batch

        # Process the batch
        reconstructed_X, mu, logvar, logy = model.apply(params, X, rng_key)
        
        # Get losses
        recon_loss = binary_cross_entropy_loss(reconstructed_X, X)
        kl_loss = gaussian_kl(mu, logvar)

        total_loss = recon_loss + kl_loss

        return total_loss

    grad_fn = value_and_grad(loss_fn, has_aux=False)
    total_loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, total_loss

# Initialize the training state
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

semi_supervised_loader = loader_dict["semi_supervised"]
validation_loader = loader_dict["validation"]
test_loader = loader_dict["test"]

# Training loop
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    running_loss = 0.0

    # Trainning
    for is_supervised, batch in semi_supervised_loader: 
        batch = device_put(batch)
        key, subkey = random.split(key)

        if is_supervised:
            state, total_loss = train_step_supervised(state, batch, subkey)
        else:
            state, total_loss = train_step_unsupervised(state, batch, subkey)
        
        running_loss += total_loss

    # Validation
    accuracies = []
    for batch in validation_loader:
        X, y = batch
        X = jax.device_put(X)
        key, subkey = random.split(key)
        
        _, _, _, logy = model.apply(state.params, X, subkey)
        accuracies.append(compute_accuracy(logy, y))

    print(f"Epoch {epoch}, Loss: {running_loss / len(semi_supervised_loader)}, Val Accuracy: {np.mean(accuracies)}")

 # Test
accuracies = []
for batch in test_loader:
    X, y = batch
    X = jax.device_put(X)
    key, subkey = random.split(key)
    
    _, _, _, logy = model.apply(state.params, X, subkey)
    accuracies.append(compute_accuracy(logy, y))

print(f"Test Accuracy: {np.mean(accuracies)}")

# Save parameters
with open('model_weights.pkl', 'wb') as file:
    pickle.dump(state.params, file)