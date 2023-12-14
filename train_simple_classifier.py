import jax
import flax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from tqdm import tqdm

from src.models.encoder_decoder import CIFAR10Encoder, MNISTEncoder
from src.models.simple_classifier import SimpleClassifier
from src.losses import cross_entropy_loss
from src.utils import compute_accuracy
from src.data_loading.loaders import get_data_loaders

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

seed = 42

# Data loading

# Data loading
img_shape, loader_dict = get_data_loaders(dataset_name="MNIST", 
                                          p_test=0.2, 
                                          p_val=0.1, 
                                          p_supervised=0.2, 
                                          batch_size=2, 
                                          num_workers=0, 
                                          seed=seed)

# Initialize the model parameters
model = SimpleClassifier(encoder=MNISTEncoder, num_classes=10)
params = model.init(jax.random.PRNGKey(0), jnp.ones((1,) + img_shape))['params']

# Create an optimizer
optimizer = optax.adam(learning_rate=0.001)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        # Unpack the batch
        X, y = batch

        logits = model.apply({'params': params}, X)
        loss = cross_entropy_loss(logits, y)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

# Initialize the training state
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

train_loader = loader_dict["supervised"]
validation_loader = loader_dict["validation"]
test_loader = loader_dict["test"]

# Training loop
num_epochs = 20
for epoch in tqdm(range(1, num_epochs + 1)):
    # Train

    running_loss = 0.0

    for batch in train_loader:
        # Move batch to GPU
        batch = jax.device_put(batch)

        state, loss, logits = train_step(state, batch)
        running_loss += loss

    # Validation
    accuracies = []
    for batch in validation_loader:
        # Move batch to GPU
        batch = jax.device_put(batch)
        
        logits = model.apply({'params': state.params}, batch[0])
        accuracies.append(compute_accuracy(logits, batch[1]))
    
    val_accuracy = np.mean(accuracies)
    print(f"Epoch {epoch}, Loss: {running_loss / len(train_loader)}, Val Accuracy: {val_accuracy}")

# Test
accuracies = []
for batch in test_loader:
    # Move batch to GPU
    batch = jax.device_put(batch)
    
    logits = model.apply({'params': state.params}, batch[0])
    accuracies.append(compute_accuracy(logits, batch[1]))

test_accuracy = np.mean(accuracies)
print(f"Test Accuracy: {test_accuracy}")