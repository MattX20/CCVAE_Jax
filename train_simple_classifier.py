import jax
import flax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from tqdm import tqdm

from src.models.encoder_decoder import CIFAR10Encoder
from src.models.simple_classifier import SimpleClassifier
from src.losses import cross_entropy_loss

from torch.utils.data import DataLoader, random_split
from torchvision import datasets

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

from src.data_transforms import default_transform, jax_collate_fn
from src.utils import compute_accuracy
from src.constants import CIFAR10_IMG_SHAPE

# Load CIFAR10 dataset
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)

# Split the training set into training and validation sets
temp_size = int(0.8 * len(dataset))
test_size = len(dataset) - temp_size
train_size = int(0.1 * temp_size)
temp_dataset, test_dataset = random_split(dataset, [temp_size, test_size])
train_dataset, _ = random_split(temp_dataset, [train_size, temp_size - train_size])
print(test_size, train_size, temp_size)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6, collate_fn=lambda batch : jax_collate_fn(batch, CIFAR10_IMG_SHAPE))
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, num_workers=6, collate_fn=lambda batch : jax_collate_fn(batch, CIFAR10_IMG_SHAPE))


# Initialize the model parameters
model = SimpleClassifier(encoder=CIFAR10Encoder, num_classes=10)
params = model.init(jax.random.PRNGKey(0), jnp.ones([1, 32, 32, 3]))['params']

# Create an optimizer
optimizer = optax.adam(learning_rate=0.001)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = model.apply({'params': params}, batch[0])
        loss = cross_entropy_loss(logits, batch[1])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, logits

# Initialize the training state
state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)

# Training loop
num_epochs = 20
for epoch in tqdm(range(num_epochs)):
    for batch in train_loader:
        # Move batch to GPU
        batch = jax.device_put(batch)

        state, loss, logits = train_step(state, batch)

    # Evaluation
    accuracies = []
    for batch in test_loader:
        # Move batch to GPU
        batch = jax.device_put(batch)
        
        logits = model.apply({'params': state.params}, batch[0])
        accuracies.append(compute_accuracy(logits, batch[1]))
    
    test_accuracy = np.mean(accuracies)
    print(f"Epoch {epoch}, Loss: {loss}, Test Accuracy: {test_accuracy}")