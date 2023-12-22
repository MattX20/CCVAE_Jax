import jax
import flax
import optax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from tqdm import tqdm
import argparse
from pathlib import Path
import pickle
from src.models.encoder_decoder import CIFAR10Encoder, MNISTEncoder, CELEBAEncoder, MNISTDecoder, CIFAR10Decoder, CELEBADecoder, get_encoder_decoder
from src.models.simple_classifier import SimpleClassifier
from src.losses import cross_entropy_loss, binary_cross_entropy_loss
from src.utils import compute_accuracy
from src.data_loading.loaders import get_data_loaders
from src.models.config import get_config

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
import wandb


parser = argparse.ArgumentParser(description='Train CCVAE')
parser.add_argument('--dataset', type=str, default="MNIST", help='Dataset to use (MNIST, CIFAR10, CELEBA, CELEBA128)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--lr', type=float, default=1e-3, help='Initial Learning rate')
parser.add_argument('--p_test', type=float, default=0.1, help='Proportion of test set')
parser.add_argument('--p_val', type=float, default=0.1, help='Proportion of validation set')
parser.add_argument('--p_supervised', type=float, default=0.05, help='Proportion of supervised data')
parser.add_argument('--freq_lr_change', type=int, default=20, help='Frequency of learning rate change')
parser.add_argument('--warmup', type=int, default=10, help='Number of warmup epochs')
parser.add_argument('--beta', type=float, default=1.0, help='Beta parameter for the ELBO')

args = parser.parse_args()
# Set up random seed
seed = args.seed

# DATASET
dataset_name = args.dataset  #"CELEBA" #"MNIST" # use "CIFAR10"

config = get_config(dataset_name)
encoder_class, decoder_class = get_encoder_decoder(dataset_name)

distribution = config["distribution"]

# Data loading

img_shape, loader_dict, size_dict = get_data_loaders(dataset_name=dataset_name, 
                                          p_test=args.p_test,
                                          p_val=args.p_val,
                                          p_supervised=args.p_supervised,
                                          batch_size=args.batch_size,
                                          num_workers=6, 
                                          seed=seed)

# Initialize the model parameters
model = SimpleClassifier(encoder=encoder_class, num_classes=config['num_classes'])
params = model.init(jax.random.PRNGKey(0), jnp.ones((1,) + img_shape))['params']

print('Model set up!')

# Initialize the loss function (criterion)
if config['multiclass']:
    criterion = binary_cross_entropy_loss
else:
    criterion = cross_entropy_loss

# Set up optimizer
init_lr = 3e-5
final_lr = args.lr
lr_schedule = optax.piecewise_constant_schedule(
    init_value=init_lr,
    boundaries_and_scales={
        args.warmup * len(loader_dict["semi_supervised"]): final_lr/init_lr,
        args.freq_lr_change * len(loader_dict["semi_supervised"]): 0.5
    }
)
optimizer = optax.adam(lr_schedule)

@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        # Unpack the batch
        X, y = batch
        y = y.reshape(-1, 1)

        logits = model.apply({'params': params}, X)
        loss = criterion(logits, y)
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

wandb.init(project="class", name=f"class_{dataset_name}_p_supervised_{args.p_supervised}")
wandb.config.update(args)

# Training loop
num_epochs = args.num_epochs
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
    print(f"\nEpoch {epoch}, Loss: {running_loss / len(train_loader)}, Val Accuracy: {val_accuracy}")

# Test
accuracies = []
for batch in test_loader:
    # Move batch to GPU
    batch = jax.device_put(batch)
    
    logits = model.apply({'params': state.params}, batch[0])
    accuracies.append(compute_accuracy(logits, batch[1]))

test_accuracy = np.mean(accuracies)
print(f"Test Accuracy: {test_accuracy}")

print("Save weights...")
folder_path = Path("./model_weights")

if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{folder_path}' created.")

model_path = folder_path / f"classifier_{dataset_name}_p_supervised_{args.p_supervised}_seed_{seed}.pkl"

with open(model_path, "wb") as f:
    pickle.dump(state.params, f)

print("Training finished!")
