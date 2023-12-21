from pathlib import Path
import pickle
import argparse
import jax
from jax import jit, device_put
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import optax
from numpyro.infer import SVI, Trace_ELBO
import numpy as np
import matplotlib.pyplot as plt

from src.models.CCVAE import CCVAE
from src.models.encoder_decoder import get_encoder_decoder
from src.data_loading.loaders import get_data_loaders
from src.losses import CCVAE_ELBO
from src.models.config import get_config
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


scale_factor = config['scale_factor'] * size_dict["supervised"] # IMPORTANT, maybe run a grid search (0.3 on cifar)

# Set up model
ccvae = CCVAE(encoder_class, 
               decoder_class, 
               config['num_classes'],
               config['latent_dim'], 
               img_shape, 
               distribution=distribution,
               beta=args.beta,
               multiclass=config['multiclass'],
)
print("Model set up!")

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
print("Optimizer set up!")

# Set up SVI
svi_supervised = SVI(ccvae.model_supervised, 
            ccvae.guide_supervised, 
            optim=optimizer, 
            loss=CCVAE_ELBO()
)

svi_unsupervised = SVI(ccvae.model_unsupervised, 
            ccvae.guide_unsupervised,
            optim=optimizer, 
            loss=Trace_ELBO()
)

state = svi_supervised.init(
    random.PRNGKey(seed), 
    xs=jnp.ones((1,)+img_shape), 
    ys=jnp.ones((1), dtype=jnp.int32)
)
svi_unsupervised.init(
    random.PRNGKey(seed), 
    xs=jnp.ones((1,)+img_shape)
)
print("SVI set up!")


# Train functions
@jit
def train_step_supervised(state, batch):
    x, y = batch
    state, loss_supervised = svi_supervised.update(state, xs=x, ys=y)
    
    return state, loss_supervised

@jit
def train_step_unsupervised(state, batch):
    x = batch
    state, loss_unsupervised = svi_unsupervised.update(state, xs=x)
    
    return state, loss_unsupervised

# Training
semi_supervised_loader = loader_dict["semi_supervised"]
validation_loader = loader_dict["validation"]
test_loader = loader_dict["test"]

# Set up wandb

wandb.init(project="ccvae", name=f"ccvae_{dataset_name}_p_supervised_{args.p_supervised}")
wandb.config.update(args)

print("Start training.")
loss_rec_supervised = []
loss_rec_unsupervised = []
validation_accuracy_rec = []

num_epochs = args.num_epochs
progress_bar = tqdm(range(1, num_epochs + 1))
for epoch in range(1, num_epochs + 1):
    running_loss = 0.0

    loss_rec_step_supervised = []
    loss_rec_step_unsupervised = []

    # Trainning
    for is_supervised, batch in semi_supervised_loader: 
        batch = device_put(batch)

        if is_supervised:
            state, loss_supervised = train_step_supervised(state, batch)
            loss_rec_step_supervised.append(loss_supervised)
        else:
            state, loss_unsupervised = train_step_unsupervised(state, batch)
            loss_rec_step_unsupervised.append(loss_unsupervised)
    
    loss_epoch_supervised = np.mean(loss_rec_step_supervised)
    loss_epoch_unsupervised = np.mean(loss_rec_step_unsupervised)

    loss_rec_supervised.append(loss_epoch_supervised)
    loss_rec_unsupervised.append(loss_epoch_unsupervised)
    
    validation_accuracy = 0.0

    for batch in validation_loader:
        batch = device_put(batch)
        x, y = batch
        ypred = ccvae.classify(state[0][1][0], x)
        validation_accuracy += jnp.mean(y == ypred)
    
    validation_accuracy /= len(validation_loader)
    validation_accuracy_rec.append(validation_accuracy)

    logs = {"loss_sup": loss_epoch_supervised,
                "loss_unsup": loss_epoch_unsupervised,
                "val_acc": validation_accuracy}
    wandb.log(logs)
    progress_bar.set_postfix(logs)
    progress_bar.update()

progress_bar.close()

print("Training finished!")

# Test
test_accuracy = 0.0
for batch in test_loader:
    batch = jax.device_put(batch)
    
    x, y = batch
    ypred = ccvae.classify(state[0][1][0], x)
    test_accuracy += jnp.mean(y == ypred)

test_accuracy = test_accuracy / len(test_loader)
print(f"Test Accuracy: {test_accuracy}")

# print("Plot figures...")
# plt.figure()
# plt.plot(loss_rec_supervised, color="red", label="supervised")
# plt.plot(loss_rec_unsupervised, color="green", label="unsupervised")
# plt.legend(loc="best")
# plt.savefig("result_loss.png")

# plt.figure()
# plt.plot(validation_accuracy_rec, label="accuracy")
# plt.legend(loc="best")
# plt.savefig("result_acc.png")

print("Save weights...")
folder_path = Path("./model_weights")

if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{folder_path}' created.")

save_file = "ccvae_" + dataset_name + "_p_supervised_" + str(args.p_supervised) + ".pkl"
file_path = folder_path / save_file

with open(file_path, 'wb') as file:
    pickle.dump(state[0][1][0], file)

print(f"Weights saved to {file_path}.")
print("Done.")