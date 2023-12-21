from numpyro.handlers import seed
from numpyro.infer import ELBO
import jax.numpy as jnp
from jax import random
from jax import vmap
from numpyro.infer.util import get_importance_trace


def binary_cross_entropy_loss(reconstructed_x, x):
    """
        binary_cross_entropy_loss between x and reconstructed_x.
        x and reconstructed_x are expected to be of shape (batch_size,) + img_shape.
    """
    epsilon = 1e-10  # To prevent log(0)
    reconstructed_x = jnp.clip(reconstructed_x, epsilon, 1 - epsilon)
    bce_loss = -(x * jnp.log(reconstructed_x) + (1 - x) * jnp.log(1 - reconstructed_x))
    return jnp.mean(bce_loss)

class CCVAE_ELBO(ELBO):
    def loss(self, rng_key, param_map, model, guide, *args, **kwargs):
        # Careful, slow if num particules too big (due to lqyx computation)
        def single_particle_elbo(rng_key):
            params = param_map.copy()
            model_seed, guide_seed = random.split(rng_key)
            seeded_model = seed(model, model_seed)
            seeded_guide = seed(guide, guide_seed)

            model_trace, guide_trace = get_importance_trace(
                seeded_model, seeded_guide, args, kwargs, params
            )

            lqy_x = model_trace["lqy_x"]["value"]

            log_vi = model_trace['x']['log_prob']
            log_vi += model_trace['z_class']['log_prob']
            log_vi += model_trace['z_style']['log_prob']
            
            log_vi -= guide_trace['z_class']['log_prob']
            log_vi -= guide_trace['z_style']['log_prob']
            log_vi -= guide_trace['y']['log_prob']
            
            lpy = model_trace['y']['log_prob']

            w = jnp.exp(guide_trace['y']['log_prob'] - lqy_x)

            elbo = (w * log_vi + lpy).sum()
            return elbo, lqy_x

        rng_keys = random.split(rng_key, self.num_particles)
        elbo_particles, lqy_x_particles = vmap(single_particle_elbo)(rng_keys)
        loss = -jnp.mean(elbo_particles) - jnp.sum(lqy_x_particles) / self.num_particles

        return loss
    

from typing import Type, Tuple

import jax.numpy as jnp
from jax import random
from jax.scipy.special import logsumexp
from jax.lax import stop_gradient
from flax import linen as nn

import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist


class CCVAEEncoder(nn.Module):
    """
        CCVAEEncoder takes as input an element of the dataset, and returns a loc and a scale
        to sample the latent z.
    """
    encoder_class: Type[nn.Module]
    latent_dim: int

    def setup(self):
        self.encoder = self.encoder_class()
        self.fc_loc = nn.Dense(features=self.latent_dim)
        self.fc_scale = nn.Dense(features=self.latent_dim)
    
    def __call__(self, x):
        x = self.encoder(x)
        
        loc = self.fc_loc(x)
        log_scale = self.fc_scale(x)
        scale = jnp.exp(log_scale)

        return loc, scale

class CCVAEDecoder(nn.Module):
    """
        CCVAEDecoder takes a latent variable and return a reconstructed image.
    """
    decoder_class: Type[nn.Module]

    def setup(self):
        self.decoder = self.decoder_class()
        self.fc = nn.Dense(features=self.decoder.input_dim)
    
    def __call__(self, z):
        z = self.fc(z)
        reconstructed_x = self.decoder(z)
        return reconstructed_x

class CondPrior(nn.Module):
    num_classes: int

    def setup(self):
        self.diag_loc_true = self.param('diag_loc_true', nn.initializers.ones, (self.num_classes,)) * 2
        self.diag_loc_false = self.param('diag_loc_false', nn.initializers.ones, (self.num_classes,)) * -2
        self.diag_scale_true = self.param('diag_scale_true', nn.initializers.ones, (self.num_classes,))
        self.diag_scale_false = self.param('diag_scale_false', nn.initializers.ones, (self.num_classes,))

    def __call__(self, y):
        loc = y * self.diag_loc_true + (1 - y) * self.diag_loc_false
        scale_ = y * self.diag_scale_true + (1 - y) * self.diag_scale_false
        scale = nn.activation.softplus(scale_)
        return loc, scale

class Classifier(nn.Module):
    num_classes: int

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.num_classes,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.num_classes,))

    def __call__(self, z_class):
        y = z_class * self.weight + self.bias
        y_prob = nn.sigmoid(y)
        return y_prob


class CCVAE:
    """
        Implementation of the CCVAE model.
    """

    def __init__(self,
                 encoder_class: Type[nn.Module],
                 decoder_class: Type[nn.Module],
                 num_classes: int,
                 latent_dim: int,
                 img_shape: Tuple[int, int, int],
                 scale_factor: float,
                 distribution: str,
                 multiclass: bool = False):
        
        assert latent_dim > num_classes, "The dimention of the latent must be greater than the number of classes"

        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.num_classes = num_classes
        
        self.latent_dim = latent_dim
        self.latent_class = num_classes
        self.latent_style = latent_dim - num_classes

        self.img_shape = img_shape
        self.scale_factor = scale_factor
        self.distribution = distribution

        self.multiclass = multiclass

        self.internal_encoder = CCVAEEncoder(self.encoder_class, self.latent_dim)
        self.internal_encoder.init(
            random.PRNGKey(0), 
            x=jnp.ones((1,) + self.img_shape),
        )

        self.internal_classifier = Classifier(self.num_classes)
        self.internal_classifier.init(
            random.PRNGKey(0), 
            z_class=jnp.ones((1, self.num_classes))
        )

        self.internal_cond_prior_class = CondPrior(self.latent_class)
        self.internal_cond_prior_class.init(
            random.PRNGKey(0), 
            y=jnp.ones((1, self.latent_class))
        )

        self.internal_decoder = CCVAEDecoder(self.decoder_class)
        self.internal_decoder.init(
            random.PRNGKey(0), 
            z=jnp.ones((1, self.latent_dim))
        )
    
    def model_supervised(self, xs, ys, k=100):
        batch_size = xs.shape[0]

        encoder = flax_module(
            "encoder", 
            CCVAEEncoder(self.encoder_class, self.latent_dim), 
            x=jnp.ones((1,) + self.img_shape)
        )

        classifier = flax_module(
            "classifier",
            Classifier(self.num_classes),
            z_class=jnp.ones((1, self.num_classes))
        )

        cond_prior_class = flax_module(
            "cond_prior_class", 
            CondPrior(self.latent_class), 
            y=jnp.ones((1, self.latent_class))
        ) 

        decoder = flax_module(
            "decoder", 
            CCVAEDecoder(self.decoder_class), 
            z=jnp.ones((1, self.latent_dim))
        )

        with numpyro.plate("data", batch_size):
            if self.multiclass:
                alpha_prior = jnp.ones((batch_size, self.num_classes)) / 2
                y_one_hot = numpyro.sample("y", dist.Bernoulli(alpha_prior).to_event(1), obs=ys)
            else :
                alpha_prior = jnp.ones((batch_size, self.num_classes)) / self.num_classes
                ys_p = numpyro.sample("y", dist.Categorical(alpha_prior), obs=ys)
                y_one_hot = jnp.eye(self.num_classes)[ys_p]

            z_class_prior_loc, z_class_prior_scale = cond_prior_class(y_one_hot)
            z_class = numpyro.sample("z_class", 
                                     dist.Normal(z_class_prior_loc, z_class_prior_scale).to_event(1)
            )

            z_style_prior_loc = jnp.zeros((batch_size, self.latent_style))
            z_style_prior_scale = jnp.ones((batch_size, self.latent_style))
            z_style = numpyro.sample("z_style",
                                     dist.Normal(z_style_prior_loc, z_style_prior_scale).to_event(1)
            )

            z = jnp.concatenate([z_class, z_style], axis=-1)

            loc = decoder(z)
            numpyro.deterministic("loc", loc)

            if self.distribution == "bernoulli":
                numpyro.sample("x", dist.Bernoulli(loc).to_event(3), obs=xs)
            elif self.distribution == "laplace":
                numpyro.sample("x", dist.Laplace(loc).to_event(3), obs=xs)


        with numpyro.plate("sampling", k * batch_size):
            loc_aux, scale_aux = encoder(xs)
            loc_aux = jnp.broadcast_to(loc_aux, (k, batch_size, self.latent_dim)).reshape(k * batch_size, -1)
            scale_aux = jnp.broadcast_to(scale_aux, (k, batch_size, self.latent_dim)).reshape(k * batch_size, -1)

            zs = numpyro.sample("zs", dist.Normal(loc_aux, scale_aux).to_event(1))

            z_class_aux, _ = jnp.split(zs, [self.num_classes], axis=-1)
            y_prob_aux = classifier(z_class_aux)
            
            if self.multiclass:
                d = dist.Bernoulli(y_prob_aux).to_event(1)
                ys_aux = jnp.broadcast_to(ys, (k, batch_size, self.num_classes)).reshape(k * batch_size, -1)
            else :
                d = dist.Categorical(y_prob_aux)
                ys_aux = jnp.broadcast_to(ys, (k, batch_size)).flatten()
            
            lqy_z = d.log_prob(ys_aux).reshape(k, batch_size)
            lqy_x = logsumexp(lqy_z, axis=0) - jnp.log(k)

            numpyro.deterministic("lqy_x", lqy_x)
    
    def guide_supervised(self, xs, ys):
        batch_size = xs.shape[0]

        encoder = flax_module(
            "encoder", 
            CCVAEEncoder(self.encoder_class, self.latent_dim), 
            x=jnp.ones((1,) + self.img_shape)
        )

        classifier = flax_module(
            "classifier",
            Classifier(self.num_classes),
            z_class=jnp.ones((1, self.num_classes))
        )

        with numpyro.plate("data", batch_size):
            loc, scale = encoder(xs)
            z_class_loc, z_style_loc = jnp.split(loc, [self.latent_class], axis=-1)
            z_class_scale, z_style_scale = jnp.split(scale, [self.latent_class], axis=-1)

            z_class = numpyro.sample("z_class", 
                                     dist.Normal(z_class_loc, z_class_scale).to_event(1)
            )

            numpyro.sample("z_style",
                           dist.Normal(z_style_loc, z_style_scale).to_event(1)
            )
            z_class_no_grad = stop_gradient(z_class)
            y_prob = classifier(z_class_no_grad)

            if self.multiclass:
                numpyro.sample("y", dist.Bernoulli(y_prob).to_event(1), obs=ys)
            else :
                numpyro.sample("y", dist.Categorical(y_prob), obs=ys)
    
    def model_unsupervised(self, xs):
        batch_size = xs.shape[0]

        cond_prior_class = flax_module(
            "cond_prior_class", 
            CondPrior(self.latent_class), 
            y=jnp.ones((1, self.latent_class))
        ) 

        decoder = flax_module(
            "decoder", 
            CCVAEDecoder(self.decoder_class), 
            z=jnp.ones((1, self.latent_dim))
        )

        with numpyro.plate("data", batch_size):
            if self.multiclass:
                alpha_prior = jnp.ones((batch_size, self.num_classes)) / 2
                y_one_hot = numpyro.sample("y", dist.Bernoulli(alpha_prior).to_event(1))
            else :
                alpha_prior = jnp.ones((batch_size, self.num_classes)) / self.num_classes
                ys_p = numpyro.sample("y", dist.Categorical(alpha_prior))
                y_one_hot = jnp.eye(self.num_classes)[ys_p]

            z_class_prior_loc, z_class_prior_scale = cond_prior_class(y_one_hot)
            z_class = numpyro.sample("z_class", 
                                     dist.Normal(z_class_prior_loc, z_class_prior_scale).to_event(1)
            )

            z_style_prior_loc = jnp.zeros((batch_size, self.latent_style))
            z_style_prior_scale = jnp.ones((batch_size, self.latent_style))
            z_style = numpyro.sample("z_style",
                                     dist.Normal(z_style_prior_loc, z_style_prior_scale).to_event(1)
            )

            z = jnp.concatenate([z_class, z_style], axis=-1)

            loc = decoder(z)
            numpyro.deterministic("loc", loc)

            if self.distribution == "bernoulli":
                numpyro.sample("x", dist.Bernoulli(loc).to_event(3), obs=xs)
            elif self.distribution == "laplace":
                numpyro.sample("x", dist.Laplace(loc).to_event(3), obs=xs)
    
    def guide_unsupervised(self, xs):
        batch_size = xs.shape[0]

        encoder = flax_module(
            "encoder", 
            CCVAEEncoder(self.encoder_class, self.latent_dim), 
            x=jnp.ones((1,) + self.img_shape)
        )

        classifier = flax_module(
            "classifier",
            Classifier(self.num_classes),
            z_class=jnp.ones((1, self.num_classes))
        )

        with numpyro.plate("data", batch_size):
            loc, scale = encoder(xs)
            z_class_loc, z_style_loc = jnp.split(loc, [self.latent_class], axis=-1)
            z_class_scale, z_style_scale = jnp.split(scale, [self.latent_class], axis=-1)

            z_class = numpyro.sample("z_class", 
                                     dist.Normal(z_class_loc, z_class_scale).to_event(1)
            )

            numpyro.sample("z_style",
                           dist.Normal(z_style_loc, z_style_scale).to_event(1)
            )
            z_class_no_grad = stop_gradient(z_class)
            y_prob = classifier(z_class_no_grad)

            if self.multiclass:
                numpyro.sample("y", dist.Bernoulli(y_prob).to_event(1))
            else :
                numpyro.sample("y", dist.Categorical(y_prob))
    
    def classify(self, params_dict, xs):
        loc, _ = self.internal_encoder.apply({"params": params_dict["encoder$params"]}, xs)
        loc_class, _ =  jnp.split(loc, [self.latent_class], axis=-1)
        y_prob = self.internal_classifier.apply({"params": params_dict["classifier$params"]}, loc_class)

        if self.multiclass:
            y_pred = jnp.round(y_prob)
        else:
            y_pred = jnp.argmax(y_prob, axis=1)
        
        return y_pred
    

from pathlib import Path
import pickle

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
from src.models.encoder_decoder import MNISTEncoder, MNISTDecoder, CIFAR10Encoder, CIFAR10Decoder
from src.data_loading.loaders import get_data_loaders
from src.losses import CCVAE_ELBO


# Set up random seed
seed = 42

# DATASET
dataset_name = "MNIST" # use "CIFAR10"

encoder_class = MNISTEncoder if dataset_name=="MNIST" else CIFAR10Encoder
decoder_class = MNISTDecoder if dataset_name=="MNIST" else CIFAR10Decoder
distribution = "bernoulli" if dataset_name=="MNIST" else "laplace"

# Data loading

img_shape, loader_dict, size_dict = get_data_loaders(dataset_name=dataset_name, 
                                          p_test=0.2, 
                                          p_val=0.2, 
                                          p_supervised=0.05, 
                                          batch_size=64, 
                                          num_workers=6, 
                                          seed=seed)

scale_factor = 0.1 * size_dict["supervised"] # IMPORTANT, maybe run a grid search (0.3 on cifar)

# Set up model
ccvae = CCVAE(encoder_class, 
               decoder_class, 
               10, 
               50, 
               img_shape, 
               scale_factor=scale_factor, 
               distribution=distribution,
               multiclass=False
)
print("Model set up!")

# Set up optimizer
lr_schedule = optax.piecewise_constant_schedule(
    init_value=1e-3,
    boundaries_and_scales={
        20 * len(loader_dict["semi_supervised"]): 0.1
    }
)
optimizer = optax.adam(lr_schedule)
print("Optimizer set up!")

# Set up SVI
svi_supervised = SVI(ccvae.model_supervised, 
            ccvae.guide_supervised, 
            optim=optimizer, 
            loss=Trace_ELBO()
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

print("Start training.")
loss_rec_supervised = []
loss_rec_unsupervised = []
validation_accuracy_rec = []

num_epochs = 30
for epoch in tqdm(range(1, num_epochs + 1)):
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
    
    print("\nEpoch:", 
          epoch, 
          "loss sup:", 
          loss_epoch_supervised, 
          "loss unsup:", 
          loss_epoch_unsupervised, 
          "val acc:", 
          validation_accuracy
    )

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

print("Plot figures...")
plt.figure()
plt.plot(loss_rec_supervised, color="red", label="supervised")
plt.plot(loss_rec_unsupervised, color="green", label="unsupervised")
plt.legend(loc="best")
plt.savefig("result_loss.png")

plt.figure()
plt.plot(validation_accuracy_rec, label="accuracy")
plt.legend(loc="best")
plt.savefig("result_acc.png")

print("Save weights...")
folder_path = Path("./model_weights")

if not folder_path.exists():
    folder_path.mkdir(parents=True, exist_ok=True)
    print(f"Folder '{folder_path}' created.")

save_file = "ccvae" + dataset_name + ".pkl"
file_path = folder_path / save_file

with open(file_path, 'wb') as file:
    pickle.dump(state[0][1][0], file)

print(f"Data saved to {file_path}.")
print("Done.")