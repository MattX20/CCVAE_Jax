import jax
from jax import jit, device_put
import jax.numpy as jnp
from jax import random
from tqdm import tqdm
import optax
from numpyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO
import numpy as np
import matplotlib.pyplot as plt

from src.models.M2VAE import M2VAE
from src.models.encoder_decoder import MNISTEncoder, MNISTDecoder
from src.data_loading.loaders import get_data_loaders

# Set up random seed
seed = 42

# Data loading

img_shape, loader_dict, size_dict = get_data_loaders(dataset_name="MNIST", 
                                          p_test=0.2, 
                                          p_val=0.2, 
                                          p_supervised=0.1, 
                                          batch_size=2, 
                                          num_workers=0, 
                                          seed=seed)

scale_factor = 50 / size_dict["supervised"]

# Set up model
m2_vae = M2VAE(MNISTEncoder, MNISTDecoder, 10, 30, img_shape, scale_factor=scale_factor)
print("Model set up!")

# Set up optimizer
optimizer = optax.adam(1e-3)
print("Optimizer set up!")

# Set up SVI
svi_supervised = SVI(m2_vae.model_supervised, 
            m2_vae.guide_supervised, 
            optim=optimizer, 
            loss=Trace_ELBO()
)

svi_unsupervised = SVI(m2_vae.model_unsupervised, 
            m2_vae.guide_unsupervised, #config_enumerate(m2_vae.guide_unsupervised), 
            optim=optimizer, 
            loss=Trace_ELBO() # TraceEnum_ELBO(max_plate_nesting=1) Would be better, ...
)

svi_classify = SVI(m2_vae.model_classify,
                   m2_vae.guide_classify,
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
svi_classify.init(
    random.PRNGKey(seed), 
    xs=jnp.ones((1,)+img_shape), 
    ys=jnp.ones((1), dtype=jnp.int32)
)
print("SVI set up!")


# Train functions
@jit
def train_step_supervised(state, batch):
    x, y = batch
    state, loss_supervised = svi_supervised.update(state, xs=x, ys=y)
    state, loss_classify = svi_classify.update(state, xs=x, ys=y)
    
    return state, loss_supervised, loss_classify

@jit
def train_step_unsupervised(state, batch):
    x = batch
    state, loss_unsupervised = svi_unsupervised.update(state, xs=x)
    
    return state, loss_unsupervised

# Training
semi_supervised_loader = loader_dict["semi_supervised"]

print("Start training.")
loss_rec_supervised = []
loss_rec_unsupervised = []
loss_rec_classify = []

num_epochs = 20
for epoch in tqdm(range(1, num_epochs + 1)):
    running_loss = 0.0

    loss_rec_step_supervised = []
    loss_rec_step_unsupervised = []
    loss_rec_step_classify = []

    # Trainning
    for is_supervised, batch in semi_supervised_loader: 
        batch = device_put(batch)

        if is_supervised:
            state, loss_supervised, loss_classify = train_step_supervised(state, batch)
        else:
            state, loss_unsupervised = train_step_unsupervised(state, batch)
        
        loss_rec_step_supervised.append(loss_supervised)
        loss_rec_step_classify.append(loss_classify)
        loss_rec_step_unsupervised.append(loss_unsupervised)
    
    loss_rec_supervised.append(np.mean(loss_rec_step_supervised))
    loss_rec_classify.append(np.mean(loss_rec_step_classify))
    loss_rec_unsupervised.append(np.mean(loss_rec_step_unsupervised))

plt.plot(loss_rec_supervised, color="red", label="supervised")
plt.plot(loss_rec_classify, color="blue", label="classify")
plt.plot(loss_rec_unsupervised, color="green", label="unsupervised")
plt.legend(loc="best")
plt.savefig("result.png")