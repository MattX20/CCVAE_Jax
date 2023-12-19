from typing import Type, Tuple

import jax.numpy as jnp
from jax import random

from flax import linen as nn

import numpyro
from numpyro.contrib.module import flax_module
import numpyro.distributions as dist


class M2FirstEncoder(nn.Module):
    """
        M2FirstEncoder takes as input an element of the dataset, and returns an (intermediate)
        latent variable h, and logits over the number of possible classes.
        This is different from the original M2, where the encoder and the classifier are
        separated networks.
    """
    encoder_class: Type[nn.Module]
    num_classes: int
    latent_dim: int

    def setup(self):
        self.encoder = self.encoder_class()
        self.fc_h = nn.Dense(features=self.latent_dim)
        self.fc_y = nn.Dense(features=self.num_classes)
    
    def __call__(self, x):
        x = self.encoder(x)
        h = self.fc_h(x)
        logy = self.fc_y(x)
        h = nn.relu(h)
        y = nn.softmax(logy)

        return h, y

class M2SecondEncoder(nn.Module):
    """
        M2SecondEncoder takes as input a latent variable h (outputed from M2FirstEncoder) and
        a one hot class representation, and returns a loc and a scale to sample the latent z.
    """
    latent_dim: int

    @nn.compact
    def __call__(self, h, y_one_hot):
        hy = jnp.concatenate([h, y_one_hot], axis=-1)
        loc = nn.Dense(features=self.latent_dim)(hy)
        scale_ = nn.Dense(features=self.latent_dim)(hy)
        scale_ = nn.activation.softplus(scale_)
        scale = jnp.clip(scale_, a_min=1e-3)
        return loc, scale

class M2Decoder(nn.Module):
    """
        Classical M2 decoder. Takes a latent variable and a one hot class representation, and 
        returns a reconstructed image.
    """
    decoder_class: Type[nn.Module]

    def setup(self):
        self.decoder = self.decoder_class()
        self.fc = nn.Dense(features=self.decoder.input_dim)
    
    def __call__(self, z, y_one_hot):
        zy = jnp.concatenate([z, y_one_hot], axis=-1)
        zy = self.fc(zy)
        reconstructed_x = self.decoder(zy)
        return reconstructed_x


class M2VAE:

    def __init__(self,
                 encoder_class: Type[nn.Module],
                 decoder_class: Type[nn.Module],
                 num_classes: int,
                 latent_dim: int,
                 img_shape: Tuple[int, int, int],
                 scale_factor: float,
                 distribution: str):
        
        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        self.scale_factor = scale_factor
        self.distribution = distribution

        self.internal_encoder1 = M2FirstEncoder(self.encoder_class, self.num_classes, self.latent_dim)
        self.internal_encoder1.init(
            random.PRNGKey(0), 
            x=jnp.ones((1,) + self.img_shape)
        )

        self.internal_encoder2 = M2SecondEncoder(self.latent_dim)
        self.internal_encoder2.init(
            random.PRNGKey(0), 
            h=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        )

        self.internal_decoder = M2Decoder(self.decoder_class)
        self.internal_decoder.init(
            random.PRNGKey(0), 
            z=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        )
    
    def model_supervised(self, xs, ys):
        batch_size = xs.shape[0]

        decoder = flax_module(
            "decoder", 
            M2Decoder(self.decoder_class), 
            z=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        ) 

        with numpyro.plate("data", batch_size):
            prior_loc = jnp.zeros((batch_size, self.latent_dim))
            prior_scale = jnp.ones((batch_size, self.latent_dim))
            zs = numpyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))
            alpha_prior = jnp.ones((batch_size, self.num_classes)) / self.num_classes
            ys = numpyro.sample("y", dist.Categorical(alpha_prior), obs=ys)
            y_one_hot = jnp.eye(self.num_classes)[ys]
            
            loc = decoder(zs, y_one_hot)
            numpyro.deterministic("loc", loc)

            if self.distribution == "bernoulli":
                numpyro.sample("x", dist.Bernoulli(loc).to_event(3), obs=xs)
            elif self.distribution == "laplace":
                numpyro.sample("x", dist.Laplace(loc).to_event(3), obs=xs)

    def guide_supervised(self, xs, ys):
        batch_size = xs.shape[0]

        encoder1 = flax_module(
            "encoder1",
            M2FirstEncoder(self.encoder_class, self.num_classes, self.latent_dim),
            x=jnp.ones((1,) + self.img_shape)
        )

        encoder2 = flax_module(
            "encoder2",
            M2SecondEncoder(self.latent_dim),
            h=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        )

        with numpyro.plate("data", batch_size):

            h, y_prob = encoder1(xs)
            y_one_hot = jnp.eye(self.num_classes)[ys]
            loc, scale = encoder2(h, y_one_hot)
            numpyro.sample("z", dist.Normal(loc, scale).to_event(1))
    
    def model_unsupervised(self, xs):
        batch_size = xs.shape[0]

        decoder = flax_module(
            "decoder", 
            M2Decoder(self.decoder_class), 
            z=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        ) 

        with numpyro.plate("data", batch_size):

            prior_loc = jnp.zeros((batch_size, self.latent_dim))
            prior_scale = jnp.ones((batch_size, self.latent_dim))
            zs = numpyro.sample("z", dist.Normal(prior_loc, prior_scale).to_event(1))

            alpha_prior = jnp.ones((batch_size, self.num_classes)) / self.num_classes
            ys = numpyro.sample("y", dist.Categorical(alpha_prior))
            y_one_hot = jnp.eye(self.num_classes)[ys]
            
            loc = decoder(zs, y_one_hot)
            numpyro.deterministic("loc", loc)

            if self.distribution == "bernoulli":
                numpyro.sample("x", dist.Bernoulli(loc).to_event(3), obs=xs)
            elif self.distribution == "laplace":
                numpyro.sample("x", dist.Laplace(loc).to_event(3), obs=xs)

    def guide_unsupervised(self, xs):
        batch_size = xs.shape[0]

        encoder1 = flax_module(
            "encoder1",
            M2FirstEncoder(self.encoder_class, self.num_classes, self.latent_dim),
            x=jnp.ones((1,) + self.img_shape)
        )

        encoder2 = flax_module(
            "encoder2",
            M2SecondEncoder(self.latent_dim),
            h=jnp.ones((1, self.latent_dim)),
            y_one_hot=jnp.ones((1, self.num_classes))
        )

        with numpyro.plate("data", batch_size):

            h, y_prob = encoder1(xs)
            ys = numpyro.sample("y", dist.Categorical(y_prob), infer={"enumerate": "parallel"})
            y_one_hot = jnp.eye(self.num_classes)[ys]
            loc, scale = encoder2(h, y_one_hot)
            numpyro.sample("z", dist.Normal(loc, scale).to_event(1))

    def model_classify(self, xs, ys):
        batch_size = xs.shape[0]

        encoder1 = flax_module(
            "encoder1",
            M2FirstEncoder(self.encoder_class, self.num_classes, self.latent_dim),
            x=jnp.ones((1,) + self.img_shape)
        )

        with numpyro.plate("data", batch_size):

            _, y_prob = encoder1(xs)
            
            with numpyro.handlers.scale(scale=self.scale_factor):
                numpyro.sample("y_aux", dist.Categorical(y_prob), obs=ys)

    def guide_classify(self, xs, ys):
        pass

    def classify(self, params_dict, xs):
        _, yprob = self.internal_encoder1.apply({"params": params_dict["encoder1$params"]}, xs)
        ypred = jnp.argmax(yprob, axis=1)

        return ypred