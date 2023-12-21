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
        scale_ = self.fc_scale(x)
        scale_ = nn.activation.softplus(scale_)
        scale = jnp.clip(scale_, a_min=1e-3)
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
        scale_ = nn.activation.softplus(scale_)
        scale = jnp.clip(scale_, a_min=1e-3)
        return loc, scale

class Classifier(nn.Module):
    num_classes: int
    multiclass: bool

    def setup(self):
        self.weight = self.param('weight', nn.initializers.ones, (self.num_classes,))
        self.bias = self.param('bias', nn.initializers.zeros, (self.num_classes,))
        self.activation = nn.sigmoid if self.multiclass else nn.softmax

    def __call__(self, z_class):
        y = z_class * self.weight + self.bias
        y_prob = self.activation(y)
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
                 distribution: str,
                 beta: float = 1.,
                 multiclass: bool = False):
        
        assert latent_dim > num_classes, "The dimention of the latent must be greater than the number of classes"

        self.encoder_class = encoder_class
        self.decoder_class = decoder_class
        self.num_classes = num_classes
        
        self.latent_dim = latent_dim
        self.latent_class = num_classes
        self.latent_style = latent_dim - num_classes

        self.img_shape = img_shape
        self.distribution = distribution

        self.multiclass = multiclass
        self.beta = beta
        self.internal_encoder = CCVAEEncoder(self.encoder_class, self.latent_dim)
        self.internal_encoder.init(
            random.PRNGKey(0), 
            x=jnp.ones((1,) + self.img_shape),
        )

        self.internal_classifier = Classifier(self.num_classes, self.multiclass)
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
            Classifier(self.num_classes, self.multiclass),
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
                numpyro.sample("x", dist.Laplace(loc, scale=self.beta).to_event(3), obs=xs)


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
            Classifier(self.num_classes, self.multiclass),
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
                numpyro.sample("x", dist.Laplace(loc, scale = self.beta).to_event(3), obs=xs)
    
    def guide_unsupervised(self, xs):
        batch_size = xs.shape[0]

        encoder = flax_module(
            "encoder", 
            CCVAEEncoder(self.encoder_class, self.latent_dim), 
            x=jnp.ones((1,) + self.img_shape)
        )

        classifier = flax_module(
            "classifier",
            Classifier(self.num_classes, self.multiclass),
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