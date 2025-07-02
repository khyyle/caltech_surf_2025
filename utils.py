import functools

import jax
import flax
from jax import numpy as jnp
from flax import linen as nn
from typing import Any, Callable
from flax.serialization import from_bytes
from flax.serialization import to_bytes

def shard(xs):
    """Split data into shards for multiple devices along the first dimension."""
    return jax.tree_util.tree_map(lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)

def flattened_traversal(fn):
    def mask(data):
        flat = flax.traverse_util.flatten_dict(data)
        return flax.traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()})
    return mask

class MLP(nn.Module): 
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu 
    out_channel: int = 1 
    do_skip: bool = True
  
    @nn.compact 
    def __call__(self, x):
        """A simple Multi-Layer Preceptron (MLP) network

        Parameters
        ----------
        x: jnp.ndarray(float32), 
            [batch_size * n_samples, feature], points.
        net_depth: int, 
            the depth of the first part of MLP.
        net_width: int, 
            the width of the first part of MLP.
        activation: function, 
            the activation function used in the MLP.
        out_channel: 
            int, the number of alpha_channels.
        do_skip: boolean, 
            whether or not to use a skip connection

        Returns
        -------
        out: jnp.ndarray(float32), 
            [batch_size * n_samples, out_channel].
        """
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.he_uniform())

        if self.do_skip:
            skip_layer = self.net_depth // 2 

        inputs = x
        for i in range(self.net_depth): 
            x = dense_layer(self.net_width)(x) 
            x = self.activation(x) 
            if self.do_skip:
                if i % skip_layer == 0 and i > 0:
                    x = jnp.concatenate([x, inputs], axis=-1) 
        out = dense_layer(self.out_channel)(x) 

        return out

def posenc(x, deg):
    """
    Concatenate `x` with a positional encoding of `x` with degree `deg`.
    Instead of computing [sin(x), cos(x)], we use the trig identity
    cos(x) = sin(x + pi/2) and do one vectorized call to sin([x, x+pi/2]).

    Parameters
    ----------
    x: jnp.ndarray, 
        variables to be encoded. Note that x should be in [-pi, pi].
    deg: int, 
        the degree of the encoding.

    Returns
    -------
    encoded: jnp.ndarray, 
        encoded variables.
    """
    if deg == 0:
        return x
    scales = jnp.array([2**i for i in range(deg)]) 
    xb = jnp.reshape((x[..., None, :] * scales[:, None]),
                     list(x.shape[:-1]) + [-1]) 


    four_feat = safe_sin(jnp.concatenate([xb, xb + 0.5 * jnp.pi], axis=-1)) 
    return jnp.concatenate([x] + [four_feat], axis=-1) 

# wrapper around the MLP module to predict and forward pass. 
class NeuralImage(nn.Module):
    """
    Full function to predict emission at a time step.
    
    Parameters
    ----------
    posenc_deg: int, default=3
    net_depth: int, default=4
    net_width: int, default=128
    activation: Callable[..., Any], default=nn.relu
    out_channel: int default=1
    do_skip: bool, default=True
    """
    posenc_deg: int = 3
    net_depth: int = 4
    net_width: int = 128
    activation: Callable[..., Any] = nn.relu
    out_channel: int = 1
    do_skip: bool = True
    for_bh: bool = False
    
    @nn.compact
    def __call__(self, coords):
        image_MLP = MLP(self.net_depth, self.net_width, self.activation, self.out_channel, self.do_skip)
        def predict_image(coords):
            net_output = image_MLP(posenc(coords, self.posenc_deg))
            if self.for_bh:
                image = nn.sigmoid(net_output[..., 0] - 10.)
            else:
                image = nn.sigmoid(net_output[..., 0])

            return image
        return predict_image(coords)
    
safe_sin = lambda x: jnp.sin(x % (100 * jnp.pi))

def loss_fn_bh(params, predictor_fn, target, A, sigma, coords):
    '''
    Args:
        params: pytree (nested dict) of all of the model's weights and biases. 
        predictor_fn: the model's apply function
        target: measured intensities
        A: measurement matrix
        sigma: Thermal noise per visibility
        coords: array of shape (N_pixels, 2) with all (x,y) grid coords
    
    Returns:
        image: predicted intensities -> image
        loss: 
    '''
    image = predictor_fn({'params': params}, coords) 
    vis = jnp.matmul(A, image.ravel())
    loss = jnp.mean((jnp.abs(vis - target)/sigma)**2)
    return loss, [image]

def loss_fn_identity(params, predictor_fn, target, coords):
    image = predictor_fn({'params': params}, coords)
    loss = jnp.mean((image - target)**2)
    return loss, image

@jax.jit
def train_step_batched(state, target, A, sigma, coords):
    (loss, [image]), grads = jax.value_and_grad(loss_fn_bh, argnums=(0), has_aux=True)(state.params, state.apply_fn, target, A, sigma, coords)
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return loss, state, image

@jax.jit #just in time compilation for speed
def train_step(state, target, coords):
    """
    Args:
        state: current train state
        target: known image intensity values
        coords: positional coords across image
    """
    (loss, image_pred), grads = jax.value_and_grad(loss_fn_identity, argnums=0, has_aux=True)(state.params, state.apply_fn, target, coords)
    state = state.apply_gradients(grads=grads) #ignoring paralleization for now
    return loss, state, image_pred

def loss_fn_hetero(params, predictor_fn, target, coords, sigma):
    pred = predictor_fn({'params': params}, coords)
    return jnp.mean(((pred - target) / sigma) ** 2), pred

@jax.jit
def train_step_hetero(state, target, coords, sigma):
    (loss, image_pred), grads = jax.value_and_grad(loss_fn_hetero, has_aux=True)(state.params, state.apply_fn, target, coords, sigma)
    return loss, state.apply_gradients(grads=grads), image_pred



