import jax.numpy as jnp
import jax.scipy as jscipy
from jax import random, vmap

from functools import partial

from utils import softplus, softplus_inv, smooth_where

latent_size = 8 + 8 + 2 + 4 + 2 + 1

def eval_nn(IF, idx, x, z):
  w1 = z[:8].reshape((4, 2))
  w2 = z[8:16].reshape((2, 4))
  w3 = z[16:18].reshape((1, 2))
  b1 = z[18:22]
  b2 = z[22:24]
  b3 = z[24]

  x0 = jnp.array(x)
  x1, logprob0 = IF(3 * idx + 0, w1 @ x0 + b1, 1, 0)
  x2, logprob1 = IF(3 * idx + 1, w2 @ x1 + b2, 1, 0)
  x3, logprob2 = IF(3 * idx + 2, w3 @ x2 + b3, 1, 0)

  return x3, logprob0 + logprob1 + logprob2

class XORNet:

  theta0 = jnp.concatenate([
    random.normal(random.PRNGKey(0), (latent_size,)),
    softplus_inv(jnp.ones(latent_size)),
  ])
  sample_shape = (latent_size,)

  @staticmethod
  def affine_reparam(theta):
    return theta[:latent_size], softplus(theta[latent_size:])

  @staticmethod
  def base_target(IF, theta, z):
    # prior is univariate Gaussian for each weight/bias
    log_prior = jnp.sum(jscipy.stats.norm.logpdf(z))

    def eval_likelihood(idx, x, y):
      pred, logprob = eval_nn(IF, idx, x, z)
      return jscipy.stats.norm.logpdf(y, pred, 0.01), logprob

    idxs = jnp.arange(4)
    X = jnp.array([ [0., 0.,], [0., 1.,], [1., 0.,], [1., 1.,], ])
    Y = jnp.array([ 0., 1., 1., 0. ])
    log_model, log_prob = vmap(eval_likelihood)(idxs, X, Y)
    log_model = jnp.sum(log_model)
    log_prob = jnp.sum(log_prob)

    mu, std = XORNet.affine_reparam(theta)
    log_q = jnp.sum(jscipy.stats.norm.logpdf(z, mu, std))
    return log_prior + log_model - log_q, log_prob

  @staticmethod
  def target(theta, z):
    IF = lambda idx, cond, t, f: (jnp.where(cond >= 0, t, f), 0.)
    return XORNet.base_target(IF, theta, z)[0]

  @staticmethod
  def smooth_target(eta, theta, z):
    IF = lambda idx, cond, t, f: (smooth_where(eta, cond, t, f), 0.)
    return XORNet.base_target(IF, theta, z)[0]
