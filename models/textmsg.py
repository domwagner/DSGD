from typing import Text
import jax.numpy as jnp
import jax.scipy as jscipy

from utils import softplus, softplus_inv, forced_cond, smooth_where

## LYY18 Text Message

data = jnp.array([
       13., 24.,  8., 24.,  7., 35., 14., 11., 15., 11.,
       22., 22., 11., 57., 11., 19., 29.,  6., 19., 12.,
       22., 12., 18., 72., 32.,  9.,  7., 13., 19., 23.,
       27., 20.,  6., 17., 13., 10., 14.,  6., 16., 15.,
        7.,  2., 15., 15., 19., 70., 49.,  7., 53., 22.,
       21., 31., 19., 11., 18., 20., 12., 35., 17., 23.,
       17.,  4.,  2., 31., 30., 13., 27.,  0., 39., 37.,
        5., 14., 13., 22.])
data_mean = jnp.mean(data)
lognorm_std = jnp.sqrt(jnp.log(2.))
lognorm_mean = jnp.log(data_mean) - lognorm_std**2/2

inv_cdf = jscipy.stats.norm.ppf(jnp.arange(0, 1, 1 / (len(data) + 1))[1:])
# LYY18 example only takes every 2nd data point 
inv_cdf = inv_cdf[1::2]

class TextMsg:

  # theta0 = jnp.array([ 3., 3., 0.5, 1., 1., 0.5 ])
  s = softplus_inv(lognorm_std)
  theta0 = jnp.array([ lognorm_mean, lognorm_mean, 0., s, s, 1. ])
  sample_shape = (3,)

  @staticmethod
  def affine_reparam(theta):
    return theta[:3], softplus(theta[3:])

  @staticmethod
  def base_target(IF, theta, z):
    r1, r2 = jnp.exp(z[:2])
    mu, log_pmf = IF(jnp.arange(len(inv_cdf)), z[2] - inv_cdf, r1, r2)

    prior_mu = jnp.array([ lognorm_mean, lognorm_mean, 0.])
    prior_std = jnp.array([ lognorm_std, lognorm_std, 1.])
    log_prior = jnp.sum(jscipy.stats.norm.logpdf(z, prior_mu, prior_std))

    log_model = jnp.sum(jscipy.stats.poisson.logpmf(data[1::2], mu=mu))

    reparam_mu, reparam_std = TextMsg.affine_reparam(theta)
    log_q = jnp.sum(jscipy.stats.norm.logpdf(z, reparam_mu, reparam_std))

    return log_prior + log_model - log_q, log_pmf

  @staticmethod
  def target(theta, z):
    IF = lambda i, cond, t, f: (jnp.where(cond >= 0, t, f), 0.)
    return TextMsg.base_target(IF, theta, z)[0]

  @staticmethod
  def smooth_target(eta, theta, z):
    IF = lambda idx, cond, t, f: (smooth_where(eta, cond, t, f), 0.)
    return TextMsg.base_target(IF, theta, z)[0]

  @staticmethod
  def target_surface(theta, z, if_idx, if_tf):
    IF = lambda i, cond, t, f: (forced_cond(if_idx, if_tf, i, cond >= 0, t, f), 0.)
    return TextMsg.base_target(IF, theta, z)[0]
  
  coefficients = jnp.array([[0., 0., 1.]] * len(data))
  nz_indices = jnp.array([2] * len(data))
  constants = inv_cdf
