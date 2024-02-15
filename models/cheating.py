# Based on Bayesian Methods for Hackers example from Chapter 2
# Discontinuous model, here smoothing performs quite well again.

import jax.numpy as jnp
import jax.scipy as jscipy

from functools import partial

from utils import softplus, softplus_inv, forced_cond, smooth_where

class_size = 100
data_yes = 35

class Cheating:

  theta0 = jnp.array([ 0., 1. ])
  sample_shape = ( 1 + 3 * class_size, )

  @staticmethod
  def affine_reparam(theta):
    mu = jnp.concatenate([ theta[:1], jnp.zeros(3 * class_size) ])
    std = jnp.concatenate([ softplus(theta[1:]), jnp.ones(3 * class_size) ])
    return mu, std

  @staticmethod
  def base_target(IF, theta, z):
    cheat_thres = z[0]
    log_prior = 0.0 # uniform improper prior

    log_pmf = 0.0

    # did each student cheat?
    cheat, log_prob = IF(jnp.arange(class_size), z[1:class_size+1] - cheat_thres, 1., 0.)
    log_pmf += log_prob

    # 2 coin tosses per student
    toss1, log_prob = IF(jnp.arange(class_size, 2*class_size), z[class_size+1:class_size*2+1], 1., 0.)
    log_pmf += log_prob
    toss2, log_prob = IF(jnp.arange(2*class_size, 3*class_size), z[class_size*2+1:], 1., 0.)
    log_pmf += log_prob

    obs_prop = jnp.sum(toss1 * cheat + (1 - toss1) * toss2) / class_size

    # observe binomial(class_size, obs_prop, data_yes)
    # correct up to a constant (n choose k term)
    log_post = data_yes * jnp.log(obs_prop) + (class_size - data_yes) * jnp.log(1 - obs_prop)

    log_q = jscipy.stats.norm.logpdf(cheat_thres, theta[0], softplus(theta[1]))

    return log_prior + log_post - log_q, log_pmf

  @staticmethod
  def target(theta, z):
    IF = lambda i, cond, t, f: (jnp.where(cond >= 0, t, f), 0)
    return Cheating.base_target(IF, theta, z)[0]

  @staticmethod
  def smooth_target(eta, theta, z):
    IF = lambda idx, cond, t, f: (smooth_where(eta, cond, t, f), 0.)
    return Cheating.base_target(IF, theta, z)[0]

  @staticmethod
  def target_surface(theta, z, if_idx, if_tf):
    IF = lambda i, cond, t, f: (forced_cond(if_idx, if_tf, i, cond >= 0, t, f), 0)
    return Cheating.base_target(IF, theta, z)[0]
  
  constants = jnp.zeros((3 * class_size, ))
  nz_indices = jnp.arange(1, 3 * class_size + 1)
  coefficients = jnp.eye(N=3*class_size, M=3*class_size+1, k=1).at[:class_size, 0].set(-1.0)
