import jax.numpy as jnp

from utils import smooth_where

# M = -0.5 * z^2 + (if z < 0 then 0 else 1) 

class Example:

  theta0 = jnp.array([1.])
  sample_shape = (1,)

  @staticmethod
  def affine_reparam(theta): return theta, jnp.array([1.])

  @staticmethod
  def target(theta, z):
    return -0.5 * theta ** 2 + jnp.where(z >= 0, 1, 0) 

  @staticmethod
  def smooth_target(eta, theta, z):
    return -0.5 * theta ** 2 + smooth_where(eta, z, 1, 0)
  
  @staticmethod
  def target_surface(theta, z, if_idx, if_tf):
    return -0.5 * theta ** 2 + jnp.where(if_tf, 1, 0) 
  
  coefficients = jnp.array([ [1.] ])
  nz_indices = jnp.array([ 0 ])
  constants = jnp.array([ 0. ])
