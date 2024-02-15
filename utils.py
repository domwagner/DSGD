import jax.numpy as jnp
import jax.scipy as jscipy
from jax import random

# if cond >= 0 then e_if else e_else
def smooth_where(eta, cond, e_if, e_else):
  c = jscipy.special.expit(cond / eta)
  return c * e_if + (1 - c) * e_else

def softplus(x):
  return jnp.where(x < 30, jnp.log1p(jnp.exp(x)), x)

def softplus_inv(x):
  return jnp.where(x < 30, jnp.log(jnp.expm1(x)), x)

def forced_cond(if_idx, if_tf, my_idx, cond, e_if, e_else):
  new_cond = jnp.where(if_idx == my_idx, if_tf, cond)
  return jnp.where(new_cond, e_if, e_else)
