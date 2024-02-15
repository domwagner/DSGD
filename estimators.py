from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, value_and_grad, jit, random, vmap
from jax.scipy.stats import norm

## All estimators defined in  this file

estimator_names = [ "DSGD", "Fixed",  "Score", "Reparam", "LYY18" ]

def batch_estimator(
  model,
  estimator_name,
  eta = 0.1,
  eta_decay = 0.5, # DSGD accuracy coefficient in iteration i: eta_i = eta * (it_match/(i+1.0))**eta_decay
  it_match = 4000, # for "Fixed" use accuracy coefficient corresponding to DSGD's eta_i for i=it_match
  n_sub_samples = 1 # boundary correction samples for LYY18 estimator
):
  if estimator_name == "DSGD":
    grad_fun = lambda i, theta, rng: smooth_grad(model, eta * (it_match/(i+1.0))**eta_decay, theta, rng)
  elif estimator_name == "Fixed":
    # same as DSGD after approx. it_match
    grad_fun = lambda i, theta, rng: smooth_grad(model, eta, theta, rng)
  elif estimator_name == "Reparam":
    grad_fun = lambda _, theta, rng: reparam_grad(model, theta, rng)
  elif estimator_name == "Score":
    grad_fun = lambda _, theta, rng: score_grad(model, theta, rng)
  elif estimator_name == "LYY18":
    grad_fun = lambda _, theta, rng: lyy18_grad(model, n_sub_samples, theta, rng)
  else:
    raise NotImplementedError("invalid estimator")

  batch_grad_fn = jit(vmap(grad_fun, in_axes=(None, None, 0)))
  return batch_grad_fn

## Objective function

@partial(jax.jit, static_argnums=(0,))
def sample_z(model, theta, rng):
  mu, std = model.affine_reparam(theta)
  sample = random.normal(rng, model.sample_shape)
  return mu + std * sample

@partial(jax.jit, static_argnums=(0,))
def target_sample(model, theta, rng):
  z = sample_z(model, theta, rng)
  return model.target(theta, z).reshape(())

@partial(jax.jit, static_argnums=(0,))
def batch_target(model, theta, rngs):
  targets = vmap(target_sample, in_axes=(None, None, 0))(model, theta, rngs)
  return jnp.mean(targets)

## 1. Reparam grad

@partial(jax.jit, static_argnums=(0,))
def reparam_grad(model, theta, rng):
  return grad(target_sample, argnums=1)(model, theta, rng)

## 2. Smoothed objective + grad 

@partial(jax.jit, static_argnums=(0,))
def smooth_target_sample(model, eta, theta, rng):
  z = sample_z(model, theta, rng)
  return model.smooth_target(eta, theta, z).reshape(())

@partial(jax.jit, static_argnums=(0,))
def smooth_grad(model, eta, theta, rng):
  return grad(smooth_target_sample, argnums=2)(model, eta, theta, rng)

## 3. Score grad

@partial(jax.jit, static_argnums=(0,))
def log_q(model, theta, z):
  mu, std = model.affine_reparam(theta)
  return jnp.sum(norm.logpdf(z, mu, std))

@partial(jax.jit, static_argnums=(0,))
def score_grad(model, theta, rng):
  z = sample_z(model, theta, rng)
  # grad(f) is needed here since it is not zero in general if `model.target`
  # is not in the form of an ELBO (in which case it has expectation zero anyway)
  def f(t): return model.target(t, z).reshape(())
  f_v, f_g = value_and_grad(f)(theta)
  return grad(partial(log_q, model))(theta, z) * f_v + f_g


## 4. LYY18 grad

@partial(jax.jit, static_argnums=(0,))
def inverse_reparam(model, theta, z):
  mu, std = model.affine_reparam(theta)
  return (z - mu) / std

@partial(jax.jit, static_argnums=(0,))
def marginal_q(model, theta, z, i):
  mu, std = model.affine_reparam(theta)
  return jnp.sum(norm.pdf(z[i], mu[i], std[i]))

@partial(jax.jit, static_argnums=(0,1))
def correction_grad(model, n_sub_samples, theta, rng):
  z_rng, idx_rng = random.split(rng)

  _, std = model.affine_reparam(theta)
  num_ifs = model.constants.size
  z = sample_z(model, theta, z_rng)
  indices = random.randint(idx_rng, (n_sub_samples,), 0, num_ifs)

  # matrix V from LYY18
  # |z| x |sample| matrix of partial derivatives
  v = jax.jacrev(inverse_reparam, argnums=1)(model, theta, z).transpose()

  @jit
  def step(idx):
    a = model.coefficients[idx]
    j = model.nz_indices[idx]
    c = model.constants[idx]

    g_z = z.at[j].set( (c - a.dot(z) + a[j] * z[j] ) / a[j] )

    q = marginal_q(model, theta, g_z, j)
    n = jnp.sign(-a[j]) * (a * std) / a[j]

    t_surface = model.target_surface(theta, g_z, idx, True)
    f_surface = model.target_surface(theta, g_z, idx, False)
    return (t_surface - f_surface) * q * (v @ n)

  return (num_ifs / n_sub_samples) * jnp.sum(vmap(step)(indices), axis=0)

@partial(jax.jit, static_argnums=(0,1))
def lyy18_grad(model, n_sub_samples, theta, rng):
  return reparam_grad(model, theta, rng) + correction_grad(model, n_sub_samples, theta, rng)
