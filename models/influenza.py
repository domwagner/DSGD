import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from functools import partial

from utils import softplus, forced_cond, smooth_where

## LYY18 Influenza

mortality_data_full = [
        0.2902977, 0.5702096, 0.6384278, 0.3135166, 0.2047724, 0.1885323, 
        0.1798065, 0.1867087, 0.1871117, 0.2157304, 0.2385934, 0.2572363, 
        0.6381312, 0.5195216, 0.2968231, 0.2514277, 0.2144966, 0.1964111, 
        0.2343862, 0.2095384, 0.21776, 0.2450697, 0.2554203, 0.3357194, 
        0.5968794, 0.5156948, 0.3085772, 0.2569596, 0.2194814, 0.2158074, 
        0.2240929, 0.2217431, 0.2388084, 0.2528328, 0.260521, 0.2822165, 
        0.8113721, 0.4458291, 0.3415985, 0.2774243, 0.2484958, 0.2525427, 
        0.2466902, 0.2452006, 0.2279679, 0.2610293, 0.3177998, 0.7298681, 
        0.281731, 0.3030367, 0.2915253, 0.2453148, 0.2051037, 0.1845583, 
        0.2046436, 0.1828399, 0.1960763, 0.2203844, 0.224994, 0.3036654, 
        0.8192756, 0.4376872, 0.3834813, 0.2919304, 0.255642, 0.2369918, 
        0.2495123, 0.2280816, 0.2335083, 0.2679917, 0.3030445, 0.3594536, 
        0.3743171, 0.3746391, 0.3398281, 0.2909505, 0.2403667, 0.2269297, 
        0.2134564, 0.2074597, 0.2099793, 0.2464345, 0.2768291, 0.3359459, 
        0.4932865, 0.5692177, 0.3593959, 0.2741868, 0.2424912, 0.2241473, 
        0.228084, 0.2283657, 0.2282342, 0.2579092, 0.2909701, 0.3113483, 
        0.49372, 0.4728154, 0.3094317, 0.2343683, 0.200722, 0.1906996, 
        0.1877136, 0.1990874, 0.1908602, 0.2111834, 0.2084524, 0.2510464, 
        0.3096678, 0.3330646, 0.349702, 0.3066515, 0.223837, 0.2030033, 
        0.2231299, 0.1891271, 0.1997522, 0.2269953, 0.2295186, 0.3198083, 
        0.5712671, 0.4351815, 0.282585, 0.2381904, 0.2196089, 0.190482, 
        0.1926964, 0.1853681, 0.1943872, 0.2290728, 0.225355, 0.2569408 ]

mortality_data_short = mortality_data_full[:12]
influenza_data = mortality_data_short

alpha1 = 1.406
alpha2 = -0.622
beta0 = 0.210
beta1 = -0.312
sigma1 = 0.023
sigma2 = 0.112
sigmav = 0.002

influenza_n = len(influenza_data)
influenza_samples = 3 * influenza_n + 1

class Influenza:
  
  theta0 = jnp.zeros(2 * influenza_samples)
  sample_shape = (influenza_samples,)

  @staticmethod
  def affine_reparam(theta):
    return theta[:influenza_samples], softplus(theta[influenza_samples:])

  @staticmethod
  def base_target(IF, theta, z):

    def step(acc, scan_args):
      a_prev, b_prev, c_prev, d_prev = acc
      i, v_curr, w_curr, f_prev, f_curr, data_curr = scan_args

      a_curr = v_curr + alpha1 * a_prev + alpha2 * b_prev
      b_curr = a_prev
      c_curr = w_curr + beta0 + beta1 * c_prev
      d_curr = d_prev

      mu1, logprob0 = IF(2*i-2, f_prev, 0.67, -0.67)
      # f[i] = sample(normal(mu1, 1))
      log_prior = jnp.sum(jscipy.stats.norm.logpdf(f_curr, mu1, 1))

      mu2, logprob1 = IF(2*i-1, f_curr, a_curr + d_curr, a_curr + c_curr + d_curr)
      # observe(normal(mu2, sigmav), influenza_data[i-1]))
      log_model = jnp.sum(jscipy.stats.norm.logpdf(data_curr, mu2, sigmav))
      
      acc = (a_curr, b_curr, c_curr, d_curr)
      return acc, (log_prior + log_model, logprob0 + logprob1)

    n = influenza_n
    f = z[:n+1]
    v = z[n+1:2*n+1]
    w = z[2*n+1:]

    acc0 = ( 0., 0., 0., 0. )
    _, (log_joint, log_pmf) = jax.lax.scan(step, acc0, ( jnp.arange(1, n+1), v, w, f[:-1], f[1:], jnp.array(influenza_data) ))

    log_joint = jnp.sum(log_joint)
    log_pmf = jnp.sum(log_pmf)

    # f[0] = sample(normal(0, 1))
    log_joint += jnp.sum(jscipy.stats.norm.logpdf(f[0], 0, 1))
    # v[i] = sample(normal(0, sigma1))
    log_joint += jnp.sum(jscipy.stats.norm.logpdf(v, 0, sigma1))
    # w[i] = sample(normal(0, sigma2))
    log_joint += jnp.sum(jscipy.stats.norm.logpdf(w, 0, sigma2))
    
    mu, std = Influenza.affine_reparam(theta)
    log_q = jnp.sum(jscipy.stats.norm.logpdf(z, mu, std))

    return log_joint - log_q, log_pmf

  @staticmethod
  def target(theta, z):
    IF = lambda i, cond, t, f: (jnp.where(cond >= 0, t, f), 0.)
    return Influenza.base_target(IF, theta, z)[0]

  @staticmethod
  def smooth_target(eta, theta, z):
    IF = lambda idx, cond, t, f: (smooth_where(eta, cond, t, f), 0.)
    return Influenza.base_target(IF, theta, z)[0]

  @staticmethod
  def target_surface(theta, z, if_idx, if_tf):
    IF = lambda i, cond, t, f: (forced_cond(if_idx, if_tf, i, cond >= 0, t, f), 0.)
    return Influenza.base_target(IF, theta, z)[0]
  
  constants = jnp.zeros(2 * influenza_n)
  nz_indices = jnp.array([ (i+1)//2 for i in range(2*influenza_n) ])
  coefficients = jnp.array([ jnp.zeros(influenza_samples).at[j].set(1) for j in nz_indices])
