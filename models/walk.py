import jax.numpy as jnp
import jax.scipy as jscipy


from utils import softplus, smooth_where, forced_cond

class Walk:

  count = 15
  theta0 = jnp.array([count / 3.0] + [0.]*count + [1.]*(count+1))
  sample_shape = (count+1,)

  @staticmethod
  def affine_reparam(theta): return theta[:Walk.count+1],  softplus(theta[Walk.count+1:])

  def base_target(IF, theta, z):

    def walk(pos, dist, i):
      return IF((i-1), pos,
                  jscipy.stats.norm.logpdf(1000, Walk.count/2.0, 1) if i > Walk.count else jscipy.stats.norm.logpdf(z[i], 0, 1) + walk(pos + z[i], dist + IF(Walk.count+i, z[i], z[i], -z[i]), i + 1),
                  jscipy.stats.norm.logpdf(Walk.count / 4.0, dist, 1))


    mu, std = Walk.affine_reparam(theta)
    log_q = jnp.sum(jscipy.stats.norm.logpdf(z, mu, std))

    log_p = jscipy.stats.norm.logpdf(z[0], Walk.count / 5.0, 1) + walk(z[0], 0, 1)

      
    return    log_p  - log_q

  @staticmethod
  def target(theta, z):
    IF = lambda i, cond, t, f: jnp.where(cond >= 0, t, f)
    return Walk.base_target(IF, theta, z)

  @staticmethod
  def smooth_target(eta, theta, z):
    IF = lambda i, cond, t, f: smooth_where(eta, cond, t, f)
    return Walk.base_target(IF, theta, z)

  
  @staticmethod
  def target_surface(theta, z, if_idx, if_tf):
    IF = lambda i, cond, t, f: forced_cond(if_idx, if_tf, i, cond >= 0, t, f)
    
    return Walk.base_target(IF, theta, z)
  
  coefficients = jnp.concatenate((jnp.tril(jnp.full((count+1, count+1), 1.)), jnp.delete(jnp.eye(count+1,count+1), 0, 0)))
  nz_indices = jnp.concatenate((jnp.full((count+1),0),jnp.arange(1,count+1)))
  constants = jnp.array([ 0. ]*(2*count+1))
