import jax
import jax.numpy as jnp
import jax.scipy as jscipy

from functools import partial

from utils import softplus, softplus_inv, forced_cond, smooth_where

## LYY18 Temperature Control Model
# (rename 'theta' in model definition to 't' to avoid name clash with variational params 'theta')

t_s = 20.0
t_a = 32.0
delta_d = 4.0  # 0.25
P_rate = 14.0 
R = 1.5
C = 10.0
invCR = 1. / (C * R)
sigma0 = 0.2
sigma1 = 0.22
RP_rate = R * P_rate
sqrt_time_step = 2.0

t_lower = t_s - (delta_d / 2.0)
t_upper = t_s + (delta_d / 2.0)

tcl_dataset = [ # dataset in NIPS18model-tcl.clj
    [[0, 19.999393052411417], 19.793315726903277],

    [[0, 20.458453540109634], 20.57447277591924 ],
    [[0, 20.985832899298813], 23.318166935159923],
    [[0, 22.38768406603287 ], 21.732074976392273],
    [[1, 23.550508079883446], 23.419968943481056],
    [[1, 22.11729366445944 ], 23.45150129902584 ],
    
    [[1, 21.064619167851312], 20.849568706918458],
    [[1, 19.832404549285794], 20.547137185277485],
    [[1, 19.105507091196795], 18.211539341750292],
    [[1, 18.555867908555115], 18.505305155632083],
    [[1, 17.565304931813326], 16.33716312726332 ],
    
    [[0, 16.784272508644587], 16.381854726212698],
    [[0, 18.498312350185266], 19.404931063949995],
    [[0, 19.427764633078546], 19.042584619340943],
    [[0, 20.160476022468195], 20.646440471252035],
    [[0, 21.550786234122324], 22.021474886012076],
    
    [[0, 22.051865745051266], 21.767275215344945],
    [[1, 22.856132484853177], 22.59452466844986 ],
    [[1, 21.756492085507755], 20.437738111776696],
    [[1, 20.220631832476116], 21.478557643265425],
    [[1, 19.617488064098556], 18.513733108301032]
] 
tcl_n = len(tcl_dataset)
tcl_samples = 2 * tcl_n - 1
tcl_obs = [x[1] for x in tcl_dataset]
qs_gt = [x[0][0] for x in tcl_dataset]

class Temperature:
  theta0 = jnp.array(
    [20.] + tcl_obs[:-1] + # means for `t`
    [0.5] * (tcl_n - 1) + # means for `q_noise`
    [softplus_inv(0.001)] + [softplus_inv(sigma0 * sqrt_time_step)] * (tcl_n - 1) + # std. dev for `t`
    [softplus_inv(0.001)] * (tcl_n - 1) # std. dev for `q_noise`
  )
  sample_shape = (tcl_samples,)

  @staticmethod
  def affine_reparam(theta):
    return theta[:tcl_samples], softplus(theta[tcl_samples:])

  @staticmethod
  def base_target(IF, theta, z):

    def step(q_prev, scan_args):
      i, t_prev, t_curr, q_cur_noise, obs = scan_args

      rate, logprob0 = IF(4*i-3, t_prev - t_upper, 1.0, q_prev)
      rate, logprob1 = IF(4*i-4, -t_prev + t_lower, 0.0, rate)

      # q_cur_noise = sample(normal(rate, 0.001))
      log_prior = jnp.sum(jscipy.stats.norm.logpdf(q_cur_noise, rate, 0.001))
      
      q_cur, logprob2 = IF(4*i-2, q_cur_noise - 0.5, 1.0, 0.0)
      b_cur = invCR * (t_a - (t_prev + (q_cur * RP_rate)))
      sigma_cur, logprob3 = IF(4*i-1, q_cur_noise - 0.5, sigma1, sigma0)

      # t[i] = sample(z_prev + b_cur, sigma_cur * sqrt_time_step)
      log_prior += jnp.sum(jscipy.stats.norm.logpdf(t_curr, t_prev + b_cur, sigma_cur + sqrt_time_step))

      # observe(normal(t[i], 1), tcl_obs[i])
      log_model = jnp.sum(jscipy.stats.norm.logpdf(obs, t_curr, 1))

      log_pmf = logprob0 + logprob1 + logprob2 + logprob3

      return q_cur, (log_prior + log_model, log_pmf)

    q0 = 0.0
    t = z[:tcl_n]
    q_noise = z[tcl_n:]
    
    _, (log_joint, log_pmf) = jax.lax.scan(step, q0, (jnp.arange(1, tcl_n), t[:-1], t[1:], q_noise, jnp.array(tcl_obs[1:])))

    log_joint = jnp.sum(log_joint)
    log_pmf = jnp.sum(log_pmf)

    # t[0] = sample(norma(20, 0.001))
    log_joint += jnp.sum(jscipy.stats.norm.logpdf(t[0], 20, 0.001) )
    # observe(normal(t[0], 1), tcl_obs[0])
    log_joint += jnp.sum(jscipy.stats.norm.logpdf(tcl_obs[0], t[0], 1))

    mu, std = Temperature.affine_reparam(theta)
    log_q = jnp.sum(jscipy.stats.norm.logpdf(z, mu, std))

    return log_joint - log_q, log_pmf

  @staticmethod
  def target(theta, z):
    IF = lambda i, cond, t, f: (jnp.where(cond >= 0, t, f), 0.0)
    return Temperature.base_target(IF, theta, z)[0]

  @staticmethod
  def smooth_target(eta, theta, z):
    IF = lambda idx, cond, t, f: (smooth_where(eta, cond, t, f), 0.)
    return Temperature.base_target(IF, theta, z)[0]

  @staticmethod
  def target_surface(theta, z, if_idx, if_tf):
    IF = lambda i, cond, t, f: (forced_cond(if_idx, if_tf, i, cond >= 0, t, f), 0.0)
    return Temperature.base_target(IF, theta, z)[0]
  
  constants = jnp.array([-t_lower, t_upper, 0.5, 0.5] * (tcl_n - 1))
  nz_indices = jnp.array([[ i//4, i//4, tcl_n + i//4, tcl_n + i//4 ][i % 4] for i in range((tcl_n - 1) * 4)])
  coefficients = jnp.array([
    jnp.zeros(tcl_samples).at[j].set([-1.,1.,1.,1.][i % 4]) for i,j in enumerate(nz_indices)
  ])
