import jax
from jax import jit, vmap
from jax import random
from jax import numpy as jnp
import optimizers

import numpy as np
from matplotlib import pyplot as plt

import time
import pickle

import estimators

def all_experiments(model, save_file=False, file_name=None, run_bench=True, etas=[5], eta_decay=0.5, it_match=4000, **kw_args):
  if save_file and file_name is None:
    file_name = model.__name__.lower() + "_" + str(time.time_ns() // 1000000)
  exprts = run_experiments(model, file_name=file_name, etas=etas, eta_decay=eta_decay, it_match=it_match, **kw_args)
  plot_experiments(exprts, file_name=file_name)
  if run_bench:
    bench = run_benchmarks(model, etas[0], eta_decay, it_match)
  else:
    bench = { e : 1 for e, _ in exprts }
  print_normalised(exprts, bench, etas[0], file_name=file_name)
  return exprts, bench

def decr(x):
  return 0.000001/(x+1)
  
def run_experiment(
  model,
  estimator_name, 
  iters = 10000,
  key = random.PRNGKey(0), 
  step_size = 0.0015, # for Adam optimiser
  batch_size = 16, # no. of gradient samples per iteration
  log_every = 100, # iterations to calculate ELBO/variances
  log_batch_size = 1000, # no. of samples for ELBO/variance calculations
  theta0 = None, # initial theta to start gradient descent
  eta = 0.1, 
  eta_decay = 0.5, # DSGD accuracy coefficient in iteration i: eta_i = eta * (it_match/(i+1.0))**eta_decay
  it_match = 4000, # for "Fixed" use accuracy coefficient corresponding to DSGD's eta_i for i=it_match
  n_sub_samples = 1 # boundary correction samples for LYY18 estimator
):
  """
    Run a single optimisation trajectory for the problem defined by `model` with the
    given gradient estimator optimised using ADAM for `iters` iterations.
    Every `log_every` iterations, `log_batch_size` samples are taken to estimate the model
    target function (usually ELBO) and the estimator variance.

    Returns a dictionary with keys/values:
      theta : Learned `theta`
      log   : Array of (index, target, var_cmp, var_nrm) for every log iteration
      time  : Time taken for main loop + logging
  """
  opt_init, opt_update, get_params = optimizers.adam(step_size) #sgd(decr) #step_size)

  batch_grad_fun = estimators.batch_estimator(model, estimator_name, eta, eta_decay, it_match, n_sub_samples)
  if theta0 is None:
    theta0 = model.theta0

  @jit
  def step(opt_state, args):
    idx, rng = args
    rngs = random.split(rng, batch_size)

    theta = get_params(opt_state)
    params = theta
    gs = batch_grad_fun(idx, theta, rngs)
    neg_g = -jnp.mean(gs, axis=0)
    new_state = opt_update(idx, neg_g, opt_state)

    return new_state, params
  
  def log_step(idx, params, rng):
    rngs = random.split(rng, log_batch_size)
    
    theta = params
    gs = batch_grad_fun(idx, theta, rngs)

    target_val = estimators.batch_target(model, theta, rngs)

    # variance in two ways: mean component variance and 2-norm variance
    ddof = 1
    var_cmp = jnp.mean(jnp.var(gs, axis=0, ddof=ddof))
    var_nrm = jnp.var(jnp.linalg.norm(gs, axis=1, ord=2), ddof=ddof)

    return (idx, target_val, var_cmp, var_nrm)

  t0 = time.time()

  opt_rng, log_rng = random.split(key)

  opt_state = opt_init(theta0)

  # Run main optimisation loop for `iters` iterations
  scan_args = (jnp.arange(iters), random.split(opt_rng, iters))
  final_state, iterm_params = jax.lax.scan(step, opt_state, scan_args)
  
  theta_final = get_params(final_state)
  final_params = theta_final

  # Run `log_step` on very `log_every`'th iteration
  # (vectorised to be done in parallel for each such iteration)
  num_logs = iters // log_every + 1
  indices = jnp.arange(num_logs) * log_every
  
  # add in the final state to all the accumulated states 
  iterm_params = jax.tree_map(lambda xs, x: jnp.append(xs[::log_every], jnp.array([x]), axis=0), iterm_params, final_params)

  keys = random.split(log_rng, num_logs)
  log = jit(vmap(log_step, in_axes=(0, 0, 0)))(indices, iterm_params, keys)

  t1 = time.time()

  result = { "theta": theta_final, "log": np.array(log), "time": t1 - t0 }

  return result

def run_experiments(model, file_name=None, etas=[5], it_match=4000, eta_decay=0.5, **kw_args):
  """
    Run experiments on the given `model` for each gradient estimator defined in `estimators.py` and
    batch size 16.
    Returns a dictionary from (estimator, batch_size) to experiment results.
    Optional: Write results to file if given
  """
  exprts = {}
  fmt = "{:<20}" * 4
  print(fmt.format("Estimator", "eta", "Time (s)", "||final theta||_2"))

  for estimator_name in estimators.estimator_names:
    if estimator_name == "LYY18" and not hasattr(model, "target_surface"):
      print("Skipping LYY18 estimator, not defined")
      continue

    if estimator_name != "DSGD" and estimator_name != "Fixed":
      # accuracy coefficient only relevant for smoothing-based approaches
      etas = [etas[0]]
    
    for eta in etas:
      res = run_experiment(model, estimator_name, eta=eta, eta_decay=eta_decay, it_match = it_match, **kw_args)
      exprts[(estimator_name, eta)] = res

      time = round(res["time"], 6)
      norm =res["theta"][0] #round(float(jnp.linalg.norm(res["theta"], 2)), 6)
      print(fmt.format(estimator_name, eta, time, norm))

  if file_name is not None:
    with open(file_name + ".pkl", "wb") as f:
      pickle.dump(exprts, f)
    
  return exprts

def plot_experiments(exprts, file_name=None):
  """
    Generate plot of experiments from `run_experiments`
  """
  plt.figure(figsize=(15, 10))
  plt.rc("text", usetex=True)
  plt.rc('font', size=30, family="serif")
  plt.rc('text.latex', preamble="")

  colors = { "Score": "purple", "Reparam": "r", "LYY18": "orange", "DSGD": "green", "Fixed": "b"}
  linestyles = { 0: "-", 1: "--", 2: (0, (5, 10)), 3: "-.", 4: ":", 5: (0, (3, 5, 1, 5, 1, 5)) }


  i    = 0
  prev = -1
  for estimator, eta in exprts:
    if prev != estimator:
      i = 0
    x, y, _, _ = exprts[(estimator, eta)]["log"]
    # label only for solid lines
    label = "\\textsc{" + ("DSGD (ours)" if estimator == "DSGD" else estimator) + "}" if i==0 else ""
    plt.plot(x, y, color=colors[estimator], linestyle=linestyles[i], label=label)
    i += 1
    prev = estimator

  plt.xlabel("Iteration")
  plt.ylabel("ELBO")
  plt.legend(loc="lower right")

  if file_name is not None:
    plt.savefig(file_name + ".eps", format="eps")

def find_max_iters(
  model,
  estimator_name,
  repeats = 10,
  time_budget = 0.1,
  tolerance = 0.05,
  debug = False,
  # For estimator as in `run_experiment`
  eta = 5,
  eta_decay = 0.5,
  it_match = 4000,
  n_sub_samples = 1,
):
  """
    Binary search for the maximium number of iterations of the estimator that can be run on the
    model within `time_budget` seconds. This takes into account JIT compilation overhead.
  """
  theta0 = model.theta0
  batch_grad_fn = estimators.batch_estimator(model, estimator_name, eta, eta_decay, it_match, n_sub_samples)

  def time_run(batch_size, key):
    idx = random.randint(key, (1,), 0, 10000)[0]
    rngs = random.split(key, batch_size)
    t0 = time.time()
    _ = batch_grad_fn(idx, theta0, rngs)
    t1 = time.time()
    return t1 - t0

  def time_batch_size(batch_size):
    runs = [ time_run(batch_size, random.PRNGKey(i)) for i in range(repeats) ]
    # Remove first run to avoid including JIT overhead
    return np.mean(runs[1:])
  
  # Binary search for max iters within budget
  l, u = 1000, 100000
  MAX = 10 ** 7 # JIT compilation becomes slow at large batch sizes
  while time_batch_size(u) < time_budget and u < MAX:
    l, u = u, u * 2
    if debug: print("LOG", estimator_name, u)
  if u >= MAX: print("WARNING: Time budget might be too large for model")
  while (u - l)/l > tolerance:
    m = (u + l) // 2
    v = time_batch_size(m)
    if debug: print("LOG", estimator_name, m, v)
    if v < time_budget: l = m
    else: u = m
  return l

def run_benchmarks(model, eta = 0.1, eta_decay = 0.5, it_match = 4000, time_budget = 0.1):
  """
    Compute max iterations as a benchmark for each estimator over this model.
    Returns a dictionary mapping estimator names to maximum iterations in the
    given time budget.
  """
  bench = {}
  fmt = "{:<25}" * 2
  print(fmt.format("Estimator", f"Max iterations in {time_budget} seconds"))
  for estimator_name in estimators.estimator_names:
    if estimator_name == "LYY18" and not hasattr(model, "target_surface"):
      print("Skipping LYY18 estimator, not defined")
      continue
    bench[estimator_name] = find_max_iters(model, estimator_name, eta=eta, eta_decay=eta_decay, it_match=it_match, time_budget = time_budget)
    print(fmt.format(estimator_name, bench[estimator_name]))

  return bench

def print_normalised(exprts, bench, eta, file_name=None):
  """
    Print normalised benchmarks and variances with respect to the 
    score estimator from exprts / benchmarks given
  """

  score_iters = bench["Score"]
  score_var_cmp = np.mean(exprts[("Score", eta)]["log"][2])
  score_var_nrm = np.mean(exprts[("Score", eta)]["log"][3])

  title_fmt = "{:<12}" * 4
  fmt = "{:<12}" + "{:<12.2e}" * 3

  title = title_fmt.format("Estimator", "Cost", "Avg(V(.))", "V(|.|_2)")   
  lines = [ title ]
  print(title)

  for estimator_name in bench:
    if estimator_name == "Score": continue
    my_iters = bench[estimator_name]
    _, _, var_cmps, var_nrms = exprts[(estimator_name, eta)]["log"]
    var_cmp_ratio = np.mean(var_cmps) * score_iters / (score_var_cmp * my_iters)
    var_nrm_ratio = np.mean(var_nrms) * score_iters / (score_var_nrm * my_iters)
    cost_ratio = score_iters / my_iters

    line = fmt.format(estimator_name, cost_ratio, var_cmp_ratio, var_nrm_ratio) 
    print(line)
    lines += [ line ]

  if file_name is not None:
    with open(file_name + ".txt", "w") as f:
      f.write("\n".join(lines))
  
