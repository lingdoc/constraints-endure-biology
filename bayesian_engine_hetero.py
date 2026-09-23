"""
bayesian_engine_hetero.py: PyMC model specification backend for variance-partitioned
phylogenetic mixed models using JAX sampling.
"""
import numpy as np
import pymc as pm
import arviz as az
import pytensor
import jax
import scipy.linalg as la

pytensor.config.floatX = "float64"

def fit_trivariance_pglmm_gpu(y, X, phylo_matrix, group_factor_idx, num_groups,
                              singleton_mask, monotypic_mask, class_label="vertebrate", chains=4):
    """
    Fits a phylogenetic generalized linear mixed model (PGLMM) partitioning
    residual variance across standard profiles, ecological singletons, and monotypic isolates.
    """
    jax.clear_caches()

    N, P = X.shape
    y = np.asarray(y, dtype=int).flatten()
    X = np.asarray(X, dtype=float)
    singleton_mask = np.asarray(singleton_mask, dtype=bool)
    monotypic_mask = np.asarray(monotypic_mask, dtype=bool)

    # Inject stabilizing ridge for Cholesky stability
    V = np.asarray(phylo_matrix, dtype=float) + np.eye(N) * 1e-5

    print(f"Executing Cholesky factorization of the {N}x{N} covariance matrix...")
    L_phylo = la.cholesky(V, lower=True)

    print(f"Compiling model architecture for {N} {class_label.lower()} observations...")

    with pm.Model() as model:
        # Fixed effect priors
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=P)

        # Variance component scaling priors
        sigma_dense = pm.Exponential("sigma_dense", lam=1.0)
        sigma_singleton = pm.Exponential("sigma_singleton", lam=1.0)
        sigma_phylo_base = pm.Exponential("sigma_phylo_base", lam=1.0)
        sigma_monotypic = pm.Exponential("sigma_monotypic", lam=1.0)

        # Ecological error lane configuration
        group_scale = pm.math.switch(singleton_mask, sigma_singleton, sigma_dense)
        group_effect_raw = pm.Normal("group_effect_raw", mu=0, sigma=1, shape=num_groups)
        group_effect = pm.Deterministic("group_effect", group_effect_raw * group_scale)

        # Evolutionary error lane configuration
        phylo_scale = pm.math.switch(monotypic_mask, sigma_monotypic, sigma_phylo_base)

        # Non-centered phylogenetic effect construction
        phylo_effect_raw = pm.Normal("phylo_effect_raw", mu=0, sigma=1, shape=N)
        phylo_effect = pm.Deterministic(
            "phylo_effect",
            pm.math.dot(L_phylo, phylo_effect_raw) * phylo_scale
        )

        # Combine linear predictor components
        linear_predictor = pm.math.dot(X, beta) + phylo_effect + group_effect[group_factor_idx]

        # Likelihood
        y_obs = pm.Bernoulli("y_obs", logit_p=linear_predictor, observed=y)

        print(f"Sampling chains via NumPyro JAX...")
        trace = pm.sample(
            draws=1000,
            tune=1000,
            chains=chains,
            nuts_sampler="numpyro",
            nuts_sampler_kwargs={"chain_method": "vectorized"},
            random_seed=2026,
            target_accept=0.90
        )

    return trace

def summarize_trivariance_metrics(trace):
    return az.summary(trace, var_names=["beta", "sigma_dense", "sigma_singleton", "sigma_phylo_base", "sigma_monotypic"])
