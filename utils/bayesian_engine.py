"""
utils/bayesian_engine.py: PyMC model specification backend for variance-partitioned
phylogenetic mixed models using JAX sampling. Automated Multi-GPU / Single-GPU Routing.
"""
import numpy as np
import pymc as pm
import arviz as az
import pytensor
import jax
import scipy.linalg as la
from Bio import Phylo

pytensor.config.floatX = "float64"

def build_phylogenetic_covariance(tree_path, species_list, df, family_col="family",
                                  order_col=None, id_col="match_key", blend_weights=None):
    """
    Parses a NEXUS tree and computes a normalized NxN phylogenetic covariance matrix.
    Optionally blends tree topology with discrete categorical (Family/Order) kinship networks.
    """
    # standard evolutionary tree branch-lengths
    trees = list(Phylo.parse(tree_path, "nexus"))
    target_tree = trees[0]

    for clade in target_tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    tip_objects_map = {
        tip.name.strip().lower().replace(' ', '_'): tip
        for tip in target_tree.get_terminals() if tip.name
    }
    N = len(df)

    root_distance_cache = {target_tree.root: 0.0}
    for parent_clade in target_tree.find_clades(order="preorder"):
        parent_dist = root_distance_cache[parent_clade]
        for child_clade in parent_clade.clades:
            root_distance_cache[child_clade] = parent_dist + child_clade.branch_length

    family_tree_cache = {}
    for _, row in df.iterrows():
        sp = row[id_col]
        fam = str(row[family_col]).strip().lower()
        if sp in tip_objects_map:
            if fam not in family_tree_cache:
                family_tree_cache[fam] = []
            family_tree_cache[fam].append(tip_objects_map[sp])

    lineage_paths_cache = {}
    for idx, row in df.iterrows():
        sp = row[id_col]
        fam = str(row[family_col]).strip().lower()
        if sp in tip_objects_map:
            lineage_paths_cache[sp] = [target_tree.root] + target_tree.get_path(tip_objects_map[sp])
        elif fam in family_tree_cache and len(family_tree_cache[fam]) > 0:
            full_cousin_path = [target_tree.root] + target_tree.get_path(family_tree_cache[fam][0])
            family_node_depth = max(1, len(full_cousin_path) - 2)
            lineage_paths_cache[sp] = full_cousin_path[:family_node_depth]
        else:
            lineage_paths_cache[sp] = [target_tree.root]

    phylo_cov = np.zeros((N, N))
    for i in range(N):
        path_i = lineage_paths_cache[species_list[i]]
        phylo_cov[i, i] = root_distance_cache[path_i[-1]]
        for j in range(i + 1, N):
            path_j = lineage_paths_cache[species_list[j]]
            min_length = min(len(path_i), len(path_j))
            mrca_node = target_tree.root
            for k in range(min_length):
                if path_i[k] == path_j[k]:
                    mrca_node = path_i[k]
                else:
                    break
            shared_dist = root_distance_cache[mrca_node]
            phylo_cov[i, j] = shared_dist
            phylo_cov[j, i] = shared_dist

    max_variance = np.max(phylo_cov)
    if max_variance > 0:
        phylo_cov = phylo_cov / max_variance

    # optional taxonomic matrix layer
    if blend_weights and order_col in df.columns:
        order_strings = df[order_col].astype(str).str.strip().str.lower().tolist()
        family_strings = df[family_col].astype(str).str.strip().str.lower().tolist()

        order_matrix = np.zeros((N, N))
        family_matrix = np.zeros((N, N))

        for i in range(N):
            for j in range(i, N):
                if order_strings[i] == order_strings[j]:
                    order_matrix[i, j] = 1.0; order_matrix[j, i] = 1.0
                if family_strings[i] == family_strings[j]:
                    family_matrix[i, j] = 1.0; family_matrix[j, i] = 1.0

        # apply the explicit matrix blend weights configuration
        w_tree, w_fam, w_ord = blend_weights
        phylo_cov = (w_tree * phylo_cov) + (w_fam * family_matrix) + (w_ord * order_matrix)
        np.fill_diagonal(phylo_cov, 1.0)

    return phylo_cov

def fit_trivariance_pglmm_gpu(y, X, phylo_matrix, group_factor_idx, num_groups,
                              singleton_mask, monotypic_mask, class_label="vertebrate", chains=4):
    """Fits a phylogenetic generalized linear mixed model via NumPyro JAX ports."""
    jax.clear_caches()
    N, P = X.shape
    y = np.asarray(y, dtype=int).flatten()
    X = np.asarray(X, dtype=float)
    singleton_mask = np.asarray(singleton_mask, dtype=bool)
    monotypic_mask = np.asarray(monotypic_mask, dtype=bool)

    V = np.asarray(phylo_matrix, dtype=float) + np.eye(N) * 1e-5
    print(f"Executing Cholesky factorization of the {N}x{N} covariance matrix...")
    L_phylo = la.cholesky(V, lower=True)

    try:
        visible_gpus = jax.device_count("gpu")
    except Exception:
        visible_gpus = 0

    if visible_gpus > 1:
        chosen_method = "parallel"
        active_cores = min(visible_gpus, chains)
        print(f" Multi-GPU Detected ({visible_gpus} Cards). Distributing chains...")
    else:
        chosen_method = "vectorized"
        active_cores = 1
        print(" Single GPU/Host context. Vectorizing chains...")

    print(f"Compiling model architecture for {N} {class_label.lower()} observations...")

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=2.5, shape=P)
        sigma_dense = pm.Exponential("sigma_dense", lam=1.0)
        sigma_singleton = pm.Exponential("sigma_singleton", lam=1.0)
        sigma_phylo_base = pm.Exponential("sigma_phylo_base", lam=1.0)
        sigma_monotypic = pm.Exponential("sigma_monotypic", lam=1.0)

        group_scale = pm.math.switch(singleton_mask, sigma_singleton, sigma_dense)
        group_effect_raw = pm.Normal("group_effect_raw", mu=0, sigma=1, shape=num_groups)
        group_effect = pm.Deterministic("group_effect", group_effect_raw * group_scale)

        phylo_scale = pm.math.switch(monotypic_mask, sigma_monotypic, sigma_phylo_base)
        phylo_effect_raw = pm.Normal("phylo_effect_raw", mu=0, sigma=1, shape=N)
        phylo_effect = pm.Deterministic("phylo_effect", pm.math.dot(L_phylo, phylo_effect_raw) * phylo_scale)

        linear_predictor = pm.math.dot(X, beta) + phylo_effect + group_effect[group_factor_idx]
        y_obs = pm.Bernoulli("y_obs", logit_p=linear_predictor, observed=y)

        print(f"Sampling chains via NumPyro JAX...")
        trace = pm.sample(
            draws=1000, tune=2500, cores=active_cores, chains=chains,
            nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": chosen_method},
            random_seed=2026, target_accept=0.96
        )
    return trace

def summarize_trivariance_metrics(trace):
    return az.summary(trace, var_names=["beta", "sigma_dense", "sigma_singleton", "sigma_phylo_base", "sigma_monotypic"])
