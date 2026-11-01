"""
amphibian_topology/simulate_skew.py
Benchmarking script evaluating Bayesian and frequentist model estimations
across an incremental progression of data skew under low network connectivity.
"""
import os
import sys
import re
import copy
import warnings
import pandas as pd
import numpy as np
import scipy.linalg as la
from scipy import stats
import pymc as pm
import arviz as az
import gpboost as gpb

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_topology_grounded_skew_landscape(N=1000, N_anomalies=3):
    """
    Constructs a contingency matrix mapped onto simulated spatial
    and group partitions to replicate a low-connectivity network structure.
    """
    np.random.seed(2026)

    # Establish group categories and regional assignments
    regions = [f"Region_{i}" for i in range(1, 7)]
    assigned_regions = np.random.choice(regions, size=N, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
    cluster_pool = [f"Cluster_{i}" for i in range(1, 121)]
    assigned_clusters = np.random.choice(cluster_pool, size=N)

    IV = np.zeros(N, dtype=np.int32)
    DV = np.zeros(N, dtype=np.int32)

    # Define a concentrated regional hotspot (DV=1, IV=1) with 12 observations
    region_1_idx = np.where(assigned_regions == "Region_1")[0].tolist()
    other_regions_idx = np.where(assigned_regions != "Region_1")[0].tolist()
    hotspot_indices = region_1_idx[:6] + other_regions_idx[:6]
    IV[hotspot_indices] = 1
    DV[hotspot_indices] = 1

    # Allocate the comparative baseline pool (DV=0, IV=1) with 230 observations
    remaining_pool = list(set(range(N)) - set(hotspot_indices))
    comparative_indices = remaining_pool[:230]
    IV[comparative_indices] = 1
    DV[comparative_indices] = 0

    # Nest the variable count of counterexample anomalies (DV=1, IV=0)
    anomaly_pool = list(set(range(N)) - set(hotspot_indices) - set(comparative_indices))
    anomaly_indices = anomaly_pool[:N_anomalies]
    if N_anomalies > 0:
        IV[anomaly_indices] = 0
        DV[anomaly_indices] = 1

    # Generate 3D spatial coordinates and construct the covariance matrix
    phi = np.random.uniform(0, 2 * np.pi, N)
    theta = np.random.uniform(0, np.pi, N)
    coords_3d = np.column_stack((6371.0 * np.sin(theta) * np.cos(phi),
                                 6371.0 * np.sin(theta) * np.sin(phi),
                                 6371.0 * np.cos(theta)))

    dist_matrix = np.sqrt(np.sum((coords_3d[:, np.newaxis, :] - coords_3d[np.newaxis, :, :]) ** 2, axis=-1))
    spatial_cov = np.exp(-dist_matrix / 500.0) + np.eye(N) * 1e-4
    L_spatial = la.cholesky(spatial_cov, lower=True)

    # Assign branch network routing indices
    sub_routing_idx = np.random.randint(1, 40, size=N)
    if N_anomalies > 0:
        sub_routing_idx[anomaly_indices] = 0  # Isolate anomalies into unique singletons

    return coords_3d, L_spatial, IV, DV, assigned_regions.tolist(), assigned_clusters.tolist(), sub_routing_idx

def execute_bayesian_node(V_matrix, L_spatial, IV, DV, family_idx, num_f, macro_idx, num_m, parameterization, scenario_name, trace_dir):
    """
    Fits a Bayesian binomial generalized linear mixed model via MCMC.
    Supports centered and non-centered random effect parameterizations.
    """
    N = len(IV)
    with pm.Model() as model:
        beta_intercept = pm.StudentT("beta_intercept", nu=3, mu=0, sigma=2.5)
        beta_slope = pm.StudentT("beta_slope", nu=3, mu=0, sigma=2.5)

        sigma_family = pm.Exponential("sigma_family", lam=1.0)
        sigma_phylo = pm.Exponential("sigma_phylo", lam=1.0)
        sigma_spatial = pm.Exponential("sigma_spatial", lam=1.0)

        sd_dist = pm.Exponential.dist(lam=1.0, shape=2)
        chol, _, _ = pm.LKJCholeskyCov("chol_macro", n=2, eta=1.0, sd_dist=sd_dist)

        if parameterization == "centered":
            u_macro = pm.MvNormal("u_macro", mu=0, chol=chol, shape=(num_m, 2))
            u_macro_int = u_macro[:, 0]
            u_macro_slope = u_macro[:, 1]
            u_family = pm.Normal("u_family", mu=0, sigma=sigma_family, shape=num_f)
        else:
            z_macro = pm.Normal("z_macro", mu=0, sigma=1, shape=(num_m, 2))
            u_macro = pm.math.dot(z_macro, chol.T)
            u_macro_int = u_macro[:, 0]
            u_macro_slope = u_macro[:, 1]
            z_family = pm.Normal("z_family", mu=0, sigma=1, shape=num_f)
            u_family = z_family * sigma_family

        z_spatial = pm.Normal("z_spatial", mu=0, sigma=1, shape=N)
        u_spatial = pm.math.dot(L_spatial, z_spatial) * sigma_spatial

        z_phylo = pm.Normal("z_phylo", mu=0, sigma=1, shape=N)
        u_phylo = pm.math.dot(la.cholesky(V_matrix, lower=True), z_phylo) * sigma_phylo

        linear_predictor = (
            beta_intercept + u_macro_int[macro_idx] + u_family[family_idx] + u_phylo + u_spatial
        ) + (beta_slope + u_macro_slope[macro_idx]) * IV

        y_obs = pm.Bernoulli("y_obs", logit_p=linear_predictor, observed=DV)

        trace = pm.sample(draws=600, tune=1000, chains=2, target_accept=0.99,
                          nuts_sampler="numpyro", nuts_sampler_kwargs={"chain_method": "vectorized"},
                          progress_bar=False, compute_convergence_stat=False)

    # Save full MCMC trace output as a NetCDF file
    netcdf_filename = f"trace_{scenario_name}_{parameterization}.nc"
    az.to_netcdf(trace, os.path.join(trace_dir, netcdf_filename))

    # Extract 1D global effects and hyper-priors to summary CSV
    target_vars = ["beta_intercept", "beta_slope", "sigma_family", "sigma_phylo", "sigma_spatial"]
    summary_dict = {"chain": trace.posterior["chain"].values.repeat(600), "draw": np.tile(trace.posterior["draw"].values, 2)}
    for var in target_vars:
        if var in trace.posterior:
            summary_dict[var] = trace.posterior[var].values.flatten()

    df_clean_summary = pd.DataFrame(summary_dict)
    df_clean_summary.to_csv(os.path.join(trace_dir, f"summary_trace_{scenario_name}_{parameterization}.csv"), index=False)

    slope_draws = trace.posterior["beta_slope"].values.flatten()
    hdi = az.hdi(slope_draws, hdi_prob=0.95)
    return np.median(slope_draws), float(hdi[0]), float(hdi[1])

def execute_frequentist_node(coords_3d, IV, DV, family_idx, branch_idx, macro_idx):
    """
    Fits a frequentist generalized linear mixed model via profile maximum likelihood.
    """
    loop_matrix_state = np.column_stack((family_idx, macro_idx, branch_idx)).astype(float)
    X_design = np.column_stack((np.ones(len(IV)), IV))
    gp_model = gpb.GPModel(
        group_data=loop_matrix_state, group_rand_coef_data=IV.astype(float), ind_effect_group_rand_coef=[2],
        gp_coords=coords_3d, cov_function="exponential", likelihood="bernoulli_logit", num_parallel_threads=4
    )
    gp_model.set_optim_params(params={"optimizer_cov": "lbfgs", "maxit": 45, "init_cov_pars": [0.2, 0.2, 0.2, 0.2, 0.5, 500.0]})

    try:
        gp_model.fit(y=DV.astype(float), X=X_design)
        coef_dict = gp_model.get_coef(std_err=True, format_pandas=True).to_dict()
        mean_val = float(coef_dict["Covariate_2"].get("Param.", np.nan))
        se_val = float(coef_dict["Covariate_2"].get("Std. err.", np.nan))
        if np.isnan(se_val) or np.isinf(se_val) or se_val <= 0 or se_val < 1e-2: se_val = 1.0
        p_val = float(2 * stats.norm.cdf(-np.abs(mean_val / se_val)))
    except Exception:
        mean_val, se_val, p_val = np.nan, 1.0, np.nan
    return mean_val, mean_val / se_val, p_val

def main():
    print("Initializing Skew Simulation Engine...")
    output_dir = "output"
    trace_dir = os.path.join(output_dir, "mcmc_traces")
    os.makedirs(trace_dir, exist_ok=True)

    scenarios = {
        "skew_0_anom": 0,
        "skew_1_anom": 1,
        "skew_2_anom": 2,
        "skew_3_anom": 3,
        "skew_4_anom": 4,
        "skew_5_anom": 5,
        "skew_8_anom": 8,
        "skew_10_anom": 10
    }

    global_synthesis_rows = []

    for name, count in scenarios.items():
        print(f"Running scenario: {name} ({count} anomalies)")

        coords_3d, L_spatial, IV, DV, regions, clusters, branch_idx = generate_topology_grounded_skew_landscape(1000, count)

        family_idx = pd.Series(clusters).astype('category').cat.codes.to_numpy()
        macro_idx = pd.Series(regions).astype('category').cat.codes.to_numpy()
        num_f = len(np.unique(family_idx))
        num_m = len(np.unique(macro_idx))

        phylo_cov = (branch_idx[:, np.newaxis] == branch_idx[np.newaxis, :]).astype(float) + np.eye(1000) * 1e-4
        V_matrix = phylo_cov / np.max(phylo_cov)

        # Execute Centered Bayesian Node
        bc_mean, bc_low, bc_high = execute_bayesian_node(V_matrix, L_spatial, IV, DV, family_idx, num_f, macro_idx, num_m, "centered", name, trace_dir)
        global_synthesis_rows.append({"Scenario": name, "Framework": "Bayesian Centered", "Estimate": bc_mean, "Diag_A": np.nan, "Diag_B": np.nan, "CI_Low": bc_low, "CI_High": bc_high, "Decision": "CONFIRMED (False Positive)" if bc_low > 0 else "NOT CONFIRMED (Accurate)"})

        # Execute Non-Centered Bayesian Node
        bnc_mean, bnc_low, bnc_high = execute_bayesian_node(V_matrix, L_spatial, IV, DV, family_idx, num_f, macro_idx, num_m, "non-centered", name, trace_dir)
        global_synthesis_rows.append({"Scenario": name, "Framework": "Bayesian Non-Centered", "Estimate": bnc_mean, "Diag_A": np.nan, "Diag_B": np.nan, "CI_Low": bnc_low, "CI_High": bnc_high, "Decision": "CONFIRMED (False Positive)" if bnc_low > 0 else "NOT CONFIRMED (Accurate)"})

        # Execute Frequentist GP-GLMM Node
        f_mean, f_z, f_p = execute_frequentist_node(coords_3d, IV, DV, family_idx, branch_idx, macro_idx)
        global_synthesis_rows.append({
            "Scenario": name,
            "Framework": "Frequentist GP-GLMM",
            "Estimate": f_mean,
            "Diag_A": f_z,
            "Diag_B": f_p,
            "CI_Low": np.nan,
            "CI_High": np.nan,
            "Decision": "NOT CONFIRMED (Accurate)" if f_p >= 0.05 else "CONFIRMED (False Positive)"
        })

    df_final = pd.DataFrame(global_synthesis_rows)
    # Save the consolidated table to disk so the plotting script can ingest it
    master_log_path = os.path.join(output_dir, "framework_skew_matrix.csv")
    df_final.to_csv(master_log_path, index=False)
    print(f"Master framework validation log saved to: {master_log_path}")

    print("\nSummary Matrix Evaluation:")
    print("=" * 120)
    print(df_final.to_string(index=False, formatters={
        "Estimate": lambda x: f"{x:.4f}" if not np.isnan(x) else "NaN",
        "Diag_A": lambda x: f"{x:.4f}" if not np.isnan(x) else "  -  ",
        "Diag_B": lambda x: f"{x:.4f}" if not np.isnan(x) else "  -  ",
        "CI_Low": lambda x: f"{x:.4f}" if not np.isnan(x) else "  -  ",
        "CI_High": lambda x: f"{x:.4f}" if not np.isnan(x) else "  -  "
    }))
    print("=" * 120)

if __name__ == "__main__":
    main()
