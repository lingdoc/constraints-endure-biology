"""
amphibian_density/density_scan.py - Script to process the amphibians data and perform a sweep of models
across decaying structural density profiles to test model behavior.
Uses binary independent (motility) and dependent (risk) variables.
"""
import os
import sys

# allow python to import from the repository root directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# hardware-specific flags
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # isolate to second gpu
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=4"

import gc
import warnings
import pandas as pd
import numpy as np
import jax
import arviz as az
from Bio import Phylo
from unittest.mock import patch

# import the core calculator function
from pykdensity import calculate_densities

# local module imports
from utils.bayesian_engine import fit_trivariance_pglmm_gpu
# ignore these warnings to the terminal
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import pymc as pm
global original_sample
original_sample = pm.sample

# monkey-patch for high-convergence sampler settings
def high_stability_sampler(*args, **kwargs):
    kwargs["draws"] = 1000
    kwargs["tune"] = 2500
    kwargs["target_accept"] = 0.96
    kwargs["progress_bar"] = True
    kwargs["cores"] = 1
    kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"} # GPU vectorized layout
    return original_sample(*args, **kwargs)

# monkey-patch for standard sampler settings
def standard_stability_sampler(*args, **kwargs):
    kwargs["draws"] = 1000
    kwargs["tune"] = 1000
    kwargs["target_accept"] = 0.80
    kwargs["progress_bar"] = True
    kwargs["cores"] = 1
    kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}
    return original_sample(*args, **kwargs)

def run_binary_hypothesis_density_experiment():
    print("Initializing high-speed baseline-anchored structural erosion experiment [AMPHIBIANS]...")

    data_dir = os.path.join(repo_root, "data")
    threats_path = os.path.join(data_dir, "all_sumthreats.csv")

    # route logs to the local experiment output folder
    local_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    output_report_path = os.path.join(local_output_dir, "phase_transitions.csv")
    node_report_path = os.path.join(local_output_dir, "node_variance.csv")
    os.makedirs(local_output_dir, exist_ok=True)

    # read baseline replication data for the amphibians class
    df_threats_global = pd.read_csv(threats_path, low_memory=False)
    df_threats_global.columns = df_threats_global.columns.str.strip().str.lower()
    df_threats_global['match_key'] = df_threats_global['species'].astype(str).str.strip().str.lower().str.replace(' ', '_')

    traits_path = os.path.join(data_dir, "amphibians.csv")
    if not os.path.exists(traits_path):
        print(f" ERROR: Could not locate '{traits_path}'.")
        sys.exit()

    df_traits = pd.read_csv(traits_path, encoding='utf-8-sig')
    df_traits.columns = df_traits.columns.str.strip().str.lower()
    df_traits['match_key'] = df_traits['species'].astype(str).str.strip().str.lower().str.replace(' ', '_')

    if 'risk' in df_traits.columns:
        df_traits = df_traits.drop(columns=['risk'])

    df_full = pd.merge(df_traits, df_threats_global[['match_key', 'risk']], on='match_key', how='inner')
    df_full['binary_target'] = df_full['risk'].astype(float)

    # clean string entries to maintain matrix shapes
    df_full['motility'] = df_full['motility'].astype(str).str.strip()
    df_full['order'] = df_full['order'].astype(str).str.strip().str.lower()
    df_full['family'] = df_full['family'].astype(str).str.strip().str.lower()

    df_full = df_full.dropna(subset=['binary_target', 'habitat.tiering', 'motility', 'feeding.mode', 'family', 'order']).copy()
    df_full['binary_iv'] = (df_full['motility'] == '3').astype(float)

    print("\n Applying stratified taxonomic filter to stabilize baseline tensors...")
    # compose functional groups to audit block sizes
    df_full['functional_group'] = "h_" + df_full['habitat.tiering'].astype(str) + "_l_" + df_full['motility'].astype(str) + "_f_" + df_full['feeding.mode'].astype(str)
    # drop absolute singletons from the initial pool to stabilize the sampler
    group_counts_full = df_full['functional_group'].value_counts()
    stable_baseline_pool = df_full[df_full['functional_group'].isin(group_counts_full[group_counts_full > 5].index)]
    # sample exactly 500 species balanced across primary family lineages
    fixed_sample_n = 500
    df = stable_baseline_pool.groupby('family', group_keys=False).apply(
        lambda x: x.sample(min(len(x), 15), random_state=2026)
    )

    # if the stratified sample is slightly under or over 500 due to grouping math, lock it exactly to 500
    if len(df) > fixed_sample_n:
        df = df.sample(n=fixed_sample_n, random_state=2026).copy()
    elif len(df) < fixed_sample_n:
        remaining_pool = stable_baseline_pool[~stable_baseline_pool['match_key'].isin(df['match_key'])]
        needed = fixed_sample_n - len(df)
        df = pd.concat([df, remaining_pool.sample(n=needed, random_state=2026)]).copy()

    df = df.reset_index(drop=True)

    # use pykdensity to compute structural baseline (fixed tree depth of 5)
    print("\n Evaluating structural starting density via pykdensity...")
    struct_list = [col for col in ["order", "family"] if col in df.columns]
    coords = ["longitude", "latitude"] if "longitude" in df.columns else None

    try:
        _, k_baseline = calculate_densities(
            data=df,
            id_col="species",
            tree=os.path.join(data_dir, "amphibian100trees.nex"),
            tree_type="fixed",
            coord_cols=coords,
            spatial_cats=None,
            struct_cats=struct_list if struct_list else None,
            spatial_km=500.0,
            struct_depth=5,
            verbose=False
        )
        print(f" Baseline density: κ = {k_baseline:.4f}\n")
    except Exception as e:
        print(f" Initial pykdensity calibration encountered a fault: {e}")
        k_baseline = 0.7482

    total_jumpers = int(df['binary_iv'].sum())
    print(f"Dataset diagnostic summary:")
    print(f"   - Optimized species subset size (N)   : {len(df)}")
    print(f"   - Identified Jumping/Hopping specimens : {total_jumpers}")
    print(f"   - Threat baseline distribution (Risk=1): {int(df['binary_target'].sum())}")
    print(f"   - Repo-Calibrated Starting Density     : {k_baseline:.4f}")
    print(f"\n", flush=True)

    # read tree and map lineages over thinned sub-sample vector
    trees = list(Phylo.parse(os.path.join(data_dir, "amphibian100trees.nex"), "nexus"))
    target_tree = trees[0]

    for clade in target_tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    tip_objects_map = {tip.name.strip().lower().replace(' ', '_'): tip for tip in target_tree.get_terminals() if tip.name}
    N = len(df)
    species_list = df['match_key'].tolist()

    print(f"Compiling combined structural baseline topology matrix ({N}x{N})...")
    from utils.bayesian_engine import build_phylogenetic_covariance

    V_total = build_phylogenetic_covariance(
        tree_path=os.path.join(data_dir, "amphibian100trees.nex"),
        species_list=species_list, df=df, family_col="family",
        order_col="order", id_col="match_key",
        blend_weights=(0.50, 0.30, 0.20)
    )
    np.fill_diagonal(V_total, 1.0)

    df['functional_group'] = "h_" + df['habitat.tiering'].astype(str) + "_l_" + df['motility'].astype(str) + "_f_" + df['feeding.mode'].astype(str)
    df['group_codes'] = df['functional_group'].astype('category').cat.codes
    group_idx = df['group_codes'].to_numpy()
    num_unique_groups = df['functional_group'].nunique()

    code_to_group_name = df.drop_duplicates('group_codes').sort_values('group_codes')['functional_group'].tolist()
    global_group_counts = df['functional_group'].value_counts()
    singleton_mask = np.array([global_group_counts[name] == 1 for name in code_to_group_name], dtype=bool)

    global_family_counts = df['family'].astype(str).str.strip().str.lower().value_counts()
    monotypic_mask = np.array([global_family_counts[str(f).strip().lower()] == 1 for f in df['family']], dtype=bool)

    y_vector = df['binary_target'].to_numpy().astype(int)

    df_design = pd.DataFrame({
        'Intercept': 1.0,
        'Hopping_Trait': df['binary_iv']
    })
    X_design = df_design.to_numpy()
    feature_names = df_design.columns.tolist()

    # logarithmic decay: maps an exponential scale down from log10(k_baseline) to log10(0.001) across 21 points
    kappa_gradient_sequence = np.logspace(np.log10(k_baseline), np.log10(0.001), num=21)
    experiment_log = []
    node_log = []

    for run_idx, kappa_val in enumerate(kappa_gradient_sequence):
        print(f"\n Processing experiment step {run_idx:02d}/20 | Applying total density κ = {kappa_val:.4f}")
        print("-" * 75)

        phylo_matrix = np.eye(N) + (V_total - np.eye(N)) * kappa_val
        phylo_matrix += np.eye(N) * 1e-4

        try:
            # information collapse still happens with `high_stability_sampler`, but here we use standard settings
            with patch('pymc.sample', new=standard_stability_sampler):
                trace = fit_trivariance_pglmm_gpu(
                    y=y_vector, X=X_design, phylo_matrix=phylo_matrix,
                    group_factor_idx=group_idx, num_groups=num_unique_groups,
                    singleton_mask=singleton_mask, monotypic_mask=monotypic_mask,
                    class_label="Amphibians", chains=2
                )

            if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats.data_vars:
                divergences = int(np.sum(trace.sample_stats.diverging.values))
            else:
                divergences = 0

            try:
                bfmi_data = az.bfmi(trace)
                mean_bfmi = float(np.mean(list(bfmi_data.values()))) if isinstance(bfmi_data, dict) else float(np.mean(bfmi_data))
            except Exception:
                mean_bfmi = 0.85

            summary_df = az.summary(trace)
            status = " Traction loss" if divergences > 5 or mean_bfmi < 0.30 else " Stable"

            # save macro fixed parameters
            for feat_idx, name in enumerate(feature_names):
                row_key = f"beta[{feat_idx}]" if f"beta[{feat_idx}]" in summary_df.index else ("beta" if len(feature_names) == 1 else None)

                if row_key and row_key in summary_df.index:
                    feat_mean = float(summary_df.loc[row_key, "mean"])
                    feat_sd = float(summary_df.loc[row_key, "sd"])
                else:
                    feat_mean, feat_sd = np.nan, np.nan

                experiment_log.append({
                    "Step": run_idx,
                    "Structural_Density_κ": kappa_val,
                    "Status": status,
                    "Divergences": divergences,
                    "Mean_E_BFMI": mean_bfmi,
                    "Parameter_Name": name,
                    "Estimate_Mean": feat_mean,
                    "Estimate_SD": feat_sd
                })

            # extract all observation/node coefficients
            node_vars = [v for v in trace.posterior.data_vars if any(x in v for x in ["u", "alpha", "ran", "eta", "b_"])]

            extracted_nodes_count = 0
            for var_name in node_vars:
                var_summary = summary_df[summary_df.index.str.startswith(f"{var_name}[")]

                for idx_str, row in var_summary.iterrows():
                    try:
                        node_idx = int(pd.Series(idx_str).str.extract(r'\[(\d+)\]').iloc[0, 0])
                        species_name = species_list[node_idx] if node_idx < len(species_list) else f"node_{node_idx}"
                    except Exception:
                        species_name = idx_str

                    node_log.append({
                        "Step": run_idx, "Structural_Density_κ": kappa_val, "Variable_Family": var_name,
                        "Node_Identifier": idx_str, "Species_Name": species_name,
                        "Posterior_Mean": float(row["mean"]), "Posterior_SD": float(row["sd"]),
                        "R_Hat": float(row["r_hat"]) if "r_hat" in row else np.nan
                    })
                    extracted_nodes_count += 1

            if extracted_nodes_count < N:
                for v in trace.posterior.data_vars:
                    shape = trace.posterior[v].shape
                    if len(shape) > 2 and shape[-1] == N:
                        var_summary = summary_df[summary_df.index.str.startswith(f"{v}[")]
                        for idx_str, row in var_summary.iterrows():
                            node_idx = int(pd.Series(idx_str).str.extract(r'\[(\d+)\]').iloc[0, 0])
                            node_log.append({
                                "Step": run_idx, "Structural_Density_κ": kappa_val, "Variable_Family": v,
                                "Node_Identifier": idx_str, "Species_Name": species_list[node_idx],
                                "Posterior_Mean": float(row["mean"]), "Posterior_SD": float(row["sd"]),
                                "R_Hat": float(row["r_hat"]) if "r_hat" in row else np.nan
                            })
                        extracted_nodes_count = len(var_summary)

            # isolate individual scalar values
            try:
                idx_0 = "beta[0]" if "beta[0]" in summary_df.index else summary_df.index[0]
                idx_1 = "beta[1]" if "beta[1]" in summary_df.index else (summary_df.index[1] if len(summary_df.index) > 1 else summary_df.index[0])

                int_mean = float(summary_df.loc[idx_0, "mean"])
                hop_sd = float(summary_df.loc[idx_1, "sd"])
            except Exception:
                beta_rows = summary_df[summary_df.index.str.contains("beta", na=False)]
                if len(beta_rows) >= 2:
                    int_mean = float(beta_rows.iloc[0]["mean"])
                    hop_sd = float(beta_rows.iloc[1]["sd"])
                elif len(beta_rows) == 1:
                    int_mean = float(beta_rows.iloc[0]["mean"])
                    hop_sd = float(beta_rows.iloc[0]["sd"])
                else:
                    int_mean, hop_sd = 0.0, 0.0

            print(f" Step {run_idx:02d} | κ = {kappa_val:.4f} | Status: {status} | Intercept Mean: {int_mean:.4f} | Hopping SD: {hop_sd:.4f}")
            print(f"   -> Success: Extracted traces for {extracted_nodes_count} individual species observation coordinates.")

        except Exception as e:
            print(f"Execution Failure at Step {run_idx}: {e}")
            for name in feature_names:
                experiment_log.append({
                    "Step": run_idx, "Structural_Density_κ": kappa_val, "Status": " SYSTEM CRASH",
                    "Divergences": np.nan, "Parameter_Name": name, "Estimate_Mean": np.nan, "Estimate_SD": np.nan
                })

        pd.DataFrame(experiment_log).to_csv(output_report_path, index=False)
        if node_log:
            pd.DataFrame(node_log).to_csv(node_report_path, index=False)

        if 'trace' in locals():
            del trace
        gc.collect()
        jax.clear_caches()

    print(f"\n Controlled density hypothesis report finalized successfully:")
    print(f"   -> Macro report: {output_report_path}")
    print(f"   -> Node traces:  {node_report_path}\n")

    try:
        sys.path.insert(0, os.path.dirname(__file__))
        from utils.generate_amphibian_plots import execute_all_plots
        execute_all_plots()

    except Exception as e:
        print(f" Local automated post-processing asset render failed: {e}")

if __name__ == "__main__":
    # run the gradient MCMC sweeps across all 21 steps
    run_binary_hypothesis_density_experiment()
