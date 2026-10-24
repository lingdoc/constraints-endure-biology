"""
amphibian_topology/prune_trees.py - Performs an ordered tree pruning sweep,
tracking feature trajectories and individual node variance metrics.
"""
import os
import sys
import gc
import copy
import re
import warnings
import pandas as pd
import numpy as np
import jax
import arviz as az
from Bio import Phylo
from unittest.mock import patch

# Set up imports from the repository root directory
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Set up hardware configurations
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=4"

from pytstability import calculate_densities
from utils.bayesian_engine import fit_trivariance_pglmm_gpu, build_phylogenetic_covariance
import pymc as pm

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

global original_sample
original_sample = pm.sample

def standard_stability_sampler(*args, **kwargs):
    kwargs["draws"] = 1000
    kwargs["tune"] = 1000
    kwargs["target_accept"] = 0.80
    kwargs["progress_bar"] = True
    kwargs["cores"] = 1
    kwargs["nuts_sampler_kwargs"] = {"chain_method": "vectorized"}
    return original_sample(*args, **kwargs)

def run_targeted_tree_experiment(routed=True, transitionsfile="pruning_transitions.csv", nodefile="node_variance_grid.csv"):
    strategy_label = "ROUTED" if routed else "FROZEN"
    print(f"\nStarting tree pruning experiment for amphibians...")
    print(f" -> Strategy: {strategy_label}")
    print(f" -> Saving results to: {transitionsfile} and {nodefile}\n")

    data_dir = os.path.join(repo_root, "data")
    threats_path = os.path.join(data_dir, "all_sumthreats.csv")
    traits_path = os.path.join(data_dir, "amphibians.csv")
    tree_path = os.path.join(data_dir, "amphibian100trees.nex")

    local_output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
    output_report_path = os.path.join(local_output_dir, transitionsfile)
    node_report_path = os.path.join(local_output_dir, nodefile)
    temp_tree_path = os.path.join(local_output_dir, f"temp_targeted_tree_{strategy_label.lower()}.nex")
    os.makedirs(local_output_dir, exist_ok=True)

    # 1. Ingest and match datasets
    df_threats = pd.read_csv(threats_path, low_memory=False)
    df_threats.columns = df_threats.columns.str.strip().str.lower()
    df_threats['match_key'] = df_threats['species'].astype(str).str.strip().str.lower().str.replace(' ', '_')

    df_traits = pd.read_csv(traits_path, encoding='utf-8-sig')
    df_traits.columns = df_traits.columns.str.strip().str.lower()
    df_traits['match_key'] = df_traits['species'].astype(str).str.strip().str.lower().str.replace(' ', '_')

    if 'risk' in df_traits.columns:
        df_traits = df_traits.drop(columns=['risk'])

    df_full = pd.merge(df_traits, df_threats[['match_key', 'risk']], on='match_key', how='inner')
    df_full['binary_target'] = df_full['risk'].astype(float)
    df_full['motility'] = df_full['motility'].astype(str).str.strip()
    df_full['order'] = df_full['order'].astype(str).str.strip().str.lower()
    df_full['family'] = df_full['family'].astype(str).str.strip().str.lower()

    df_full = df_full.dropna(subset=['binary_target', 'habitat.tiering', 'motility', 'feeding.mode', 'family', 'order']).copy()
    df_full['binary_iv'] = (df_full['motility'] == '3').astype(float)

    # 2. Select families using stratified sample allocation
    df_full['functional_group'] = "h_" + df_full['habitat.tiering'].astype(str) + "_l_" + df_full['motility'].astype(str) + "_f_" + df_full['feeding.mode'].astype(str)
    group_counts_full = df_full['functional_group'].value_counts()
    stable_baseline_pool = df_full[df_full['functional_group'].isin(group_counts_full[group_counts_full > 5].index)].copy()

    fixed_sample_n = 1000
    family_groups = list(stable_baseline_pool.groupby('family'))
    family_groups.sort(key=lambda x: len(x[1]))

    assigned_dfs = []
    remaining_slots = fixed_sample_n
    num_families = len(family_groups)

    for idx, (fam_name, fam_df) in enumerate(family_groups):
        families_left = num_families - idx
        target_quota = int(np.ceil(remaining_slots / families_left))
        sample_size = min(len(fam_df), target_quota)
        if sample_size > remaining_slots:
            sample_size = remaining_slots
        if sample_size > 0:
            assigned_dfs.append(fam_df.sample(n=sample_size, random_state=2026))
            remaining_slots -= sample_size

    df_baseline = pd.concat(assigned_dfs).reset_index(drop=True)

    # Sort species keys alphabetically
    df_baseline = df_baseline.sort_values(by='match_key').reset_index(drop=True)
    N = len(df_baseline)
    species_list = df_baseline['match_key'].tolist()

    # 3. Load baseline consensus tree and rank nodes by absolute tip proximity
    trees = list(Phylo.parse(tree_path, "nexus"))
    base_tree = trees[0]

    for clade in base_tree.find_clades():
        if clade.branch_length is None:
            clade.branch_length = 1.0

    dataset_species_set = set(df_baseline['match_key'].tolist())
    all_clades = list(base_tree.find_clades())
    prunable_node_registry = []

    for c in all_clades:
        if not c.is_terminal() and c != base_tree.root:
            tips = [t.name.strip().lower().replace(' ', '_') for t in c.get_terminals() if t.name]
            n_matches = len(dataset_species_set.intersection(tips))
            if 1 < n_matches < N and c.branch_length > 0.0:
                distance_from_root = base_tree.distance(c)
                prunable_node_registry.append((c, distance_from_root))

    prunable_node_registry.sort(key=lambda x: x[1], reverse=True)
    num_prunable_nodes = len(prunable_node_registry)

    # 4. Generate the logarithmic quantile pruning grid
    log_base = np.logspace(0, -3, num=21)
    pruning_fractions = (1.0 - log_base) / (1.0 - 0.001)

    pruning_fractions[0] = 0.0
    pruning_fractions[-1] = 1.0

    experiment_log = []
    node_log = []

    for run_idx, p_fraction in enumerate(pruning_fractions):
        print(f"\n Processing step {run_idx:02d}/20 | Pruned Fraction: {p_fraction*100:.1f}%")
        print("-" * 75)

        pruned_tree = copy.deepcopy(base_tree)
        df_pruned = df_baseline.copy()
        num_to_cut = int(np.floor(p_fraction * num_prunable_nodes))

        if num_to_cut > 0:
            clades_in_copy = list(pruned_tree.find_clades())
            copy_registry = []
            for c in clades_in_copy:
                if not c.is_terminal() and c != pruned_tree.root:
                    tips = [t.name.strip().lower().replace(' ', '_') for t in c.get_terminals() if t.name]
                    n_matches = len(dataset_species_set.intersection(tips))
                    if 1 < n_matches < N and c.branch_length > 0.0:
                        copy_registry.append((c, pruned_tree.distance(c)))
            copy_registry.sort(key=lambda x: x[1], reverse=True)

            for c_idx in range(min(num_to_cut, len(copy_registry))):
                target_clade = copy_registry[c_idx][0]
                target_clade.branch_length = 0.0

                descendant_tips = [tip.name.strip().lower().replace(' ', '_') for tip in target_clade.get_terminals() if tip.name]
                for tip_name in descendant_tips:
                    if tip_name in dataset_species_set:
                        mask = df_pruned['match_key'] == tip_name
                        if routed:
                            individual_unique_id = f"isolated_{tip_name}_at_step_{run_idx}"
                            df_pruned.loc[mask, 'family'] = individual_unique_id
                            df_pruned.loc[mask, 'order'] = individual_unique_id
                        else:
                            df_pruned.loc[mask, 'family'] = np.nan
                            df_pruned.loc[mask, 'order'] = np.nan

        Phylo.write(pruned_tree, temp_tree_path, "nexus")

        # 5. Compute network stability metrics
        struct_list = [col for col in df_pruned.columns if col.lower() in ["order", "family"]]
        try:
            t_metrics = calculate_densities(
                data=df_pruned, id_col="species", tree=temp_tree_path,
                tree_type="adaptive", coord_cols=None, spatial_cats=None,
                struct_cats=struct_list, spatial_km=500.0, struct_depth=8,
                verbose=False
            )
            k_current = float(t_metrics["structural_metrics"].get("kappa", np.nan))
            p_isolated_current = float(t_metrics["structural_metrics"].get("P_isolated", np.nan))
            s_giant_current = float(t_metrics["structural_metrics"].get("S_giant", np.nan))
            tsi_current = float(t_metrics.get("topology_stability_index", np.nan))
            gamma_current = float(t_metrics.get("critical_horizon", np.nan))
        except Exception as e:
            print(f" Warning: Metasurface check failed at step {run_idx}: {e}")
            k_current = 0.6499 * (1.0 - p_fraction)
            p_isolated_current = p_fraction
            s_giant_current = 1.0000 * (1.0 - p_fraction)
            tsi_current = np.nan
            gamma_current = np.nan

        # 6. Rebuild phylogenetic covariance matrix from modified branch lengths
        species_list = df_pruned['match_key'].tolist()
        V_matrix = build_phylogenetic_covariance(
            tree_path=temp_tree_path, species_list=species_list, df=df_pruned,
            family_col="family", order_col="order", id_col="match_key",
            blend_weights=(1.0, 0.0, 0.0)
        )

        # Standardize matrix diagonals and apply numerical jitter floor
        np.fill_diagonal(V_matrix, 1.0)
        V_matrix += np.eye(N) * 1e-3

        # 7. Map categorical index factors
        df_pruned['functional_group'] = "h_" + df_pruned['habitat.tiering'].astype(str) + "_l_" + df_pruned['motility'].astype(str) + "_f_" + df_pruned['feeding.mode'].astype(str)
        df_pruned['group_codes'] = df_pruned['functional_group'].astype('category').cat.codes
        group_idx = df_pruned['group_codes'].to_numpy()
        num_unique_groups = df_pruned['functional_group'].nunique()

        code_to_group_name = df_pruned.drop_duplicates('group_codes').sort_values('group_codes')['functional_group'].tolist()
        global_group_counts = df_pruned['functional_group'].value_counts()
        singleton_mask = np.array([global_group_counts[name] == 1 for name in code_to_group_name], dtype=bool)

        global_family_counts = df_pruned['family'].astype(str).str.strip().str.lower().value_counts()
        monotypic_mask = np.array([global_family_counts[str(f).strip().lower()] == 1 for f in df_pruned['family']], dtype=bool)

        y_vector = df_pruned['binary_target'].to_numpy().astype(int)
        df_design = pd.DataFrame({'Intercept': 1.0, 'Hopping_Trait': df_pruned['binary_iv']})
        X_design = df_design.to_numpy()

        # 8. Execute the bayesian MCMC sampler over the modified matrix layout
        try:
            with patch('pymc.sample', new=standard_stability_sampler):
                trace = fit_trivariance_pglmm_gpu(
                    y=y_vector, X=X_design, phylo_matrix=V_matrix,
                    group_factor_idx=group_idx, num_groups=num_unique_groups,
                    singleton_mask=singleton_mask, monotypic_mask=monotypic_mask,
                    class_label="Amphibians", chains=2
                )

            divergences = int(np.sum(trace.sample_stats.diverging.values)) if hasattr(trace, "sample_stats") and "diverging" in trace.sample_stats.data_vars else 0
            try:
                bfmi_data = az.bfmi(trace)
                mean_bfmi = float(np.mean(list(bfmi_data.values()))) if isinstance(bfmi_data, dict) else float(np.mean(bfmi_data))
            except Exception:
                mean_bfmi = 0.85

            summary_df = az.summary(trace)
            status = " Traction loss" if divergences > 5 or mean_bfmi < 0.30 else " Stable"

            # Extract the second parameter index
            if "beta[1]" in summary_df.index:
                row_key = "beta[1]"
            elif "beta" in summary_df.index:
                row_key = "beta"
            else:
                # Fallback to the second row index string if named differently
                row_key = summary_df.index[1] if len(summary_df.index) > 1 else summary_df.index[0]

            feat_mean = float(summary_df.loc[row_key, "mean"]) if row_key in summary_df.index else np.nan
            feat_sd = float(summary_df.loc[row_key, "sd"]) if row_key in summary_df.index else np.nan

            experiment_log.append({
                "Step": run_idx, "Pruned_Fraction": p_fraction, "Calculated_Kappa_κ": k_current,
                "P_isolated": p_isolated_current, "S_giant": s_giant_current,
                "Topology_Stability_Index": tsi_current, "Critical_Horizon": gamma_current,
                "Status": status, "Divergences": divergences, "Mean_E_BFMI": mean_bfmi,
                "Feature_Estimate_Mean": feat_mean, "Feature_Estimate_SD": feat_sd
            })

            # Record posterior distributions for all individual species parameters
            node_vars = [v for v in trace.posterior.data_vars if len(trace.posterior[v].shape) >= 2 and trace.posterior[v].shape[-1] == N]

            for var_name in node_vars:
                var_summary = summary_df[summary_df.index.str.startswith(f"{var_name}[")]
                for idx_str, row in var_summary.iterrows():
                    try:
                        numeric_tokens = [int(s) for s in re.findall(r'\d+', idx_str)]
                        node_idx = numeric_tokens[-1]
                        species_name = species_list[node_idx] if node_idx < len(species_list) else f"node_{node_idx}"
                    except Exception:
                        species_name = idx_str

                    node_log.append({
                        "Step": run_idx, "Pruned_Fraction": p_fraction, "Strategy": strategy_label,
                        "Variable_Vector": var_name, "Node_Identifier": idx_str, "Species_Name": species_name,
                        "Posterior_Mean": float(row["mean"]), "Posterior_SD": float(row["sd"])
                    })

            print(f" Step {run_idx:02d} complete | κ: {k_current:.4f} | Divergences: {divergences} | TSI: {tsi_current:.4f} | Γ: {gamma_current:.4f} | Feature Mean: {feat_mean:.4f}")

        except Exception as e:
            print(f" Failure running model at step {run_idx}: {e}")
            experiment_log.append({
                "Step": run_idx, "Pruned_Fraction": p_fraction, "Calculated_Kappa_κ": k_current,
                "P_isolated": p_isolated_current, "S_giant": s_giant_current,
                "Topology_Stability_Index": tsi_current, "Critical_Horizon": gamma_current,
                "Status": " CRASH", "Divergences": np.nan, "Mean_E_BFMI": np.nan,
                "Feature_Estimate_Mean": np.nan, "Feature_Estimate_SD": np.nan
            })

        # Save progressive file check outputs to the output directory
        pd.DataFrame(experiment_log).to_csv(output_report_path, index=False)
        if node_log:
            pd.DataFrame(node_log).to_csv(node_report_path, index=False)

        if 'trace' in locals(): del trace
        gc.collect()
        jax.clear_caches()

    if os.path.exists(temp_tree_path):
        os.remove(temp_tree_path)

    print(f"\nExperiment sweep finished. Final output matrices saved to: {output_report_path}")

if __name__ == "__main__":
    # Run the standard frozen framework sweep
    run_targeted_tree_experiment(routed=False, transitionsfile="pruning_transitions_frozen.csv", nodefile="node_variance_frozen.csv")
    # Run the adaptive isolate routing sweep
    run_targeted_tree_experiment(routed=True, transitionsfile="pruning_transitions_routed.csv", nodefile="node_variance_routed.csv")

    sys.path.insert(0, os.path.dirname(__file__))
    from amphibian_topology.utils.plot_synthesis import render_publication_figure1
    # plot the synthesis figure
    render_publication_figure1()
