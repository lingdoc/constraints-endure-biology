"""
Main pipeline script to load data, build phylogenetic covariance matrices,
and run Bayesian mixed models across vertebrate classes.
"""
import os
import gc
import warnings
import pandas as pd
import numpy as np
from Bio import Phylo
import jax
import scipy.stats as stats

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# local module imports
from utils.bayesian_engine import fit_trivariance_pglmm_gpu, summarize_trivariance_metrics
from utils.check_hypotheses import run_standalone_hypothesis_check

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def execute_global_vertebrate_bayesian_pipeline():
    print("Initializing model pipeline...")

    data_dir = "data"
    threats_path = os.path.join(data_dir, "all_sumthreats.csv")

    if not os.path.exists(threats_path):
        print(f"Error: Threat file missing at {threats_path}")
        return

    df_threats_global = pd.read_csv(threats_path, low_memory=False)
    df_threats_global.columns = df_threats_global.columns.str.strip().str.lower()
    df_threats_global['match_key'] = df_threats_global['species'].astype(str).str.strip().str.lower().str.replace(' ', '_')

    class_pipeline_schema = [
        {"label": "Mammals",    "traits": "mammals.csv",    "tree": "mammal100trees.nex"},
        {"label": "Birds",      "traits": "birds.csv",      "tree": "bird100trees.nex"},
        {"label": "Squamates",  "traits": "reptiles.csv",   "tree": "squamate100trees.nex"},
        {"label": "Amphibians", "traits": "amphibians.csv", "tree": "amphibian100trees.nex"}
    ]

    print("\n Data diagnostics: Isolated taxonomic groups")
    print("-"*60)
    for schema in class_pipeline_schema:
        traits_path = os.path.join(data_dir, schema["traits"])
        if os.path.exists(traits_path):
            df_diag = pd.read_csv(traits_path, encoding='utf-8-sig')
            df_diag.columns = df_diag.columns.str.strip().str.lower()
            null_families = df_diag['family'].isna().sum() if 'family' in df_diag.columns else len(df_diag)
            string_missing = df_diag['family'].astype(str).str.strip().str.lower().isin(['nan', 'none', 'unknown', '', 'incertae_sedis']).sum() if 'family' in df_diag.columns else len(df_diag)
            total_unclassified_isolates = max(int(null_families), int(string_missing))

            if 'family' in df_diag.columns:
                fam_counts = df_diag['family'].value_counts()
                monotypic_family_count = sum(fam_counts == 1)
            else:
                monotypic_family_count = 0
            print(f"Class: {schema['label']:<10} | Rows: {len(df_diag):<5} | Unclassified Families: {total_unclassified_isolates:<3} | Monotypic Families (N=1): {monotypic_family_count}")
        else:
            print(f"Class: {schema['label']:<10} | Local file missing.")
    print("-"*60 + "\n", flush=True)

    master_results_log = []

    for schema in class_pipeline_schema:
        label = schema["label"]
        print(f"\n Running model for: {label.upper()}")
        print("-"*60)

        traits_path = os.path.join(data_dir, schema["traits"])
        tree_path = os.path.join(data_dir, schema["tree"])

        if not os.path.exists(traits_path) or not os.path.exists(tree_path):
            print(f" -> Skipping {label}: Data files missing.")
            continue

        df_traits = pd.read_csv(traits_path, encoding='utf-8-sig')
        df_traits.columns = df_traits.columns.str.strip().str.lower()
        df_traits['match_key'] = df_traits['species'].astype(str).str.strip().str.lower().str.replace(' ', '_')

        if 'risk' in df_traits.columns:
            df_traits = df_traits.drop(columns=['risk'])

        df = pd.merge(df_traits, df_threats_global[['match_key', 'risk']], on='match_key', how='inner')
        df['binary_target'] = df['risk'].astype(float)
        df = df.dropna(subset=['binary_target', 'habitat.tiering', 'motility', 'feeding.mode', 'family']).copy()

        # Parse the phylogenetic tree structures
        trees = list(Phylo.parse(tree_path, "nexus"))
        target_tree = trees[0]

        for clade in target_tree.find_clades():
            if clade.branch_length is None:
                clade.branch_length = 1.0

        tip_objects_map = {tip.name.strip().lower().replace(' ', '_'): tip for tip in target_tree.get_terminals() if tip.name}
        N = len(df)
        species_list = df['match_key'].tolist()

        print(f"Building phylogenetic covariance matrix ({N}x{N})...")
        from utils.bayesian_engine import build_phylogenetic_covariance
        phylo_cov = build_phylogenetic_covariance(
            tree_path=tree_path, species_list=species_list, df=df,
            family_col="family", id_col="match_key"
        )

        # add baseline small adjustment ridge
        phylo_cov = phylo_cov + np.eye(N) * 1e-4

        # group traits to find singletons and monotypic lineages
        df['functional_group'] = "h_" + df['habitat.tiering'].astype(str) + "_l_" + df['motility'].astype(str) + "_f_" + df['feeding.mode'].astype(str)
        df['group_codes'] = df['functional_group'].astype('category').cat.codes
        group_idx = df['group_codes'].to_numpy()
        num_unique_groups = df['functional_group'].nunique()

        code_to_group_name = df.drop_duplicates('group_codes').sort_values('group_codes')['functional_group'].tolist()
        global_group_counts = df['functional_group'].value_counts()
        singleton_mask = np.array([global_group_counts[name] == 1 for name in code_to_group_name], dtype=bool)
        strict_singletons = int(sum(singleton_mask))

        global_family_counts = df['family'].astype(str).str.strip().str.lower().value_counts()
        monotypic_mask = np.array([global_family_counts[str(f).strip().lower()] == 1 for f in df['family']], dtype=bool)
        strict_monotypic_isolates = int(sum(monotypic_mask))

        print(f"\n Dataset metrics for {label}:")
        print(f"  - Observations (N)              : {N}")
        print(f"  - Ecological singletons         : {strict_singletons}")
        print(f"  - Evolutionary isolates         : {strict_monotypic_isolates}")
        print("-"*45 + "\n", flush=True)

        y_vector = df['binary_target'].to_numpy().astype(int)
        X_design = np.ones((N, 1))

        # sample posterior distributions using PyMC engine
        trace_output = fit_trivariance_pglmm_gpu(
            y=y_vector,
            X=X_design,
            phylo_matrix=phylo_cov,
            group_factor_idx=group_idx,
            num_groups=num_unique_groups,
            singleton_mask=singleton_mask,
            monotypic_mask=monotypic_mask,
            class_label=label,
            chains=4
        )

        summary_df = summarize_trivariance_metrics(trace_output)
        print(summary_df)

        if "beta[0]" in summary_df.index:
            beta_key = "beta[0]"
        elif "beta" in summary_df.index:
            beta_key = "beta"
        else:
            beta_key = summary_df.index[0]

        master_results_log.append({
            "Vertebrate_Class": label,
            "Total_N_Obs": N,
            "Eco_Singletons": strict_singletons,
            "Evo_Isolates": strict_monotypic_isolates,
            "Mean_Beta_Intercept": summary_df.loc[beta_key, "mean"],
            "Mean_Sigma_Dense": summary_df.loc["sigma_dense", "mean"],
            "Mean_Sigma_Singleton": summary_df.loc["sigma_singleton", "mean"] if strict_singletons > 0 else np.nan,
            "Mean_Sigma_Phylo_Base": summary_df.loc["sigma_phylo_base", "mean"],
            "Mean_Sigma_Monotypic": summary_df.loc["sigma_monotypic", "mean"] if strict_monotypic_isolates > 0 else np.nan
        })

        del phylo_cov, trace_output, summary_df, df, df_traits
        gc.collect()
        jax.clear_caches()
        print(f" -> Finished processing {label}. Memory cleared.")

    print("\n Pipeline model estimation summary:")
    print("-"*100)
    df_global_summary = pd.DataFrame(master_results_log)
    print(df_global_summary.to_string(index=False))
    df_global_summary.to_csv(os.path.join("output", "global_bayesian_trivar_metrics.csv"), index=False)

    # check hypotheses from original study
    run_standalone_hypothesis_check(threats_path)

if __name__ == "__main__":
    execute_global_vertebrate_bayesian_pipeline()
