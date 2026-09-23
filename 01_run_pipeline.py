"""
01_run_pipeline.py: Master execution script running data integration, matrix
construction, and comparative Bayesian regressions across vertebrate classes.
"""
import os
import gc
import warnings
import pandas as pd
import numpy as np
from Bio import Phylo
import jax
import scipy.stats as stats

from bayesian_engine_hetero import fit_trivariance_pglmm_gpu, summarize_trivariance_metrics

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def replicate_anthropogenic_multistressor_findings(threats_path):
    print("\n" + "="*80)
    print("REPLICATING ANTHROPOGENIC MULTISTRESSOR FINDINGS (MUNSTERMANN ET AL.)")
    print("="*80)

    if not os.path.exists(threats_path):
        print(f"Error: File not found at {threats_path}")
        return

    df_global = pd.read_csv(threats_path, low_memory=False)

    explicit_threat_cols = [
        'Development', 'Agriculture', 'Energy & Mining', 'Transportation',
        'Hunting & Trade', 'Logging', 'Fishing', 'Human disturbance', 'Fire',
        'Dams', 'Invasive species &d Disease', 'Pollution', 'Geological Events',
        'Climate change & severe weather', 'Other'
    ]

    df_clean_threats = df_global[explicit_threat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    print("\n Global Primary Threat Rankings (Top 5):")
    threat_counts = df_clean_threats.sum().sort_values(ascending=False)
    for rank, (threat_name, count) in enumerate(threat_counts.head(5).items(), 1):
        print(f"  {rank}. {threat_name.upper():<35} : Impacts {int(count)} species")

    df_global['sum'] = pd.to_numeric(df_global['sum'], errors='coerce').fillna(0)
    df_global['NTVUENCR_num'] = pd.to_numeric(df_global['NTVUENCR'], errors='coerce')
    df_test = df_global.dropna(subset=['NTVUENCR_num']).copy()

    pool_low_risk = df_test[df_test['NTVUENCR_num'] == 0.0]
    pool_high_risk = df_test[df_test['NTVUENCR_num'] == 1.0]

    if len(pool_low_risk) > 0 and len(pool_high_risk) > 0:
        print(f"\n Testing Multistressor Accumulation Hypothesis:")
        print(f"  - Low-Risk Pool (0.0) Median Threats    : {pool_low_risk['sum'].median()}")
        print(f"  - High-Risk Pool (1.0) Median Threats   : {pool_high_risk['sum'].median()}")

        u_stat, p_val = stats.mannwhitneyu(pool_low_risk['sum'], pool_high_risk['sum'], alternative='two-sided')
        print(f"  - Mann-Whitney U Statistic             : {u_stat}")
        print(f"  - Calculated p-value                   : {p_val:.6e}")

        if p_val < 0.05:
            print("  STATUS: SUCCESS. High-risk species face a significantly higher number of cumulative threats (p < 0.05).")
        else:
            print("  STATUS: FAIL. No significant cumulative threat accumulation detected.")

        print("\n Class-Specific Multistressor Breakdown:")
        for cls in df_test['CLASS'].dropna().unique():
            df_cls = df_test[df_test['CLASS'] == cls]
            p_low = df_cls[df_cls['NTVUENCR_num'] == 0.0]
            p_high = df_cls[df_cls['NTVUENCR_num'] == 1.0]
            if len(p_low) > 0 and len(p_high) > 0:
                _, cls_p = stats.mannwhitneyu(p_low['sum'], p_high['sum'], alternative='two-sided')
                print(f"    * {str(cls):<12} -> Low Median: {p_low['sum'].median()} | High Median: {p_high['sum'].median()} | p-val: {cls_p:.4e}")
    else:
        print("\n Skipping hypothesis test: Target risk indicators could not be split safely.")
    print("="*80 + "\n")

def execute_global_vertebrate_bayesian_pipeline():
    print("=== INITIALIZING MODEL PIPELINE ===")

    data_dir = "data"
    threats_path = os.path.join(data_dir, "all_sumthreats.csv")

    if not os.path.exists(threats_path):
        print(f"ERROR: Threat file missing at {threats_path}")
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

    print("\n" + "="*60)
    print("DIAGNOSTIC REPORT: UNCLASSIFIED TAXONOMIC ISOLATES")
    print("="*60)
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
            print(f"Class: {schema['label']:<10} | Total Rows: {len(df_diag):<5} | Unclassified Families: {total_unclassified_isolates:<3} | Monotypic Families (N=1): {monotypic_family_count}")
        else:
            print(f"Class: {schema['label']:<10} | Local file missing.")
    print("="*60 + "\n", flush=True)

    master_results_log = []

    for schema in class_pipeline_schema:
        label = schema["label"]
        print(f"\n" + "="*60)
        print(f"ESTIMATING MODEL FOR VERTEBRATE CLASS: {label.upper()}")
        print("="*60)

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

        # Extract consensus phylogeny structure
        trees = list(Phylo.parse(tree_path, "nexus"))
        target_tree = trees[0]

        for clade in target_tree.find_clades():
            if clade.branch_length is None:
                clade.branch_length = 1.0

        tip_objects_map = {tip.name.strip().lower().replace(' ', '_'): tip for tip in target_tree.get_terminals() if tip.name}
        N = len(df)
        species_list = df['match_key'].tolist()

        root_distance_cache = {target_tree.root: 0.0}
        for parent_clade in target_tree.find_clades(order="preorder"):
            parent_dist = root_distance_cache[parent_clade]
            for child_clade in parent_clade.clades:
                root_distance_cache[child_clade] = parent_dist + child_clade.branch_length

        family_tree_cache = {}
        for _, row in df.iterrows():
            sp = row['match_key']
            fam = str(row['family']).strip().lower()
            if sp in tip_objects_map:
                if fam not in family_tree_cache:
                    family_tree_cache[fam] = []
                family_tree_cache[fam].append(tip_objects_map[sp])

        lineage_paths_cache = {}
        for idx, row in df.iterrows():
            sp = row['match_key']
            fam = str(row['family']).strip().lower()
            if sp in tip_objects_map:
                lineage_paths_cache[sp] = [target_tree.root] + target_tree.get_path(tip_objects_map[sp])
            elif fam in family_tree_cache and len(family_tree_cache[fam]) > 0:
                full_cousin_path = [target_tree.root] + target_tree.get_path(family_tree_cache[fam][0])
                family_node_depth = max(1, len(full_cousin_path) - 2)
                lineage_paths_cache[sp] = full_cousin_path[:family_node_depth]
            else:
                lineage_paths_cache[sp] = [target_tree.root]

        print(f"Calculating phylogenetic covariance matrix ({N}x{N})...")
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
            phylo_cov = (phylo_cov / max_variance) + np.eye(N) * 1e-4

        # Categorize indices for ecological singletons and evolutionary isolates
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

        print("\n" + "-"*45)
        print(f"SAMPLE DATA METRICS ({label.upper()}):")
        print(f"  - Observations (N)              : {N}")
        print(f"  - Ecological Singletons         : {strict_singletons}")
        print(f"  - Evolutionary Isolates         : {strict_monotypic_isolates}")
        print("-"*45 + "\n", flush=True)

        y_vector = df['binary_target'].to_numpy().astype(int)
        X_design = np.ones((N, 1))

        # Fit Bayesian regression graph via PyMC engine
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
        print(f" -> Completed analysis for {label}. System memory registers cleared.")

    print("\n" + "="*100)
    print("=== PIPELINE MODEL ESTIMATION SUMMARY ===")
    print("="*100)
    df_global_summary = pd.DataFrame(master_results_log)
    print(df_global_summary.to_string(index=False))
    df_global_summary.to_csv(os.path.join("output", "global_bayesian_trivar_metrics.csv"), index=False)
    replicate_anthropogenic_multistressor_findings(threats_path)

if __name__ == "__main__":
    execute_global_vertebrate_bayesian_pipeline()
