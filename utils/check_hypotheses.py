"""
check_hypotheses.py: Standalone test verifying anthropogenic multi-stressor
accumulation metrics using baseline statistical methods.
"""
import os
import pandas as pd
import scipy.stats as stats

def run_standalone_hypothesis_check(threats_path):
    print("\n Replicating multi-stressor findings (Munstermann et al.)")
    print("-"*80)

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

    print("\nTop 5 primary threat rankings:")
    threat_counts = df_clean_threats.sum().sort_values(ascending=False)
    for rank, (threat_name, count) in enumerate(threat_counts.head(5).items(), 1):
        display_name = threat_name.upper().replace('&D', '&').replace('_', ' ')
        print(f"  {rank}. {display_name:<35} : Impacts {int(count)} species")

    df_global['sum'] = pd.to_numeric(df_global['sum'], errors='coerce').fillna(0)
    df_global['NTVUENCR_num'] = pd.to_numeric(df_global['NTVUENCR'], errors='coerce')
    df_test = df_global.dropna(subset=['NTVUENCR_num']).copy()

    pool_low_risk = df_test[df_test['NTVUENCR_num'] == 0.0]
    pool_high_risk = df_test[df_test['NTVUENCR_num'] == 1.0]

    if len(pool_low_risk) > 0 and len(pool_high_risk) > 0:
        print(f"\nTesting multi-stressor accumulation hypothesis:")
        print(f"  - Low-risk pool median threats   : {pool_low_risk['sum'].median()}")
        print(f"  - High-risk pool median threats  : {pool_high_risk['sum'].median()}")

        u_stat, p_val = stats.mannwhitneyu(pool_low_risk['sum'], pool_high_risk['sum'], alternative='two-sided')
        print(f"  - Mann-Whitney U statistic       : {u_stat}")
        print(f"  - p-value                        : {p_val:.6e}")

        if p_val < 0.05:
            print("  Result: Significant cumulative threat accumulation confirmed (p < 0.05).")
        else:
            print("  Result: No significant difference in threat accumulation.")

        print("\nBreakdown by taxonomic class:")
        for cls in df_test['CLASS'].dropna().unique():
            df_cls = df_test[df_test['CLASS'] == cls]
            p_low = df_cls[df_cls['NTVUENCR_num'] == 0.0]
            p_high = df_cls[df_cls['NTVUENCR_num'] == 1.0]
            if len(p_low) > 0 and len(p_high) > 0:
                _, cls_p = stats.mannwhitneyu(p_low['sum'], p_high['sum'], alternative='two-sided')
                print(f"    * {str(cls):<12} -> Low Median: {p_low['sum'].median()} | High Median: {p_high['sum'].median()} | p-val: {cls_p:.4e}")
    else:
        print("\nSkipping hypothesis test: Insufficient risk category splits.")
    print("-"*80 + "\n")

if __name__ == "__main__":
    # Allows researchers to run it as a lightweight standalone command
    DEFAULT_PATH = os.path.join("data", "all_sumthreats.csv")
    run_standalone_hypothesis_check(DEFAULT_PATH)
