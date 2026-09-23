"""
00_verify_hypotheses.py: Standalone test verifying anthropogenic multi-stressor
accumulation metrics using baseline statistical methods.
"""
import os
import pandas as pd
import scipy.stats as stats

def run_standalone_hypothesis_check():
    print("=" * 80)
    print("RUNNING STANDALONE VERTEBRATE MULTISTRESSOR HYPOTHESIS VALIDATION")
    print("=" * 80)

    threats_path = os.path.join("data", "all_sumthreats.csv")

    if not os.path.exists(threats_path):
        print(f"Error: Raw threats file missing at {threats_path}")
        print("Please ensure your data directory contains 'all_sumthreats.csv'.")
        return

    # Ingest threat data
    df_global = pd.read_csv(threats_path, low_memory=False)

    explicit_threat_cols = [
        'Development', 'Agriculture', 'Energy & Mining', 'Transportation',
        'Hunting & Trade', 'Logging', 'Fishing', 'Human disturbance', 'Fire',
        'Dams', 'Invasive species &d Disease', 'Pollution', 'Geological Events',
        'Climate change & severe weather', 'Other'
    ]

    df_clean_threats = df_global[explicit_threat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    print("\nGlobal Primary Threat Rankings (Top 5):")
    threat_counts = df_clean_threats.sum().sort_values(ascending=False)
    for rank, (threat_name, count) in enumerate(threat_counts.head(5).items(), 1):
        print(f"  {rank}. {threat_name.upper():<35} : Impacts {int(count)} species")

    df_global['sum'] = pd.to_numeric(df_global['sum'], errors='coerce').fillna(0)
    df_global['NTVUENCR_num'] = pd.to_numeric(df_global['NTVUENCR'], errors='coerce')
    df_test = df_global.dropna(subset=['NTVUENCR_num']).copy()

    pool_low_risk = df_test[df_test['NTVUENCR_num'] == 0.0]
    pool_high_risk = df_test[df_test['NTVUENCR_num'] == 1.0]

    if len(pool_low_risk) > 0 and len(pool_high_risk) > 0:
        print(f"\nTesting Global Multistressor Accumulation Hypothesis:")
        print(f"  - Low-Risk Pool (0.0) Median Threats    : {pool_low_risk['sum'].median()}")
        print(f"  - High-Risk Pool (1.0) Median Threats   : {pool_high_risk['sum'].median()}")

        u_stat, p_val = stats.mannwhitneyu(pool_low_risk['sum'], pool_high_risk['sum'], alternative='two-sided')
        print(f"  - Mann-Whitney U Statistic             : {u_stat}")
        print(f"  - Calculated p-value                   : {p_val:.6e}")

        if p_val < 0.05:
            print("  STATUS: SUCCESS. High-risk pools exhibit higher stressor loading (p < 0.05).")
        else:
            print("  STATUS: FAIL. No significant difference in stressor loading detected.")

        print("\nClass-Specific Multistressor Breakdown:")
        for cls in df_test['CLASS'].dropna().unique():
            df_cls = df_test[df_test['CLASS'] == cls]
            p_low = df_cls[df_cls['NTVUENCR_num'] == 0.0]
            p_high = df_cls[df_cls['NTVUENCR_num'] == 1.0]
            if len(p_low) > 0 and len(p_high) > 0:
                _, cls_p = stats.mannwhitneyu(p_low['sum'], p_high['sum'], alternative='two-sided')
                print(f"    * {str(cls):<12} -> Low Median: {p_low['sum'].median()} | High Median: {p_high['sum'].median()} | p-val: {cls_p:.4e}")
    else:
        print("\nSkipping test: Low-risk and high-risk vectors could not be isolated.")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run_standalone_hypothesis_check()
