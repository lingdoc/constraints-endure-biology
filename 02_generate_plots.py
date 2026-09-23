"""
02_generate_plots.py: Generates publication-ready figures replicating the primary
threat distributions and multi-stressor patterns from Munstermann et al. (2022).
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set clean academic plotting configurations
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

def generate_primary_threat_chart(df_global, output_dir):
    """Replicates Figure 2 from the original paper: Global Threat Rankings."""
    print("Generating Figure 1: Primary Threat Rankings...")

    explicit_threat_cols = [
        'Development', 'Agriculture', 'Energy & Mining', 'Transportation',
        'Hunting & Trade', 'Logging', 'Fishing', 'Human disturbance', 'Fire',
        'Dams', 'Invasive species &d Disease', 'Pollution', 'Geological Events',
        'Climate change & severe weather', 'Other'
    ]

    # Calculate impacts cleanly
    df_clean_threats = df_global[explicit_threat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    threat_counts = df_clean_threats.sum().sort_values(ascending=True) # Ascending for horizontal bar chart

    plt.figure(figsize=(10, 6))
    colors = ['#4A5568' if i < len(threat_counts)-3 else '#C53030' for i in range(len(threat_counts))]

    threat_counts.plot(kind='barh', color=colors, edgecolor='none')
    plt.xlabel('Number of Impacted Species')
    plt.ylabel('IUCN Threat Category')
    plt.title('Global Distribution of Anthropogenic Threat Pressures')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig1_primary_threats.png"), dpi=300)
    plt.close()

def generate_multistressor_violin(df_global, output_dir):
    """Replicates the cumulative threat loading split across vertebrate lineages."""
    print("Generating Figure 2: Multistressor Class Breakdown...")

    df_global['sum'] = pd.to_numeric(df_global['sum'], errors='coerce').fillna(0)
    df_global['NTVUENCR_num'] = pd.to_numeric(df_global['NTVUENCR'], errors='coerce')
    df_plot = df_global.dropna(subset=['NTVUENCR_num', 'CLASS']).copy()

    # Clean up labels for presentation
    df_plot['Risk Group'] = df_plot['NTVUENCR_num'].map({0.0: 'Low Risk (0.0)', 1.0: 'High Risk (1.0)'})
    df_plot['CLASS'] = df_plot['CLASS'].str.capitalize()

    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=df_plot,
        x='CLASS',
        y='sum',
        hue='Risk Group',
        palette=['#2B6CB0', '#C53030'],
        fliersize=1,
        linewidth=1.2
    )

    plt.xlabel('Vertebrate Taxonomic Class')
    plt.ylabel('Cumulative Anthropogenic Stressors Count')
    plt.title('Multistressor Accumulation Profiles by Risk Pool')
    plt.legend(title='Extinction Risk Category')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig2_multistressor_loading.png"), dpi=300)
    plt.close()

def generate_variance_profile_chart(metrics_path, output_dir):
    """Plots the estimated tri-variance lanes to show heteroskedastic divergence."""
    if not os.path.exists(metrics_path):
        print("Skipping Figure 3: global_bayesian_trivar_metrics.csv not found. Run 01_run_pipeline.py first.")
        return

    print("Generating Figure 3: Bayesian Variance Lane Profiles...")
    df_metrics = pd.read_csv(metrics_path)

    classes = df_metrics['Vertebrate_Class'].tolist()
    x = np.arange(len(classes))
    width = 0.25

    plt.figure(figsize=(10, 6))

    # Render partitioned variance bars side-by-side
    plt.bar(x - width, df_metrics['Mean_Sigma_Dense'], width, label='Dense Community Background (σ_dense)', color='#4A5568')
    plt.bar(x, df_metrics['Mean_Sigma_Singleton'].fillna(0), width, label='Ecological Singletons (σ_singleton)', color='#2B6CB0')
    plt.bar(x + width, df_metrics['Mean_Sigma_Monotypic'], width, label='Evolutionary Isolates (σ_monotypic)', color='#C53030')

    plt.xlabel('Vertebrate Lineage')
    plt.ylabel('Estimated Variance Parameter Means')
    plt.title('Heteroskedastic Variance Component Partitions Across Domains')
    plt.xticks(x, classes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig3_variance_profiles.png"), dpi=300)
    plt.close()

def main():
    threats_path = os.path.join("data", "all_sumthreats.csv")
    metrics_path = os.path.join("output", "global_bayesian_trivar_metrics.csv")
    output_dir = "output"

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(threats_path):
        print(f"Error: Data file missing at {threats_path}")
        return

    df_global = pd.read_csv(threats_path, low_memory=False)

    generate_primary_threat_chart(df_global, output_dir)
    generate_multistressor_violin(df_global, output_dir)
    generate_variance_profile_chart(metrics_path, output_dir)
    print("All figures successfully saved to the 'output/' directory.")

if __name__ == "__main__":
    main()
