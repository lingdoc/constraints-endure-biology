"""
generates figures mapping primary threat distributions, cumulative stressor
accumulations, and estimated variance components across vertebrate classes.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
    """
    plots the global count of impacted species across standard iucn threat categories.
    uses clear grayscale and colorblind safe tone adjustments for publication visibility.
    """
    print("Generating Figure 1: Primary Threat Rankings...")

    explicit_threat_cols = [
        'Development', 'Agriculture', 'Energy & Mining', 'Transportation',
        'Hunting & Trade', 'Logging', 'Fishing', 'Human disturbance', 'Fire',
        'Dams', 'Invasive species &d Disease', 'Pollution', 'Geological Events',
        'Climate change & severe weather', 'Other'
    ]

    df_clean_threats = df_global[explicit_threat_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    threat_counts = df_clean_threats.sum().sort_values(ascending=True)

    plt.figure(figsize=(10, 6))

    # okabe-ito colorblind safe: neutral dark gray background with vermilion accents for top threats
    colors = ['#56B4E9' if i < len(threat_counts)-3 else '#D55E00' for i in range(len(threat_counts))]

    ax = threat_counts.plot(kind='barh', color=colors, edgecolor='none')

    # add structural hatch textures to guarantee visibility in grayscale prints
    for idx, bar in enumerate(ax.patches):
        if idx >= len(threat_counts) - 3:
            bar.set_hatch('//') # distinct texture for critical top threat markers

    plt.xlabel('Number of Impacted Species')
    plt.ylabel('IUCN Threat Category')
    plt.title('Global Distribution of Anthropogenic Threat Pressures')
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig1_primary_threats.png"), dpi=300)
    plt.close()

def generate_multistressor_boxplot(df_global, output_dir):
    """
    plots boxplots comparing cumulative stressor counts between low and high risk pools.
    incorporates colorblind safe tones backed by geometric hatch textures.
    """
    print("Generating Figure 2: Multistressor Class Breakdown...")

    df_global['sum'] = pd.to_numeric(df_global['sum'], errors='coerce').fillna(0)
    df_global['NTVUENCR_num'] = pd.to_numeric(df_global['NTVUENCR'], errors='coerce')
    df_plot = df_global.dropna(subset=['NTVUENCR_num', 'CLASS']).copy()

    df_plot['Risk Group'] = df_plot['NTVUENCR_num'].map({0.0: 'Low Risk (0.0)', 1.0: 'High Risk (1.0)'})
    df_plot['CLASS'] = df_plot['CLASS'].str.capitalize()

    plt.figure(figsize=(10, 6))

    # okabe-ito safe palette definitions: sky blue vs vermilion orange
    color_palette = ['#56B4E9', '#D55E00']

    ax = sns.boxplot(
        data=df_plot,
        x='CLASS',
        y='sum',
        hue='Risk Group',
        palette=color_palette,
        fliersize=1,
        linewidth=1.2
    )

    # loop through boxes to inject distinct textures for color-independent evaluation
    box_patches = [patch for patch in ax.patches if patch.get_label() == '']
    for idx, patch in enumerate(box_patches):
        # alternating textures across low risk and high risk boxes
        if idx % 2 == 0:
            patch.set_hatch('..')
        else:
            patch.set_hatch('\\\\')

    plt.xlabel('Vertebrate Taxonomic Class')
    plt.ylabel('Cumulative Anthropogenic Stressors Count')
    plt.title('Multistressor Accumulation Profiles by Risk Pool')
    plt.legend(title='Extinction Risk Category')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig2_multistressor_loading.png"), dpi=300)
    plt.close()

def generate_variance_profile_chart(metrics_path, output_dir):
    """
    plots the full 4-partition variance components across vertebrate classes.
    uses distinct okabe-ito tones cross-matched with standalone line patterns.
    """
    if not os.path.exists(metrics_path):
        print(f"Skipping Figure 3: {metrics_path} not found. Run model entry script first.")
        return

    print("Generating Figure 3: Variance Component Profiles...")
    df_metrics = pd.read_csv(metrics_path)

    classes = df_metrics['Vertebrate_Class'].tolist()
    x = np.arange(len(classes))
    width = 0.20

    fig, ax = plt.subplots(figsize=(11, 6))

    # complete colorblind safe okabe-ito assignments with varying textures for structural contrast
    b1 = ax.bar(x - 1.5*width, df_metrics['Mean_Sigma_Dense'], width,
                label='Dense Community Background (σ_dense)', color='#0072B2', edgecolor='black', linewidth=0.5)

    b2 = ax.bar(x - 0.5*width, df_metrics['Mean_Sigma_Singleton'].fillna(0), width,
                label='Ecological Singletons (σ_singleton)', color='#56B4E9', edgecolor='black', linewidth=0.5, hatch='//')

    b3 = ax.bar(x + 0.5*width, df_metrics['Mean_Sigma_Phylo_Base'], width,
                label='Lineage Variance: Baseline Clades (σ_phylo_base)', color='#E69F00', edgecolor='black', linewidth=0.5, hatch='\\\\')

    b4 = ax.bar(x + 1.5*width, df_metrics['Mean_Sigma_Monotypic'], width,
                label='Evolutionary Isolates (σ_monotypic)', color='#CC79A7', edgecolor='black', linewidth=0.5, hatch='..')

    plt.xlabel('Vertebrate Lineage', labelpad=10)
    plt.ylabel('Estimated Variance Parameter Means')
    plt.title('Heteroskedastic Variance Component Partitions Across Domains', pad=12)

    plt.xticks(x, classes)
    plt.ylim(0, 4.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    plt.legend(frameon=False, fontsize=9, loc='upper left')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "fig3_variance_profiles.png"), dpi=300)
    plt.close()

def main():
    """coordinates the production graphics generation loop sequence."""
    threats_path = os.path.join("data", "all_sumthreats.csv")
    metrics_path = os.path.join("output", "global_bayesian_trivar_metrics.csv")
    figures_dir = os.path.join("output", "figures")

    os.makedirs(figures_dir, exist_ok=True)

    if not os.path.exists(threats_path):
        print(f"Error: Missing threat dataset at {threats_path}")
        return

    df_global = pd.read_csv(threats_path, low_memory=False)

    generate_primary_threat_chart(df_global, figures_dir)
    generate_multistressor_boxplot(df_global, figures_dir)
    generate_variance_profile_chart(metrics_path, figures_dir)
    print(f"All verification plots successfully saved to: {figures_dir}")

if __name__ == "__main__":
    main()
