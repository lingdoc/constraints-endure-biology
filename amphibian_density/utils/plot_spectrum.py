"""
Script to plot results from `density_scan.py` on the amphibians dataset.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_figure1_sampler_diagnostics(macro_csv, output_dir):
    """Plots MCMC Sampler Diagnostics: E-BFMI vs Divergences over a true Log Scale."""
    output_png = os.path.join(output_dir, "sampler_diagnostics.png")
    if not os.path.exists(macro_csv): return

    df = pd.read_csv(macro_csv)
    df_step = df.groupby("Structural_Density_κ").first().reset_index().sort_values("Structural_Density_κ", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    color_bfmi = '#1f77b4'
    ax1.set_xlabel("Structural Network Connectivity ($\kappa$, True Log Scale)", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Mean E-BFMI Score", color=color_bfmi, fontweight="bold")
    ax1.plot(df_step["Structural_Density_κ"], df_step["Mean_E_BFMI"], marker='o', color=color_bfmi, linewidth=2, label="Mean E-BFMI")
    ax1.tick_params(axis='y', labelcolor=color_bfmi)
    ax1.axhline(y=0.30, color='darkred', linestyle='--', alpha=0.7, label="Critical Energy Floor (0.30)")
    ax1.set_ylim(0.05, 0.55)

    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2 = ax1.twinx()
    color_div = '#d62728'
    ax2.set_ylabel("Total Divergent Transitions", color=color_div, fontweight="bold")
    ax2.plot(df_step["Structural_Density_κ"], df_step["Divergences"], marker='x', linestyle=':', color=color_div, linewidth=1.5, label="Divergences")
    ax2.tick_params(axis='y', labelcolor=color_div)

    max_div = df_step["Divergences"].max()
    ax2.set_ylim(-10, max_div * 1.10)

    ax1.axvspan(0.13, 0.07, color='orange', alpha=0.08, label="Stage 1: Geometric Shear")
    ax1.axvspan(0.07, df_step["Structural_Density_κ"].min(), color='red', alpha=0.08, label="Stage 2: Thermodynamic Freeze & Failure")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.title("MCMC Phase Transition Across Evolutionary Networks\n(Dual-Stage Non-Linear Model Failure)", fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" Figure 1: Log-scaled diagnostics exported to: {output_png}")

def generate_figure2_parameter_trajectories(macro_csv, output_dir):
    """Plots parameter structural inversion curves over a clean Log Scale."""
    output_png = os.path.join(output_dir, "parameter_transitions.png")
    if not os.path.exists(macro_csv): return

    df = pd.read_csv(macro_csv)
    df_trait = df[df["Parameter_Name"] == "Hopping_Trait"].sort_values("Structural_Density_κ", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    color_sd = '#1f77b4'
    ax1.set_xlabel("Structural Network Connectivity ($\kappa$, Log Scale)", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Parameter Uncertainty (Estimate_SD)", color=color_sd, fontweight="bold")
    ax1.plot(df_trait["Structural_Density_κ"], df_trait["Estimate_SD"], marker='o', color=color_sd, linewidth=2, label="Hopping Trait SD")
    ax1.tick_params(axis='y', labelcolor=color_sd)

    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2 = ax1.twinx()
    color_mean = '#2ca02c'
    ax2.set_ylabel("Effect Size (Estimate_Mean)", color=color_mean, fontweight="bold")
    ax2.plot(df_trait["Structural_Density_κ"], df_trait["Estimate_Mean"], marker='s', linestyle='--', color=color_mean, linewidth=1.5, label="Hopping Trait Mean")
    ax2.tick_params(axis='y', labelcolor=color_mean)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.title("Amphibian Phase Transitions: Parameter Trajectory vs Structural Decay", fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" Figure 2: Parameter trajectory exported to: {output_png}")


def generate_figure3_node_spectrum(node_csv, output_dir):
    """Plots micro 3-panel spectrum with high-contrast, color-matched error bars."""
    output_png = os.path.join(output_dir, "spectrum_3panel.png")
    if not os.path.exists(node_csv): return

    df = pd.read_csv(node_csv)
    df_phylo = df[df["Variable_Family"].str.startswith("phylo_effect")].copy()
    if df_phylo.empty: return

    # Deduplicate by species to secure unique node tracking arrays
    df_phylo = df_phylo.sort_values("Variable_Family").drop_duplicates(subset=["Step", "Species_Name"])

    steps = sorted(df_phylo["Step"].unique())
    baseline_step = steps[0]
    terminal_step = steps[-1]

    df_baseline = df_phylo[df_phylo["Step"] == baseline_step].sort_values("Posterior_SD").reset_index(drop=True)
    df_baseline["spectrum_coordinate"] = df_baseline.index
    sorting_map = df_baseline.set_index("Species_Name")["spectrum_coordinate"].to_dict()

    unique_kappas = df_phylo[["Step", "Structural_Density_κ"]].drop_duplicates()
    closest_idx = (unique_kappas['Structural_Density_κ'] - 0.25).abs().idxmin()
    mid_step = int(unique_kappas.loc[closest_idx, 'Step'])

    df_midpoint = df_phylo[df_phylo["Step"] == mid_step].copy()
    df_midpoint["spectrum_coordinate"] = df_midpoint["Species_Name"].map(sorting_map)
    df_midpoint = df_midpoint.dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)

    df_terminal = df_phylo[df_phylo["Step"] == terminal_step].copy()
    df_terminal["spectrum_coordinate"] = df_terminal["Species_Name"].map(sorting_map)
    df_terminal = df_terminal.dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)

    se_base = df_baseline["Posterior_SD"].mean()
    se_mid  = df_midpoint["Posterior_SD"].mean()
    se_term = df_terminal["Posterior_SD"].mean()

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True, dpi=300)
    plt.subplots_adjust(hspace=0.28)

    # Functional zone cuts
    core_limit = 150
    margin_limit = 400

    def plot_layer(ax, data_frame, title_text, mean_se):
        ax.set_title(title_text, fontsize=11, fontweight="bold", color="#2c3e50")

        # Split data by horizontal spectrum zones
        core_mask = data_frame["spectrum_coordinate"] <= core_limit
        margin_mask = (data_frame["spectrum_coordinate"] > core_limit) & (data_frame["spectrum_coordinate"] <= margin_limit)
        singleton_mask = data_frame["spectrum_coordinate"] > margin_limit

        # Color palettes mapping both scatter points and lines dynamically
        zones = [
            {"mask": core_mask, "color": "#1f77b4", "bar_color": "#90caf9", "label": "Structural Core"},
            {"mask": margin_mask, "color": "#f57c00", "bar_color": "#ffcc80", "label": "Transition Margin"},
            {"mask": singleton_mask, "color": "#e377c2", "bar_color": "#f48fb1", "label": "Isolated Singletons"}
        ]

        # Plot error bars and scatter points grouped by zone to match colors flawlessly
        for zone in zones:
            sub_df = data_frame[zone["mask"]]
            if not sub_df.empty:
                # 📍 FIXED: High-contrast, color-matched vertical interval bars
                ax.vlines(sub_df["spectrum_coordinate"],
                          sub_df["Posterior_Mean"] - sub_df["Posterior_SD"],
                          sub_df["Posterior_Mean"] + sub_df["Posterior_SD"],
                          colors=zone["bar_color"], alpha=0.5, linewidth=0.8, zorder=1)

                # Overlay scatter dots on top of the bars
                ax.scatter(sub_df["spectrum_coordinate"], sub_df["Posterior_Mean"],
                           s=8, color=zone["color"], alpha=0.8, zorder=2, label=zone["label"] if ax == ax1 else "")

        ax.axvline(x=core_limit, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=margin_limit, color='gray', linestyle=':', alpha=0.4)

        ax.text(0.97, 0.08, f"Mean Node SE ($\overline{{\sigma}}$): {mean_se:.3f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3'), fontsize=9, fontweight='bold', ha='right')
        ax.set_ylabel("Latent Effect", fontweight="bold", fontsize=9)
        ax.set_ylim(-12, 12)
        ax.grid(True, linestyle="--", alpha=0.2)

    # Render panels
    plot_layer(ax1, df_baseline, f"Panel A: Baseline Taxonomy Specification ($\kappa$ = {df_baseline['Structural_Density_κ'].iloc[0]:.4f})", se_base)
    plot_layer(ax2, df_midpoint, f"Panel B: Moderate Topology Boundary ($\kappa$ = {df_midpoint['Structural_Density_κ'].iloc[0]:.4f})", se_mid)
    plot_layer(ax3, df_terminal, f"Panel C: Eroded Density Specification ($\kappa$ = {df_terminal['Structural_Density_κ'].iloc[0]:.4f})", se_term)

    # Legend layout setup
    ax1.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='none', fontsize=9)

    ax1.text(core_limit/2, 10, "Structural Core\n(Low Baseline SE)", fontsize=8, color="#555", ha='center', fontweight='bold')
    ax1.text((core_limit+margin_limit)/2, 10, "Transition Margin\n(Moderate Network Support)", fontsize=8, color="#555", ha='center', fontweight='bold')
    ax1.text((margin_limit+500)/2, 10, "Isolated Singletons\n(High Baseline SE)", fontsize=8, color="#555", ha='center', fontweight='bold')

    ax3.set_xlabel("Amphibian Continuity Spectrum (Ordered by Baseline Variance $\longrightarrow$)", fontweight="bold", fontsize=10, labelpad=12)
    ax3.set_xlim(-15, len(df_baseline) + 15)

    plt.suptitle("Amphibian Node Phase Transitions: Macroecological Fragmentation and Pooling Collapse", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f" Figure 3: High-visibility interval plot successfully exported to: {output_png}")

def execute_all_plots():
    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    macro_ledger = os.path.join(local_dir, "output", "phase_transitions.csv")
    node_ledger = os.path.join(local_dir, "output", "node_variance.csv")
    output_dir = os.path.join(local_dir, "output")

    generate_figure1_sampler_diagnostics(macro_ledger, output_dir)
    generate_figure2_parameter_trajectories(macro_ledger, output_dir)
    generate_figure3_node_spectrum(node_ledger, output_dir)
    print(" All production graphics compiled successfully.")

if __name__ == "__main__":
    execute_all_plots()
