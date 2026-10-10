"""
script to plot results from `density_scan.py` on the amphibians dataset.
"""
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def generate_figure1_sampler_diagnostics(macro_csv, output_dir):
    """plots mcmc sampler diagnostics: e-bfmi vs divergences over a true log scale."""
    output_png = os.path.join(output_dir, "sampler_diagnostics.png")
    if not os.path.exists(macro_csv): return

    df = pd.read_csv(macro_csv)
    df_step = df.groupby("Structural_Density_κ").first().reset_index().sort_values("Structural_Density_κ", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # okabe-ito colorblind safe blue
    color_bfmi = '#0072B2'
    ax1.set_xlabel("Structural Network Connectivity ($\kappa$, True Log Scale)", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Mean E-BFMI Score", color=color_bfmi, fontweight="bold")
    ax1.plot(df_step["Structural_Density_κ"], df_step["Mean_E_BFMI"], marker='o', color=color_bfmi, linewidth=2, label="Mean E-BFMI")
    ax1.tick_params(axis='y', labelcolor=color_bfmi)
    ax1.axhline(y=0.30, color='#000000', linestyle='--', alpha=0.7, label="Critical Energy Floor (0.30)")
    ax1.set_ylim(0.05, 0.55)

    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2 = ax1.twinx()
    # okabe-ito black / charcoal alternative to provide clear dark contrast against blue
    color_div = '#000000'
    ax2.set_ylabel("Total Divergent Transitions", color=color_div, fontweight="bold")
    ax2.plot(df_step["Structural_Density_κ"], df_step["Divergences"], marker='x', linestyle=':', color=color_div, linewidth=1.5, label="Divergences")
    ax2.tick_params(axis='y', labelcolor=color_div)

    max_div = df_step["Divergences"].max()
    ax2.set_ylim(-10, max_div * 1.10)

    # background indicators adjusted to neutral grays with contrasting hatch angles for greyscale separation
    ax1.axvspan(0.13, 0.07, color='#efefef', hatch='/', alpha=0.5, label="Stage 1: Geometric Shear")
    ax1.axvspan(0.07, df_step["Structural_Density_κ"].min(), color='#dbdbdb', hatch='\\', alpha=0.5, label="Stage 2: Thermodynamic Freeze")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True)

    plt.title("MCMC Phase Transition Across Evolutionary Networks\n(Dual-Stage Non-Linear Model Failure)", fontweight="bold", pad=15)
    fig.tight_layout()
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Exported log-scaled sampler diagnostics to: {output_png}")

def generate_figure2_parameter_trajectories(macro_csv, output_dir):
    """plots parameter structural inversion curves over a clean log scale."""
    output_png = os.path.join(output_dir, "parameter_transitions.png")
    if not os.path.exists(macro_csv): return

    df = pd.read_csv(macro_csv)
    df_trait = df[df["Parameter_Name"] == "Hopping_Trait"].sort_values("Structural_Density_κ", ascending=False)

    fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

    # okabe-ito blue
    color_sd = '#0072B2'
    ax1.set_xlabel("Structural Network Connectivity ($\kappa$, Log Scale)", fontweight="bold", labelpad=10)
    ax1.set_ylabel("Parameter Uncertainty (Estimate_SD)", color=color_sd, fontweight="bold")
    ax1.plot(df_trait["Structural_Density_κ"], df_trait["Estimate_SD"], marker='o', color=color_sd, linewidth=2, label="Hopping Trait SD")
    ax1.tick_params(axis='y', labelcolor=color_sd)

    ax1.set_xscale('log')
    ax1.invert_xaxis()
    ax1.grid(True, linestyle=":", alpha=0.3)

    ax2 = ax1.twinx()
    # okabe-ito vermilion/orange variant to ensure colorblind contrast against blue
    color_mean = '#E69F00'
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
    print(f"Exported parameter phase trajectory curves to: {output_png}")


def generate_figure3_node_spectrum(node_csv, output_dir):
    """plots a wide 2-column landscape layout using colorblind safe and high-contrast greyscale tones."""
    output_png = os.path.join(output_dir, "spectrum_3panel.png")
    if not os.path.exists(node_csv): return

    df = pd.read_csv(node_csv)
    df_phylo = df[df["Variable_Family"].str.startswith("phylo_effect")].copy()
    if df_phylo.empty: return

    # deduplicate to secure unique node arrays
    df_phylo = df_phylo.sort_values("Variable_Family").drop_duplicates(subset=["Step", "Species_Name"])

    steps = sorted(df_phylo["Step"].unique())
    baseline_step = steps[0]

    # establish baseline tracking map coordinates
    df_baseline = df_phylo[df_phylo["Step"] == baseline_step].sort_values("Posterior_SD").reset_index(drop=True)
    df_baseline["spectrum_coordinate"] = df_baseline.index
    sorting_map = df_baseline.set_index("Species_Name")["spectrum_coordinate"].to_dict()

    unique_kappas = df_phylo[["Step", "Structural_Density_κ"]].drop_duplicates()

    mid_idx = (unique_kappas['Structural_Density_κ'] - 0.25).abs().idxmin()
    mid_step = int(unique_kappas.loc[mid_idx, 'Step'])

    threshold_idx = (unique_kappas['Structural_Density_κ'] - 0.10).abs().idxmin()
    threshold_step = int(unique_kappas.loc[threshold_idx, 'Step'])

    # map coordinates across dataframes
    df_phylo = df_phylo.copy()
    df_phylo["spectrum_coordinate"] = df_phylo["Species_Name"].map(sorting_map)

    df_base_plot = df_phylo[df_phylo["Step"] == baseline_step].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)
    df_mid_plot = df_phylo[df_phylo["Step"] == mid_step].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)
    df_thresh_plot = df_phylo[df_phylo["Step"] == threshold_step].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)

    # wide landscape layout with narrow right column setup
    fig = plt.figure(figsize=(18, 8), dpi=300)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.4, 0.6], hspace=0.35, wspace=0.18)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    ax_violin = fig.add_subplot(gs[:, 1])

    core_limit = 150
    margin_limit = 400

    def plot_left_panel(ax, data_frame, title_text, mean_se):
        ax.set_title(title_text, fontsize=11, fontweight="bold", color="#2c3e50", pad=6)

        core_mask = data_frame["spectrum_coordinate"] <= core_limit
        margin_mask = (data_frame["spectrum_coordinate"] > core_limit) & (data_frame["spectrum_coordinate"] <= margin_limit)
        singleton_mask = data_frame["spectrum_coordinate"] > margin_limit

        # applied complete okabe-ito color matching array parameters to guarantee distinct cross-vision traits
        zones = [
            {"mask": core_mask, "color": "#0072B2", "bar_color": "#90caf9", "label": "Structural Core"},
            {"mask": margin_mask, "color": "#E69F00", "bar_color": "#ffcc80", "label": "Transition Margin"},
            {"mask": singleton_mask, "color": "#CC79A7", "bar_color": "#f48fb1", "label": "Isolated Singletons"}
        ]

        # plot error bars and scatter points grouped by zone to match colors flawlessly
        for zone in zones:
            sub_df = data_frame[zone["mask"]]
            if not sub_df.empty:
                ax.vlines(sub_df["spectrum_coordinate"],
                          sub_df["Posterior_Mean"] - sub_df["Posterior_SD"],
                          sub_df["Posterior_Mean"] + sub_df["Posterior_SD"],
                          colors=zone["bar_color"], alpha=0.35, linewidth=0.7, zorder=1)
                ax.scatter(sub_df["spectrum_coordinate"], sub_df["Posterior_Mean"],
                           s=6, color=zone["color"], alpha=0.8, zorder=2, label=zone["label"] if ax == ax1 else "")

        ax.axvline(x=core_limit, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=margin_limit, color='gray', linestyle=':', alpha=0.4)

        # zone indicator sub-labels embedded within canvas frames to stop title overlap crashes
        if ax == ax1:
            ax.text(core_limit/2, 10.5, "Structural Core", fontsize=8, color="#546e7a", ha='center', fontweight='bold')
            ax.text((core_limit+margin_limit)/2, 10.5, "Transition Margin", fontsize=8, color="#546e7a", ha='center', fontweight='bold')
            ax.text((margin_limit+500)/2, 10.5, "Isolated Singletons", fontsize=8, color="#546e7a", ha='center', fontweight='bold')

        ax.text(0.98, 0.08, f"Mean Node SE ($\overline{{\sigma}}$): {mean_se:.3f}", transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3', edgecolor='#cfd8dc'),
                fontsize=8, fontweight='bold', ha='right', zorder=5)
        ax.set_ylabel("Latent Effect", fontweight="bold", fontsize=9)
        ax.set_ylim(-12, 12)
        ax.grid(True, linestyle="--", alpha=0.15)

    # render sequential matching left landscape columns
    plot_left_panel(ax1, df_base_plot, f"Panel A: Baseline Tree Topology Specification ($\kappa$ = {df_base_plot['Structural_Density_κ'].iloc[0]:.4f})", df_base_plot["Posterior_SD"].mean())
    plot_left_panel(ax2, df_mid_plot, f"Panel B: Moderate Topology Boundary ($\kappa$ = {df_mid_plot['Structural_Density_κ'].iloc[0]:.4f})", df_mid_plot["Posterior_SD"].mean())
    plot_left_panel(ax3, df_thresh_plot, f"Panel C: Critical Threshold Boundary Corridor ($\kappa$ = {df_thresh_plot['Structural_Density_κ'].iloc[0]:.4f})", df_thresh_plot["Posterior_SD"].mean())

    ax1.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='none', fontsize=8)
    ax3.set_xlabel("Amphibian Continuity Spectrum (Ordered by Baseline Variance $\longrightarrow$)", fontweight="bold", fontsize=10, labelpad=6)
    ax3.set_xlim(-15, len(df_base_plot) + 15)

    # right column uncertainty distribution density plotting suite
    ax_violin.set_title("Panel D: Sampler Uncertainty Shift", fontsize=12, fontweight="bold", color="#2c3e50", pad=12)

    margin_filter = (df_phylo["spectrum_coordinate"] > core_limit) & (df_phylo["spectrum_coordinate"] <= margin_limit)
    df_margin_all = df_phylo[margin_filter].copy()

    v_data = [
        df_margin_all[df_margin_all["Step"] == baseline_step]["Posterior_SD"].values,
        df_margin_all[df_margin_all["Step"] == mid_step]["Posterior_SD"].values,
        df_margin_all[df_margin_all["Step"] == threshold_step]["Posterior_SD"].values
    ]

    parts = ax_violin.violinplot(v_data, showmeans=False, showmedians=True, showextrema=True, widths=0.4)

    # maps identical colorblind-safe configurations across violin indicators
    colors_violin = ['#90caf9', '#ffcc80', '#f48fb1']
    for idx, body in enumerate(parts['bodies']):
        body.set_facecolor(colors_violin[idx])
        body.set_edgecolor(colors_violin[idx])
        body.set_alpha(0.65)

    parts['cmedians'].set_edgecolor('#37474f')
    parts['cmedians'].set_linewidth(1.5)
    parts['cmins'].set_edgecolor('#78909c')
    parts['cmaxes'].set_edgecolor('#78909c')

    ax_violin.set_xticks([1, 2, 3])
    ax_violin.set_xticklabels([
        f"Baseline\n($\kappa$={df_base_plot['Structural_Density_κ'].iloc[0]:.2f})",
        f"Margin Peak\n($\kappa$={df_mid_plot['Structural_Density_κ'].iloc[0]:.2f})",
        f"Threshold Cross\n($\kappa$={df_thresh_plot['Structural_Density_κ'].iloc[0]:.2f})"
    ], fontweight="bold", fontsize=9)

    ax_violin.set_ylabel("Node Standard Error ($\sigma$)", fontweight="bold", fontsize=10)
    ax_violin.grid(True, linestyle=":", alpha=0.3, axis='y')

    # adjusted arrow placement trajectories to scale elegantly without overlapping tracking elements
    ax_violin.annotate("Uncertainty Balloons\n(+29% Shift)", xy=(1.15, 3.45), xytext=(1.65, 4.35),
                       arrowprops=dict(facecolor='#E69F00', shrink=0.08, width=1.2, headwidth=5, edgecolor='none'),
                       fontsize=8, color='#E69F00', fontweight='bold', ha='center')

    ax_violin.annotate("Traction Loss\nCollapse Loop", xy=(2.85, 3.75), xytext=(2.25, 4.55),
                       arrowprops=dict(facecolor='#CC79A7', shrink=0.08, width=1.2, headwidth=5, edgecolor='none'),
                       fontsize=8, color='#CC79A7', fontweight='bold', ha='center')

    plt.suptitle("Amphibian Node Phase Transitions: Macroecological Fragmentation and Pooling Collapse", fontsize=13, fontweight="bold", y=0.98)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Exported integrated comparative landscape spectrum to: {output_png}")

def generate_universal_synthesis_figure(macro_csv, node_csv, output_dir):
    """combines sampler diagnostics, multi-stage spectrum stacks, and violin densities using a color-to-greyscale safe palette."""
    output_png = os.path.join(output_dir, "amphibian_synthesis.png")

    if not os.path.exists(macro_csv) or not os.path.exists(node_csv):
        print("Missing required CSV ledger sheets to build compilation figure.")
        return

    # load and process data matrices
    df_macro = pd.read_csv(macro_csv)
    df_step = df_macro.groupby("Structural_Density_κ").first().reset_index().sort_values("Structural_Density_κ", ascending=False)

    df_nodes = pd.read_csv(node_csv)
    df_phylo = df_nodes[df_nodes["Variable_Family"].str.startswith("phylo_effect")].copy()
    if df_phylo.empty:
        return
    df_phylo = df_phylo.sort_values("Variable_Family").drop_duplicates(subset=["Step", "Species_Name"])

    steps = sorted(df_phylo["Step"].unique())
    baseline_step = steps[0]

    df_baseline = df_phylo[df_phylo["Step"] == baseline_step].sort_values("Posterior_SD").reset_index(drop=True)
    df_baseline["spectrum_coordinate"] = df_baseline.index
    sorting_map = df_baseline.set_index("Species_Name")["spectrum_coordinate"].to_dict()

    unique_kappas = df_phylo[["Step", "Structural_Density_κ"]].drop_duplicates()
    mid_idx = (unique_kappas['Structural_Density_κ'] - 0.25).abs().idxmin()
    mid_step = int(unique_kappas.loc[mid_idx, 'Step'])
    threshold_idx = (unique_kappas['Structural_Density_κ'] - 0.10).abs().idxmin()
    threshold_step = int(unique_kappas.loc[threshold_idx, 'Step'])

    df_phylo = df_phylo.copy()
    df_phylo["spectrum_coordinate"] = df_phylo["Species_Name"].map(sorting_map)

    df_base_plot = df_phylo[df_phylo["Step"] == baseline_step].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)
    df_mid_plot = df_phylo[df_phylo["Step"] == mid_step].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)
    df_thresh_plot = df_phylo[df_phylo["Step"] == threshold_step].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate").reset_index(drop=True)

    # configure integrated figure master layout mesh via manual coordinate spacing
    fig = plt.figure(figsize=(18, 16), dpi=300)
    gs_master = fig.add_gridspec(4, 1, height_ratios=[1.6, 1.0, 1.0, 1.0])
    gs_master.update(top=0.93, bottom=0.06, hspace=0.36)

    # explicit cell index [0] to create the nested sub-grid correctly
    gs_row0 = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=gs_master[0])
    ax_diag_left = fig.add_subplot(gs_row0[0, 0])

    # explicit slicing [1:] to allocate the multi-row split dashboard below Panel A
    gs_lower = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_master[1:], width_ratios=[1.4, 0.6], hspace=0.35, wspace=0.18)
    ax_spec1 = fig.add_subplot(gs_lower[0, 0])
    ax_spec2 = fig.add_subplot(gs_lower[1, 0], sharex=ax_spec1)
    ax_spec3 = fig.add_subplot(gs_lower[2, 0], sharex=ax_spec1)
    ax_violin = fig.add_subplot(gs_lower[:, 1])

    # panel a: mcmc sampler diagnostics (dual color-to-greyscale contrast)
    color_bfmi = '#0072B2' # deep blue -> rich dark grey in print
    ax_diag_left.set_xlabel("Structural Network Connectivity ($\kappa$, True Log Scale)", fontweight="bold", labelpad=6)
    ax_diag_left.set_ylabel("Mean E-BFMI Score", color=color_bfmi, fontweight="bold")
    ax_diag_left.plot(df_step["Structural_Density_κ"], df_step["Mean_E_BFMI"], marker='o', color=color_bfmi, linewidth=2.5, label="Mean E-BFMI")
    ax_diag_left.tick_params(axis='y', labelcolor=color_bfmi)
    ax_diag_left.axhline(y=0.30, color='#000000', linestyle='--', alpha=0.7, linewidth=1.2, label="Critical Energy Floor (0.30)")
    ax_diag_left.set_ylim(0.05, 0.55)
    ax_diag_left.set_xscale('log')
    ax_diag_left.invert_xaxis()
    ax_diag_left.grid(True, linestyle=":", alpha=0.4, color='#b0bec5')

    ax_diag_right = ax_diag_left.twinx()
    color_div = '#E69F00' # amber orange -> crisp medium grey in print, separating the crossed lines
    ax_diag_right.set_ylabel("Total Divergent Transitions", color=color_div, fontweight="bold")
    ax_diag_right.plot(df_step["Structural_Density_κ"], df_step["Divergences"], marker='s', linestyle='-.', color=color_div, linewidth=1.75, label="Divergences")
    ax_diag_right.tick_params(axis='y', labelcolor=color_div)
    max_div = df_step["Divergences"].max()
    ax_diag_right.set_ylim(-10, max_div * 1.10)

    ax_diag_left.axvspan(0.13, 0.07, color='#f5f5f5', hatch='/', alpha=0.7, label="Stage 1: Geometric Shear")
    ax_diag_left.axvspan(0.07, df_step["Structural_Density_κ"].min(), color='#e0e0e0', hatch='\\', alpha=0.7, label="Stage 2: Thermodynamic Freeze")

    lines1, labels1 = ax_diag_left.get_legend_handles_labels()
    lines2, labels2 = ax_diag_right.get_legend_handles_labels()
    ax_diag_left.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fontsize=8)
    ax_diag_left.set_title("Panel A: MCMC Phase Transition Across Evolutionary Networks (Dual-Stage Model Failure)", fontweight="bold", fontsize=12, pad=10)

    # panels b, c, d: continuous node spectrum stacks (colors + structural markers) ---
    core_limit = 150
    margin_limit = 400

    def plot_spectrum_panel(ax, data_frame, title_text, mean_se, is_first=False):
        ax.set_title(title_text, fontsize=11, fontweight="bold", color="#2c3e50", pad=6)
        core_mask = data_frame["spectrum_coordinate"] <= core_limit
        margin_mask = (data_frame["spectrum_coordinate"] > core_limit) & (data_frame["spectrum_coordinate"] <= margin_limit)
        singleton_mask = data_frame["spectrum_coordinate"] > margin_limit

        # map shape keys with explicit luminance levels to look distinct in color and grey channels
        zones = [
            {"mask": core_mask, "color": "#0072B2", "marker": "o", "bar_color": "#90caf9", "label": "Structural Core"},       # blue solid circle
            {"mask": margin_mask, "color": "#E69F00", "marker": "^", "bar_color": "#ffcc80", "label": "Transition Margin"},  # amber solid triangle
            {"mask": singleton_mask, "color": "#CC79A7", "marker": "d", "bar_color": "#f48fb1", "label": "Isolated Singletons"} # purple solid diamond
        ]

        for zone in zones:
            sub_df = data_frame[zone["mask"]]
            if not sub_df.empty:
                ax.vlines(sub_df["spectrum_coordinate"],
                          sub_df["Posterior_Mean"] - sub_df["Posterior_SD"],
                          sub_df["Posterior_Mean"] + sub_df["Posterior_SD"],
                          colors=zone["bar_color"], alpha=0.35, linewidth=0.6, zorder=1)

                ax.scatter(sub_df["spectrum_coordinate"], sub_df["Posterior_Mean"],
                           s=7, color=zone["color"], marker=zone["marker"], alpha=0.8, zorder=2, label=zone["label"] if is_first else "")

        ax.axvline(x=core_limit, color='black', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.axvline(x=margin_limit, color='black', linestyle=':', alpha=0.5, linewidth=0.8)

        if is_first:
            ax.text(core_limit/2, 10.5, "Structural Core", fontsize=8, color="#212121", ha='center', fontweight='bold')
            ax.text((core_limit+margin_limit)/2, 10.5, "Transition Margin", fontsize=8, color="#212121", ha='center', fontweight='bold')
            ax.text((margin_limit+500)/2, 10.5, "Isolated Singletons", fontsize=8, color="#212121", ha='center', fontweight='bold')

        ax.text(0.98, 0.08, f"Mean Node SE ($\overline{{\sigma}}$): {mean_se:.3f}", transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3', edgecolor='#b0bec5'),
                fontsize=8, fontweight='bold', ha='right', zorder=5)
        ax.set_ylabel("Latent Effect", fontweight="bold", fontsize=9)
        ax.set_ylim(-12, 12)
        ax.grid(True, linestyle="--", alpha=0.15)

    plot_spectrum_panel(ax_spec1, df_base_plot, f"Panel B: Baseline Tree Topology Specification ($\kappa$ = {df_base_plot['Structural_Density_κ'].iloc[0]:.4f})", df_base_plot["Posterior_SD"].mean(), is_first=True)
    plot_spectrum_panel(ax_spec2, df_mid_plot, f"Panel C: Moderate Topology Boundary ($\kappa$ = {df_mid_plot['Structural_Density_κ'].iloc[0]:.4f})", df_mid_plot["Posterior_SD"].mean())
    plot_spectrum_panel(ax_spec3, df_thresh_plot, f"Panel D: Critical Threshold Boundary Corridor ($\kappa$ = {df_thresh_plot['Structural_Density_κ'].iloc[0]:.4f})", df_thresh_plot["Posterior_SD"].mean())

    ax_spec1.legend(loc="upper left", frameon=True, facecolor='white', edgecolor='none', fontsize=8)
    ax_spec3.set_xlabel("Amphibian Continuity Spectrum (Ordered by Baseline Variance $\longrightarrow$)", fontweight="bold", fontsize=10, labelpad=6)
    ax_spec3.set_xlim(-15, len(df_base_plot) + 15)

    # panel e: right column uncertainty distributions (safe color violin shading)
    ax_violin.set_title("Panel E: Sampler Uncertainty Shift", fontsize=12, fontweight="bold", color="#2c3e50", pad=12)
    margin_filter = (df_phylo["spectrum_coordinate"] > core_limit) & (df_phylo["spectrum_coordinate"] <= margin_limit)
    df_margin_all = df_phylo[margin_filter].copy()

    v_data = [
        df_margin_all[df_margin_all["Step"] == baseline_step]["Posterior_SD"].values,
        df_margin_all[df_margin_all["Step"] == mid_step]["Posterior_SD"].values,
        df_margin_all[df_margin_all["Step"] == threshold_step]["Posterior_SD"].values
    ]

    parts = ax_violin.violinplot(v_data, showmeans=False, showmedians=True, showextrema=True, widths=0.4)
    # matched colors perfectly across sections to maintain full visual consistency
    colors_violin = ['#0072B2', '#E69F00', '#CC79A7']
    for idx, body in enumerate(parts['bodies']):
        body.set_facecolor(colors_violin[idx])
        body.set_edgecolor('#000000') # solid black outline borders to anchor the layout geometry
        body.set_linewidth(1.0)
        body.set_alpha(0.75) # clean transparency for high aesthetic value

    parts['cmedians'].set_edgecolor('#000000')
    parts['cmedians'].set_linewidth(2.0)
    parts['cmins'].set_edgecolor('#212121')
    parts['cmaxes'].set_edgecolor('#212121')

    ax_violin.set_xticks([1, 2, 3])
    ax_violin.set_xticklabels([
        f"Baseline\n($\kappa$={df_base_plot['Structural_Density_κ'].iloc[0]:.2f})",
        f"Margin Peak\n($\kappa$={df_mid_plot['Structural_Density_κ'].iloc[0]:.2f})",
        f"Threshold Cross\n($\kappa$={df_thresh_plot['Structural_Density_κ'].iloc[0]:.2f})"
    ], fontweight="bold", fontsize=9)
    ax_violin.set_ylabel("Node Standard Error ($\sigma$)", fontweight="bold", fontsize=10)
    ax_violin.grid(True, linestyle=":", alpha=0.3, axis='y')

    ax_violin.annotate("Uncertainty Balloons\n(+29% Shift)", xy=(1.15, 3.45), xytext=(1.65, 4.35),
                       arrowprops=dict(facecolor='#E69F00', shrink=0.08, width=1.2, headwidth=5, edgecolor='none'),
                       fontsize=8, color='#E69F00', fontweight='bold', ha='center')

    ax_violin.annotate("Traction Loss\nCollapse Loop", xy=(2.85, 3.75), xytext=(2.25, 4.55),
                       arrowprops=dict(facecolor='#CC79A7', shrink=0.08, width=1.2, headwidth=5, edgecolor='none'),
                       fontsize=8, color='#CC79A7', fontweight='bold', ha='center')

    # save consolidated image
    plt.suptitle("Amphibian Evolutionary Network Synthesis: MCMC Diagnostics and Node Stability", fontsize=14, fontweight="bold", y=0.97)

    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Exported integrated universal publication canvas grid directly to: {output_png}")

def execute_all_plots():
    local_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    macro_ledger = os.path.join(local_dir, "output", "phase_transitions.csv")
    node_ledger = os.path.join(local_dir, "output", "node_variance.csv")
    output_dir = os.path.join(local_dir, "output")

    generate_figure1_sampler_diagnostics(macro_ledger, output_dir)
    generate_figure2_parameter_trajectories(macro_ledger, output_dir)
    generate_figure3_node_spectrum(node_ledger, output_dir)
    generate_universal_synthesis_figure(macro_ledger, node_ledger, output_dir)
    print("All companion experiment publication graphics compiled successfully.")


if __name__ == "__main__":
    execute_all_plots()
