"""
amphibian_density/utils/plot_synthesis.py
Compiles Figure 1 for publication by combining macro-level topology transitions
with micro-level node uncertainty distributions for the two models.
Reviewed version plotting the dynamic mass-weighted max horizon frontier.
"""
import os
import io
import re
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set up project directories and output paths
utils_dir = os.path.dirname(os.path.abspath(__file__))
amphibian_dir = os.path.dirname(utils_dir)
output_dir = os.path.join(amphibian_dir, "output")

def render_publication_figure1():
    # Define paths for the macro and micro tracking files
    frozen_macro_path = os.path.join(output_dir, "pruning_transitions_frozen.csv")
    routed_macro_path = os.path.join(output_dir, "pruning_transitions_routed.csv")
    frozen_node_path = os.path.join(output_dir, "node_variance_frozen.csv")
    routed_node_path = os.path.join(output_dir, "node_variance_routed.csv")

    # Set up the output directory for figures
    plots_dir = os.path.join(output_dir, "figures")
    os.makedirs(plots_dir, exist_ok=True)

    final_output_png = os.path.join(plots_dir, "phase_synthesis.png")

    # Confirm all required data sheets exist before proceeding
    for path in [frozen_macro_path, routed_macro_path, frozen_node_path, routed_node_path]:
        if not os.path.exists(path):
            print(f"Error: Missing required tracking data at -> {path}")
            return

    print("Loading log data sheets...")
    df_macro_f = pd.read_csv(frozen_macro_path).sort_values("Step")
    df_macro_r = pd.read_csv(routed_macro_path).sort_values("Step")
    df_node_f = pd.read_csv(frozen_node_path)
    df_node_r = pd.read_csv(routed_node_path)

    # Map the metric stability values by step index for easy lookup
    tsi_map_frozen = df_macro_f.set_index("Step")["Topology_Stability_Index"].to_dict()
    tsi_map_routed = df_macro_r.set_index("Step")["Topology_Stability_Index"].to_dict()

    # Filter out missing values and isolate node trajectories
    def clean_node_frame(df):
        sub = df[df["Variable_Vector"].str.contains("phylo|effect|u|b", case=False, na=False)].copy()
        if sub.empty:
            sub = df.copy()
        return sub.sort_values("Variable_Vector").drop_duplicates(subset=["Step", "Species_Name"])

    df_node_f = clean_node_frame(df_node_f)
    df_node_r = clean_node_frame(df_node_r)

    # Establish uniform spectrum coordinates based on the baseline configuration
    steps_list = sorted(df_node_f["Step"].unique())
    baseline_step = steps_list[0]

    # Select step intervals to capture the network breakdown stages
    mid_step = 7
    deep_step = 20

    df_base_sort = df_node_f[df_node_f["Step"] == baseline_step].sort_values("Posterior_SD").reset_index(drop=True)
    df_base_sort["spectrum_coordinate"] = df_base_sort.index
    sorting_map = df_base_sort.set_index("Species_Name")["spectrum_coordinate"].to_dict()

    for df_target in [df_node_f, df_node_r]:
        df_target["spectrum_coordinate"] = df_target["Species_Name"].map(sorting_map)

    # Set up the master canvas dimensions and grid spacing
    fig = plt.figure(figsize=(20, 18), dpi=300)
    gs_master = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.0, 0.9])
    gs_master.update(top=0.92, bottom=0.05, hspace=0.24)

    # Color definitions using colorblind safe values
    c_tsi = '#0072B2'
    c_horizon = '#D55E00'
    c_f_div = '#CC79A7'
    c_r_div = '#009E73'
    c_bfmi = '#56B4E9'

    # Row 1: stability values and sampler errors
    gs_row1 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_master[0], wspace=0.22)
    ax_macro_f = fig.add_subplot(gs_row1[0, 0])
    ax_macro_r = fig.add_subplot(gs_row1[0, 1])

    def plot_macro_tier(ax, data, div_color, title_text, label_suffix, is_left_panel=True):
        # 1. Left Primary Axis: Topology Stability Index (TSI)
        if is_left_panel:
            ax.set_ylabel("Topology Stability Index (TSI, Log-Y)", color=c_tsi, fontweight="bold", labelpad=12)
            ax.tick_params(axis='y', labelcolor=c_tsi, pad=6)
        else:
            ax.set_ylabel("", color=c_tsi)
            ax.tick_params(axis='y', labelleft=False, labelcolor='none')

        l_tsi, = ax.plot(data["Step"], data["Topology_Stability_Index"], marker='o', color=c_tsi, linewidth=2.5, label="TSI Metric", zorder=6)

        # dynamic mass-weighted logarithmic ceiling curve
        l_horiz, = ax.plot(data["Step"], data["Critical_Horizon"], marker='s', linestyle='--', color=c_horizon, linewidth=2.0, label="Dynamic Danger Horizon ($\Gamma$)", zorder=5)

        ax.set_yscale('log')
        ax.set_ylim(0.0008, 0.60)
        ax.grid(True, linestyle=":", alpha=0.4, which="both")

        # Background hatching for Topological Stability Regimes
        # Regime 1: High-connectivity (Steps 0 -> 4) - Left blank

        # Regime 2: Intermediate-connectivity / Geometric Shear (Steps 4 -> 7)
        ax.axvspan(4, 7, facecolor='none', edgecolor='#3a3f42', hatch='//',
                   linewidth=0.5, alpha=0.35, zorder=1, label="Transitional Shear")

        # Regime 3: Low-connectivity / Fragmented Archipelago (Steps 7 -> 20)
        ax.axvspan(7, 20, facecolor='none', edgecolor='#b0bec5', hatch='\\\\\\',
                   linewidth=0.5, alpha=0.35, zorder=1, label="Fragmented")

        # 2. Right Axis 1: Total Divergent Transitions
        ax_div = ax.twinx()
        ax_div.set_ylabel(f"Total Divergent Transitions ({label_suffix})", color=div_color, fontweight="bold", labelpad=12)
        l_div, = ax_div.plot(data["Step"], data["Divergences"], marker='x', linestyle='-', color=div_color, linewidth=1.75, label="MCMC Divergences", zorder=6)
        ax_div.set_yscale('symlog', linthresh=1.0)
        ax_div.set_ylim(0, 1000.0)
        ax_div.spines['right'].set_bounds(0, 1000.0)
        ax_div.tick_params(axis='y', labelcolor=div_color, pad=6)

        # 3. Right Axis 2: Mean E-BFMI
        ax_bfmi = ax.twinx()
        l_bfmi, = ax_bfmi.plot(data["Step"], data["Mean_E_BFMI"], marker='s', linestyle=':', color=c_bfmi, linewidth=2.0, label="Mean E-BFMI", zorder=7)
        ax_bfmi.set_ylim(0.0, 1.0)

        # An E-BFMI below 0.30 is the standard indicator of poor sample mixing
        l_energy_floor = ax_bfmi.axhline(y=0.30, color=c_bfmi, linestyle=':', linewidth=1.5, label="Energy Limit (0.30)", zorder=3)

        if is_left_panel:
            ax_bfmi.spines['right'].set_position(('outward', 58))
            ax_bfmi.set_ylabel("Energy Bayesian Fraction of Missing Info (E-BFMI)", color=c_bfmi, fontweight="bold", labelpad=12)
            ax_bfmi.tick_params(axis='y', labelcolor=c_bfmi, pad=6)
        else:
            ax_bfmi.set_ylabel("", color=c_bfmi)
            ax_bfmi.spines['right'].set_visible(False)
            ax_bfmi.tick_params(axis='y', which='both', left=False, right=False, labelright=False)

        # Formatting constraints
        ax.tick_params(axis='x', which='major', pad=6)
        ax.set_xticks(data["Step"])
        ax.set_xlabel("Pruning Grid Step Index", fontweight="bold", labelpad=6)
        ax.set_title(title_text, fontweight="bold", fontsize=12, pad=12)

        # Update the legend parameters
        handles = [l_tsi, l_horiz, l_div, l_bfmi, l_energy_floor]
        ax.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=True, framealpha=0.9, fontsize=7)

        ax.set_box_aspect(0.35)
        ax_div.set_box_aspect(0.35)
        ax_bfmi.set_box_aspect(0.35)

    # Call functions explicitly passing structural layout indicators
    plot_macro_tier(ax_macro_f, df_macro_f, c_f_div, "Panel A: Frozen Architecture Phase Interface", "FROZEN", is_left_panel=True)
    plot_macro_tier(ax_macro_r, df_macro_r, c_r_div, "Panel B: Adaptive Routed Isolate Architecture Phase Interface", "ROUTED", is_left_panel=False)

    # Row 2: Node parameter spectrum grids
    gs_row2 = gridspec.GridSpecFromSubplotSpec(3, 2, subplot_spec=gs_master[1], hspace=0.35, wspace=0.18)

    core_limit, margin_limit = 150, 400
    z_colors = ["#0072B2", "#E69F00", "#CC79A7"]
    z_bars = ["#90caf9", "#ffcc80", "#f48fb1"]
    z_markers = ["o", "^", "d"]
    z_labels = ["Structural Core", "Transition Margin", "Isolated Singletons"]

    def plot_spectrum_cell(ax, node_df, step_idx, mode_label, is_left, tsi_lookup_dict):
        sub_df = node_df[node_df["Step"] == step_idx].dropna(subset=["spectrum_coordinate"]).sort_values("spectrum_coordinate")

        # Retrieve stability index value for the active step
        tsi_val = tsi_lookup_dict.get(step_idx, 0.0)

        # Standardize sub-panel letter designations
        panel_letter = "C" if step_idx == 0 else ("D" if step_idx == mid_step else "E")
        panel_suffix = "F" if "Frozen" in mode_label else "R"
        ax.set_title(f"Panel {panel_letter}_{panel_suffix}: {mode_label} Spectrum (Step {step_idx} | TSI = {tsi_val:.4f})", fontsize=9, fontweight="bold")

        c_mask = sub_df["spectrum_coordinate"] <= core_limit
        m_mask = (sub_df["spectrum_coordinate"] > core_limit) & (sub_df["spectrum_coordinate"] <= margin_limit)
        s_mask = sub_df["spectrum_coordinate"] > margin_limit
        masks = [c_mask, m_mask, s_mask]

        for i, mask in enumerate(masks):
            slice_df = sub_df[mask]
            if not slice_df.empty:
                ax.vlines(slice_df["spectrum_coordinate"], slice_df["Posterior_Mean"] - slice_df["Posterior_SD"],
                          slice_df["Posterior_Mean"] + slice_df["Posterior_SD"], colors=z_bars[i], alpha=0.25, linewidth=0.5)
                ax.scatter(slice_df["spectrum_coordinate"], slice_df["Posterior_Mean"], s=4, color=z_colors[i], marker=z_markers[i], alpha=0.7,
                           label=z_labels[i] if (step_idx == baseline_step and is_left) else "")

        ax.axvline(x=core_limit, color='gray', linestyle=':', alpha=0.4)
        ax.axvline(x=margin_limit, color='gray', linestyle=':', alpha=0.4)
        ax.set_ylim(-6.0, 6.0)
        ax.set_ylabel("Latent Effect", fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.15)

        # Extract the average standard deviation within the margin parameters
        overall_m_df = sub_df[(sub_df["spectrum_coordinate"] > core_limit) & (sub_df["spectrum_coordinate"] <= margin_limit)]
        mean_sd = overall_m_df["Posterior_SD"].mean() if not overall_m_df.empty else 0.0
        ax.text(0.98, 0.06, f"Margin Mean $\sigma$: {mean_sd:.3f}", transform=ax.transAxes, bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', edgecolor='#cfd8dc'), fontsize=7, ha='right')

    # Construct columns for comparison: Left side is Frozen, Right side is Routed
    steps_to_plot = [baseline_step, mid_step, deep_step]
    for row_idx, target_step in enumerate(steps_to_plot):
        ax_left = fig.add_subplot(gs_row2[row_idx, 0])
        ax_right = fig.add_subplot(gs_row2[row_idx, 1], sharex=ax_left, sharey=ax_left)

        plot_spectrum_cell(ax_left, df_node_f, target_step, "Frozen Frame", is_left=True, tsi_lookup_dict=tsi_map_frozen)
        plot_spectrum_cell(ax_right, df_node_r, target_step, "Routed Frame", is_left=False, tsi_lookup_dict=tsi_map_routed)

        if row_idx == 0:
            ax_left.legend(loc="upper left", fontsize=7, frameon=True)
        if row_idx == 2:
            ax_left.set_xlabel("Amphibian Continuity Alignment", fontweight="bold", fontsize=9)
            ax_right.set_xlabel("Amphibian Continuity Alignment", fontweight="bold", fontsize=9)

    # Row 3: Standard error distributions
    gs_row3 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs_master[2], wspace=0.22)
    ax_viol_f = fig.add_subplot(gs_row3[0, 0])
    ax_viol_r = fig.add_subplot(gs_row3[0, 1], sharey=ax_viol_f)

    def render_violin_suite(ax, node_df, title_text, bodies_color, hatch_pat):
        m_filter = (node_df["spectrum_coordinate"] > core_limit) & (node_df["spectrum_coordinate"] <= margin_limit)
        df_m = node_df[m_filter].copy()
        violin_data = [df_m[df_m["Step"] == s]["Posterior_SD"].values for s in steps_to_plot if not df_m[df_m["Step"] == s].empty]

        parts = ax.violinplot(violin_data, showmeans=False, showmedians=True, showextrema=True, widths=0.4)
        for idx, body in enumerate(parts['bodies']):
            body.set_facecolor(bodies_color)
            body.set_edgecolor('#212121')
            if hatch_pat:
                body.set_hatch(hatch_pat)
            body.set_alpha(0.70)

        parts['cmedians'].set_edgecolor('#000000')
        parts['cmedians'].set_linewidth(1.5)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels([f"Baseline\n(Step {baseline_step})",
                            f"Inflection Collapse\n(Step {mid_step})",
                            f"Deep Fragmented\n(Step {deep_step})"],
                           fontweight="bold", fontsize=8)
        ax.set_ylabel("Node Standard Error ($\sigma$)", fontweight="bold", fontsize=9)
        ax.set_title(title_text, fontweight="bold", fontsize=11, color="#2c3e50")
        ax.grid(True, linestyle=":", alpha=0.3, axis='y')

    render_violin_suite(ax_viol_f, df_node_f, "Panel F: Frozen Framework Uncertainty Inflation Profiles", c_f_div, "/")
    render_violin_suite(ax_viol_r, df_node_r, "Panel G: Routed Framework Isolate Uncertainty Track Flattening", c_r_div, None)
    ax_viol_f.set_ylim(0.5, 3.2)

    plt.suptitle("[Amphibian Network Paradigm: Naive Frozen Taxonomy vs Adaptive Isolate Sub-Graph Routing]",
                 fontsize=15, fontweight="bold", y=0.97)

    plt.savefig(final_output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Consolidated universal publication Figure 1 graphic compiled at: {final_output_png}")

if __name__ == "__main__":
    render_publication_figure1()
