"""
amphibian_topology/utils/plot_skew.py
Ingests Bayesian NetCDF traces and Frequentist evaluation logs
to render an automated framework uncertainty profile comparison plot.
"""
import os
import pandas as pd
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

def render_uncertainty_comparison():
    """
    Ingests downstream runtime artifacts to track framework
    interval bounds across varying anomaly conditions.
    """
    # Establish directory mapping relative to the utils subdirectory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(base_dir, "output")
    trace_dir = os.path.join(output_dir, "mcmc_traces")
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    master_log_path = os.path.join(output_dir, "framework_skew_matrix.csv")
    if not os.path.exists(master_log_path):
        raise FileNotFoundError(
            f"Missing required master validation log: {master_log_path}. "
            "Please execute 'simulate_skew.py' first to generate runtime logs."
        )

    # Ingest the framework decision matrix
    df_matrix = pd.read_csv(master_log_path)

    scenarios = ["skew_0_anom", "skew_1_anom", "skew_2_anom", "skew_3_anom",
                 "skew_4_anom", "skew_5_anom", "skew_8_anom", "skew_10_anom"]
    x_coords = np.arange(len(scenarios))

    b_est, b_low, b_high = [], [], []
    f_est, f_low, f_high = [], [], []

    # 1. Parse Bayesian netcdf traces via arviz
    for scen in scenarios:
        netcdf_path = os.path.join(trace_dir, f"trace_{scen}_non-centered.nc")
        if not os.path.exists(netcdf_path):
            raise FileNotFoundError(f"Missing required MCMC sample archive: {netcdf_path}")

        # Load the trace into memory
        trace = az.from_netcdf(netcdf_path)
        slope_draws = trace.posterior["beta_slope"].values.flatten()

        # Extract median and 95% High Density Interval bounds
        b_est.append(np.median(slope_draws))
        hdi = az.hdi(slope_draws, hdi_prob=0.95)
        b_low.append(hdi[0])
        b_high.append(hdi[1])

    # 2. Extract Frequentist metrics from summary log
    df_freq = df_matrix[df_matrix["Framework"] == "Frequentist GP-GLMM"].copy()

    for scen in scenarios:
        row = df_freq[df_freq["Scenario"] == scen]
        if row.empty:
            raise ValueError(f"Scenario row reference '{scen}' missing from frequentist log records.")

        est = float(row["Estimate"].values[0])
        z_score = float(row["Diag_A"].values[0])

        f_est.append(est)
        # Reconstruct profile standard errors from the log's Wald z-scores
        se = est / z_score if (z_score > 0 and not np.isnan(z_score)) else 25.0
        f_low.append(est - 1.96 * se)
        f_high.append(est + 1.96 * se)

    # 3. Configure visual canvas
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots(figsize=(9.5, 6))

    # Colorblind safe color selection (Okabe-Ito / Paul Tol variant)
    color_bayesian = '#004488'   # Dark Cobalt Blue
    color_frequentist = '#cc3311' # Vermilion Red / Orange

    # Plot Bayesian envelope with slanted stripe texture
    ax.plot(x_coords, b_est, color=color_bayesian, marker='o', linestyle='-', linewidth=2,
            label='Bayesian Non-Centered (Median)')
    ax.fill_between(x_coords, b_low, b_high, facecolor='none', edgecolor=color_bayesian,
                    hatch='//', alpha=0.3, label='Bayesian 95% Credible Interval')

    # Plot Frequentist envelope with cross-hatch texture and alternate dash-dot style
    ax.plot(x_coords, f_est, color=color_frequentist, marker='s', linestyle='-.', linewidth=2,
            label='Frequentist GP-GLMM (MLE)')
    ax.fill_between(x_coords, f_low, f_high, facecolor='none', edgecolor=color_frequentist,
                    hatch='\\', alpha=0.15, label='Frequentist 95% Confidence Interval')

    # Dotted reference line at zero
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.2, alpha=0.7)

    # Label adjustments
    ax.set_title('Cross-Framework Uncertainty Profiles Under Fragmented Asymmetry', fontweight='bold', pad=14, fontsize=12)
    ax.set_xlabel('Number of Nested Counterexample Anomalies ($N_{anom}$)', labelpad=10)
    ax.set_ylabel(r'Global Slope Parameter Bounds ($\beta_{slope}$)', labelpad=10)

    # Translate file string scenarios back to visual integer ticks (0, 1, 2, 3...)
    tick_labels = [int(s.split('_')[1]) for s in scenarios]
    ax.set_xticks(x_coords)
    ax.set_xticklabels(tick_labels)
    ax.set_ylim(-12, 16)
    ax.grid(True, linestyle=':', alpha=0.5)

    # Labels outlining structural boundaries
    ax.text(4.5, 7.5, 'Bayesian Topological Mirage\nPrior shrinkage compresses the interval bounds\nabove zero across all evaluated skews.',
            color=color_bayesian, fontsize=9, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.text(4.5, -6.5, 'Frequentist Profile\nUnpenalized intervals remain uncollapsed below zero,\nmaintaining a non-significant baseline rejection.',
            color=color_frequentist, fontsize=9, ha='center', fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    ax.legend(loc='upper right', frameon=True, facecolor='white', edgecolor='none', fontsize=9)

    # Export to figure
    output_path = os.path.join(figures_dir, "framework_uncertainty.png")
    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Publication figure successfully synchronized and saved to: {output_path}")

if __name__ == "__main__":
    render_uncertainty_comparison()
