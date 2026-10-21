"""
calculate_connectivity.py - Checks data clustering across the biology dataset layers.
It calculates spatial and structural network density metrics for vertebrate classes
using the pytstability library.
"""

import os
import pandas as pd
import numpy as np
from pytstability import calculate_densities

# Set up project directories and output path
data_dir = "data"
output_csv = os.path.join("output", "vertebrate_connectivity_summary.csv")
os.makedirs("output", exist_ok=True)

print("\nStarting connectivity analysis for vertebrate classes...")

# Define the dataset files and phylogenetic trees to process
target_classes = [
    {"name": "Amphibians", "csv": "amphibians.csv", "tree": "amphibian100trees.nex"},
    {"name": "Birds",      "csv": "birds.csv",      "tree": "bird100trees.nex"},
    {"name": "Mammals",    "csv": "mammals.csv",    "tree": "mammal100trees.nex"},
    {"name": "Reptiles",   "csv": "reptiles.csv",   "tree": "squamate100trees.nex"}
]

compiled_results = []
total_valid_spatial_species = 0
total_valid_structural_species = 0
running_weighted_spatial_sum = 0.0
running_weighted_structural_sum = 0.0

# Process each group sequentially
for target in target_classes:
    csv_path = os.path.join(data_dir, target["csv"])
    tree_path = os.path.join(data_dir, target["tree"])

    # Skip the group if required files are missing
    if not os.path.exists(csv_path) or not os.path.exists(tree_path):
        print(f"\nSkipping {target['name']}: Data files not found.")
        continue

    print(f"\nProcessing group: {target['name']}")
    df_class = pd.read_csv(csv_path)
    df_class.columns = df_class.columns.str.strip()  # Clean up column names
    sample_size = len(df_class)

    # Identify geographic coordinate columns case-insensitively
    lon_col = next((c for c in df_class.columns if c.lower() == 'longitude'), None)
    lat_col = next((c for c in df_class.columns if c.lower() == 'latitude'), None)
    coords = [lat_col, lon_col] if (lat_col and lon_col) else None

    # Identify structural taxonomic classification columns
    struct_list = [col for col in df_class.columns if col.lower() in ["order", "family"]]

    try:
        # Compute spatial and structural densities using standard baseline settings
        t_metrics = calculate_densities(
            data=df_class,
            id_col="Species",
            tree=tree_path,
            tree_type="adaptive",
            coord_cols=coords,
            spatial_cats=None,
            struct_cats=struct_list if struct_list else None,
            spatial_km=500.0,
            struct_depth=8,  # baseline threshold ratio
            verbose=False
        )

        k_spatial = float(t_metrics["spatial_metrics"].get("kappa", np.nan))
        k_structural = float(t_metrics["structural_metrics"].get("kappa", np.nan))
        tsi_current = float(t_metrics.get("topology_stability_index", np.nan))
        gamma_current = float(t_metrics.get("critical_horizon", np.nan))

    except Exception as e:
        print(f"  Warning: Could not compute densities for {target['name']}: {e}")
        k_spatial = np.nan
        k_structural = np.nan
        tsi_current = np.nan
        gamma_current = np.nan

    # Categorize the topological stability using the 0.0018 threshold boundary
    if not pd.isna(tsi_current) and tsi_current <= 0.0018:
        tier_status = "critical funnel zone"
    elif not pd.isna(tsi_current) and tsi_current <= 0.0500:
        tier_status = "moderate geometric shear"
    else:
        tier_status = "stable pooling backbone"

    # Store results for the current class
    compiled_results.append({
        "Class": target["name"],
        "Sample_Size_N": sample_size,
        "Spatial_Density": k_spatial if not pd.isna(k_spatial) else np.nan,
        "Structural_Density": k_structural if not pd.isna(k_structural) else np.nan,
        "Topology_Stability_Index": tsi_current if not pd.isna(tsi_current) else np.nan,
        "Critical_Horizon": gamma_current if not pd.isna(gamma_current) else np.nan,
        "Operational_Status": tier_status
    })

    # Accumulate data for overall sample averages
    if not pd.isna(k_spatial):
        total_valid_spatial_species += sample_size
        running_weighted_spatial_sum += (k_spatial * sample_size)

    if not pd.isna(k_structural):
        total_valid_structural_species += sample_size
        running_weighted_structural_sum += (k_structural * sample_size)

# Save the final data and print the summary report
print("  Analysis complete")

df_out = pd.DataFrame(compiled_results)
df_out.to_csv(output_csv, index=False)
print(f"Summary sheet saved to: {output_csv}\n")

print(df_out.to_string(index=False, formatters={
    "Spatial_Density": lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A",
    "Structural_Density": lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A",
    "Topology_Stability_Index": lambda x: f"{x:.5f}" if not pd.isna(x) else "N/A",
    "Critical_Horizon": lambda x: f"{x:.5f}" if not pd.isna(x) else "N/A"
}))
print("-" * 80)

# Print weighted global averages across all validated classes
if total_valid_spatial_species > 0:
    avg_spatial = running_weighted_spatial_sum / total_valid_spatial_species
    print(f"Overall dataset spatial density score    : {avg_spatial:.4f}")
else:
    print("Overall dataset spatial density score    : N/A")

if total_valid_structural_species > 0:
    avg_structural = running_weighted_structural_sum / total_valid_structural_species
    print(f"Overall dataset structural density score : {avg_structural:.4f}")
else:
    print("Overall dataset structural density score : N/A")

print("\n")
