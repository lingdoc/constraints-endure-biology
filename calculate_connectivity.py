"""
calculate_connectivity.py

this script checks data clustering across the biology dataset layers.
it uses the installed 'pykdensity' library to measure spatial and structural
connections, then calculates an average score for the whole dataset.
"""

import os
import pandas as pd
import numpy as np
from pykdensity import calculate_densities

# 0. project setup
# folders to read/write data
data_dir = "data"
output_csv = os.path.join("output", "vertebrate_connectivity_summary.csv")
os.makedirs("output", exist_ok=True)

print("Starting data connectivity analysis\n")

# define the data files matching the folders in this repository
target_classes = [
    {"name": "Amphibians", "csv": "amphibians.csv", "tree": "amphibian100trees.nex"},
    {"name": "Birds",      "csv": "birds.csv",      "tree": "bird100trees.nex"},
    {"name": "Mammals",    "csv": "mammals.csv",    "tree": "mammal100trees.nex"},
    {"name": "Reptiles",   "csv": "reptiles.csv",   "tree": "squamate100trees.nex"}
]

compiled_results = []
total_valid_species = 0
running_weighted_structural_sum = 0.0

# 1. Data files processing loop
for target in target_classes:
    csv_path = os.path.join(data_dir, target["csv"])
    tree_path = os.path.join(data_dir, target["tree"])

    if not os.path.exists(csv_path) or not os.path.exists(tree_path):
        print(f"\nskipping {target['name']}: file paths not found.")
        continue

    print(f"\nProcessing file: {target['name']}")
    df_class = pd.read_csv(csv_path)
    sample_size = len(df_class)

    # run the calculator function using standard family tree steps
    k_spatial, k_structural = calculate_densities(
        data=df_class,
        id_col="Species",
        tree=tree_path,
        tree_type="fixed",
        structural_depth_threshold=5
    )

    # store the findings for the final table
    compiled_results.append({
        "Class": target["name"],
        "Sample_Size_N": sample_size,
        "Spatial_Density": k_spatial if not pd.isna(k_spatial) else "N/A",
        "Structural_Density": k_structural if not pd.isna(k_structural) else "N/A"
    })

    # track the counts to make sure small datasets do not skew the final average
    if not pd.isna(k_structural):
        total_valid_species += sample_size
        running_weighted_structural_sum += (k_structural * sample_size)

# 2. Save and show summary logs
print("\n Connectivity check complete")
print("-" * 80)

df_out = pd.DataFrame(compiled_results)
df_out.to_csv(output_csv, index=False)
print(f" Summary file saved to: {output_csv}\n")

# print the final table directly to the screen
print(df_out.to_string(index=False))

if total_valid_species > 0:
    compound_score = running_weighted_structural_sum / total_valid_species
    print("-" * 80)
    print(f"Dataset average connectivity score (k): {compound_score:.4f}")
    print("-" * 80)

    # evaluate the score using the limits in our documentation guidelines
    if compound_score > 0.15:
        print("Result: High structural data clustering found.")
        print("Rule  : Ideal for structure-aware models (PGLMM).")
    elif compound_score <= 0.005:
        print("Result: Very sparse data connections found.")
        print("Rule  : high risk of parameter errors. Simpler flat models recommended.")
    else:
        print("Result: Balanced, stable connectivity signal.")
else:
    print("\nError: Could not calculate an average dataset score.")
print("\n")
