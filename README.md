# Constraints that endure: Assessing model robustness in vertebrate ecology

This repository contains the code to run Bayesian phylogenetic generalized linear mixed models (PGLMMs) evaluating global extinction risks in terrestrial vertebrates. The pipeline builds on data and phylogenies from Munstermann et al. (2022) to separate baseline ecological traits from macro-evolutionary structures across four vertebrate classes.

## Repository structure

```text
├── amphibian_topology/                      # network isolation and skew experiments
│   ├── output/                              # stability logs and plots
│   │   ├── figures/                          
│   │   │   ├── framework_uncertainty.png    # plot of model certainties
│   │   │   └── phase_synthesis.png          # plots macro curves and node errors
│   │   ├── mcmc_traces/                     # skew experiment traces for Bayesian models
│   │   ├── framework_skew_matrix.csv        # results from the skew experiment
│   │   ├── node_variance_frozen.csv         # logs node results for the frozen model
│   │   ├── node_variance_routed.csv         # logs node results for the routed model
│   │   ├── pruning_transitions_frozen.csv   # logs macro results for the frozen model
│   │   └── pruning_transitions_routed.csv   # logs macro results for the routed model
│   ├── utils/                               
│   │   ├── plot_skew.py                     # plot skew uncertainties
│   │   └── plot_synthesis.py                # plot pruning results
│   ├── prune_trees.py                       # tree pruning experiment script
│   └── prune_trees.py                       # skew simulation script
├── data/                                    # original trait tables and phylogenetic trees
│   ├── all_sumthreats.csv                   
│   ├── amphibian100trees.nex                
│   ├── amphibians.csv                       
│   ├── bird100trees.nex                     
│   ├── birds.csv                            
│   ├── mammal100trees.nex                   
│   ├── mammals.csv                          
│   ├── reptiles.csv                         
│   └── squamate100trees.nex                 
├── output/                                  # main replication outputs and results
│   ├── figures/                             # plots generated from metrics
│   │   ├── fig1_primary_threats.png         # assessment of threats to vertebrates
│   │   ├── fig2_multistressor_loading.png   # vertebrate multistressors
│   │   └── fig3_variance_profiles.png       # variance by category and class                     
│   ├── vertebrate_connectivity_summary.csv  # connectivity profiles by vertebrate class
│   └── global_bayesian_trivar_metrics.csv   # metrics from main replication
├── utils/                                   # core utilities
│   ├── bayesian_engine.py                   # PyMC model configuration and tree parser
│   ├── check_hypotheses.py                  # verification test
│   └── generate_plots.py                    # figure generator
├── .gitignore                               
├── calculate_connectivity.py                # measure topology of datasets via pytstability
├── README.md                                # project documentation
├── requirements.txt                         # pinned package dependencies
└── run_pipeline.py                          # main script
```

## Background & data sources

The original study examines how ecological traits shape macro-ecological vulnerability scales globally across terrestrial vertebrates using standard, homoskedastic frequentist phylogenetic mixed models.

* **Original Paper:** Munstermann, Maya J, Heim, Noel A, McCauley, Douglas J, Payne, Jonathan L, Upham, Nathan S, Wang, Steve C, & Knope, Matthew L (2022). A global ecological signal of extinction risk in terrestrial vertebrates. *Conservation Biology*, 36(3), e13852. https://doi.org/10.1111/cobi.13852.
* **Data Source:** Munstermann, Maya; Heim, Noel; McCauley, Douglas et al. (2021). *A global ecological signal of extinction risk in terrestrial vertebrates [Dataset].* Dryad Digital Repository. https://doi.org/10.5061/dryad.4b8gthtcq.

### Key differences in this version
This framework adjusts the uniform variance assumptions standard in frequentist models by introducing a heteroskedastic partitioning structure:

| Architectural Component | Original Study Engine (Frequentist PGLMM) | Replicated Pipeline Engine (PyMC / JAX) |
| :--- | :--- | :--- |
| **Statistical Framework** | Frequentist maximum likelihood estimation. | Full Bayesian MCMC joint variance parameter mapping. |
| **Ecological Variance** | Assumes equal variance across all terminal tips. | Isolates unique ecological singletons to prevent overall uncertainty inflation. |
| **Evolutionary Variance** | Fixed tree topology branch scales. | Models ancient, monotypic isolates via separate variance multipliers to protect baseline trees. |

## Results of the main study replication

### Estimated model parameters
The table below displays the estimated parameter means generated by the pipeline and saved to `output/global_bayesian_trivar_metrics.csv`:

| Vertebrate Class | Total N Obs | Eco Singletons | Evo Isolates | Mean Beta Intercept | Mean Sigma Dense | Mean Sigma Singleton | Mean Sigma Phylo Base | Mean Sigma Monotypic |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mammals** | 2,718 | 4 | 19 | -0.513 | 0.342 | 0.900 | 3.144 | 1.755 |
| **Birds** | 7,582 | 3 | 25 | -0.908 | 0.123 | 0.844 | 2.010 | 1.670 |
| **Squamates** | 4,185 | 1 | 10 |  0.982 | 0.286 | 0.956 | 3.499 | 2.691 |
| **Amphibians** | 3,531 | 0 | 7  | -0.402 | 0.237 |   N/A | 4.050 | 1.343 |

* `Mean_Sigma_Dense`: Residual variance for standard clustered ecological profiles.
* `Mean_Sigma_Singleton`: Residual variance assigned to hyper-rare, unique ecological profiles.
* `Mean_Sigma_Phylo_Base`: Baseline covariance scaling factor for standard tree topology.
* `Mean_Sigma_Monotypic`: Covariance scaling multiplier for ancient, taxonomically isolated lineages.

<details>
<summary><b>View expected console log output for the multi-stressor replication</b></summary>

```text
Replicating multi-stressor findings (Munstermann et al.)
--------------------------------------------------------------------------------

Top 5 primary threat rankings:
 1. AGRICULTURE                         : Impacts 4491 species
 2. LOGGING                             : Impacts 3187 species
 3. INVASIVE SPECIES & DISEASE          : Impacts 2053 species
 4. DEVELOPMENT                         : Impacts 1903 species
 5. HUNTING & TRADE                     : Impacts 1702 species

Testing multi-stressor accumulation hypothesis:
 - Low-risk pool median threats   : 2.0
 - High-risk pool median threats  : 2.0
 - Mann-Whitney U statistic       : 3002832.0
 - p-value                        : 1.770602e-16
 Result: Significant cumulative threat accumulation confirmed (p < 0.05).

Breakdown by taxonomic class:
   * REPTILIA     -> Low Median: 2.0 | High Median: 2.0 | p-val: 4.9173e-02
   * AVES         -> Low Median: 2.0 | High Median: 2.0 | p-val: 1.8607e-02
   * AMPHIBIA     -> Low Median: 2.0 | High Median: 3.0 | p-val: 1.1581e-09
   * MAMMALIA     -> Low Median: 2.0 | High Median: 2.0 | p-val: 3.2302e-01
--------------------------------------------------------------------------------
```
</details>

## Environment setup & dependencies

The core Bayesian models use JAX-accelerated sampling. While they can run on a CPU, an NVIDIA GPU is highly recommended for faster sampling times.

### Dependencies
Install the required packages using the project requirements file. This process automatically installs the [`pytstability` package](https://github.com/lingdoc/pytstability) to handle the auditing of data connectivity:
```bash
pip install -r requirements.txt
```

## How to run the replication

### 1. Audit dataset connectivity
Run the diagnostic script to check background spatial and structural data clustering across all classes before fitting models. This script uses the `pytstability` package to calculate structural density and TSI margins, logging the final class rankings to the terminal:
```bash
python calculate_connectivity.py
```

### 2. Fast-track hypothesis verification (Optional)
Run the verification script to check basic threat counts and run background ranking tests:
```bash
python utils/check_hypotheses.py
```

### 3. Fit the Bayesian model
Run the main pipeline entry script to clean the species tables, read the phylogenetic files, and execute the full tri-variance regression graphs:
```bash
python run_pipeline.py
```

### 4. Generate charts
Compile figures and variance profile charts:
```bash
python utils/generate_plots.py
```

## Companion experiments: Connectivity and skew diagnostics
*Located in the `amphibian_topology/` subdirectory.*

### 1. Lineage fragmentation (`prune_trees.py`)
This sub-experiment tracks how the core Bayesian engine responds as data connectivity degrades. In macroecology, phylogenies typically provide dense, highly connected paths of information sharing. However, other domains (such as historical linguistics) deal with deeply fractured data networks containing isolated singletons that lack group context.

To observe how this fragmentation impacts inference, `prune_trees.py` takes a stable baseline pool of 1,000 amphibian species and systematically slices their ancestral branches over 21 steps. It contrasts a standard tracking layout (**Frozen**) against an isolate-insulated parameter track (**Routed**), mapping the critical phase boundary where ancestral backbone degradation triggers a complete sampling traction collapse.

### 2. Matrix skew (`simulate_skew.py`)
This script evaluates the interaction between severe network fragmentation and extreme matrix asymmetry under a conditional implication hypothesis (X ⇒ Y) where any counterexample should logically invalidate the claim. Holding network connectivity fixed at a highly degraded regime (TSI ≤ Γ, extracted directly from Step 7 of the pruning sweep), this experiment systematically varies the number of nested counterexample anomalies across eight incremental steps (N_anom ∈ [0, 1, 2, 3, 4, 5, 8, 10]).

For each step, it runs a side-by-side benchmark comparing a Centered Bayesian model, a Non-Centered Bayesian model, and a frequentist profile maximum likelihood optimizer (GP-GLMM) to map out exactly where different model architectures succumb to parameter erasure. The consolidated results are automatically logged to `output/framework_skew_matrix.csv`.

### Key findings
* **The Topological Mirage:** When counterexamples are rare singletons nested within massive cladal structures dominated by the global background population (N_anom ≤ 4), both the Centered and Non-Centered Bayesian models succumb to the Topological Mirage. Hierarchical prior shrinkage gravity forces the group variance container (σ_family) to collapse toward zero, mathematically erasing the exceptions and manufacturing a highly confident, false-positive universal statement (95% CIs entirely above zero).
* **Profile Likelihood Resilience:** Conversely, the frequentist GP-GLMM engine successfully resists this shrinkage pressure. Operating via an unpenalized profile likelihood surface, its regional random slope coefficients absorb localized variance spikes. This shields the global fixed-effect slope parameter (β_slope) and preserves a mathematically accurate, non-significant baseline rejection (p ≥ 0.51).
* **Uncertainty Divergence Mapping:** These results are visualized via a script (`plot_skew.py`) that extracts these runtime artifacts directly from the NetCDF traces and framework summary logs. The chart (`output/figures/framework_uncertainty.png`) illustrates the divergence between Bayesian interval collapse and frequentist interval coverage.

### Running the diagnostic scripts
To run the tree-pruning loop, execute the asymmetry sweep, and compile the companion graphics, run the following commands sequentially from the project root directory:
```bash
# Run the continuous lineage pruning sweep
python amphibian_topology/prune_trees.py

# Generate the phase transition comparison
python amphibian_topology/utils/plot_synthesis.py

# Run the fine-grained data asymmetry matrix sweep
python amphibian_topology/simulate_skew.py

# Generate the uncertainty divergence figure
python amphibian_topology/utils/plot_skew.py
```
