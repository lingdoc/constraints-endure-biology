# Constraints that endure: Assessing model robustness in vertebrate ecology

This repository contains the code to run Bayesian phylogenetic generalized linear mixed models (PGLMMs) evaluating global extinction risks in terrestrial vertebrates. The pipeline builds on data and phylogenies from Munstermann et al. (2022) to separate baseline ecological traits from macro-evolutionary structures across four vertebrate classes.

## Repository structure

```text
├── amphibian_density/                       # companion network isolation experiment
│   ├── output/                              # stability logs and plots
│   │   ├── node_variance.csv                # logs for all 500 species for each model
│   │   ├── parameter_transitions.png        # displays parameter trajectories
│   │   ├── phase_transitions.csv            # logs macro results for each model
│   │   ├── sampler_diagnostics.png          # plots model performance
│   │   └── spectrum_3panel.png              # plots global parameters for 3 models
│   ├── utils/                               
│   │   └── plot_spectrum.py                 # plot generator
│   └── density_scan.py                      # main experiment script
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
│   ├── vertebrate_connectivity_summary.csv  # density scores by vertebrate class
│   └── global_bayesian_trivar_metrics.csv   # metrics from main replication
├── utils/                                   # core utilities
│   ├── bayesian_engine.py                   # PyMC model configuration and tree parser
│   ├── check_hypotheses.py                  # verification test
│   └── generate_plots.py                    # figure generator
├── .gitignore                               
├── calculate_connectivity.py                # measure network density of datsets
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

## Companion experiment: Amphibian density scan
*Located in the `amphibian_density/` subdirectory.*

### What this experiment does
This self-contained sub-experiment tests the limits of a Bayesian statistical model by observing how it breaks down under extreme data conditions.

In nature, evolutionary trees typically provide a dense, connected map of information. However, other fields (like historical linguistics) deal with deeply fractured data environments containing unique isolates that lack group context. To understand how statistical models handle this fragmentation, the `density_scan.py` script takes a stable group of 500 amphibian species and **systematically breaks apart their evolutionary connections over 21 steps on a logarithmic scale**.

### Key findings
As data connectivity erodes from a natural unified structure into more disconnected points, the model undergoes three clear phases:
* **The Stable Phase:** At high connectivity, the model functions normally, sharing information across related species to keep estimate adjustments steady.
* **The High-Uncertainty Corridor:** As connections begin to snap, the model struggles to balance unique traits against the shrinking groups. This creates an inflation zone where error bars expand dramatically.
* **The Traction Collapse:** When connectivity drops near zero, the model hits a mechanical breaking point. It loses traction on the underlying network structure and triggers an artificial over-pooling illusion, falsely collapsing the error boundaries because it can no longer detect information variance.

![MCMC Phase Transitions](./amphibian_density/output/sampler_diagnostics.png)

![Macroecological Fragmentation](./amphibian_density/output/spectrum_3panel.png)

### Running the Sub-Experiment
To run the automated degradation loop and re-compile the color-coded 3-panel interval graphics, execute the following from the root directory:
```bash
python amphibian_density/density_scan.py
```

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
Install the required packages using the project requirements file. This process automatically installs the [`pykdensity` package](https://github.com/lingdoc/pykdensity) to handle the auditing of data connectivity:
```bash
pip install -r requirements.txt
```

## How to run the replication

### 1. Audit dataset connectivity (Recommended)
Run the diagnostic script to measure background spatial and structural data clustering before fitting any models. This script loops through all vertebrate classes, calculates individual density scores using the `pykdensity` package, and logs a weighted compound average score (κ) to the terminal:
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
