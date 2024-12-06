# Mutations and Methylation Aging
This repositiory contains code used for the manuscript: Koch et al. 2024: Somatic mutations as an explanation for epigenetic aging.

# Getting started
1. Clone the repository, e.g., `git clone https://github.com/zanekoch/MutationsAndMethylationAging.git`
2. Download the data by running `./download_data/download_internal.sh`. Optionally, download raw data from the TCGA and ICGC consortia by running following commands in `./download_data/download_external.sh` (this is necessary for the replication of some figures, but not others).
3. Create the conda environment by running `conda env create -f ./env/mutationsAndMethylationAging_noversions.yml` and activate it by running `conda activate mutationsAndMethylationAging`. Note: this environment does not include versions to maximize flexibility across systems, however the specific versions used in the analysis are listed in `mutationsAndMethylationAging_noversions.yml`.
4. Run the notebooks in the `./notebooks` directory to generate the figures.

# Environment
To create a conda environment with all dependencies, run:
- `conda env create -f ./env/mutationsAndMethylationAging.yml`

# Data
- Data to replicate the main and supplementary figures can be found on [figshare](link). `./download_data/download_internal.sh` contains commands for downloading this data.
- Data to re-run the methylation disturbance and clock training can be found on the respective consortia websites. In particular, the TCGA pan-cancer data can be accessed on [xena browser](https://xenabrowser.net/datapages/?cohort=TCGA%20Pan-Cancer%20(PANCAN)&removeHub=https%3A%2F%2Fxena.treehouse.gi.ucsc.edu%3A443) and the ICGC PCAWG data can be by following instructions [here](https://docs.icgc-argo.org/docs/data-access/icgc-25k-data#open-release-data---object-bucket-details). Examples of downloading these and other files can be found int `./download_data/download_external.sh`.

# Notebooks
Jupyter notebooks containing code for generating figures can be found in the `./notebooks` directory. Code is not provided for figures 1 and 3 as these were created using Biorender.
- `figure2.ipynb` contains code for generating Figure 2.
- `figure4.ipynb` contains code for generating Figure 4.
- `figure5.ipynb` contains code for generating Figure 5.
- `supplementary_figures.ipynb` contains code for generating Supplementary Figures.

# Source code
Source code used for calculating the methylation disturbance, training clocks, and plotting can be foudn in `./source`.
- `analysis.py` contains functions relevant to figure 2.
- `get_data.py` contains functions for loading and processing the TCGA and ICGC data.
- `mutation_clocks.py` contains functions for calculating the mutation burden clock and plotting.
- `utils.py` contains utility functions to, e.g., calculate mutual information, join mutation and methylation data, and quantile normalize.
- `compute_comethylation.py` contains functions for calculating the methylation disturbance.
- `methyl_and_mut_clocks.py` contains further functions for training the methylation and mutation clocks.

# Submission scripts
The `submission_scripts` directory contains submission scripts for running source code on a slurm HPC.
- `run_compute_comethylation.py` is a submission script for running `compute_comethylation.py`.
- `run_get_mean_metrics_comethylation.py` is a submission script for processing the methylation disturbance results.
