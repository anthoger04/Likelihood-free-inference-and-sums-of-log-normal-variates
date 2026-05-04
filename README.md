# Simulation and Monte Carlo: Likelihood-Free Inference

## Project Overview
This repository contains the Python implementation for the project on "Likelihood-Free Inference and Sums of Log-Normal Variates".

**Important Note regarding our approach:** 
Following the updated project guidelines, **the main approach we present for Questions 1 and 2 utilizes the new conjugate priors** ($\sigma^2 \sim \text{IG}(a, b)$ and $\mu|\sigma^2 \sim \mathcal{N}(0, \kappa\sigma^2)$).

However, we have also preserved our initial implementation using independent priors (Gaussian and Log-Normal) as a baseline. This allows us to empirically illustrate the structural inefficiency and severe prior bias of Likelihood-Free Inference when exploring non-conjugate, highly diffuse spaces.

For the Exact MCMC (Question 3), we utilize conditionally conjugate independent priors ($\mu \sim \mathcal{N}$ and $\sigma^2 \sim \text{IG}$). As demonstrated in our report, this independent structure perfectly yields the closed-form full conditionals required for the Gibbs sampler.

## Project Structure
All source code is located in the `src/` directory, as well as execution and analysis that are handled in the Jupyter Notebooks.

### 1. Main Presented Approach: Conjugate Priors (Updated Framework)
**This is our primary submission for Questions 1 & 2.**
* **`results_inv_gamma_prior.ipynb`**: **Main execution notebook.** Contains the empirical results and figures for the conjugate prior approach.
* **`src/q1_b.py`** (Question 1): Rejection-ABC algorithm using the conjugate priors.
* **`src/ABCMCMC_bis.py`** (Question 2): MCMC-ABC algorithm (Marjoram et al.) using the conjugate priors.

### 2. Alternative Approach: Independent Priors (Baseline Comparison)
Kept for pedagogical comparison to demonstrate ABC bias/variance trade-offs.
* **`results_normal_prior.ipynb`**: Execution notebook for the baseline experiments.
* **`src/q1.py`**: Rejection-ABC algorithm using the independent priors.
* **`src/ABCMCMC.py`**: MCMC-ABC algorithm using the independent priors.

### 3. Exact MCMC & Shared Utilities (Questions 3 & 4)
* **`src/MonteCarlo.py`**: Contains the Exact MCMC (Gibbs Sampler with Data Augmentation) targeting the true posterior using closed-form conditionals. It also evaluates the Monte Carlo standard error (MCSE).
* **`src/Figures.py`**: Helper functions for generating comprehensive MCMC diagnostics (Trace plots, ESS, ACF).

## Instructions for Use
To reproduce our primary results and figures, simply open and run all cells sequentially in `results_inv_gamma_prior.ipynb`. 

**Dependencies:**
* `numpy`
* `scipy`
* `matplotlib`
* `statsmodels`