import numpy as np

def simulate_lognormal_sum(n, L, mu, sigma, rng=None):
    """
    Generates independent observations, where each observation is the sum 
    of independent and identically distributed log-normal random variables.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    X = rng.normal(loc=mu, scale=sigma, size=(n, L))
    Y = np.sum(np.exp(X), axis=1)
    return Y

def wasserstein_1d(Y_obs, Z_sim):
    """
    Computes the 1-Wasserstein distance between two univariate empirical measures 
    of equal size using their order statistics.
    """
    return np.mean(np.abs(np.sort(Y_obs) - np.sort(Z_sim)))

def reject_abc(Y_obs, L, a, b, kappa, epsilon, num_samples=100, max_attempts=10000, rng=None):
    """
    Executes the Rejection Approximate Bayesian Computation (ABC) algorithm 
    to approximate the joint posterior distribution of the parameters.
    
    Parameters:
    -----------
    Y_obs : numpy.ndarray
        A 1D array containing the observed dataset.
    L : int
        The structural parameter of the generative model (number of log-normals summed).
    a, b : float
        The shape and scale parameters for the Inverse-Gamma prior on sigma^2.
    kappa : float
        The scaling factor for the conditional Gaussian prior on mu (mu | sigma^2 ~ N(0, kappa * sigma^2)).
    epsilon : float
        The tolerance threshold defining the acceptance region for the Wasserstein distance.
    num_samples : int, default=100
        The target number of accepted samples required from the posterior approximation.
    max_attempts : int, default=10000
        A computational safety limit to prevent infinite execution.
    rng : numpy.random.Generator, optional
        An isolated random number generator instance to guarantee exact reproducibility.
        
    Returns:
    --------
    samples : numpy.ndarray
        A 2D array of shape (accepted_samples, 2) containing the accepted (mu, sigma) pairs.
    acceptance_rate : float
        The ratio of accepted samples to total simulated candidate parameters.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    n = len(Y_obs)
    accepted_params = []
    
    attempts = 0
    while len(accepted_params) < num_samples and attempts < max_attempts:
        attempts += 1
        
        # Sample candidate parameters from conjugate priors
        # 1. sigma^2 ~ Inverse-Gamma(a, b)
        # In numpy, standard_gamma generates Gamma(a, 1). To get IG(a, b), we use b / standard_gamma(a)
        sigma2_star = b / rng.standard_gamma(a)
        sigma_star = np.sqrt(sigma2_star)
        
        # 2. mu | sigma^2 ~ N(0, kappa * sigma^2)
        mu_star = rng.normal(0, np.sqrt(kappa * sigma2_star))
        
        # Generate synthetic dataset
        Z_sim = simulate_lognormal_sum(n, L, mu_star, sigma_star, rng=rng)
        
        # Evaluate the 1-Wasserstein discrepancy
        dist = wasserstein_1d(Y_obs, Z_sim)
        
        # Apply the acceptance kernel
        if dist <= epsilon:
            accepted_params.append([mu_star, sigma_star])
            
    # Convergence check
    if attempts == max_attempts and len(accepted_params) < num_samples:
        print(f"    [!] Computational limit reached ({max_attempts} iterations). Only {len(accepted_params)} samples accepted.")
        
    acceptance_rate = len(accepted_params) / attempts
    return np.array(accepted_params), acceptance_rate


if __name__ == "__main__":
    # STRICT REPRODUCIBILITY: Initialize an isolated random number generator
    seed_value = 42
    main_rng = np.random.default_rng(seed_value)
    
    L_true = 10
    mu_true = 0.0
    sigma_true = 0.3
    n_obs = 200  
    
    print("Generating strictly defined observational data...")
    Y_obs = simulate_lognormal_sum(n_obs, L_true, mu_true, sigma_true, rng=main_rng)
    
    epsilon_val = 0.5 
    
    # New configurations adapted for the Conjugate Prior (IG and Normal)
    prior_configs = [
        {"a": 3.0, "b": 0.2, "kappa": 5.0, "name": "Well-calibrated"},
        {"a": 0.1, "b": 0.1, "kappa": 100.0, "name": "Highly diffuse (Efficiency drop)"},
        {"a": 100.0, "b": 100.0, "kappa": 0.01, "name": "Highly concentrated (Prior bias risk)"}
    ]
    
    print(f"\n--- Rejection-ABC Empirical Results (Target: mu={mu_true}, sigma={sigma_true}) ---")
    for config in prior_configs:
        print(f"\nEvaluating '{config['name']}' prior specification (a={config['a']}, b={config['b']}, kappa={config['kappa']}):")
        
        # Pass the isolated generator down the execution pipeline
        samples, acc_rate = reject_abc(
            Y_obs, L_true, config['a'], config['b'], config['kappa'], epsilon_val, num_samples=50, rng=main_rng
        )
        
        print(f"  -> Acceptance Rate : {acc_rate:.4%}")
        
        if len(samples) > 0:
            mu_est = np.mean(samples[:, 0])
            sigma_est = np.mean(samples[:, 1])
            print(f"  -> Estimated mu    : {mu_est:.4f}")
            print(f"  -> Estimated sigma : {sigma_est:.4f}")
        else:
            print("  -> Estimated mu    : N/A (0 samples accepted)")
            print("  -> Estimated sigma : N/A (0 samples accepted)")