import numpy as np

def simulate_lognormal_sum(n, L, mu, sigma, rng=None):
    """
    Generates independent observations, where each observation is the sum 
    of independent and identically distributed log-normal random variables.
    (Generative model derived from Bernton et al., 2019, Section 4.2).
    
    Parameters:
    -----------
    n : int
        The number of independent observations to simulate.
    L : int
        The number of log-normal variables summed per observation.
    mu : float
        The location parameter of the underlying normal distribution.
    sigma : float
        The scale parameter of the underlying normal distribution.
    rng : numpy.random.Generator, optional
        An isolated random number generator instance for strict reproducibility.
        If None, falls back to a default, unseeded generator.
        
    Returns:
    --------
    Y : numpy.ndarray
        A 1D array of length 'n' containing the simulated sums.
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
    
    Parameters:
    -----------
    Y_obs : numpy.ndarray
        A 1D array containing the observed data points.
    Z_sim : numpy.ndarray
        A 1D array containing the synthetically generated data points.
        Must be of the same length as Y_obs.
        
    Returns:
    --------
    distance : float
        The computed 1-Wasserstein distance.
    """
    return np.mean(np.abs(np.sort(Y_obs) - np.sort(Z_sim)))

def reject_abc(Y_obs, L, s, t, epsilon, num_samples=100, max_attempts=10000, rng=None):
    """
    Executes the Rejection Approximate Bayesian Computation (ABC) algorithm 
    to approximate the joint posterior distribution of the parameters.
    
    Parameters:
    -----------
    Y_obs : numpy.ndarray
        A 1D array containing the observed dataset.
    L : int
        The structural parameter of the generative model (number of log-normals summed).
    s : float
        The standard deviation of the Gaussian prior centered at 0 for the location parameter (mu).
    t : float
        The standard deviation of the Gaussian prior centered at 0 for the log-variance (log sigma^2).
    epsilon : float
        The tolerance threshold defining the acceptance region for the Wasserstein distance.
    num_samples : int, default=100
        The target number of accepted samples required from the posterior approximation.
    max_attempts : int, default=10000
        A computational safety limit to prevent infinite execution when the 
        acceptance probability converges to zero.
    rng : numpy.random.Generator, optional
        An isolated random number generator instance to guarantee exact 
        reproducibility across independent execution environments.
        
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
        
        # Sample candidate parameters from independent Gaussian priors
        mu_star = rng.normal(0, s)
        log_sigma2_star = rng.normal(0, t)
        
        # Numerical stability safeguard against exponential overflow
        if log_sigma2_star > 10:
            continue
            
        sigma_star = np.sqrt(np.exp(log_sigma2_star))
        
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
    
    prior_configs = [
        {"s": 1.0, "t": 1.0, "name": "Well-calibrated"},
        {"s": 10.0, "t": 10.0, "name": "Highly diffuse (Efficiency drop)"},
        {"s": 0.01, "t": 0.01, "name": "Highly concentrated (Prior bias risk)"}
    ]
    
    print(f"\n--- Rejection-ABC Empirical Results (Target: mu={mu_true}, sigma={sigma_true}) ---")
    for config in prior_configs:
        print(f"\nEvaluating '{config['name']}' prior specification (s={config['s']}, t={config['t']}):")
        
        # Pass the isolated generator down the execution pipeline
        samples, acc_rate = reject_abc(
            Y_obs, L_true, config['s'], config['t'], epsilon_val, num_samples=50, rng=main_rng
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