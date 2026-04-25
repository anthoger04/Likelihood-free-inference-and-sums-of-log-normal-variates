

import numpy as np
from q1 import simulate_lognormal_sum, wasserstein_1d, reject_abc


n_obs = 200  

def ABCMCMC(Y_obs, L, s, t, epsilon, step_mu, step_log_sigma2, num_samples=100, max_attempts=10000, rng=None):
    """
    Tirer θ ~ p(θ)
Simuler x_sim ~ p(· | θ)
Calculer d = ρ(S(x_sim), S(x_obs))
Si d ≤ ε → bon point de départ, on lance la chaîne
Sinon → retourner en 1 et recommencer
    
    
    """
    # Step 1, let's sample mu and sigma
    # Sample candidate parameters from independent Gaussian priors
    
    dist=np.inf
    while not(dist <= epsilon):
        mu_star = rng.normal(0, s)
        log_sigma2_star = rng.normal(0, t)
        
        # Numerical stability safeguard against exponential overflow
        if log_sigma2_star > 10:
            continue
            
        sigma_star = np.sqrt(np.exp(log_sigma2_star))
        
        # Generate synthetic dataset
        Z_sim = simulate_lognormal_sum(n_obs, L, mu_star, sigma_star, rng=rng)
        
        # Evaluate the 1-Wasserstein discrepancy
        dist = wasserstein_1d(Y_obs, Z_sim)
        
        # Apply the acceptance kernel
    
        theta_initial=(mu_star, log_sigma2_star)

    #### Etape 2 : 
    accepted = 0

    theta_step=theta_initial
    chain = np.zeros((num_samples, 2))

    for i in range(num_samples):

        mu=theta_step[0]
        logsigma2=theta_step[1]

        ecart_mu = rng.normal(0, step_mu)
        ecart_sigma = rng.normal(0, step_log_sigma2)
    
        mu_prime = mu+ecart_mu
        logsima_prime=ecart_sigma+logsigma2
        sigma_prime = np.sqrt(np.exp(logsima_prime))

        Z_sim_prime = simulate_lognormal_sum(n_obs, L, mu_prime, sigma_prime, rng=rng)

        dist_prime = wasserstein_1d(Y_obs, Z_sim_prime)

        if dist_prime <=epsilon : 
            log_ratio = (-(mu_prime**2 - mu**2) / (2 * s**2)-(logsima_prime**2 - logsigma2**2) / (2 * t**2))
            alpha = min(1, np.exp(log_ratio))
            U=rng.uniform(0,1)
            if U <= alpha:
                accepted += 1
                theta_step = (mu_prime, logsima_prime)
            else:
                theta_step = (mu, logsigma2)  # ← ici
        else:
            theta_step = (mu, logsigma2)  

        chain[i] = [theta_step[0], theta_step[1]]
    acc_rate = accepted / num_samples

    return chain, acc_rate



