

import numpy as np
from q1 import simulate_lognormal_sum, wasserstein_1d, reject_abc


n_obs = 200  
def find_warm_start(Y_obs, L, s, t, rng, n_trials=200):
    """
    Sample n_trials candidates from the prior, return the one
    minimising the Wasserstein distance. No epsilon constraint.
    """
    best_theta = (1,1)
    best_dist  = np.inf

    for _ in range(n_trials):
        mu_cand         = rng.normal(0, s)
        log_sigma2_cand = rng.normal(0, t)

        if log_sigma2_cand > 10:
            continue

        sigma_cand = np.sqrt(np.exp(log_sigma2_cand))
        Z_sim      = simulate_lognormal_sum(n_obs, L, mu_cand, sigma_cand, rng=rng)
        dist       = wasserstein_1d(Y_obs, Z_sim)

        if dist < best_dist:
            best_dist  = dist
            best_theta = (mu_cand, log_sigma2_cand)

    return best_theta, best_dist

def ABCMCMC(Y_obs, L, s, t, epsilon, step_mu, step_log_sigma2, num_samples=100, max_attempts=10000, rng=None):
    """

    
    
    """
    # Step 1, let's sample mu and sigma
    

    
    theta_initial = find_warm_start(Y_obs, L, s, t, rng, n_trials=200)[0]


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



