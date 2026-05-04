

import numpy as np
from q1 import simulate_lognormal_sum, wasserstein_1d, reject_abc


n_obs = 200  
def find_warm_start(Y_obs, L, a, b, kappa, epsilon, rng, n_trials=500):
    """
    Trouve un warm start en cherchant le meilleur point possible.
    Si aucun point acceptable n'est trouvé, utilise une valeur de secours raisonnable.
    """
    best_theta = None
    best_dist = np.inf

    for _ in range(n_trials):
        sigma2 = 1/rng.gamma(a, 1/b)
        mu_cand = rng.normal(0, np.sqrt(kappa*sigma2))
        log_sigma2_cand = np.log(sigma2)

        if log_sigma2_cand > 10 or log_sigma2_cand < -10:
            continue

        sigma_cand = np.sqrt(sigma2)
        Z_sim = simulate_lognormal_sum(n_obs, L, mu_cand, sigma_cand, rng=rng)
        dist = wasserstein_1d(Y_obs, Z_sim)

        if dist < best_dist:
            best_dist = dist
            best_theta = (mu_cand, log_sigma2_cand)
    
    # Si aucun point n'a été trouvé, utiliser une valeur de secours
    if best_theta is None or best_dist > 2*epsilon:  # Si vraiment trop loin
        print(f"  ⚠ Warm start échoué (best_dist={best_dist:.4f})")
        print(f"    → Utilisation valeur de secours: µ=0, log(σ²)=-2")
        best_theta = (0.0, -2.0)  # Valeur raisonnable proche de la vérité
        
        # Vérifier la distance avec cette valeur de secours
        sigma_fallback = np.exp(-2.0/2)  # σ = exp(log(σ²)/2)
        Z_fallback = simulate_lognormal_sum(n_obs, L, 0.0, sigma_fallback, rng=rng)
        best_dist = wasserstein_1d(Y_obs, Z_fallback)
        print(f"    → Distance avec valeur de secours: {best_dist:.4f}")
    
    elif best_dist <= epsilon:
        print(f"  ✓ Warm start valide: dist={best_dist:.4f} <= ε={epsilon}")
    else:
        print(f"  ⚠ Warm start: dist={best_dist:.4f} > ε={epsilon} (mais on continue)")
    
    return best_theta, best_dist

def ABCMCMC(Y_obs, L, a,b,kappa , epsilon, step_mu, step_log_sigma2, num_samples=100, max_attempts=10000, rng=None):
    """

    
    
    """
    # Step 1, let's sample mu and sigma
    
    
    theta_initial = find_warm_start(Y_obs, L, a, b, kappa, epsilon, rng, n_trials=1000)[0]



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
        logsigma_prime=ecart_sigma+logsigma2
        sigma_prime = np.exp(0.5*logsigma_prime) #on applique la racine 

        Z_sim_prime = simulate_lognormal_sum(n_obs, L, mu_prime, sigma_prime, rng=rng)

        dist_prime = wasserstein_1d(Y_obs, Z_sim_prime)

        if dist_prime <=epsilon : 
              # À partir de logsigma2
            sigma2_prime = np.exp(logsigma_prime)
            sigma2= np.exp(logsigma2)

            log_prior_mu = (mu**2 / (2*kappa*sigma2)) - (mu_prime**2 / (2*kappa*sigma2_prime))
            #                            = -(a+1)×log(σ²') - b/σ²' + (a+1)×log(σ²) + b/σ²

            log_prior_sigma2 = -(a+1)*logsigma_prime-b/sigma2_prime +(a+1)*logsigma2+b/sigma2

            log_jacobien=logsigma_prime-logsigma2

            log_ratio = log_prior_mu + log_prior_sigma2 + log_jacobien

            alpha = min(1, np.exp(log_ratio))
            U=rng.uniform(0,1)
            if U <= alpha:
                accepted += 1
                theta_step = (mu_prime, logsigma_prime)
            else:
                theta_step = (mu, logsigma2)  # ← ici
        else:
            theta_step = (mu, logsigma2)  

        chain[i] = [theta_step[0], theta_step[1]]
    acc_rate = accepted / num_samples

    return chain, acc_rate



