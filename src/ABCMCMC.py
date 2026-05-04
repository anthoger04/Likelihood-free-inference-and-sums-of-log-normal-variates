import numpy as np
from q1 import simulate_lognormal_sum, wasserstein_1d


def find_warm_start(Y_obs, L, s, t, epsilon, rng, n_trials=500):
    """
    Cherche un warm start en échantillonnant depuis le prior.
    Si aucun point acceptable n'est trouvé, utilise une valeur de secours.
    
    Parameters
    ----------
    Y_obs : np.ndarray
        Données observées
    L : int
        Nombre de log-normales
    s, t : float
        Écart-types des priors normaux sur µ et log(σ²)
    epsilon : float
        Seuil de tolérance ABC
    rng : np.random.Generator
        Générateur aléatoire
    n_trials : int
        Nombre d'essais
        
    Returns
    -------
    best_theta : tuple (float, float)
        Meilleur (µ, log(σ²)) trouvé
    best_dist : float
        Distance correspondante
    """
    n_obs = len(Y_obs)  
    best_dist = np.inf

    for _ in range(n_trials):
        mu_cand = rng.normal(0, s)
        log_sigma2_cand = rng.normal(0, t)

        # Rejeter les valeurs extrêmes
        if log_sigma2_cand > 10 or log_sigma2_cand < -10:
            continue

        sigma_cand = np.sqrt(np.exp(log_sigma2_cand))
        Z_sim = simulate_lognormal_sum(n_obs, L, mu_cand, sigma_cand, rng=rng)
        dist = wasserstein_1d(Y_obs, Z_sim)

        if dist < best_dist:
            best_dist = dist
            best_theta = (mu_cand, log_sigma2_cand)
    
    
    if best_theta is None or best_dist > 2 * epsilon:
     
        best_theta = (0.0, -2.0)  # µ=0, log(σ²)=-2 → σ ≈ 0.37
        
        # Vérifier la distance avec cette valeur
        sigma_fallback = np.sqrt(np.exp(-2.0))
        Z_fallback = simulate_lognormal_sum(n_obs, L, 0.0, sigma_fallback, rng=rng)
        best_dist = wasserstein_1d(Y_obs, Z_fallback)
    
    return best_theta, best_dist


def ABCMCMC(Y_obs, L, s, t, epsilon, step_mu, step_log_sigma2, 
            num_samples=10000, rng=None):
    """
    ABC-MCMC avec priors normaux indépendants et fallback de sécurité.
    
    Priors : µ ~ N(0, s²) et log(σ²) ~ N(0, t²)
    
    Parameters
    ----------
    Y_obs : np.ndarray
        Données observées
    L : int
        Nombre de log-normales sommées
    s, t : float
        Écart-types des priors normaux
    epsilon : float
        Seuil ABC
    step_mu, step_log_sigma2 : float
        Tailles des pas pour Random Walk
    num_samples : int
        Nombre d'échantillons MCMC
    rng : np.random.Generator
        Générateur aléatoire
        
    Returns
    -------
    results : dict
        Contient 'chain', 'acc_rate', 'init_dist'
    """
    if rng is None:
        rng = np.random.default_rng()
    
    n_obs = len(Y_obs)  # ← CORRECTION : déduire de Y_obs
    
    # ═══════════════════════════════════════════════════════════════
    # Étape 1 : Warm start avec fallback
    # ═══════════════════════════════════════════════════════════════
    theta_initial, init_dist = find_warm_start(Y_obs, L, s, t, epsilon, rng, n_trials=1000)
    
    # ═══════════════════════════════════════════════════════════════
    # Étape 2 : MCMC
    # ═══════════════════════════════════════════════════════════════
    accepted = 0
    theta_step = theta_initial
    chain = np.zeros((num_samples, 2))

    for i in range(num_samples):
        mu, logsigma2 = theta_step
        
        # Proposition Random Walk
        mu_prime = mu + rng.normal(0, step_mu)
        logsigma_prime = logsigma2 + rng.normal(0, step_log_sigma2)  # ← CORRECTION typo
        
        # Simuler et calculer distance
        sigma_prime = np.sqrt(np.exp(logsigma_prime))
        Z_sim_prime = simulate_lognormal_sum(n_obs, L, mu_prime, sigma_prime, rng=rng)
        dist_prime = wasserstein_1d(Y_obs, Z_sim_prime)
        
        # Test ABC + Metropolis
        if dist_prime <= epsilon:
            # Prior ratio (priors normaux indépendants)
            log_ratio = (
                -(mu_prime**2 - mu**2) / (2 * s**2)
                -(logsigma_prime**2 - logsigma2**2) / (2 * t**2)
            )
            
            alpha = min(1, np.exp(log_ratio))
            U = rng.uniform(0, 1)
            
            if U <= alpha:
                accepted += 1
                theta_step = (mu_prime, logsigma_prime)
            else:
                theta_step = (mu, logsigma2)
        else:
            theta_step = (mu, logsigma2)
        
        chain[i] = [theta_step[0], theta_step[1]]
    
    acc_rate = accepted / num_samples
    
    return {
        'chain': chain,
        'acc_rate': acc_rate,
        'init_dist': init_dist
    }