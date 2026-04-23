import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma, norm
import time

np.random.seed(42)

# Data Generation
L = 10
n = 200
mu_true = 0.0
sigma_true = 0.3

# Generate true latent variables
X_true = np.random.normal(mu_true, sigma_true, size=(n, L))
Y_obs = np.sum(np.exp(X_true), axis=1)

# Sort Y_obs for Wasserstein distance
Y_obs_sorted = np.sort(Y_obs)

# Priors
mu_prior_mean = 0.0
mu_prior_var = 1.0

# For sigma^2, we use Inverse Gamma(alpha, beta) for conjugacy
alpha_prior = 2.0
beta_prior = 0.2 # mean = beta / (alpha - 1) = 0.2 / 1 = 0.2, true var = 0.09

# MCMC Settings
n_iter = 10000
burnin = 2000
M_pairs = 5 # Number of pairs to update per observation per MCMC step
tau = 1.0 # Random walk standard deviation for w

# Initialize state
mu = 0.0
sigma2 = 0.1
X = np.log(Y_obs[:, None] / L) * np.ones((n, L))

mu_samples = np.zeros(n_iter)
sigma2_samples = np.zeros(n_iter)
acceptances = 0
proposals = 0

start_time = time.time()
for iter in range(n_iter):
    # Step (a): Sample theta | X
    # Sample mu
    post_var_mu = 1.0 / (1.0/mu_prior_var + n*L/sigma2)
    post_mean_mu = post_var_mu * (mu_prior_mean/mu_prior_var + np.sum(X)/sigma2)
    mu = np.random.normal(post_mean_mu, np.sqrt(post_var_mu))
    
    # Sample sigma^2
    alpha_post = alpha_prior + n*L/2.0
    beta_post = beta_prior + np.sum((X - mu)**2)/2.0
    sigma2 = invgamma.rvs(a=alpha_post, scale=beta_post)
    
    # Step (b): Sample blocks of two variables
    for i in range(n):
        for _ in range(M_pairs):
            # Pick two distinct indices
            l1, l2 = np.random.choice(L, 2, replace=False)
            
            # Current values
            x1 = X[i, l1]
            x2 = X[i, l2]
            S = np.exp(x1) + np.exp(x2)
            w = x1 - x2
            
            # Propose new w
            w_star = w + np.random.normal(0, tau)
            
            # Compute proposed x1*, x2*
            x1_star = np.log(S) + w_star - np.log(1 + np.exp(w_star))
            x2_star = np.log(S) - np.log(1 + np.exp(w_star))
            
            # Acceptance probability
            log_prob_current = - (x1 - mu)**2 / (2*sigma2) - (x2 - mu)**2 / (2*sigma2)
            log_prob_star = - (x1_star - mu)**2 / (2*sigma2) - (x2_star - mu)**2 / (2*sigma2)
            
            alpha = np.exp(log_prob_star - log_prob_current)
            proposals += 1
            if np.random.rand() < alpha:
                X[i, l1] = x1_star
                X[i, l2] = x2_star
                acceptances += 1
                
    mu_samples[iter] = mu
    sigma2_samples[iter] = sigma2
    
    if (iter+1) % 1000 == 0:
        print(f"Iter {iter+1}/{n_iter}, Acc rate: {acceptances/proposals:.3f}")
        acceptances = 0
        proposals = 0

print(f"MCMC Time: {time.time() - start_time:.2f} s")

mu_exact = np.mean(mu_samples[burnin:])
sigma_exact = np.mean(np.sqrt(sigma2_samples[burnin:]))
print(f"Exact MCMC Posterior Mean: mu={mu_exact:.4f}, sigma={sigma_exact:.4f}")

# ABC Rejection
def abc_rejection(epsilon, N_abc=10000):
    accepted_mu = []
    accepted_sigma = []
    for _ in range(N_abc):
        # Sample prior
        mu_star = np.random.normal(mu_prior_mean, np.sqrt(mu_prior_var))
        sigma2_star = invgamma.rvs(a=alpha_prior, scale=beta_prior)
        sigma_star = np.sqrt(sigma2_star)
        
        # Simulate data
        X_star = np.random.normal(mu_star, sigma_star, size=(n, L))
        Y_star = np.sum(np.exp(X_star), axis=1)
        
        # Distance
        dist = np.mean(np.abs(np.sort(Y_star) - Y_obs_sorted))
        
        if dist < epsilon:
            accepted_mu.append(mu_star)
            accepted_sigma.append(sigma_star)
            
    return np.mean(accepted_mu), np.mean(accepted_sigma), len(accepted_mu)/N_abc

epsilons = [2.0, 1.0, 0.5, 0.25]
print("Running ABC Rejection...")
print(f"True Params: mu={mu_true}, sigma={sigma_true}")
print(f"Gibbs Exact: mu={mu_exact:.4f}, sigma={sigma_exact:.4f}")
print("---")
for eps in epsilons:
    mean_mu, mean_sigma, acc_rate = abc_rejection(eps, 50000)
    bias_mu = mean_mu - mu_exact
    bias_sigma = mean_sigma - sigma_exact
    print(f"eps={eps}: Acc Rate={acc_rate*100:.3f}% | mu_mean={mean_mu:.4f} (bias={bias_mu:.4f}) | sigma_mean={mean_sigma:.4f} (bias={bias_sigma:.4f})")
    
# Save trace plots for MCMC
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(mu_samples, alpha=0.7)
plt.axhline(mu_true, color='r', linestyle='--')
plt.title(r'Trace plot of $\mu$')
plt.subplot(122)
plt.plot(np.sqrt(sigma2_samples), alpha=0.7)
plt.axhline(sigma_true, color='r', linestyle='--')
plt.title(r'Trace plot of $\sigma$')
plt.tight_layout()
plt.savefig('mcmc_traces.png')
