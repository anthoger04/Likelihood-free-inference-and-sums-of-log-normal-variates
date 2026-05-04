import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


def effective_sample_size(chain_1d):
    """
    ESS = N / (1 + 2 * sum(autocorrelations))
    On tronque la somme quand l'ACF devient négative (méthode de Geyer).
    """
    n = len(chain_1d)
    acf_vals = acf(chain_1d, nlags=min(500, n // 2), fft=True)
    # Méthode de Geyer : on arrête à la première ACF négative
    cutoff = next((i for i, v in enumerate(acf_vals) if v < 0), len(acf_vals))
    ess = n / (1 + 2 * np.sum(acf_vals[1:cutoff]))
    return max(1.0, ess)


def plot_diagnostics(chain, param_names=None, burnin=0, true_values=None):
    """
    Diagnostics complets pour une chaîne ABC-MCMC.

    Paramètres
    ----------
    chain        : np.ndarray (num_samples, n_params)
    param_names  : liste de noms (ex. ['mu', 'log_sigma2'])
    burnin       : nombre d'itérations à ignorer
    true_values  : valeurs vraies (optionnel, pour référence visuelle)
    """
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(chain.shape[1])]

    chain_post = chain[burnin:]
    n_params = chain.shape[1]

    fig = plt.figure(figsize=(16, 4 * n_params))
    gs = gridspec.GridSpec(n_params, 4, figure=fig, hspace=0.5, wspace=0.4)

    for p in range(n_params):
        x = chain_post[:, p]
        name = param_names[p]
        tv = true_values[p] if true_values is not None else None
        ess = effective_sample_size(x)

        # ── 1. Trace plot ──────────────────────────────────────────────────
        ax_trace = fig.add_subplot(gs[p, 0])
        ax_trace.plot(x, lw=0.6, alpha=0.8, color="#4C72B0")
        if tv is not None:
            ax_trace.axhline(tv, color="crimson", lw=1.2, ls="--", label=f"vrai = {tv}")
            ax_trace.legend(fontsize=8)
        ax_trace.set_title(f"Trace — {name}", fontsize=10)
        ax_trace.set_xlabel("Itération")
        ax_trace.set_ylabel(name)

        # ── 2. Histogramme de la postérieure ──────────────────────────────
        ax_hist = fig.add_subplot(gs[p, 1])
        ax_hist.hist(x, bins=40, color="#4C72B0", alpha=0.75, edgecolor="white", lw=0.3)
        ax_hist.axvline(np.mean(x), color="orange", lw=1.5, ls="-",  label=f"moyenne = {np.mean(x):.3f}")
        ax_hist.axvline(np.median(x), color="green",  lw=1.5, ls="--", label=f"médiane = {np.median(x):.3f}")
        if tv is not None:
            ax_hist.axvline(tv, color="crimson", lw=1.5, ls=":", label=f"vrai = {tv}")
        ax_hist.set_title(f"Postérieure — {name}", fontsize=10)
        ax_hist.set_xlabel(name)
        ax_hist.legend(fontsize=7)

        # ── 3. Autocorrélation ────────────────────────────────────────────
        ax_acf = fig.add_subplot(gs[p, 2])
        plot_acf(x, ax=ax_acf, lags=60, alpha=0.05, color="#4C72B0", zero=False)
        ax_acf.set_title(f"ACF — {name}\nESS = {ess:.0f}", fontsize=10)
        ax_acf.set_xlabel("Lag")
        ax_acf.set_ylim(-0.3, 1.05)

        # ── 4. Running mean (convergence) ─────────────────────────────────
        ax_run = fig.add_subplot(gs[p, 3])
        running_mean = np.cumsum(x) / np.arange(1, len(x) + 1)
        ax_run.plot(running_mean, lw=1, color="#4C72B0")
        if tv is not None:
            ax_run.axhline(tv, color="crimson", lw=1.2, ls="--", label=f"vrai = {tv}")
            ax_run.legend(fontsize=8)
        ax_run.set_title(f"Moyenne cumulée — {name}", fontsize=10)
        ax_run.set_xlabel("Itération")
        ax_run.set_ylabel(f"E[{name}]")

    fig.suptitle(
        f"Diagnostics ABC-MCMC  |  burn-in = {burnin}  |  échantillons retenus = {len(chain_post)}",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.savefig("diagnostics_abcmcmc.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Figure sauvegardée : diagnostics_abcmcmc.png")


def print_summary(chain, param_names=None, burnin=0, true_values=None):
    """Affiche ESS, moyenne, écart-type, IC 95% pour chaque paramètre."""
    if param_names is None:
        param_names = [f"theta_{i}" for i in range(chain.shape[1])]
    chain_post = chain[burnin:]

    print(f"\n{'─'*60}")
    print(f"{'Paramètre':<15} {'Moyenne':>8} {'Std':>8} {'IC 95% bas':>11} {'IC 95% haut':>12} {'ESS':>6}")
    print(f"{'─'*60}")
    for p, name in enumerate(param_names):
        x = chain_post[:, p]
        ess = effective_sample_size(x)
        
        lo, hi = np.percentile(x, [2.5, 97.5])
        tv_str = f"  (vrai = {true_values[p]:.3f})" if true_values is not None else ""
        print(f"{name:<15} {np.mean(x):>8.4f} {np.std(x):>8.4f} {lo:>11.4f} {hi:>12.4f} {ess:>6.0f}{tv_str}")
    print(f"{'─'*60}\n")