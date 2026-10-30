"""
Équation de la chaleur u_t = nu u_xx : entraînement du PINN avec plusieurs optimiseurs.
Chaque run (optimiseur, lr, graine) est sauvegardé dans results/heat/ :
  - un fichier .json avec l'historique des métriques
  - un fichier .pt avec les poids finaux du réseau (pour les cartes d'erreur)
"""
import json
import math
import os
import time
import torch

from pinn import PINN
from data_gen_heat import (NU, t_f, x_f, t_0, x_0, u_0, t_b, x_b, u_b,
                           t_test, x_test, u_exact)

# ============================================================
# Configuration de l'expérience
# ============================================================
MAX_EVALS = 4000     # budget, en évaluations complètes de la perte


LOG_EVERY = 50       # mesures toutes les 50 évaluations
SEEDS = [0, 1, 2]
RESULTS_DIR = "results/heat"
N_F = t_f.shape[0]   # nombre de points de collocation (2000)


# ============================================================
# Réseau, perte et métriques
# ============================================================

def pde_residual(model, t, x):
    u = model( t, x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t - NU * u_xx


def loss_fn(model, batch_size=None):
    """Perte totale. batch_size=None : tous les points de collocation ;
    sinon, un mini-batch tiré au hasard (pour les méthodes stochastiques)."""
    if batch_size is None:
        t, x = t_f, x_f
    else:
        idx = torch.randint(0, N_F, (batch_size,))# # detach : nouveau tenseur sans historique, pour dériver par rapport aux points tirés
        t = t_f[idx].detach().requires_grad_(True)
        x = x_f[idx].detach().requires_grad_(True)

    loss_pde = torch.mean(pde_residual(model, t, x) ** 2)
    loss_ic = torch.mean((model(t_0, x_0) - u_0) ** 2)
    loss_bc = torch.mean((model(t_b, x_b) - u_b) ** 2)
    return loss_pde + loss_ic + loss_bc, loss_pde, loss_ic, loss_bc


def l2_error(model):
    with torch.no_grad():
        u_pred = model(t_test, x_test)
        return (torch.linalg.norm(u_pred - u_exact) / torch.linalg.norm(u_exact)).item()


def test_residual(model):
    """Résidu moyen de l'EDP sur la grille de test (points jamais vus)."""
    t = t_test.clone().requires_grad_(True)
    x = x_test.clone().requires_grad_(True)
    return torch.mean(pde_residual(model, t, x) ** 2).item()


def full_loss_and_grad_norm(model):# norme du gradient : indicateur de stationnarité (convergence ou plateau) 
    """Perte sur tous les points et norme du gradient (hors budget)."""
    model.zero_grad()
    loss = loss_fn(model)[0]
    loss.backward()
    gn = math.sqrt(sum((p.grad ** 2).sum().item()
                       for p in model.parameters() if p.grad is not None))
    model.zero_grad()
    return loss.item(), gn


# ============================================================
# GD avec recherche linéaire d'Armijo
# ============================================================
class ArmijoGD(torch.optim.Optimizer):
    """Descente de gradient avec backtracking d'Armijo :
    on cherche t tel que L(θ - t g) <= L(θ) - c t ||g||²,
    en partant de t = lr et en multipliant t par beta à chaque échec."""

    def __init__(self, params, lr=1.0, c=1e-4, beta=0.5, max_backtracks=30):
        super().__init__(params, dict(lr=lr, c=c, beta=beta,
                                      max_backtracks=max_backtracks))

    def step(self, closure):
        loss = closure()                                   # perte + gradient
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.grad is not None]
        grads = [p.grad.detach().clone() for p in params]
        old = [p.detach().clone() for p in params]
        g2 = sum((g ** 2).sum() for g in grads).item()

        t = group["lr"]
        for _ in range(group["max_backtracks"]):
            with torch.no_grad():
                for p, p0, g in zip(params, old, grads):
                    p.copy_(p0 - t * g)
            new_loss = closure(backward=False)             # perte seule
            if new_loss.item() <= loss.item() - group["c"] * t * g2:
                break
            t *= group["beta"]
        return loss


# ============================================================
# Liste des optimiseurs
# ============================================================
OPTIMIZERS = {
    "GD-Armijo":    lambda p, lr: ArmijoGD(p, lr=lr),
    "SGD":          lambda p, lr: torch.optim.SGD(p, lr=lr),
    "SGD-momentum": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "Adam":         lambda p, lr: torch.optim.Adam(p, lr=lr),
    "L-BFGS":       lambda p, lr: torch.optim.LBFGS(p, lr=lr, max_iter=20,
                                                    line_search_fn="strong_wolfe"),
}

# Taux d'apprentissage par défaut (pas initial pour Armijo et L-BFGS)
DEFAULT_LR = {"GD-Armijo": 1.0, "SGD": 1e-2, "SGD-momentum": 1e-3,
              "Adam": 1e-3, "L-BFGS": 1.0}

# Valeurs de lr testées (Armijo et L-BFGS choisissent leur pas : une seule valeur)
LR_GRID = {
    "GD-Armijo":    [1.0],
    "SGD":          [1e-3, 1e-2, 1e-1, 1.0],
    "SGD-momentum": [1e-4, 1e-3, 1e-2, 1e-1],
    "Adam":         [1e-4, 1e-3, 1e-2, 1e-1],
    "L-BFGS":       [1.0],
}

# Mini-batch pour les méthodes stochastiques, tous les points pour les autres
BATCH_SIZE = {"GD-Armijo": None, "SGD": 256, "SGD-momentum": 256,
              "Adam": 256, "L-BFGS": None}


# ============================================================
# Entraînement générique
# ============================================================
def train(name, lr, seed, max_evals=MAX_EVALS):
    torch.manual_seed(seed)          # même graine = même initialisation pour tous
    model = PINN(in_dim=2)
    opt = OPTIMIZERS[name](model.parameters(), lr)
    batch = BATCH_SIZE[name]
    cost = 1.0 if batch is None else batch / N_F   # coût d'un appel, en évaluations complètes

    n_evals = 0.0

    def closure(backward=True):
        nonlocal n_evals
        n_evals += cost
        opt.zero_grad()
        loss = loss_fn(model, batch)[0]
        if backward:
            loss.backward()
        return loss

    hist = {k: [] for k in ["evals", "iter", "time", "loss",
                            "grad_norm", "l2", "residual"]}
    it, train_time, diverged = 0, 0.0, False

    def n_iterations():
        # L-BFGS fait plusieurs itérations par appel à step() : on lit son compteur interne
        if isinstance(opt, torch.optim.LBFGS):
            return opt.state[opt._params[0]].get("n_iter", 0)
        return it
    
    def log():
        loss, gn = full_loss_and_grad_norm(model)
        values = [n_evals, n_iterations(), train_time, loss, gn, l2_error(model), test_residual(model)]
        for k, v in zip(hist, values):
            hist[k].append(v)
        return loss

    log()                            # état initial
    next_log = LOG_EVERY
    while n_evals < max_evals:
        t0 = time.perf_counter()
        opt.step(closure)
        train_time += time.perf_counter() - t0
        it += 1

        if n_evals >= next_log or n_evals >= max_evals:
            loss = log()
            while next_log <= n_evals:
                next_log += LOG_EVERY
            if not math.isfinite(loss):
                diverged = True
                break

    hist.update(optimizer=name, lr=lr, seed=seed, batch_size=batch,
                iterations=n_iterations(), diverged=diverged)
    return hist, model


def run_name(name, lr, seed):
    return f"{name}_lr{lr:g}_seed{seed}"


# ============================================================
# Programme principal
# ============================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name in OPTIMIZERS:
        for lr in LR_GRID[name]:
            for seed in SEEDS:
                tag = run_name(name, lr, seed)
                if os.path.exists(f"{RESULTS_DIR}/{tag}.json"):
                    print(f"{tag} : déjà calculé, ignoré")
                    continue

                print(f"{name:13s} lr={lr:g} seed={seed} ...", flush=True)
                hist, model = train(name, lr, seed)

                with open(f"{RESULTS_DIR}/{tag}.json", "w") as f:
                    json.dump(hist, f)
                torch.save(model.state_dict(), f"{RESULTS_DIR}/{tag}.pt")

                status = "DIVERGED" if hist["diverged"] else ""
                print(f"   loss={hist['loss'][-1]:.2e}  L2={hist['l2'][-1]:.2e}  "
                      f"it={hist['iterations']}  time={hist['time'][-1]:.1f}s {status}")