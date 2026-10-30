"""
Burgers equation u_t + u u_x = nu u_xx: PINN training with several optimizers.

The protocol is IDENTICAL to train_heat.py (network, optimizers, budget,
learning-rate grid, seeds): only the data and the PDE residual change.

Each run (optimizer, learning rate, seed) is saved in results/burgers/:
  - a .json file with the history of the metrics
  - a .pt file with the final network weights (used for the error maps)
"""
import json
import math
import os
import time
import torch

from pinn import PINN
from data_gen_burgers import (NU, t_f, x_f, t_0, x_0, u_0, t_b, x_b, u_b,   # <-- BURGERS
                              t_test, x_test, u_exact)

# ============================================================
# Experiment configuration
# ============================================================
MAX_EVALS = 4000     # budget, in full loss evaluations
LOG_EVERY = 50       # record the metrics every 50 evaluations
SEEDS = [0, 1, 2]
RESULTS_DIR = "results/burgers"                                              # <-- BURGERS
N_F = t_f.shape[0]   # number of collocation points


# ============================================================
# PDE residual, loss and metrics
# ============================================================
def pde_residual(model, t, x):
    """Residual of the Burgers equation, computed by automatic differentiation."""
    u = model(t, x)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - NU * u_xx                                         # <-- BURGERS


def loss_fn(model, batch_size=None):
    """Total loss. batch_size=None: all collocation points;
    otherwise a random mini-batch (for the stochastic methods)."""
    if batch_size is None:
        t, x = t_f, x_f
    else:
        idx = torch.randint(0, N_F, (batch_size,))
        # detach: new tensor without history, so that we differentiate w.r.t. the sampled points
        t = t_f[idx].detach().requires_grad_(True)
        x = x_f[idx].detach().requires_grad_(True)

    loss_pde = torch.mean(pde_residual(model, t, x) ** 2)
    loss_ic = torch.mean((model(t_0, x_0) - u_0) ** 2)
    loss_bc = torch.mean((model(t_b, x_b) - u_b) ** 2)
    return loss_pde + loss_ic + loss_bc, loss_pde, loss_ic, loss_bc


def l2_error(model):
    """Relative L2 error with respect to the reference solution."""
    with torch.no_grad():
        u_pred = model(t_test, x_test)
        return (torch.linalg.norm(u_pred - u_exact) / torch.linalg.norm(u_exact)).item()


def test_residual(model):
    """Mean squared PDE residual on the reference grid (points never used in training)."""
    t = t_test.clone().requires_grad_(True)
    x = x_test.clone().requires_grad_(True)
    return torch.mean(pde_residual(model, t, x) ** 2).item()


def full_loss_and_grad_norm(model):
    """Full loss and gradient norm (outside the budget).
    The gradient norm is a stationarity indicator (convergence or plateau)."""
    model.zero_grad()
    loss = loss_fn(model)[0]
    loss.backward()
    gn = math.sqrt(sum((p.grad ** 2).sum().item()
                       for p in model.parameters() if p.grad is not None))
    model.zero_grad()
    return loss.item(), gn


# ============================================================
# Gradient descent with Armijo line search
# ============================================================
class ArmijoGD(torch.optim.Optimizer):
    """Gradient descent with Armijo backtracking:
    find t such that L(theta - t g) <= L(theta) - c t ||g||^2,
    starting from t = lr and multiplying t by beta after each failure."""

    def __init__(self, params, lr=1.0, c=1e-4, beta=0.5, max_backtracks=30):
        super().__init__(params, dict(lr=lr, c=c, beta=beta,
                                      max_backtracks=max_backtracks))

    def step(self, closure):
        loss = closure()                                   # loss + gradient at theta_k
        group = self.param_groups[0]
        params = [p for p in group["params"] if p.grad is not None]
        grads = [p.grad.detach().clone() for p in params]  # copy of g (closure erases p.grad)
        old = [p.detach().clone() for p in params]         # copy of theta_k
        g2 = sum((g ** 2).sum() for g in grads).item()     # ||g||^2

        t = group["lr"]
        for _ in range(group["max_backtracks"]):
            with torch.no_grad():
                for p, p0, g in zip(params, old, grads):
                    p.copy_(p0 - t * g)                    # theta = theta_k - t g
            new_loss = closure(backward=False)             # loss only, no gradient
            if new_loss.item() <= loss.item() - group["c"] * t * g2:
                break                                      # Armijo condition satisfied
            t *= group["beta"]                             # otherwise reduce the step
        return loss


# ============================================================
# Optimizers (identical to train_heat.py)
# ============================================================
OPTIMIZERS = {
    "GD-Armijo":    lambda p, lr: ArmijoGD(p, lr=lr),
    "SGD":          lambda p, lr: torch.optim.SGD(p, lr=lr),
    "SGD-momentum": lambda p, lr: torch.optim.SGD(p, lr=lr, momentum=0.9),
    "Adam":         lambda p, lr: torch.optim.Adam(p, lr=lr),
    "L-BFGS":       lambda p, lr: torch.optim.LBFGS(p, lr=lr, max_iter=20,
                                                    line_search_fn="strong_wolfe"),
}

# Default learning rates (initial step for Armijo and L-BFGS)
DEFAULT_LR = {"GD-Armijo": 1.0, "SGD": 1e-2, "SGD-momentum": 1e-3,
              "Adam": 1e-3, "L-BFGS": 1.0}

# Tested learning rates (Armijo and L-BFGS choose their own step: a single value)
LR_GRID = {
    "GD-Armijo":    [1.0],
    "SGD":          [1e-3, 1e-2, 1e-1, 1.0],
    "SGD-momentum": [1e-4, 1e-3, 1e-2, 1e-1],
    "Adam":         [1e-4, 1e-3, 1e-2, 1e-1],
    "L-BFGS":       [1.0],
}

# Mini-batches for the stochastic methods, full loss for the line-search methods
BATCH_SIZE = {"GD-Armijo": None, "SGD": 256, "SGD-momentum": 256,
              "Adam": 256, "L-BFGS": None}


# ============================================================
# Generic training loop (identical to train_heat.py)
# ============================================================
def train(name, lr, seed, max_evals=MAX_EVALS):
    torch.manual_seed(seed)          # same seed = same initial network for every optimizer
    model = PINN(in_dim=2)
    opt = OPTIMIZERS[name](model.parameters(), lr)
    batch = BATCH_SIZE[name]
    cost = 1.0 if batch is None else batch / N_F   # cost of one loss call, in full evaluations

    n_evals = 0.0

    def closure(backward=True):
        """Computes the loss (and its gradient) and charges its cost to the budget."""
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
        """True number of iterations: L-BFGS performs several per call to step()."""
        if isinstance(opt, torch.optim.LBFGS):
            return opt.state[opt._params[0]].get("n_iter", 0)
        return it

    def log():
        """Records the current state of the network (does not consume budget)."""
        loss, gn = full_loss_and_grad_norm(model)
        values = [n_evals, n_iterations(), train_time, loss, gn,
                  l2_error(model), test_residual(model)]
        for k, v in zip(hist, values):
            hist[k].append(v)
        return loss

    log()                            # initial state
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
# Main program
# ============================================================
if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for name in OPTIMIZERS:
        for lr in LR_GRID[name]:
            for seed in SEEDS:
                tag = run_name(name, lr, seed)
                if os.path.exists(f"{RESULTS_DIR}/{tag}.json"):
                    print(f"{tag}: already computed, skipped")
                    continue

                print(f"{name:13s} lr={lr:g} seed={seed} ...", flush=True)
                hist, model = train(name, lr, seed)

                with open(f"{RESULTS_DIR}/{tag}.json", "w") as f:
                    json.dump(hist, f)
                torch.save(model.state_dict(), f"{RESULTS_DIR}/{tag}.pt")

                status = "DIVERGED" if hist["diverged"] else ""
                print(f"   loss={hist['loss'][-1]:.2e}  L2={hist['l2'][-1]:.2e}  "
                      f"it={hist['iterations']}  time={hist['time'][-1]:.1f}s {status}")