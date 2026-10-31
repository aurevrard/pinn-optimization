"""
Point 4 - Effect of the choice of the collocation points (Burgers equation).

Same protocol as train_burgers.py (network, loss, budget, seeds), except the
collocation points: half of them are sampled uniformly in the whole domain and
half in a band around the shock (|x| < BAND). The total number N_f = 2000 is
unchanged, so one loss evaluation has the same cost as in point 3.

The runs with UNIFORM points are those of point 3 (results/burgers/), so only
the shock-refined runs are computed here, for the 3 selected optimizers with
their best learning rate on Burgers.

Results are saved in results/burgers_shock/ (same format as point 3).
Run from the project root:   python train_burgers_shock.py
"""
import json
import os

import torch

import train_burgers as tb     # reuse the loss, the optimizers and the training loop

# ============================================================
# Configuration
# ============================================================
BAND = 0.1              # half-width of the refined band around x = 0
N_F = tb.N_F            # same total number of collocation points (2000)
N_BAND = N_F // 2       # points in the band, the others in the whole domain
SAMPLING_SEED = 1       # fixed: the same refined points for every run
RESULTS_DIR = "results/burgers_shock"

# Optimizers kept for point 4, with their best learning rate on Burgers (point 3)
SELECTED = {"SGD-momentum": 1e-2, "Adam": 1e-3, "L-BFGS": 1.0}


# ============================================================
# Shock-refined collocation points
# ============================================================
def shock_refined_points():
    g = torch.Generator().manual_seed(SAMPLING_SEED)
    n_domain = N_F - N_BAND
    # half in the whole domain [0, 1] x [-1, 1]
    t_domain = torch.rand(n_domain, 1, generator=g)
    x_domain = -1.0 + 2.0 * torch.rand(n_domain, 1, generator=g)
    # half in the band [0, 1] x [-BAND, BAND] around the shock
    t_band = torch.rand(N_BAND, 1, generator=g)
    x_band = -BAND + 2.0 * BAND * torch.rand(N_BAND, 1, generator=g)
    t = torch.cat([t_domain, t_band]).requires_grad_(True)
    x = torch.cat([x_domain, x_band]).requires_grad_(True)
    return t, x


def count_in_layer(t, x, width=0.003, t_min=0.3):
    """Number of points inside the thin shock layer (same count as in data_gen_burgers)."""
    return int(((x.abs() < width) & (t > t_min)).sum())


# ============================================================
if __name__ == "__main__":
    t_new, x_new = shock_refined_points()
    print(f"Points in the shock layer: uniform = {count_in_layer(tb.t_f, tb.x_f)}, "
          f"shock-refined = {count_in_layer(t_new, x_new)} (out of {N_F})")

    # Replace the collocation points used by train_burgers.loss_fn.
    # loss_fn reads the module variables t_f and x_f, so changing them here is enough.
    tb.t_f, tb.x_f = t_new, x_new

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, lr in SELECTED.items():
        for seed in tb.SEEDS:
            tag = tb.run_name(name, lr, seed)
            if os.path.exists(f"{RESULTS_DIR}/{tag}.json"):
                print(f"{tag}: already computed, skipped")
                continue

            print(f"{name:13s} lr={lr:g} seed={seed} ...", flush=True)
            hist, model = tb.train(name, lr, seed)
            hist["sampling"] = "shock-refined"

            with open(f"{RESULTS_DIR}/{tag}.json", "w") as f:
                json.dump(hist, f)
            torch.save(model.state_dict(), f"{RESULTS_DIR}/{tag}.pt")

            status = "DIVERGED" if hist["diverged"] else ""
            print(f"   loss={hist['loss'][-1]:.2e}  L2={hist['l2'][-1]:.2e}  "
                  f"it={hist['iterations']}  time={hist['time'][-1]:.1f}s {status}")