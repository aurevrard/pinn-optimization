"""
Complement to points 3 and 4 - More collocation points (Burgers equation).

Same protocol as train_burgers.py, except the number of collocation points:
N_f = 10 000 points sampled uniformly (instead of 2000), as in Raissi et al.
Only L-BFGS is run (best method on Burgers), with the 3 seeds.

Warning: one loss evaluation now uses 5 times more points, so it costs about
5 times more. Compare these runs with the others in TRAINING TIME, not in
number of evaluations.

Results are saved in results/burgers_nf10000/ (same format as point 3).
Run from the project root:   python train_burgers_nf10k.py
"""
import json
import os

import torch

import train_burgers as tb     # reuse the loss, the optimizers and the training loop

# ============================================================
# Configuration
# ============================================================
N_F_NEW = 10_000        # number of collocation points (instead of 2000)
SAMPLING_SEED = 2       # fixed: the same points for the 3 runs
RESULTS_DIR = "results/burgers_nf10000"
SELECTED = {"L-BFGS": 1.0}


def uniform_points(n):
    """n collocation points sampled uniformly in [0, 1] x [-1, 1]."""
    g = torch.Generator().manual_seed(SAMPLING_SEED)
    t = torch.rand(n, 1, generator=g).requires_grad_(True)
    x = (-1.0 + 2.0 * torch.rand(n, 1, generator=g)).requires_grad_(True)
    return t, x


def count_in_layer(t, x, width=0.003, t_min=0.3):
    """Number of points inside the thin shock layer."""
    return int(((x.abs() < width) & (t > t_min)).sum())


# ============================================================
if __name__ == "__main__":
    t_new, x_new = uniform_points(N_F_NEW)
    print(f"Points in the shock layer: N_f = {tb.N_F} -> {count_in_layer(tb.t_f, tb.x_f)}, "
          f"N_f = {N_F_NEW} -> {count_in_layer(t_new, x_new)}")

    # Replace the collocation points (and their number) used by train_burgers.loss_fn
    tb.t_f, tb.x_f, tb.N_F = t_new, x_new, N_F_NEW

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for name, lr in SELECTED.items():
        for seed in tb.SEEDS:
            tag = tb.run_name(name, lr, seed)
            if os.path.exists(f"{RESULTS_DIR}/{tag}.json"):
                print(f"{tag}: already computed, skipped")
                continue

            print(f"{name:13s} lr={lr:g} seed={seed} N_f={N_F_NEW} ...", flush=True)
            hist, model = tb.train(name, lr, seed)
            hist["n_f"] = N_F_NEW

            with open(f"{RESULTS_DIR}/{tag}.json", "w") as f:
                json.dump(hist, f)
            torch.save(model.state_dict(), f"{RESULTS_DIR}/{tag}.pt")

            status = "DIVERGED" if hist["diverged"] else ""
            print(f"   loss={hist['loss'][-1]:.2e}  L2={hist['l2'][-1]:.2e}  "
                  f"it={hist['iterations']}  time={hist['time'][-1]:.1f}s {status}")