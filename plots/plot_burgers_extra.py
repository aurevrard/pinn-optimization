"""
Extra figures for the Burgers equation.
  1. L-BFGS with N_f = 2000 (point 3) vs N_f = 10 000: error and PDE residual
     on the test grid as a function of the TRAINING TIME (one evaluation does not
     cost the same in both cases, so the number of evaluations is not comparable).
  2. Error map of L-BFGS with N_f = 10 000 (seed 0), in plots/figures/burgers/nf10000/.
  3. Profiles u(x) at a few times: reference vs Adam, L-BFGS (N_f = 2000) and
     L-BFGS (N_f = 10 000), to see whether the front is smoothed or shifted.
Run from the project root:   python plots/plot_burgers_extra.py
Figures are written to plots/figures/burgers/.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "plots"))
from pinn import PINN
import data_gen_burgers as data
from plot_optimizers import fig_error_map

FIGS = ROOT / "plots" / "figures" / "burgers"
DIR_2K = ROOT / "results" / "burgers"
DIR_10K = ROOT / "results" / "burgers_nf10000"
SEEDS = [0, 1, 2]


def run_name(name, lr, seed):
    return f"{name}_lr{lr:g}_seed{seed}"


def load(directory, name, lr):
    return [json.loads((directory / f"{run_name(name, lr, s)}.json").read_text())
            for s in SEEDS]


def load_model(directory, name, lr, seed):
    model = PINN(in_dim=2)
    path = directory / f"{run_name(name, lr, seed)}.pt"
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


# ============================================================
# Figure 1: N_f = 2000 vs N_f = 10 000 (L-BFGS), versus training time
# ============================================================
def fig_nf_comparison():
    configs = {"$N_f = 2000$": (load(DIR_2K, "L-BFGS", 1.0), "tab:purple"),
               "$N_f = 10\\,000$": (load(DIR_10K, "L-BFGS", 1.0), "tab:brown")}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "l2", "relative $L^2$ error"),
                           (axes[1], "residual", "PDE residual (test grid)")]:
        for name, (runs, color) in configs.items():
            t_max = max(r["time"][-1] for r in runs)
            grid = np.linspace(0, t_max, 300)
            Y = np.array([np.interp(grid, r["time"], np.log10(r[key]), right=np.nan)
                          for r in runs])
            med, lo, hi = np.nanmedian(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0)
            ax.semilogy(grid, 10 ** med, color=color, label=f"L-BFGS, {name}")
            ax.fill_between(grid, 10 ** lo, 10 ** hi, color=color, alpha=0.2)
        ax.set_xlabel("training time [s]")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "lbfgs_nf_comparison.png", dpi=150)

    # Summary in the terminal (median [min, max] over seeds)
    for name, (runs, _) in configs.items():
        err = [r["l2"][-1] for r in runs]
        res = [r["residual"][-1] for r in runs]
        tim = [r["time"][-1] for r in runs]
        print(f"L-BFGS {name:16s} error = {np.median(err):.2e} [{min(err):.1e}, {max(err):.1e}]"
              f"   residual = {np.median(res):.2e}   time = {np.median(tim):.0f} s")


# ============================================================
# Figure 3: profiles u(x) at a few times
# ============================================================
def fig_profiles(times=(0.25, 0.5, 0.75), seed=0):
    t_ref, x_ref, u_ref = data.t_ref, data.x_ref, data.u_ref
    models = {"Adam ($N_f = 2000$)": (load_model(DIR_2K, "Adam", 1e-3, seed), "tab:red", "--"),
              "L-BFGS ($N_f = 2000$)": (load_model(DIR_2K, "L-BFGS", 1.0, seed), "tab:purple", "--"),
              "L-BFGS ($N_f = 10\\,000$)": (load_model(DIR_10K, "L-BFGS", 1.0, seed), "tab:brown", ":")}

    fig, axes = plt.subplots(1, len(times), figsize=(4 * len(times), 3.6), sharey=True)
    x = torch.tensor(x_ref, dtype=torch.float32).reshape(-1, 1)
    for ax, t_target in zip(axes, times):
        k = int(np.argmin(np.abs(t_ref - t_target)))          # closest time of the grid
        t = torch.full_like(x, float(t_ref[k]))
        ax.plot(x_ref, u_ref[k], "k-", lw=2, label="reference")
        for name, (model, color, style) in models.items():
            with torch.no_grad():
                u = model(t, x).numpy().ravel()
            ax.plot(x_ref, u, color=color, linestyle=style, lw=1.8, label=name)
        ax.set_title(f"t = {t_ref[k]:.2f}")
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("u")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(FIGS / "profiles.png", dpi=150)


# ============================================================
if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_nf_comparison()
    # Error map of L-BFGS with N_f = 10 000 (same function as for the other maps).
    # Saved in a subfolder so that it does not overwrite the map of the N_f = 2000 run.
    figs_10k = FIGS / "nf10000"
    figs_10k.mkdir(exist_ok=True)
    fig_error_map(data, DIR_10K, run_name, "L-BFGS", 0, 1.0, figs_10k)
    fig_profiles()
    print(f"Figures saved in {FIGS}")