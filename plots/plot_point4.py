"""
Point 4 - Figures for the choice of the collocation points (Burgers equation).

Compares, for the 3 selected optimizers, the runs of point 3 (uniform points,
results/burgers/) with the shock-refined runs (results/burgers_shock/).
The training loss is NOT compared: it is computed on different points in the
two cases. The comparison uses quantities computed on the same reference grid:
  - the relative L2 error on the whole grid,
  - the relative L2 error restricted to the band |x| < BAND around the shock,
  - the PDE residual on the test grid.

Run from the project root:   python plots/plot_point4.py
Figures and table are written to plots/figures/point4/.
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
from plot_optimizers import on_grid

FIGS = ROOT / "plots" / "figures" / "point4"
BAND = 0.1
SEEDS = [0, 1, 2]
SELECTED = {"SGD-momentum": 1e-2, "Adam": 1e-3, "L-BFGS": 1.0}
CONFIGS = {"uniform": (ROOT / "results" / "burgers", "0.45"),
           "shock-refined": (ROOT / "results" / "burgers_shock", "tab:orange")}
MAX_EVALS = 4000

# Points of the reference grid inside the band around the shock
MASK = (data.x_test.abs() < BAND).ravel()


def run_name(name, lr, seed):
    return f"{name}_lr{lr:g}_seed{seed}"


def evaluate(directory, name, lr, seed):
    """Loads one run and computes the errors on the reference grid."""
    tag = run_name(name, lr, seed)
    hist = json.loads((directory / f"{tag}.json").read_text())
    model = PINN(in_dim=2)
    model.load_state_dict(torch.load(directory / f"{tag}.pt", weights_only=True))
    with torch.no_grad():
        diff = (model(data.t_test, data.x_test) - data.u_exact).ravel()
    u = data.u_exact.ravel()
    return {"hist": hist, "model": model, "diff": diff,
            "global": (torch.linalg.norm(diff) / torch.linalg.norm(u)).item(),
            "band": (torch.linalg.norm(diff[MASK]) / torch.linalg.norm(u[MASK])).item(),
            "residual": hist["residual"][-1],
            "time": hist["time"][-1]}


def load_all():
    """{(config, optimizer): [result seed 0, seed 1, seed 2]}"""
    return {(cfg, name): [evaluate(directory, name, lr, s) for s in SEEDS]
            for cfg, (directory, _) in CONFIGS.items()
            for name, lr in SELECTED.items()}


# ============================================================
# Figure 1: global error and error in the band, uniform vs shock-refined
# ============================================================
def fig_errors(res):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = list(SELECTED)
    for ax, key, label in [(axes[0], "global", "relative $L^2$ error (whole domain)"),
                           (axes[1], "band", f"relative $L^2$ error ($|x| < {BAND}$)")]:
        for j, (cfg, (_, color)) in enumerate(CONFIGS.items()):
            offset = -0.12 if j == 0 else 0.12
            for i, name in enumerate(names):
                v = np.array([r[key] for r in res[(cfg, name)]])
                m = np.median(v)
                ax.errorbar(i + offset, m, yerr=[[m - v.min()], [v.max() - m]],
                            fmt="o", color=color, capsize=4, markersize=7,
                            label=cfg if i == 0 else None)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names)
        ax.set_yscale("log")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "errors_uniform_vs_refined.png", dpi=150)


# ============================================================
# Figure 2: error during training, one panel per optimizer
# ============================================================
def fig_error_curves(res):
    grid = np.linspace(0, MAX_EVALS, 201)
    fig, axes = plt.subplots(1, len(SELECTED), figsize=(13, 3.8), sharey=True)
    for ax, name in zip(axes, SELECTED):
        for cfg, (_, color) in CONFIGS.items():
            Y = np.array([on_grid(r["hist"], "l2", grid) for r in res[(cfg, name)]])
            med, lo, hi = np.nanmedian(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0)
            ax.semilogy(grid, 10 ** med, color=color, label=cfg)
            ax.fill_between(grid, 10 ** lo, 10 ** hi, color=color, alpha=0.2)
        ax.set_title(name)
        ax.set_xlabel("loss evaluations")
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("relative $L^2$ error")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIGS / "error_curves_uniform_vs_refined.png", dpi=150)


# ============================================================
# Figure 3: error maps of one optimizer, uniform vs shock-refined (same color scale)
# ============================================================
def fig_maps(res, name="L-BFGS", seed=0):
    nt = len(torch.unique(data.t_test))
    nx = len(data.t_test) // nt
    T = data.t_test.reshape(nt, nx).numpy()
    X = data.x_test.reshape(nt, nx).numpy()
    E = {cfg: res[(cfg, name)][seed]["diff"].reshape(nt, nx).numpy() for cfg in CONFIGS}
    vmax = max(np.abs(e).max() for e in E.values())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, cfg in zip(axes, CONFIGS):
        im = ax.pcolormesh(T, X, E[cfg], shading="auto", cmap="RdBu_r",
                           vmin=-vmax, vmax=vmax)
        r = res[(cfg, name)][seed]
        ax.text(0.02, 0.97, f"{name}, {cfg}, seed {seed}\n"
                f"error = {r['global']:.2e}, band error = {r['band']:.2e}",
                transform=ax.transAxes, ha="left", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(im, ax=axes, label=r"$u_\theta - u$")
    fig.savefig(FIGS / f"error_maps_{name}_uniform_vs_refined.png", dpi=150,
                bbox_inches="tight")


# ============================================================
# Table: median [min, max] over seeds
# ============================================================
def table(res):
    def mr(values):
        v = np.array(values)
        return f"{np.median(v):.1e} [{v.min():.1e}, {v.max():.1e}]"

    header = ["Optimizer", "Sampling", "Rel. $L^2$ error", "Error in band",
              "PDE residual", "Time [s]"]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule", " & ".join(header) + r" \\",
             r"\midrule"]
    for name in SELECTED:
        for cfg in CONFIGS:
            rs = res[(cfg, name)]
            row = [name, cfg, mr([r["global"] for r in rs]), mr([r["band"] for r in rs]),
                   mr([r["residual"] for r in rs]), f"{np.median([r['time'] for r in rs]):.0f}"]
            lines.append(" & ".join(row) + r" \\")
            print("  ".join(row))
        lines.append(r"\midrule" if name != list(SELECTED)[-1] else r"\bottomrule")
    lines.append(r"\end{tabular}")
    (FIGS / "table_point4.tex").write_text("\n".join(lines))


# ============================================================
if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    res = load_all()
    table(res)
    fig_errors(res)
    fig_error_curves(res)
    fig_maps(res, "L-BFGS")
    fig_maps(res, "Adam")
    print(f"Figures saved in {FIGS}")