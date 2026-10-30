"""
Comparison figures for the optimizers, for either PDE.
Run from the project root:
    python plots/plot_optimizers.py --pde heat
    python plots/plot_optimizers.py --pde burgers
    python plots/plot_optimizers.py --pde burgers --map L-BFGS --seed 1
Results are read from results/<pde>/ and figures are written to plots/figures/<pde>/.
"""
import argparse
import importlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch

# Access to the modules at the project root
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pinn import PINN

warnings.filterwarnings("ignore", category=RuntimeWarning)  # medians of empty columns


# ============================================================
# Loading the results
# ============================================================
def load_all(results_dir):
    """All runs: {(optimizer, lr): [run seed 0, seed 1, ...]}."""
    all_runs = {}
    for f in sorted(results_dir.glob("*.json")):
        r = json.loads(f.read_text())
        all_runs.setdefault((r["optimizer"], r["lr"]), []).append(r)
    return all_runs


def final(run, key):
    """Final value of a metric (NaN if the run diverged)."""
    return np.nan if run.get("diverged") else run[key][-1]


def best_lrs(all_runs, optimizers):
    """For each optimizer, the learning rate with the lowest median final LOSS.
    The loss is used (not the error) because the exact solution is unknown in practice."""
    best = {}
    for name in optimizers:
        candidates = {lr: rs for (n, lr), rs in all_runs.items() if n == name}
        if not candidates:
            continue

        def score(rs):
            v = np.array([final(r, "loss") for r in rs], dtype=float)
            return np.inf if np.all(np.isnan(v)) else np.nanmedian(v)

        best[name] = min(candidates, key=lambda lr: score(candidates[lr]))
    return best


def on_grid(run, key, grid):
    """Interpolates a metric (in log10) on a common grid of evaluations.
    Beyond the last measurement (diverged run), the value is NaN."""
    x = np.array(run["evals"], dtype=float)
    y = np.log10(np.array(run[key], dtype=float))
    return np.interp(grid, x, y, right=np.nan)


# ============================================================
# Figure 1: metrics versus number of loss evaluations
# ============================================================
def plot_metric(ax, runs, key, ylabel, grid, colors):
    for name, rs in runs.items():
        Y = np.array([on_grid(r, key, grid) for r in rs])
        med, lo, hi = np.nanmedian(Y, 0), np.nanmin(Y, 0), np.nanmax(Y, 0)
        ax.plot(grid, 10 ** med, color=colors[name], label=name)
        ax.fill_between(grid, 10 ** lo, 10 ** hi, color=colors[name], alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("loss evaluations")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.3)


def fig_metrics(runs, grid, colors, figs, title):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    plot_metric(axes[0, 0], runs, "loss", "training loss", grid, colors)
    plot_metric(axes[0, 1], runs, "l2", "relative $L^2$ error", grid, colors)
    plot_metric(axes[1, 0], runs, "grad_norm", r"$\|\nabla_\theta \mathcal{L}\|$", grid, colors)
    plot_metric(axes[1, 1], runs, "residual", "PDE residual (test grid)", grid, colors)
    axes[0, 0].legend()
    fig.suptitle(f"{title} — median over seeds, shaded: min–max")
    fig.tight_layout()
    fig.savefig(figs / "metrics_vs_evals.png", dpi=150)


# ============================================================
# Figure 2: sensitivity to the learning rate
# ============================================================
def fig_lr_sensitivity(all_runs, optimizers, colors, figs, title):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, label in [(axes[0], "loss", "final training loss"),
                           (axes[1], "l2", "final relative $L^2$ error")]:
        for name in optimizers:
            lrs = sorted(lr for (n, lr) in all_runs if n == name)
            if not lrs:
                continue
            vals = [np.array([final(r, key) for r in all_runs[(name, lr)]]) for lr in lrs]
            med = np.array([np.nanmedian(v) for v in vals])
            lo = np.array([np.nanmin(v) for v in vals])
            hi = np.array([np.nanmax(v) for v in vals])

            if len(lrs) == 1:   # no sweep: horizontal reference line
                ax.axhline(med[0], color=colors[name], linestyle="--",
                           label=f"{name} (adaptive step)")
            else:
                ax.errorbar(lrs, med, yerr=[med - lo, hi - med], marker="o",
                            capsize=3, color=colors[name], label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("learning rate")
        ax.set_ylabel(label)
        ax.grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{title} — sensitivity to the learning rate (median, bars: min–max)")
    fig.tight_layout()
    fig.savefig(figs / "lr_sensitivity.png", dpi=150)


# ============================================================
# Figure 3: loss versus computational time
# ============================================================
def fig_time(runs, colors, figs, title):
    plt.figure(figsize=(6, 4))
    for name, rs in runs.items():
        for i, r in enumerate(rs):
            plt.semilogy(r["time"], r["loss"], color=colors[name], alpha=0.7,
                         label=name if i == 0 else None)
    plt.xlabel("training time [s]")
    plt.ylabel("training loss")
    plt.title(f"{title} — loss vs computational time")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "loss_vs_time.png", dpi=150)


# ============================================================
# Figure 4: final error versus final loss
# ============================================================
def fig_loss_vs_error(runs, colors, figs, title):
    plt.figure(figsize=(5.5, 4.5))
    for name, rs in runs.items():
        L = [final(r, "loss") for r in rs]
        E = [final(r, "l2") for r in rs]
        plt.loglog(L, E, "o", color=colors[name], label=name, markersize=7)
    plt.xlabel("final training loss")
    plt.ylabel("final relative $L^2$ error")
    plt.title(f"{title} — does a low loss imply an accurate solution?")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(figs / "loss_vs_error.png", dpi=150)


# ============================================================
# Figure 5: evaluations needed to reach a target loss
# ============================================================
def evals_to_target(run, target):
    for e, l in zip(run["evals"], run["loss"]):
        if l <= target:
            return e
    return np.nan


def fig_evals_to_target(runs, colors, figs, title, target, max_evals):
    names = list(runs)
    vals = [np.array([evals_to_target(r, target) for r in runs[n]]) for n in names]

    plt.figure(figsize=(6, 4))
    for i, (n, v) in enumerate(zip(names, vals)):
        if np.all(np.isnan(v)):
            plt.text(i, max_evals * 0.05, "not\nreached", ha="center", fontsize=8)
            continue
        m = np.nanmedian(v)
        plt.bar(i, m, color=colors[n])
        plt.errorbar(i, m, yerr=[[m - np.nanmin(v)], [np.nanmax(v) - m]],
                     color="k", capsize=4)
    plt.xticks(range(len(names)), names, rotation=20)
    plt.ylabel("loss evaluations")
    plt.title(f"{title} — evaluations to reach loss $\\leq$ {target:.0e}")
    plt.tight_layout()
    plt.savefig(figs / "evals_to_target.png", dpi=150)


# ============================================================
# Summary table (console + LaTeX)
# ============================================================
def summary_table(runs, figs):
    def ms(values, fmt):
        v = np.array(values, dtype=float)
        if np.all(np.isnan(v)):
            return "diverged"
        return f"{np.nanmean(v):{fmt}} $\\pm$ {np.nanstd(v):{fmt}}"

    rows = []
    for name, rs in runs.items():
        rows.append([name, f"{rs[0]['lr']:g}",
                     ms([final(r, "loss") for r in rs], ".2e"),
                     ms([final(r, "l2") for r in rs], ".2e"),
                     ms([r["iterations"] for r in rs], ".0f"),
                     ms([r["time"][-1] for r in rs], ".1f")])

    header = ["Optimizer", "lr", "Final loss", "Rel. $L^2$ error", "Iterations", "Time [s]"]
    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (figs / "summary_table.tex").write_text("\n".join(lines))

    for row in rows:
        print("  ".join(s.replace(" $\\pm$ ", " ± ") for s in row))


# ============================================================
# Figure 6: signed error map for a chosen optimizer
# ============================================================
def fig_error_map(data, results_dir, run_name, name, seed, lr, figs, title):
    path = results_dir / f"{run_name(name, lr, seed)}.pt"
    model = PINN(in_dim=2)
    model.load_state_dict(torch.load(path, weights_only=True))

    with torch.no_grad():
        u_pred = model(data.t_test, data.x_test)
    err = (torch.linalg.norm(u_pred - data.u_exact) / torch.linalg.norm(data.u_exact)).item()

    # Grid shape: rows = time instants, columns = positions (works for any NT x NX grid)
    nt = len(torch.unique(data.t_test))
    nx = len(data.t_test) // nt
    T = data.t_test.reshape(nt, nx).numpy()
    X = data.x_test.reshape(nt, nx).numpy()
    E = (u_pred - data.u_exact).reshape(nt, nx).numpy()
    vmax = np.abs(E).max()

    plt.figure(figsize=(6, 4))
    plt.pcolormesh(T, X, E, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(label=r"$u_\theta - u$")
    plt.xlabel("t")
    plt.ylabel("x")
    plt.title(f"{title} — signed error, {name} (seed {seed})")
    ax = plt.gca()
    ax.text(0.02, 0.97, f"Relative $L^2$ error = {err:.2e}",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    plt.tight_layout()
    plt.savefig(figs / f"error_map_{name}_seed{seed}.png", dpi=150)


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pde", choices=["heat", "burgers"], required=True,
                        help="which PDE to plot")
    parser.add_argument("--map", default="Adam", help="optimizer shown in the error map")
    parser.add_argument("--seed", type=int, default=0, help="seed shown in the error map")
    parser.add_argument("--target", type=float, default=1e-3,
                        help="target loss for the 'evaluations to target' figure")
    args = parser.parse_args()

    # Modules of the chosen PDE: train_heat / train_burgers and data_gen_heat / data_gen_burgers
    train_mod = importlib.import_module(f"train_{args.pde}")
    data_mod = importlib.import_module(f"data_gen_{args.pde}")

    title = {"heat": "Heat equation", "burgers": "Burgers equation"}[args.pde]
    results_dir = ROOT / train_mod.RESULTS_DIR
    figs = ROOT / "plots" / "figures" / args.pde
    figs.mkdir(parents=True, exist_ok=True)

    optimizers = list(train_mod.OPTIMIZERS)
    colors = {name: plt.cm.tab10(i) for i, name in enumerate(optimizers)}
    grid = np.linspace(0, train_mod.MAX_EVALS, 201)   # common axis of evaluations

    all_runs = load_all(results_dir)
    if not all_runs:
        sys.exit(f"No results found in {results_dir}")
    best = best_lrs(all_runs, optimizers)
    print("Best learning rate per optimizer:", best)

    runs = {name: all_runs[(name, best[name])] for name in optimizers if name in best}

    fig_lr_sensitivity(all_runs, optimizers, colors, figs, title)
    fig_metrics(runs, grid, colors, figs, title)
    fig_time(runs, colors, figs, title)
    fig_loss_vs_error(runs, colors, figs, title)
    fig_evals_to_target(runs, colors, figs, title, args.target, train_mod.MAX_EVALS)
    summary_table(runs, figs)
    fig_error_map(data_mod, results_dir, train_mod.run_name, args.map, args.seed,
                  best[args.map], figs, title)

    print(f"Figures saved in {figs}")
    plt.show()