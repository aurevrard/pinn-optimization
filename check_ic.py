# check_ic.py - error on the initial condition, uniform vs shock-refined
from pathlib import Path
import torch
from pinn import PINN

x = torch.linspace(-1, 1, 256).reshape(-1, 1)
t = torch.zeros_like(x)
u0 = -torch.sin(torch.pi * x)
runs = {"SGD-momentum": 0.01, "Adam": 0.001, "L-BFGS": 1.0}

for name, lr in runs.items():
    for folder in ["results/burgers", "results/burgers_shock"]:
        errs = []
        for seed in [0, 1, 2]:
            model = PINN(in_dim=2)
            model.load_state_dict(torch.load(Path(folder) / f"{name}_lr{lr:g}_seed{seed}.pt",
                                             weights_only=True))
            with torch.no_grad():
                errs.append((torch.linalg.norm(model(t, x) - u0) / torch.linalg.norm(u0)).item())
        print(f"{name:13s} {folder:24s} IC error = {sorted(errs)[1]:.2e}  (seeds {errs})")