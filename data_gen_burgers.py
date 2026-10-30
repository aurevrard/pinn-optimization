"""
Équation de Burgers u_t + u u_x = nu u_xx sur x ∈ [-1, 1], t ∈ [0, 1],
avec u(0, x) = -sin(pi x) et u(t, ±1) = 0.
Points d'entraînement tirés au hasard, et solution de référence lue dans
data/burgers_shock.mat (solution numérique fournie par Raissi et al.).
"""
from pathlib import Path

import numpy as np
import scipy.io
import torch

torch.manual_seed(0)

# ============================================================
# Paramètres du problème
# ============================================================
NU = 0.01 / np.pi            # viscosité
X_MIN, X_MAX = -1.0, 1.0     # domaine spatial
T_MAX = 1.0                  # domaine temporel [0, T_MAX]

N_f = 2000                   # points de collocation (intérieur)
N_0 = 100                    # points de condition initiale
N_b = 100                    # points de bord (50 par côté)


def uniform(n, a, b):
    """n valeurs tirées uniformément dans [a, b], en colonne (n, 1)."""
    return a + (b - a) * torch.rand(n, 1)


# ============================================================
# Points d'entraînement
# ============================================================
# Collocation : tirage uniforme dans [0, T_MAX] x [X_MIN, X_MAX]
t_f = uniform(N_f, 0.0, T_MAX).requires_grad_(True)
x_f = uniform(N_f, X_MIN, X_MAX).requires_grad_(True)

# Condition initiale : t = 0, u(0, x) = -sin(pi x)
x_0 = uniform(N_0, X_MIN, X_MAX)
t_0 = torch.zeros_like(x_0)
u_0 = -torch.sin(torch.pi * x_0)

# Bords : x = -1 et x = 1 pour des t aléatoires, u = 0
t_b = uniform(N_b, 0.0, T_MAX)
x_b = torch.cat([torch.full((N_b // 2, 1), X_MIN),
                 torch.full((N_b - N_b // 2, 1), X_MAX)])
u_b = torch.zeros(N_b, 1)

# ============================================================
# Solution de référence (grille de test)
# ============================================================
REF_FILE = Path(__file__).resolve().parent / "data" / "burgers_shock.mat"
ref = scipy.io.loadmat(REF_FILE)

t_ref = ref["t"].flatten()             # instants, forme (NT,)
x_ref = ref["x"].flatten()             # positions, forme (NX,)
u_ref = np.real(ref["usol"]).T         # solution, forme (NT, NX) : lignes = t, colonnes = x
NT, NX = u_ref.shape

T_grid, X_grid = np.meshgrid(t_ref, x_ref, indexing="ij")   # grilles (NT, NX)

# Conversion en tenseurs float32 (le .mat est en float64, le réseau en float32)
t_test = torch.tensor(T_grid.reshape(-1, 1), dtype=torch.float32)
x_test = torch.tensor(X_grid.reshape(-1, 1), dtype=torch.float32)
u_exact = torch.tensor(u_ref.reshape(-1, 1), dtype=torch.float32)


# ============================================================
# Vérifications (lancer : python data_gen_burgers.py)
# ============================================================
if __name__ == "__main__":
    print("Clés du fichier :", [k for k in ref if not k.startswith("__")])
    print(f"Grille de référence : NT = {NT} instants, NX = {NX} positions")
    print(f"t dans [{t_ref.min():.3f}, {t_ref.max():.3f}],  x dans [{x_ref.min():.3f}, {x_ref.max():.3f}]")
    print("Formes :", t_f.shape, x_0.shape, x_b.shape, t_test.shape, u_exact.shape)

    # 1. La référence à t = 0 doit coïncider avec -sin(pi x)
    ecart_ic = np.abs(u_ref[0] + np.sin(np.pi * x_ref)).max()
    print(f"Écart max à t = 0 entre référence et -sin(pi x) : {ecart_ic:.2e}")

    # 2. La référence doit être nulle sur les bords
    ecart_bc = max(np.abs(u_ref[:, 0]).max(), np.abs(u_ref[:, -1]).max())
    print(f"Valeur max de |u| sur les bords : {ecart_bc:.2e}")

    # 3. Le choc : pente maximale à t = 1
    pente = np.abs(np.diff(u_ref[-1]) / np.diff(x_ref)).max()
    print(f"Pente max |u_x| à t = {t_ref[-1]:.2f} : {pente:.1f}")
    # 4. Number of collocation points inside the shock layer
    in_shock = (x_f.abs() < 0.003) & (t_f > 0.3)
    print(f"Collocation points in the shock layer: {int(in_shock.sum())} / {N_f}")