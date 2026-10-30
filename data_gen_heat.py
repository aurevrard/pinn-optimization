import torch

torch.manual_seed(0)

NU = 0.1          # paramètre de l'EDP
N_f = 2000        # points de collocation (intérieur)
N_0 = 100         # points de condition initiale
N_b = 100         # points de bord (50 par côté)

def exact_solution(t, x, nu=NU):
    return torch.exp(-nu * torch.pi**2 * t) * torch.sin(torch.pi * x)

# --- Collocation : tirage uniforme dans [0,1] x [0,1] ---
t_f = torch.rand(N_f, 1).requires_grad_(True)
x_f = torch.rand(N_f, 1).requires_grad_(True)

# --- Condition initiale : t = 0, u(0,x) = sin(pi x) ---
x_0 = torch.rand(N_0, 1)
t_0 = torch.zeros_like(x_0)
u_0 = torch.sin(torch.pi * x_0)

# --- Bords : x = 0 et x = 1 pour des t aléatoires, u = 0 ---
t_b = torch.rand(N_b, 1)
x_b = torch.cat([torch.zeros(N_b // 2, 1), torch.ones(N_b - N_b // 2, 1)])
u_b = torch.zeros(N_b, 1)

# --- Grille de test régulière 100 x 100 ---
t_grid, x_grid = torch.meshgrid(torch.linspace(0, 1, 100),
                                torch.linspace(0, 1, 100), indexing="ij")
t_test = t_grid.reshape(-1, 1)
x_test = x_grid.reshape(-1, 1)
u_exact = exact_solution(t_test, x_test)

if __name__ == "__main__":
    print(t_f.shape, x_0.shape, x_b.shape, t_test.shape, u_exact.shape)