import torch

x = torch.linspace(0, 1, 50).reshape(-1, 1).requires_grad_(True)
u = x**3

u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]

print(u_x, u_xx)


