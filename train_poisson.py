import torch
import matplotlib.pyplot as plt
from pinn import PINN             # ta classe de réseau
from data_gen_poisson import x_f, x_b, u_b, x_test,u_exact  # tes points et ta solution exacte


torch.manual_seed(0)
model = PINN()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

def loss_fn():
    u = model(x_f)
    u_x = torch.autograd.grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
    r = u_xx + torch.pi**2*torch.sin(torch.pi*x_f)
    loss_pde = torch.mean(r**2)
    loss_bc = torch.mean ((model(x_b) - u_b)**2)
    return loss_pde + loss_bc 

for it in range (5000):
    optimizer.zero_grad()
    loss = loss_fn()
    loss.backward()
    optimizer.step()
    if it % 500 ==0:
        print(it, loss.item())


with torch.no_grad():
    u_pred = model(x_test)

err_L2 = torch.norm(u_pred - u_exact) / torch.norm(u_exact)
print("Erreur L2 relative :", err_L2.item())

plt.plot(x_test, u_exact, label="exacte sin(πx)")
plt.plot(x_test, u_pred, "--", label="PINN")
plt.legend()
plt.show()