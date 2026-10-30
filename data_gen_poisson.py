import torch
import numpy as np

Nf = 100
x_f = torch.linspace(0,1,Nf).reshape(-1,1).requires_grad_(True)
print(x_f.shape)

x_b = torch.tensor([[0.],[1.]])
u_b = torch.tensor([[0.],[0.]])


x_test= torch.linspace(0,1,10*Nf).reshape(-1,1)
u_exact = torch.sin(torch.pi*x_test)
t_f = torch.linspace(0,1,Nf).reshape(-1,1).requires_grad_(True)

if __name__ == "__main__":
    print(x_test.shape, u_exact.shape)