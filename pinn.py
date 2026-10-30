import torch
import torch.nn as nn


class PINN (nn.Module):
    def __init__(self, in_dim=1):
        super().__init__()
        self.net = nn.Sequential(
    nn.Linear(in_dim, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 1),
    )

    def forward(self, *inputs):
        return self.net(torch.cat(inputs, dim=1))

if __name__ == "__main__":
    model = PINN()
    x = torch.linspace(0,1,100).reshape(-1,1)
    u = model(x)
    print(u.shape)

