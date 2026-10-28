import torch
import torch.nn as nn


class PINN (nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
    nn.Linear(1, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 20),
    nn.Tanh(),
    nn.Linear(20, 1),
    )

    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    model = PINN()
    x = torch.linspace(0,1,100).reshape(-1,1)
    u = model(x)
    print(u.shape)

