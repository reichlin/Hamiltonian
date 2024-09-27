import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, tanh=False):
        super().__init__()

        hidden = 64
        n_layers = 2
        self.tanh = tanh

        self.f = nn.ModuleList()
        self.f.append(nn.Linear(input_dim, hidden))
        self.f.append(nn.ReLU())
        for _ in range(n_layers):
            self.f.append(nn.Linear(hidden, hidden))
            self.f.append(nn.ReLU())
        self.f.append(nn.Linear(hidden, output_dim))

    def forward(self, x):
        h = x
        for layer in self.f:
            h = layer(h)
        return h if not self.tanh else F.tanh(h)


class Arnold_Liouville(nn.Module):

    def __init__(self, input_size):
        super().__init__()

        self.phi = MLP(input_size, input_size)
        self.psi = MLP(input_size, input_size)

        self.distance = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        z = self.phi(x).view(-1, x.shape[-1] // 2, 2)
        return z

    def get_next_state(self, z):
        
        alpha = 1 / (torch.norm(z, dim=-1))
        
        s = torch.sin(alpha)
        c = torch.cos(alpha)
        rot = torch.stack([torch.stack([c, -s], -1),
                           torch.stack([s, c], -1)], -1)

        return torch.squeeze(torch.matmul(torch.unsqueeze(z, -2), rot), -2)

    def get_energy(self, z):
        return torch.exp(-(torch.linalg.norm(z, dim=-1))/1.0)

    def optimize(self, batch):

        x, x1, H = batch
        B = x.shape[0]

        z = self(x)
        z1 = self(x1)

        z1_hat = self.get_next_state(z)

        energy = torch.mean(self.get_energy(z) + self.get_energy(z1))

        x_hat = self.psi(z.view(B, -1))
        x1_hat = self.psi(z1.view(B, -1))

        psi_loss = (torch.mean((x - x_hat)**2) + torch.mean((x1 - x1_hat)**2))
        phi_loss = torch.mean((z1 - z1_hat) ** 2)

        self.opt.zero_grad()
        (psi_loss+phi_loss+energy).backward()
        self.opt.step()

        return psi_loss.detach().cpu().item(), phi_loss.detach().cpu().item()
