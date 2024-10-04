import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Residual_Block(nn.Module):

    def __init__(self, in_dim):
        super().__init__()

        hidden = 64
        self.f = nn.Sequential(nn.Linear(in_dim, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, hidden),
                               nn.ReLU(),
                               nn.Linear(hidden, in_dim))

    def forward(self, x):
        h = self.f(x)
        return h + x


class MLP(nn.Module):

    def __init__(self, input_dim, output_dim, params_network=None):
        super().__init__()

        hidden = 64 if params_network is None else params_network['n_neurons']
        n_layers = 2 if params_network is None else params_network['n_layers']

        self.f = nn.ModuleList()
        self.f.append(nn.Linear(input_dim, hidden))
        self.f.append(nn.ReLU())
        if params_network is not None and params_network['residual'] == 0:
            for _ in range(n_layers):
                self.f.append(Residual_Block(hidden))
                self.f.append(nn.ReLU())
        else:
            for _ in range(n_layers):
                self.f.append(nn.Linear(hidden, hidden))
                self.f.append(nn.ReLU())
        self.f.append(nn.Linear(hidden, output_dim))

    def forward(self, x):
        h = x
        for layer in self.f:
            h = layer(h)
        return h


class Arnold_Liouville(nn.Module):

    def __init__(self, input_size, latent_size, reg_energy, params_network=None):
        super().__init__()

        assert latent_size % 2 == 0  # torus representation requires even number of dimensions

        self.latent_size = latent_size
        self.reg_energy = reg_energy

        self.phi = MLP(input_size, latent_size, params_network)
        self.psi = MLP(latent_size, input_size, params_network)

        self.distance = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        z = self.phi(x).view(-1, self.latent_size // 2, 2)
        return z

    def get_next_state(self, z):
        
        alpha = 1 / (torch.norm(z, dim=-1))
        
        s = torch.sin(alpha)
        c = torch.cos(alpha)
        rot = torch.stack([torch.stack([c, -s], -1),
                           torch.stack([s, c], -1)], -1)

        return torch.squeeze(torch.matmul(torch.unsqueeze(z, -2), rot), -2)

    def get_energy(self, z):
        return torch.exp(-(torch.linalg.norm(z, dim=-1))/self.reg_energy)

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





class Koopman(nn.Module):

    def __init__(self, input_size, latent_size, params_network=None, device=None):
        super().__init__()

        self.latent_size = latent_size

        self.phi = MLP(input_size, latent_size, params_network)
        self.psi = MLP(latent_size, input_size, params_network)
        self.K = nn.Parameter(torch.eye(latent_size), requires_grad=True)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

        print()

    def forward(self, x):
        z = self.phi(x)
        return z

    def get_next_state(self, z):
        return z @ self.K

    def optimize(self, batch):
        x, x1, H = batch

        z = self(x)
        z1 = self(x1)

        z1_hat = self.get_next_state(z)

        x_hat = self.psi(z)
        x1_hat = self.psi(z1)

        psi_loss = (torch.mean((x - x_hat) ** 2) + torch.mean((x1 - x1_hat) ** 2))
        phi_loss = torch.mean((z1 - z1_hat) ** 2)

        self.opt.zero_grad()
        (psi_loss + phi_loss).backward()
        self.opt.step()

        return psi_loss.detach().cpu().item(), phi_loss.detach().cpu().item()
