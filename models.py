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

        self.phi = MLP(input_size, 2) #int(input_size/2))
        self.Integrator = MLP(int(input_size/2), int(input_size/2)) #nn.Linear(1, int(input_size/2))
        self.Actor = MLP(int(input_size / 2), int(input_size / 2), True)
        self.psi = MLP(3, 2) #input_size, input_size)

        self.distance = nn.CosineSimilarity(dim=1, eps=1e-6)

        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)

    def get_repr(self, x):
        v = self.phi(x)
        return F.normalize(v, p=2.0, dim=-1)

    def get_next_state(self, z, K):

        # return (z + K) % 2*np.pi


        s = torch.sin(K)
        c = torch.cos(K)
        rot = torch.stack([torch.stack([c, -s], -1),
                           torch.stack([s, c], -1)], -1)
        return z @ rot #torch.squeeze(torch.bmm(torch.unsqueeze(z, 1), rot))


    def optimize(self, batch):

        x, x1, H = batch

        I = self.Integrator(H)

        K = torch.squeeze(self.Actor(H))
        s = torch.sin(K)
        c = torch.cos(K)
        rot = torch.stack([torch.stack([c, -s], -1),
                           torch.stack([s, c], -1)], -1)

        z = self.get_repr(x)
        z1 = self.get_repr(x1)

        z1_hat = torch.squeeze(torch.bmm(torch.unsqueeze(z, 1), rot))

        x_hat = self.psi(torch.cat([z, I], -1))
        x1_hat = self.psi(torch.cat([z1, I], -1))
        x1_dyn_hat = 0 #self.psi(torch.cat([z1_hat, I], -1))

        psi_loss = (torch.mean((x - x_hat)**2) + torch.mean((x1 - x1_hat)**2) + torch.mean((x1 - x1_dyn_hat)**2))
        phi_loss = -torch.log(torch.mean(self.distance(z1, z1_hat)) + 1)
        # phi_loss = torch.mean((z1 - z1_hat) ** 2)
        # phi_loss = torch.mean((z1 - z - K)**2)

        # idx = np.arange(z1.shape[0])
        # np.random.shuffle(idx)
        # phi_loss_neg = torch.mean(self.distance(z1, z1_hat[idx]))

        self.opt.zero_grad()
        (psi_loss+phi_loss).backward()
        self.opt.step()

        return psi_loss.detach().cpu().item(), phi_loss.detach().cpu().item()
