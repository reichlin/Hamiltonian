import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataloader import Pendulum
from models import Arnold_Liouville


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("./logs/boh_harmonic_overfit_1traj_eucl")

T_trj = 100
N_trj = 1#100
input_dim = 2
EPOCHS = 10000000000000000

dataset = Pendulum(N_trj, T_trj, device)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Arnold_Liouville(input_dim).to(device)

log_idx = 0
for epoch in tqdm(range(EPOCHS)):

    for batch in dataloader:
        psi_loss, phi_loss = model.optimize(batch)

        writer.add_scalar("psi_loss", psi_loss, log_idx)
        writer.add_scalar("dynamics_loss", phi_loss, log_idx)

        log_idx += 1


    # try latent integration

    x, h = dataset.get_traj()
    
    #i = model.Integrator(h[0:1, 0:1])
    #k = torch.squeeze(model.Actor(h[0:1, 0:1]))**2
    z = model.get_repr(x[0:1],h[0:1, 0:1])
    #x_hat = [model.psi(torch.cat([z, i], -1))]
    #x_hat_pred = [model.psi(torch.cat([z, i], -1))]
    x_hat = [model.psi(z)]
    x_hat_pred = [model.psi(z)]
    for t in range(x.shape[0]-1):
        z = model.get_next_state(z)
        # z += k
        x_hat.append(model.psi(model.get_repr(x[t:t+1],h[0:1, 0:1])))

        x_hat_pred.append(model.psi(z))
    x_hat = torch.cat(x_hat, 0)
    x_hat_pred = torch.cat(x_hat_pred, 0)

    fig = plt.figure()
    plt.plot(x[:, 0].detach().cpu().numpy())
    plt.plot(x_hat[:, 0].detach().cpu().numpy())
    plt.plot(x_hat_pred[:, 0].detach().cpu().numpy())
    writer.add_figure("angle", fig, epoch)
    plt.close(fig)
    fig = plt.figure()
    plt.plot(x[:, 1].detach().cpu().numpy())
    plt.plot(x_hat[:, 1].detach().cpu().numpy())
    plt.plot(x_hat_pred[:, 1].detach().cpu().numpy())
    writer.add_figure("angular vel", fig, epoch)
    plt.close(fig)


    # try energy jump


print()




