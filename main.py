import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from dataloader import Pendulum, DoublePendulum
from models import Arnold_Liouville


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter("./logs/double_debug2")

T_trj = 100
N_trj = 100
input_dim = 4 #2
EPOCHS = 10000000000000000

save_gif = False

dataset = DoublePendulum(N_trj, T_trj, device) #Pendulum(N_trj, T_trj, device) #Pendulum(N_trj, T_trj, device) #DoublePendulum(N_trj, T_trj, device) #Pendulum(N_trj, T_trj, device)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

model = Arnold_Liouville(input_dim).to(device)

imgs = []
log_idx = 0
for epoch in tqdm(range(EPOCHS)):

    for batch in dataloader:
        psi_loss, phi_loss = model.optimize(batch)

        writer.add_scalar("psi_loss", psi_loss, log_idx)
        writer.add_scalar("dynamics_loss", phi_loss, log_idx)

        log_idx += 1

    x, h = dataset.get_traj()


    ''' TEST '''

    z = model(x[0:1])
    x_hat = [model.psi(z.view(1, -1))]
    x_hat_pred = [model.psi(z.view(1, -1))]
    for t in range(x.shape[0]-1):
        z = model.get_next_state(z)
        x_hat.append(model.psi(model(x[t:t+1]).view(1, -1)))
        x_hat_pred.append(model.psi(z.view(1, -1)))
    x_hat = torch.cat(x_hat, 0)
    x_hat_pred = torch.cat(x_hat_pred, 0)

    theta_1 = x.detach().cpu().numpy()[:, 0]
    theta_2 = x.detach().cpu().numpy()[:, 2] if input_dim == 4 else None
    real_end_effector_x, real_end_effector_y = dataset.get_end_effector(theta_1, theta_2)
    theta_1_pred, theta_2_pred = x_hat_pred.detach().cpu().numpy()[:, 0], x_hat_pred.detach().cpu().numpy()[:, 2]
    pred_end_effector_x, pred_end_effector_y = dataset.get_end_effector(theta_1_pred, theta_2_pred)

    writer.add_scalar("pred_error_5", np.mean((real_end_effector_x[:5] - pred_end_effector_x[:5])**2 + (real_end_effector_y[:5] - pred_end_effector_y[:5])**2), epoch)
    writer.add_scalar("pred_error_10", np.mean((real_end_effector_x[:10] - pred_end_effector_x[:10])**2 + (real_end_effector_y[:10] - pred_end_effector_y[:10])**2), epoch)
    writer.add_scalar("pred_error_20", np.mean((real_end_effector_x[:20] - pred_end_effector_x[:20])**2 + (real_end_effector_y[:20] - pred_end_effector_y[:20])**2), epoch)
    writer.add_scalar("pred_error_50", np.mean((real_end_effector_x[:50] - pred_end_effector_x[:50])**2 + (real_end_effector_y[:50] - pred_end_effector_y[:50])**2), epoch)

    fig = plt.figure()
    plt.scatter(real_end_effector_x[:20], real_end_effector_y[:20])
    plt.plot(real_end_effector_x[:20], real_end_effector_y[:20])
    plt.scatter(pred_end_effector_x[:20], pred_end_effector_y[:20])
    plt.plot(pred_end_effector_x[:20], pred_end_effector_y[:20])
    writer.add_figure("end_effector", fig, epoch)
    plt.close(fig)

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

    if input_dim == 2:
        all_s = np.stack(dataset.s, 0)
        all_s_tensor = torch.from_numpy(all_s).float().to(device)
        all_z = torch.squeeze(model(all_s_tensor)).detach().cpu().numpy()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.scatter(all_s[:, 0], all_s[:, 1], s=1)
        ax1.set_title('Phase Space')
        ax2.scatter(all_z[:, 0], all_z[:, 1], s=1)
        max_z = np.max(np.abs(all_z))
        max_z += 0.2*np.max(np.abs(all_z))
        ax2.set_ylim(-max_z, max_z)
        ax2.set_xlim(-max_z, max_z)
        ax2.set_title('Latent Representation')
        writer.add_figure("representation", fig, epoch)
        plt.close(fig)

    if input_dim == 2 and save_gif:

        if not os.path.exists('imgs'):
            os.makedirs('imgs')

        fig = plt.figure()
        plt.scatter(all_z[:, 0], all_z[:, 1], s=1)
        max_z = np.max(np.abs(all_z))
        max_z += 0.2*np.max(np.abs(all_z))
        plt.ylim(-max_z, max_z)
        plt.xlim(-max_z, max_z)
        plt.savefig("imgs/" + str(epoch) + ".png")
        imgs.append(Image.open("imgs/" + str(epoch) + ".png"))

        imgs[0].save("latent_evolution.gif", save_all=True, append_images=imgs, optimize=False, duration=100, loop=0)


print()




