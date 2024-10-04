import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import argparse

from dataloader import Pendulum, DoublePendulum, CoupledHarmonicOscillator
from models import Arnold_Liouville, Koopman


parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--model', default=0, type=int)
parser.add_argument('--experiment', default=2, type=int)
parser.add_argument('--N_dim', default=10, type=int)
parser.add_argument('--T_trj', default=100, type=int)
parser.add_argument('--N_trj', default=1000, type=int)
parser.add_argument('--latent_size', default=64, type=int)
parser.add_argument('--n_neurons', default=64, type=int)
parser.add_argument('--n_layers', default=3, type=int)
parser.add_argument('--residual', default=0, type=int)
parser.add_argument('--reg_energy', default=0.0001, type=float)
args = parser.parse_args()


seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.random.manual_seed(seed)
torch.cuda.manual_seed(seed)


all_experiments = {0: "SinglePendulum", 1: "DoublePendulum", 2: "Oscillator"}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

''' HYPERPARAMETERS '''
T_trj = args.T_trj #100
N_trj = args.N_trj #100
N_dim = args.N_dim
latent_size = args.latent_size #input_dim
reg_energy = args.reg_energy
EPOCHS = 10000
experiment = all_experiments[args.experiment]
params_network = {'n_neurons': args.n_neurons, 'n_layers': args.n_layers, 'residual': args.residual}
batch_size = 64
save_gif = False
model_name = "AL" if args.model == 0 else "Koopman"

name_exp = model_name + "_"
name_exp += experiment + "_N=" + str(N_dim) + "_T_trj=" + str(T_trj) + "_N_trj=" + str(N_trj) + "_latent_size=" + str(latent_size) + "_reg_energy=" + str(reg_energy)
name_exp += "_n_neurons=" + str(args.n_neurons) + "_n_layers=" + str(args.n_layers) + "_residual=" + str(args.residual) + "_seed=" + str(seed)

writer = SummaryWriter("./logs_pendulum/" + name_exp)

if experiment == "SinglePendulum":
    dataset = Pendulum(N_trj, T_trj, device)
    input_dim = 2
elif experiment == "DoublePendulum":
    dataset = DoublePendulum(N_trj, T_trj, device)
    input_dim = 4
elif experiment == "Oscillator":
    dataset = CoupledHarmonicOscillator(N_trj, T_trj, N_dim, device)
    input_dim = N_dim*2
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

if model_name == "AL":
    model = Arnold_Liouville(input_dim, latent_size, reg_energy, params_network).to(device)
elif model_name == "Koopman":
    model = Koopman(input_dim, latent_size, params_network, device).to(device)


imgs = []
log_idx = 0
for epoch in tqdm(range(EPOCHS)):

    for batch in dataloader:
        psi_loss, phi_loss = model.optimize(batch)

        writer.add_scalar("psi_loss", psi_loss, log_idx)
        writer.add_scalar("dynamics_loss", phi_loss, log_idx)

        log_idx += 1

    ''' TEST '''
    x, h = dataset.get_traj()
    z = model(x[0:1])
    x_hat = [model.psi(z.view(1, -1))]
    x_hat_pred = [model.psi(z.view(1, -1))]
    for t in range(1, x.shape[0]-1):
        z = model.get_next_state(z)
        x_hat.append(model.psi(model(x[t:t+1]).view(1, -1)))
        x_hat_pred.append(model.psi(z.view(1, -1)))
    x_hat = torch.cat(x_hat, 0)
    x_hat_pred = torch.cat(x_hat_pred, 0)

    # theta_1 = x.detach().cpu().numpy()[:-1, 0]
    # theta_2 = x.detach().cpu().numpy()[:-1, 2] if input_dim == 4 else None
    # real_end_effector_x, real_end_effector_y = dataset.get_end_effector(theta_1, theta_2)
    # theta_1_pred = x_hat_pred.detach().cpu().numpy()[:-1, 0]
    # theta_2_pred = x_hat_pred.detach().cpu().numpy()[:-1, 2] if input_dim == 4 else None
    # pred_end_effector_x, pred_end_effector_y = dataset.get_end_effector(theta_1_pred, theta_2_pred)
    # writer.add_scalar("pred_error_0_5", np.mean((real_end_effector_x[0:5] - pred_end_effector_x[0:5])**2 + (real_end_effector_y[0:5] - pred_end_effector_y[0:5])**2), epoch)
    # writer.add_scalar("pred_error_5_10", np.mean((real_end_effector_x[5:10] - pred_end_effector_x[5:10])**2 + (real_end_effector_y[5:10] - pred_end_effector_y[5:10])**2), epoch)
    # writer.add_scalar("pred_error_15_20", np.mean((real_end_effector_x[15:20] - pred_end_effector_x[15:20])**2 + (real_end_effector_y[15:20] - pred_end_effector_y[15:20])**2), epoch)
    # writer.add_scalar("pred_error_45_50", np.mean((real_end_effector_x[45:50] - pred_end_effector_x[45:50])**2 + (real_end_effector_y[45:50] - pred_end_effector_y[45:50])**2), epoch)
    # writer.add_scalar("pred_error_95_100", np.mean((real_end_effector_x[95:100] - pred_end_effector_x[95:100]) ** 2 + (real_end_effector_y[95:100] - pred_end_effector_y[95:100]) ** 2), epoch)
    # writer.add_scalar("pred_error_100_200", np.mean((real_end_effector_x[100:200] - pred_end_effector_x[100:200]) ** 2 + (real_end_effector_y[100:200] - pred_end_effector_y[100:200]) ** 2), epoch)
    # writer.add_scalar("pred_error_400_498", np.mean((real_end_effector_x[400:498] - pred_end_effector_x[400:498]) ** 2 + (real_end_effector_y[400:498] - pred_end_effector_y[400:498]) ** 2), epoch)

    writer.add_scalar("test_mse", np.mean((x[:-1].detach().cpu().numpy() - x_hat_pred.detach().cpu().numpy()) ** 2), epoch)

    writer.add_scalar("test_mse_0_5", np.mean((x[0:5].detach().cpu().numpy() - x_hat_pred[0:5].detach().cpu().numpy())**2), epoch)
    writer.add_scalar("test_mse_5_10", np.mean((x[5:10].detach().cpu().numpy() - x_hat_pred[5:10].detach().cpu().numpy())**2), epoch)
    writer.add_scalar("test_mse_15_20", np.mean((x[15:20].detach().cpu().numpy() - x_hat_pred[15:20].detach().cpu().numpy())**2), epoch)
    writer.add_scalar("test_mse_45_50", np.mean((x[45:50].detach().cpu().numpy() - x_hat_pred[45:50].detach().cpu().numpy())**2), epoch)
    writer.add_scalar("test_mse_95_100", np.mean((x[95:100].detach().cpu().numpy() - x_hat_pred[95:100].detach().cpu().numpy()) ** 2), epoch)


    if epoch % 1 == 0:
        # fig = plt.figure()
        # # plt.scatter(real_end_effector_x[:20], real_end_effector_y[:20])
        # # plt.plot(real_end_effector_x[:20], real_end_effector_y[:20])
        # # plt.scatter(pred_end_effector_x[:20], pred_end_effector_y[:20])
        # # plt.plot(pred_end_effector_x[:20], pred_end_effector_y[:20])
        # plt.scatter(real_end_effector_x, real_end_effector_y)
        # plt.plot(real_end_effector_x, real_end_effector_y)
        # plt.scatter(pred_end_effector_x, pred_end_effector_y)
        # plt.plot(pred_end_effector_x, pred_end_effector_y)
        # max_l = dataset.l1+dataset.l2
        # plt.ylim(-(max_l + 1), max_l + 1)
        # plt.xlim(-(max_l + 1), max_l + 1)
        # writer.add_figure("end_effector", fig, epoch)
        # plt.close(fig)

        fig = plt.figure()
        plt.plot(x[:, 2].detach().cpu().numpy())
        plt.plot(x_hat[:, 2].detach().cpu().numpy())
        plt.plot(x_hat_pred[:, 2].detach().cpu().numpy())
        writer.add_figure("angle", fig, epoch)
        plt.close(fig)
        fig = plt.figure()
        plt.plot(x[:, 3].detach().cpu().numpy())
        plt.plot(x_hat[:, 3].detach().cpu().numpy())
        plt.plot(x_hat_pred[:, 3].detach().cpu().numpy())
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




