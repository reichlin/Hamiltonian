import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm


class Pendulum(Dataset):

    def __init__(self, N_trj, T_trj, device):

        self.device = device
        self.T_trj = T_trj

        self.m, self.l, self.g = 1, 2, 10
        self.delta_t = 0.1

        self.s = []
        self.s1 = []
        self.H = []

        for i in tqdm(range(N_trj)):

            theta_t, theta_prime_t = np.random.random()*np.pi-np.pi/2, np.random.random()*2-1
            # theta_t, theta_prime_t = -1.198, -2.1239

            all_h = []
            for t in range(self.T_trj):

                # theta_t1 = (theta_t + theta_prime_t * delta_t) % (2*np.pi)
                # theta_prime_t1 = theta_prime_t - (g / l) * np.sin(theta_t) * delta_t
                # H = 0.5 * m * (l*theta_prime_t1)**2 + m * g * l * (1 - np.cos(theta_t1))

                # LeapFrog (grazie er vate)
                theta_prime_t_half = theta_prime_t - (self.g / self.l) * np.sin(theta_t) * (self.delta_t / 2)
                theta_t1 = (theta_t + theta_prime_t_half * self.delta_t) #% (2 * np.pi)
                theta_prime_t1 = theta_prime_t_half - (self.g / self.l) * np.sin(theta_t1) * (self.delta_t / 2)
                H = 0.5 * self.m * (self.l*theta_prime_t1)**2 + self.m * self.g * self.l * (1 - np.cos(theta_t1))

                self.s.append(np.array([theta_t, theta_prime_t]))
                self.s1.append(np.array([theta_t1, theta_prime_t1]))
                #self.H.append(H)
                all_h.append(H)

                theta_t = theta_t1
                theta_prime_t = theta_prime_t1

            self.H.extend([np.mean(np.array(all_h)) for _ in range(len(all_h))])



    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):

        s = torch.from_numpy(self.s[idx]).float().to(self.device)
        s1 = torch.from_numpy(self.s1[idx]).float().to(self.device)
        h = torch.from_numpy(np.array([self.H[idx]])).float().to(self.device)

        return s, s1, h

    # def get_traj(self):
    #      return torch.from_numpy(np.stack(self.s[:self.T_trj], 0)).float().to(self.device), torch.from_numpy(np.array([self.H[:self.T_trj]])).float().to(self.device)
    

    def get_traj(self):

        trajectory = []
        Hamiltonian = []

        # theta_t, theta_prime_t = np.random.random() * 2 * np.pi, np.random.random() * 2 - 1
        theta_t, theta_prime_t = np.random.random() * np.pi - np.pi / 2, np.random.random() * 2 - 1

        all_h = []
        for t in range(self.T_trj):

            # LeapFrog (grazie er vate)
            theta_prime_t_half = theta_prime_t - (self.g / self.l) * np.sin(theta_t) * (self.delta_t / 2)
            theta_t1 = (theta_t + theta_prime_t_half * self.delta_t) #% (2 * np.pi)
            theta_prime_t1 = theta_prime_t_half - (self.g / self.l) * np.sin(theta_t1) * (self.delta_t / 2)
            H = 0.5 * self.m * (self.l * theta_prime_t1) ** 2 + self.m * self.g * self.l * (1 - np.cos(theta_t1))

            trajectory.append(np.array([theta_t, theta_prime_t]))
            all_h.append(H)

            theta_t = theta_t1
            theta_prime_t = theta_prime_t1

        Hamiltonian = np.expand_dims(np.stack(([np.mean(np.array(all_h)) for _ in range(len(all_h))])), -1)

        return torch.from_numpy(np.stack(trajectory, 0)).float().to(self.device), torch.from_numpy(Hamiltonian).float().to(self.device)
