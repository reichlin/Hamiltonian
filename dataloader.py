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

        self.init_angle = [np.pi, -np.pi]
        self.init_vel = [2, -2]

        self.s = []
        self.s1 = []
        self.H = []

        for i in tqdm(range(N_trj)):

            s, H, s1 = self.generate_traj()

            self.s.append(s)
            self.s1.append(H)
            self.H.append(s1)

        self.s = np.concatenate(self.s, 0)
        self.H = np.concatenate(self.H, 0)
        self.s1 = np.concatenate(self.s1, 0)


    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):

        s = torch.from_numpy(self.s[idx]).float().to(self.device)
        s1 = torch.from_numpy(self.s1[idx]).float().to(self.device)
        h = torch.from_numpy(np.array([self.H[idx]])).float().to(self.device)

        return s, s1, h

    def generate_traj(self):

        all_s, all_h, all_s1 = [], [], []

        theta_t = np.random.random() * (self.init_angle[0]-self.init_angle[1]) + self.init_angle[1]
        theta_prime_t = np.random.random() * (self.init_vel[0]-self.init_vel[1]) + self.init_vel[1]

        for t in range(self.T_trj):

            # LeapFrog (grazie er vate)
            theta_prime_t_half = theta_prime_t - (self.g / self.l) * np.sin(theta_t) * (self.delta_t / 2)
            theta_t1 = (theta_t + theta_prime_t_half * self.delta_t)
            theta_prime_t1 = theta_prime_t_half - (self.g / self.l) * np.sin(theta_t1) * (self.delta_t / 2)
            H = 0.5 * self.m * (self.l * theta_prime_t1) ** 2 + self.m * self.g * self.l * (1 - np.cos(theta_t1))

            all_s.append(np.array([theta_t, theta_prime_t]))
            all_h.append(H)
            all_s1.append(np.array([theta_t1, theta_prime_t1]))

            theta_t = theta_t1
            theta_prime_t = theta_prime_t1

        Hamiltonian = np.expand_dims(np.stack(([np.mean(np.array(all_h)) for _ in range(len(all_h))])), -1)

        return np.stack(all_s, 0), Hamiltonian, np.stack(all_s1, 0)


    def get_traj(self):

        trajectory, Hamiltonian, _ = self.generate_traj()

        return torch.from_numpy(trajectory).float().to(self.device), torch.from_numpy(Hamiltonian).float().to(self.device)

    def get_end_effector(self, theta_1, theta_2=None):

        x1 = self.l * np.sin(theta_1)
        y1 = -self.l * np.cos(theta_1)

        return x1, y1


class DoublePendulum(Dataset):

    def __init__(self, N_trj, T_trj, device):

        self.device = device
        self.T_trj = T_trj

        self.m1, self.m2, self.l1, self.l2, self.g = 1, 0.5, 2, 3, 10
        self.delta_t = 0.01

        self.init_angle1 = [np.pi, -np.pi]
        self.init_vel1 = [2, -2]
        self.init_angle2 = [np.pi, -np.pi]
        self.init_vel2 = [2, -2]

        self.s = []
        self.s1 = []
        self.H = []

        for i in tqdm(range(N_trj)):

            s, H, s1 = self.generate_traj()

            self.s.append(s)
            self.s1.append(s1)
            self.H.append(H)

        self.s = np.concatenate(self.s, 0)
        self.H = np.concatenate(self.H, 0)
        self.s1 = np.concatenate(self.s1, 0)

    # def f(self, theta1, theta2, p1, p2):
    #
    #     l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g
    #
    #     A1 = (p1 * p2 * np.sin(theta1 - theta2)) / (l1 * l2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2))
    #     A2 = 1 / (2 * l1 ** 2 * l2 ** 2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2)) ** 2 * (p1 ** 2 * m2 * l2 ** 2 - 2 * p1 * p2 * m2 * l1 * l2 * np.cos(theta1 - theta2) + p2 ** 2 * (m1 + m2) * l1 ** 2) * np.sin(2 * (theta1 - theta2))
    #
    #     theta1_prime = (p1 * l2 - p2 * l1 * np.cos(theta1 - theta2)) / (l1 ** 2 * l2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2))
    #     theta2_prime = (p2 * (m1 + m2) * l1 - p1 * m2 * l2 * np.cos(theta1 - theta2)) / (m2 * l1 * l2 ** 2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2))
    #
    #     p1_prime = -(m1 + m2) * g * l1 * np.sin(theta1) - A1 + A2
    #     p2_prime = -m2 * g * l2 * np.sin(theta2) + A1 - A2
    #
    #     new_theta1 = theta1 + self.delta_t*theta1_prime
    #     new_theta2 = theta2 + self.delta_t * theta2_prime
    #     new_p1 = p1 + self.delta_t * p1_prime
    #     new_p2 = p2 + self.delta_t * p2_prime
    #
    #     return new_theta1, new_theta2, new_p1, new_p2, theta1_prime, theta2_prime

    def update_momenta_half_step(self, theta1, theta2, p1, p2):
        # Calculate the half-step updates for momenta
        p1_prime, p2_prime = self.p_prime(theta1, theta2, p1, p2)
        new_p1 = p1 + 0.5 * self.delta_t * p1_prime
        new_p2 = p2 + 0.5 * self.delta_t * p2_prime
        return new_p1, new_p2

    def theta_prime(self, theta1, theta2, p1, p2):
        l1, l2, m1, m2 = self.l1, self.l2, self.m1, self.m2
        return (p1 * l2 - p2 * l1 * np.cos(theta1 - theta2)) / (l1 ** 2 * l2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2))

    def p_prime(self, theta1, theta2, p1, p2):
        l1, l2, m1, m2, g = self.l1, self.l2, self.m1, self.m2, self.g

        A1 = (p1 * p2 * np.sin(theta1 - theta2)) / (l1 * l2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2))
        A2 = (p1 ** 2 * m2 * l2 ** 2 - 2 * p1 * p2 * m2 * l1 * l2 * np.cos(theta1 - theta2) +
              p2 ** 2 * (m1 + m2) * l1 ** 2) * np.sin(2 * (theta1 - theta2)) / (
                2 * l1 ** 2 * l2 ** 2 * (m1 + m2 * np.sin(theta1 - theta2) ** 2) ** 2)

        p1_prime = -(m1 + m2) * g * l1 * np.sin(theta1) - A1 + A2
        p2_prime = -m2 * g * l2 * np.sin(theta2) + A1 - A2

        return p1_prime, p2_prime

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):

        s = torch.from_numpy(self.s[idx]).float().to(self.device)
        s1 = torch.from_numpy(self.s1[idx]).float().to(self.device)
        h = torch.from_numpy(self.H[idx]).float().to(self.device)

        return s, s1, h

    def generate_traj(self):

        all_s, all_h, all_s1 = [], [], []

        theta1 = np.random.random() * (self.init_angle1[0] - self.init_angle1[1]) + self.init_angle1[1]
        p1 = np.random.random() * (self.init_vel1[0] - self.init_vel1[1]) + self.init_vel1[1]
        theta2 = np.random.random() * (self.init_angle2[0] - self.init_angle2[1]) + self.init_angle2[1]
        p2 = np.random.random() * (self.init_vel2[0] - self.init_vel2[1]) + self.init_vel2[1]

        for t in range(self.T_trj):

            #new_theta1, new_theta2, new_p1, new_p2, new_theta1_prime, new_theta2_prime = self.f(theta1, theta2, p1, p2)

            p1_half, p2_half = self.update_momenta_half_step(theta1, theta2, p1, p2)
            new_theta1 = theta1 + self.delta_t * self.theta_prime(theta1, theta2, p1_half, p2_half)
            new_theta2 = theta2 + self.delta_t * self.theta_prime(theta2, theta1, p2_half, p1_half)
            new_p1, new_p2 = self.update_momenta_half_step(new_theta1, new_theta2, p1_half, p2_half)
            new_theta1_prime = self.theta_prime(new_theta1, new_theta2, new_p1, new_p2)
            new_theta2_prime = self.theta_prime(new_theta2, new_theta1, new_p2, new_p1)

            H1 = 0.5 * self.m1 * (self.l1 * new_theta1_prime) ** 2 + self.m1 * self.g * self.l1 * (1 - np.cos(new_theta1))
            H2 = 0.5 * self.m2 * (self.l2 * new_theta2_prime) ** 2 + self.m2 * self.g * self.l2 * (1 - np.cos(new_theta2))

            all_s.append(np.array([theta1, theta2, p1, p2]))
            all_h.append(np.array([H1, H2]))
            all_s1.append(np.array([new_theta1, new_theta2, new_p1, new_p2]))

            theta1, theta2, p1, p2 = new_theta1, new_theta2, new_p1, new_p2

        Hamiltonian = np.repeat(np.mean(np.stack(all_h), 0, keepdims=True), len(all_h), 0)

        return np.stack(all_s, 0), Hamiltonian, np.stack(all_s1, 0)

    def get_traj(self):

        trajectory, Hamiltonian, _ = self.generate_traj()

        return torch.from_numpy(trajectory).float().to(self.device), torch.from_numpy(Hamiltonian).float().to(self.device)


    def get_end_effector(self, theta_1, theta_2):

        x1 = self.l1 * np.sin(theta_1)
        y1 = -self.l1 * np.cos(theta_1)

        x2 = x1 + self.l2 * np.sin(theta_2)
        y2 = y1 - self.l2 * np.cos(theta_2)

        return x2, y2






















