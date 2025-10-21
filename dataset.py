import torch
from torch.utils.data import Dataset
import numpy as np
import random

class SystemIdentificationDataset(Dataset):
    def __init__(self, num_signals, horizon, input_dim, state_dim, output_dim, closed_loop, input_noise_std, output_noise_std, fixed_x0, seed, u_ext_index):
        """
        Args:
            num_signals (int): Number of independent input signals (trajectories)
            horizon (int): Duration of each signal (timesteps)
            input_dim (int): Input signal dimension
            output_dim (int): System output dimension
            closed_loop: takes input_data (Tensor) and returns system outputs (Tensor)
        """
        self.num_signals = num_signals
        self.horizon = horizon
        self.input_dim = input_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.closed_loop = closed_loop
        self.input_noise_std = input_noise_std
        self.output_noise_std = output_noise_std
        self.fixed_x0 = fixed_x0
        self.u_ext_index = u_ext_index
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Generate white noise input signals (num_signals, horizon, input_dim)
        # If u_ext_index is None, generate disturbance for all input channels
        ext_input_dim = 1 if u_ext_index is not None else input_dim
        self.external_input_data = torch.randn((self.num_signals, self.horizon, ext_input_dim)) * self.input_noise_std

        #Generate the batched initial conditions
        if self.fixed_x0 is not None:
            self.x0 = self.fixed_x0.expand(self.num_signals, 1, self.state_dim)
        else:
            self.x0 = (torch.rand((self.num_signals, 1, self.state_dim)) * 10) - 5  # Uniform initialization between -5 and 5
            self.output_noise = torch.randn((self.num_signals, self.horizon, self.output_dim)) * self.output_noise_std
        # Compute corresponding closed-loop system signals
        self.plant_input_data, self.output_data = closed_loop(self.x0, self.external_input_data, self.u_ext_index, self.output_noise_std)  # Must return a tensor

    def __len__(self):
        return self.plant_input_data.shape[0]

    def __getitem__(self, idx):
        r = self.external_input_data[idx, :, :]
        u = self.plant_input_data[idx, :, :]
        y = self.output_data[idx, :, :]
        return r, u, y

