# timegan.py - 简化版 TimeGAN for 多变量时序生成
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

class Embedder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, x):
        out, h = self.rnn(x)
        return h[-1]  # 取最后隐藏状态作为嵌入

class Recovery(nn.Module):
    def __init__(self, hidden_dim, input_dim, seq_len, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self.seq_len = seq_len

    def forward(self, h):
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # [B, T, H]
        out, _ = self.rnn(h)
        return self.fc(out)  # [B, T, D]

class Generator(nn.Module):
    def __init__(self, hidden_dim, noise_dim, seq_len, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(noise_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.seq_len = seq_len

    def forward(self, z):
        z = z.unsqueeze(1).repeat(1, self.seq_len, 1)
        out, _ = self.rnn(z)
        return torch.sigmoid(self.fc(out))  # 或 tanh，根据数据范围

class Supervisor(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

    def forward(self, h):
        out, _ = self.rnn(h)
        return out

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, num_layers=3):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, h = self.rnn(x)
        return torch.sigmoid(self.fc(h[-1]))

class TimeGAN:
    def __init__(self, input_dim, seq_len, hidden_dim=64, noise_dim=32, num_layers=3, device="cuda"):
        self.device = device
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

        self.embedder = Embedder(input_dim, hidden_dim, num_layers).to(device)
        self.recovery = Recovery(hidden_dim, input_dim, seq_len, num_layers).to(device)
        self.supervisor = Supervisor(hidden_dim, num_layers).to(device)
        self.generator = Generator(hidden_dim, noise_dim, seq_len, num_layers).to(device)
        self.discriminator = Discriminator(hidden_dim, num_layers).to(device)

        self.e_opt = optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=1e-3)
        self.s_opt = optim.Adam(self.supervisor.parameters(), lr=1e-3)
        self.g_opt = optim.Adam(list(self.generator.parameters()) + list(self.supervisor.parameters()), lr=1e-3)
        self.d_opt = optim.Adam(self.discriminator.parameters(), lr=1e-3)

        self.bce = nn.BCELoss()

    def train_autoencoder(self, loader, epochs=50):
        for epoch in range(epochs):
            for x in loader:
                x = x[0].to(self.device)  # [B, T, D]
                h = self.embedder(x)
                x_rec = self.recovery(h)
                loss_ae = torch.mean((x - x_rec)**2)
                self.e_opt.zero_grad()
                loss_ae.backward()
                self.e_opt.step()

    def train_supervisor(self, loader, epochs=50):
        for epoch in range(epochs):
            for x in loader:
                x = x[0].to(self.device)
                h = self.embedder(x)
                h_sup = self.supervisor(h)
                loss_s = torch.mean((h_sup - h)**2)
                self.s_opt.zero_grad()
                loss_s.backward()
                self.s_opt.step()

    def train_gan(self, loader, epochs=100):
        for epoch in range(epochs):
            for x in loader:
                x = x[0].to(self.device)
                batch_size = x.size(0)

                # Train Discriminator
                z = torch.randn(batch_size, self.hidden_dim).to(self.device)
                h_fake = self.generator(z)
                h_sup = self.supervisor(h_fake)
                y_fake = self.discriminator(h_sup.detach())
                y_real = self.discriminator(self.embedder(x))

                loss_d = self.bce(y_real, torch.ones_like(y_real)) + self.bce(y_fake, torch.zeros_like(y_fake))
                self.d_opt.zero_grad()
                loss_d.backward()
                self.d_opt.step()

                # Train Generator
                y_fake = self.discriminator(self.supervisor(self.generator(z)))
                loss_g = self.bce(y_fake, torch.ones_like(y_fake))
                self.g_opt.zero_grad()
                loss_g.backward()
                self.g_opt.step()

    def generate(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.hidden_dim).to(self.device)
            h_fake = self.generator(z)
            h_sup = self.supervisor(h_fake)
            synth = self.recovery(h_sup)
        return synth.cpu().numpy()