#!/usr/bin/env python3
"""
autoencoders.py - LSTM和GRU自动编码器模型
"""
import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    """LSTM自动编码器"""
    def __init__(self, input_dim, enc_hidden_dim=64, latent_dim=2, dec_hidden_dim=2):
        super().__init__()
        assert latent_dim == dec_hidden_dim, \
            "latent_dim must equal decoder LSTM hidden_dim for size compatibility"

        self.latent_dim = latent_dim
        self.dec_hidden_dim = dec_hidden_dim

        # ---------- 编码器 ----------
        self.encoder_lstm = nn.LSTM(
            input_dim,
            enc_hidden_dim,
            batch_first=True
        )
        self.fc_enc = nn.Linear(enc_hidden_dim, latent_dim)

        # ---------- 解码器 ----------
        self.decoder_lstm = nn.LSTM(
            input_dim,
            dec_hidden_dim,
            batch_first=True
        )
        self.fc_dec = nn.Linear(dec_hidden_dim, input_dim)

    def encode(self, x):
        """编码器前向传播"""
        # x: (B, T, F)
        _, (h, _) = self.encoder_lstm(x)      # h = (1, B, enc_hidden_dim)
        h = h[0]                               # → (B, enc_hidden_dim)
        latent = self.fc_enc(h)                # → (B, latent_dim)
        return latent

    def decode(self, latent, seq_len):
        """解码器前向传播"""
        B = latent.size(0)
        # 解码初始隐状态
        h0 = latent.unsqueeze(0)                           # (1, B, latent_dim=dec_hidden_dim)
        c0 = torch.zeros_like(h0)

        # 输入：全部是 0
        dec_input = torch.zeros(B, seq_len, self.latent_dim, device=latent.device)

        out, _ = self.decoder_lstm(dec_input, (h0, c0))    # (B, T, dec_hidden_dim)
        recon = self.fc_dec(out)                           # (B, T, input_dim)
        return recon

    def forward(self, x):
        """完整前向传播"""
        seq_len = x.size(1)
        latent = self.encode(x)
        recon = self.decode(latent, seq_len)
        return recon, latent

class GRUAutoencoder(nn.Module):
    """GRU自动编码器"""
    def __init__(self, input_dim: int, hidden_dim: int = 64,
                 latent_dim: int = 32, num_layers: int = 1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.encoder_gru = nn.GRU(input_dim, hidden_dim,
                                  num_layers=num_layers, batch_first=True)
        self.hidden2latent = nn.Linear(hidden_dim, latent_dim)

        self.latent2hidden = nn.Linear(latent_dim, hidden_dim)
        self.decoder_gru = nn.GRU(hidden_dim, input_dim,
                                  num_layers=num_layers, batch_first=True)

    def encode(self, x):
        """编码器前向传播"""
        _, h_n = self.encoder_gru(x)
        h = h_n[-1]
        latent = self.hidden2latent(h)
        return latent

    def decode(self, latent, seq_len):
        """解码器前向传播"""
        h = self.latent2hidden(latent).unsqueeze(0).repeat(self.num_layers, 1, 1)
        dec_input = torch.zeros((latent.size(0), seq_len, self.hidden_dim),
                                device=latent.device)
        out, _ = self.decoder_gru(dec_input, h)
        if out.size(-1) != self.input_dim:
            proj = nn.Linear(out.size(-1), self.input_dim).to(out.device)
            out = proj(out)
        return out

    def forward(self, x):
        """完整前向传播"""
        latent = self.encode(x)
        recon = self.decode(latent, x.size(1))
        return recon, latent