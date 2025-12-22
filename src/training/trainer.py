#!/usr/bin/env python3
"""
trainer.py - 训练相关函数
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List


def train_autoencoder(
        model: nn.Module,
        arr3d: np.ndarray,
        mask: np.ndarray,
        epochs: int = 20,
        batch_size: int = 128,
        device: str = "cpu"
) -> nn.Module:
    """训练自动编码器"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss(reduction="none")

    # 创建数据加载器
    dataset = TensorDataset(
        torch.tensor(arr3d, dtype=torch.float32),
        torch.tensor(mask, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 训练循环
    for ep in range(epochs):
        model.train()
        total_loss = 0

        for xb, mb in loader:
            xb = xb.to(device)
            mb = mb.to(device)

            recon, _ = model(xb)

            loss = loss_fn(recon, xb)                # (B, T, F)
            loss = loss.mean(dim=-1)                 # → (B, T)
            loss = (loss * mb).mean()                # mask loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"[Epoch {ep+1}/{epochs}] loss = {total_loss:.4f}")

    return model

def extract_embeddings(
        model: nn.Module,
        data: np.ndarray,
        device: str = 'cpu',
        batch_size: int = 128
) -> Tuple[np.ndarray, np.ndarray]:
    """提取嵌入和重建数据"""
    model = model.to(device)
    model.eval()
    X = torch.tensor(data, dtype=torch.float32)
    loader = DataLoader(X, batch_size=batch_size)

    embs = []
    recons = []

    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            recon, latent = model(xb)
            embs.append(latent.cpu().numpy())
            recons.append(recon.cpu().numpy())

    embs = np.vstack(embs)
    recons = np.vstack(recons)

    return embs, recons

def create_reconstruction_map(
        reconstructions: np.ndarray,
        sequences: List[np.ndarray],
        job_ids: List[int]
) -> dict:
    """创建重建映射字典"""
    recon_map = {}
    for i, jid in enumerate(job_ids):
        recon_map[jid] = reconstructions[i][:sequences[i].shape[0]]
    return recon_map