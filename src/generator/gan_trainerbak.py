# src/generator/gan_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from log.logger import get_logger

logger = get_logger(__name__)

class SimpleGenerator(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, z):
        return self.net(z)

class SimpleDiscriminator(nn.Module):
    def __init__(self, in_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class GANTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.latent_dim = getattr(cfg, 'gan_latent_dim', 32)
        # 根据数据模式确定输出维度
        self.data_mode = getattr(cfg, 'data_mode', 'job_stats')

        if self.data_mode == 'job_stats':
            # job统计特征模式：输出4个基本特征
            self.out_dim = 4  # cpu, mem, disk_io, duration
        else:
            # 时间序列模式：需要更多特征
            self.out_dim = getattr(cfg, 'gan_out_dim', 15)  # 默认15个job特征

        self.generator = SimpleGenerator(self.latent_dim, self.out_dim,
                                         getattr(cfg, 'gan_hidden_dim', 128))
        self.discriminator = SimpleDiscriminator(self.out_dim,
                                                 getattr(cfg, 'gan_hidden_dim', 128))

        self.optim_G = optim.Adam(self.generator.parameters(),
                                  lr=getattr(cfg, 'gan_lr', 1e-4))
        self.optim_D = optim.Adam(self.discriminator.parameters(),
                                  lr=getattr(cfg, 'gan_lr', 1e-4))
        self.loss_fn = nn.BCELoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        logger.info(f"GAN初始化: latent_dim={self.latent_dim}, out_dim={self.out_dim}, device={self.device}")

    def train(self, seqs, epochs=None):
        """训练GAN"""
        if seqs is None or len(seqs) == 0:
            logger.warning("训练数据为空，跳过GAN训练")
            return

        # 确保数据是2D的
        if len(seqs.shape) == 3:
            # 展平时间序列数据
            N, T, F = seqs.shape
            seqs = seqs.reshape(N * T, F)
            logger.info(f"展平序列数据: {seqs.shape}")

        if seqs.shape[1] != self.out_dim:
            logger.warning(f"数据特征维度 {seqs.shape[1]} 与GAN输出维度 {self.out_dim} 不匹配")
            # 如果数据维度大于输出维度，截断
            if seqs.shape[1] > self.out_dim:
                seqs = seqs[:, :self.out_dim]
                logger.info(f"截断数据到前 {self.out_dim} 个特征")
            # 如果数据维度小于输出维度，填充
            else:
                padding = np.zeros((seqs.shape[0], self.out_dim - seqs.shape[1]))
                seqs = np.hstack([seqs, padding])
                logger.info(f"填充数据到 {self.out_dim} 个特征")

        epochs = epochs or getattr(self.cfg, 'gan_epochs', 100)
        batch_size = getattr(self.cfg, 'gan_batch_size', 64)
        n_samples = len(seqs)

        # 转换数据为tensor
        real_data = torch.FloatTensor(seqs).to(self.device)

        logger.info(f"开始GAN训练: {n_samples} 样本, {epochs} 轮次")

        for epoch in range(epochs):
            # 训练判别器
            self.discriminator.zero_grad()

            # 真实数据
            real_labels = torch.ones(batch_size, 1).to(self.device)
            idx = torch.randint(0, n_samples, (batch_size,))
            real_batch = real_data[idx]
            real_output = self.discriminator(real_batch)
            d_real_loss = self.loss_fn(real_output, real_labels)

            # 生成数据
            z = torch.randn(batch_size, self.latent_dim).to(self.device)
            fake_batch = self.generator(z)
            fake_labels = torch.zeros(batch_size, 1).to(self.device)
            fake_output = self.discriminator(fake_batch.detach())
            d_fake_loss = self.loss_fn(fake_output, fake_labels)

            # 判别器总损失
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.optim_D.step()

            # 训练生成器
            self.generator.zero_grad()
            gen_labels = torch.ones(batch_size, 1).to(self.device)
            gen_output = self.discriminator(fake_batch)
            g_loss = self.loss_fn(gen_output, gen_labels)
            g_loss.backward()
            self.optim_G.step()

            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: D_loss={d_loss.item():.4f}, G_loss={g_loss.item():.4f}")

        logger.info("GAN训练完成")

    def sample(self, n=8):
        """从 generator 采样 n 条样本"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(self.device)
            out = self.generator(z).cpu().numpy()

        # 根据数据模式进行后处理
        if self.data_mode == 'job_stats':
            # job统计模式：输出基本特征
            out[:, 0] = np.clip(out[:, 0], 0.0, 1.0)  # cpu
            out[:, 1] = np.clip(out[:, 1], 0.0, 1.0)  # mem
            out[:, 2] = np.clip(out[:, 2], 0.0, 10.0)  # disk_io
            out[:, 3] = np.clip(np.abs(out[:, 3]), 1.0, 3600.0)  # duration

            samples = []
            for row in out:
                samples.append({
                    "cpu": float(row[0]),
                    "mem": float(row[1]),
                    "disk_io": float(row[2]),
                    "duration": float(row[3])
                })
        else:
            # 时间序列模式：输出所有特征
            samples = []
            for row in out:
                sample = {}
                for i in range(min(self.out_dim, 15)):  # 最多15个特征
                    if i == 0:  # cpu_mean
                        sample['cpu'] = float(np.clip(row[i], 0.0, 1.0))
                    elif i == 4:  # mem_mean
                        sample['mem'] = float(np.clip(row[i], 0.0, 1.0))
                    elif i == 8:  # duration_sec
                        sample['duration'] = float(np.clip(np.abs(row[i]), 1.0, 3600.0))
                    elif i == 9:  # cpu_intensity
                        sample['cpu_intensity'] = float(row[i])
                    elif i == 10:  # mem_intensity
                        sample['mem_intensity'] = float(row[i])

                # 确保有基本特征
                if 'cpu' not in sample:
                    sample['cpu'] = 0.5
                if 'mem' not in sample:
                    sample['mem'] = 0.3
                if 'duration' not in sample:
                    sample['duration'] = 300.0
                sample['disk_io'] = 0.2  # 默认值

                samples.append(sample)

        return samples