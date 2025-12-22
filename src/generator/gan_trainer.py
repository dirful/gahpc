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

        # 关键修改：动态确定输出维度
        self.data_mode = getattr(cfg, 'data_mode', 'job_stats')

        if self.data_mode == 'job_stats':
            # job统计特征模式：输出15个job-level特征
            # 这是根据DBLoader提取的特征确定的
            self.out_dim = 15  # 修改：从4改为15
        else:
            # 时间序列模式：需要更多特征
            self.out_dim = getattr(cfg, 'gan_out_dim', 15)

        # 也可以从配置中读取
        if hasattr(cfg, 'gan_out_dim'):
            self.out_dim = cfg.gan_out_dim

        logger.info(f"GAN输出维度设置为: {self.out_dim}")

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

        # 动态调整GAN输出维度以匹配数据
        data_dim = seqs.shape[1]
        if data_dim != self.out_dim:
            logger.warning(f"数据特征维度 {data_dim} 与GAN输出维度 {self.out_dim} 不匹配")

            # 方案1: 调整GAN输出维度并重新初始化网络
            self.out_dim = data_dim
            logger.info(f"调整GAN输出维度为: {self.out_dim}")

            # 重新初始化网络
            self.generator = SimpleGenerator(self.latent_dim, self.out_dim,
                                             getattr(self.cfg, 'gan_hidden_dim', 128))
            self.discriminator = SimpleDiscriminator(self.out_dim,
                                                     getattr(self.cfg, 'gan_hidden_dim', 128))

            self.generator.to(self.device)
            self.discriminator.to(self.device)

            self.optim_G = optim.Adam(self.generator.parameters(),
                                      lr=getattr(self.cfg, 'gan_lr', 1e-4))
            self.optim_D = optim.Adam(self.discriminator.parameters(),
                                      lr=getattr(self.cfg, 'gan_lr', 1e-4))

            logger.info(f"已重新初始化GAN网络以适应维度 {self.out_dim}")

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

        # 关键修改：根据输出维度进行后处理
        samples = []

        # 定义15个job-level特征的标准顺序
        job_features = [
            'cpu_mean', 'cpu_std', 'cpu_max', 'cpu_95th',
            'mem_mean', 'mem_std', 'mem_max',
            'machines_count', 'task_count',
            'duration_sec', 'cpu_intensity', 'mem_intensity',
            'task_density', 'cpu_cv', 'mem_cv'
        ]

        for row_idx, row in enumerate(out):
            sample = {}

            # 根据实际输出维度处理
            for i in range(min(self.out_dim, len(row))):
                if i < len(job_features):
                    feat_name = job_features[i]
                else:
                    feat_name = f'feature_{i}'

                value = row[i]

                # 根据特征类型进行适当的后处理
                if 'cpu' in feat_name and '_cv' not in feat_name:
                    # CPU相关特征应该在0-1之间
                    sample[feat_name] = float(np.clip(value, 0.0, 1.0))
                elif 'mem' in feat_name and '_cv' not in feat_name:
                    # 内存特征应该是正数（MB）
                    sample[feat_name] = float(np.abs(value) * 1000)
                elif '_cv' in feat_name:
                    # 变异系数应该是正数
                    sample[feat_name] = float(np.abs(value))
                elif feat_name in ['machines_count', 'task_count']:
                    # 计数特征应该是整数
                    sample[feat_name] = int(max(1, np.abs(value)))
                elif feat_name == 'duration_sec':
                    # 持续时间应该是正数
                    sample[feat_name] = float(max(1, np.abs(value)))
                elif 'intensity' in feat_name or 'density' in feat_name:
                    # 强度和密度特征
                    sample[feat_name] = float(value)
                else:
                    sample[feat_name] = float(value)

            # 确保有基本的cpu、mem、duration字段（用于向后兼容）
            if 'cpu_mean' in sample:
                sample['cpu'] = sample['cpu_mean']
            else:
                sample['cpu'] = 0.5

            if 'mem_mean' in sample:
                sample['mem'] = sample['mem_mean'] / 10000.0  # 归一化到0-1
            else:
                sample['mem'] = 0.3

            if 'duration_sec' in sample:
                sample['duration'] = sample['duration_sec']
            else:
                sample['duration'] = 300.0

            sample['disk_io'] = 0.2  # 默认值

            samples.append(sample)

        return samples

    def sample_raw(self, n=8):
        """返回原始样本，不进行后处理（用于评估）"""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim).to(self.device)
            out = self.generator(z).cpu().numpy()
        return out