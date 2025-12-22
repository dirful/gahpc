import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerVAE(nn.Module):
    def __init__(self, dyn_dim, static_dim, hidden_dim, latent_dim, nhead, num_layers):
        super(TransformerVAE, self).__init__()

        # 1. 编码器部分 (Encoder)
        self.embedding = nn.Linear(dyn_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 2. 生成式核心：变分层 (Variational Layers)
        # 将 Transformer 提取的特征映射为均值 (mu) 和 对数方差 (logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 3. FiLM 模块：静态特征约束 (Conditioning)
        # 用于生成 FiLM 的偏移和缩放系数
        self.film_gen = nn.Sequential(
            nn.Linear(static_dim, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2) # 输出 gamma 和 beta
        )

        # 4. 解码器部分 (Decoder)
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)
        decoder_layers = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layers, num_layers=num_layers)
        self.output_layer = nn.Linear(hidden_dim, dyn_dim)

    def reparameterize(self, mu, logvar):
        """重参数化技巧：从分布中采样 z"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_dyn, x_static):
        # --- Encoder ---
        # x_dyn shape: (batch, seq_len, dyn_dim)
        out = self.embedding(x_dyn)
        out = self.transformer_encoder(out)

        # 取最后一个时间步或进行平均池化作为隐变量基础
        h = out.mean(dim=1)

        # --- Variational Bottleneck ---
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar) # 采样得到原始隐变量

        # --- FiLM Conditioning (注入 Priority 等静态特征) ---
        # 将静态特征调制到采样出的 z 上
        film_params = self.film_gen(x_static)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        z_cond = gamma * z + beta

        # --- Decoder ---
        # 将采样并调制后的 z 映射回 Transformer 隐藏维度
        z_hidden = self.latent_to_hidden(z_cond).unsqueeze(1) # (batch, 1, hidden_dim)

        # 这里的解码过程可以用 z 作为 memory，或者作为输入序列
        # 为了保持时序重构，我们用 z 引导解码
        decoded = self.transformer_decoder(tgt=self.embedding(x_dyn), memory=z_hidden)
        x_recon = self.output_layer(decoded)

        return x_recon, mu, logvar

# 损失函数定义 (供训练循环使用)
def vae_loss_function(recon_x, x, mu, logvar, dynamic_weights, beta_kld=0.01):
    """
    recon_x: 重构值
    x: 原始值
    mu, logvar: 变分参数
    dynamic_weights: PPO 算出来的特征维度权重 (batch, dyn_dim)
    beta_kld: KL 散度的调节系数 (建议初始设小一点)
    """
    # 1. 加权 MSE (结合 PPO 权重)
    # 扩展 weights 到 (batch, seq_len, dyn_dim)
    weights = dynamic_weights.unsqueeze(1).expand_as(x)
    recon_loss = (F.mse_loss(recon_x, x, reduction='none') * weights).mean()

    # 2. KL 散度 (生成式约束)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta_kld * (kld_loss / x.size(0))