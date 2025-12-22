# ==================== utils/LearnableFeatureWeights.py ====================
import torch
import torch.nn as nn

class LearnableFeatureWeights(nn.Module):
    """
    Expert Prior + Learnable Calibration
    w_i = w_i(expert) * exp(delta_i)
    """
    def __init__(self, expert_weights: torch.Tensor):
        super().__init__()
        self.register_buffer("expert_w", expert_weights.clone())
        self.delta = nn.Parameter(torch.zeros_like(expert_weights))

    def forward(self, feat_dim=None):
        w = self.expert_w * torch.exp(self.delta)
        if feat_dim is not None:
            # 自动裁剪（No-Static / No-CPI 必须）
            w = w[:feat_dim]
        return w

class ConstrainedWeightModule(nn.Module):
    def __init__(self, base_weights, learnable_mask, min_weight=0.1, max_weight=5.0):
        """
        优化后的权重约束模块
        :param base_weights: 基础权重张量 (例如全 1.0)
        :param learnable_mask: 哪些特征是可学习的 (0或1的张量)
        :param min_weight: 全局或逐特征的最小值限制
        :param max_weight: 全局或逐特征的最大值限制 (建议设大一点，给 5090 发挥空间)
        """
        super().__init__()

        # 确保基础参数不参与反向传播
        self.register_buffer('base_weights', base_weights.clone().detach())
        self.register_buffer('learnable_mask', learnable_mask.clone().detach().float())

        # 支持标量或张量形式的边界限制
        if not torch.is_tensor(min_weight):
            min_weight = torch.full_like(base_weights, min_weight)
        if not torch.is_tensor(max_weight):
            max_weight = torch.full_like(base_weights, max_weight)

        self.register_buffer('min_weight', min_weight)
        self.register_buffer('max_weight', max_weight)

        # 可学习的偏移量：初始化为 0
        # 注意：这里去掉了硬性的 clamp，让梯度更自由
        self.delta = nn.Parameter(torch.zeros_like(base_weights))

    def forward(self):
        # 方案：使用 sigmoid 映射 delta，确保学习过程平滑
        # 最终权重 = min + (max - min) * sigmoid(delta)
        # 这样 delta 即使很大，权重也会平滑地趋近于边界，而不是直接撞墙
        normalized_delta = torch.sigmoid(self.delta)
        weights = self.min_weight + (self.max_weight - self.min_weight) * normalized_delta

        # 仅针对可学习的部分应用
        final_weights = torch.where(self.learnable_mask.bool(), weights, self.base_weights)
        return final_weights

    def get_raw_delta(self):
        """调试用：查看未经限制的原始偏移量"""
        return self.delta.detach()

class DualObjectiveWeight(nn.Module):
    """
    Dual weight heads for:
    - w_tc  : temporal consistency
    - w_sil : cluster separation
    """
    def __init__(self, dim):
        super().__init__()

        # ===== 共享基础权重（你原来的机制）=====
        self.base = ConstrainedWeightModule(
            base_weights=torch.ones(dim),
            learnable_mask=torch.ones(dim),
            min_weight=0.1,
            max_weight=3.0
        )

        # ===== Silhouette 导向（允许极化）=====
        self.sil_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )

        # ===== Temporal Consistency 导向（偏稳定）=====
        self.tc_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Sigmoid()
        )

    def forward(self, z):
        """
        z: [B, T, D]
        """
        z_mean = z.mean(dim=1)          # [B, D]
        w_base = self.base()            # [D]

        w_sil = w_base * self.sil_head(z_mean)
        w_tc  = w_base * self.tc_head(z_mean)

        return w_sil, w_tc

# 修改建议：在你的 PPO 权重模块中引入“概率灵敏度”
class ConstrainedWeightModuleVae(nn.Module):
    def __init__(self, input_dim, output_dim = 6):
        super(ConstrainedWeightModuleVae, self).__init__()
        self.output_dim = output_dim
        # 输入现在包含：特征原始误差 + 均值mu + 方差sigma
        self.net = nn.Sequential(
            nn.Linear(input_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Softmax(dim=-1) # 强制权重归一化，防止撞墙
        )

    def forward(self, feature_err, mu, logvar):
        # 拼接误差和分布特征
        combined_obs = torch.cat([feature_err, mu, logvar], dim=-1)
        weights = self.net(combined_obs)
        # 这里的 weights 能够反映出：哪些特征在当前分布下最“异常”
        return weights * self.output_dim # 保持权重均值为1，避免数值爆炸