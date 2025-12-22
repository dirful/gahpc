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
        """
        计算最终权重的逻辑
        """
        # 核心逻辑：权重 = 基础值 + 可学习偏移 * 掩码
        # 我们不再对 delta 进行限制，只对最终生成的 weights 进行限制
        weights = self.base_weights + self.delta * self.learnable_mask

        # 使用 clamp 确保最终结果在业务允许的物理范围内
        # 建议 max_weight 设为 5.0 甚至更高，以区分特征重要性
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)

        return weights

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
