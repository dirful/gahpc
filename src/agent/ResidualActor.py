from torch import nn


class ResidualActor(nn.Module):
    """
    PPO Residual Actor
    只输出很小的 residual（[-1, 1] → scaled）
    """
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Tanh()   # residual ∈ [-1, 1]
        )

    def forward(self, x):
        return self.net(x)
