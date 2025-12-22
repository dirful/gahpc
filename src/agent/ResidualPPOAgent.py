import torch

from agent.ResidualActor import ResidualActor


class ResidualPPOAgent:
    def __init__(
            self,
            state_dim,
            residual_scale=0.02,   # ±2%
            device="cpu"
    ):
        self.actor = ResidualActor(state_dim).to(device)
        self.residual_scale = residual_scale
        self.device = device

    @torch.no_grad()
    def select_residual(self, state, cpu_dir, mem_dir):
        """
        Residual 必须服从 safe direction
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        residual = self.actor(state).cpu().numpy()

        cpu_res = residual[0] * self.residual_scale
        mem_res = residual[1] * self.residual_scale

        # ========= 强约束：不允许反向 =========
        cpu_res *= cpu_dir
        mem_res *= mem_dir

        return cpu_res, mem_res
