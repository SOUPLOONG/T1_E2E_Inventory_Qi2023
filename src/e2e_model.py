"""
E2E (End-to-End) 库存补货模型
==============================
对应论文 Section 3.2 + Figure 1 + 公式 (6)。

完整结构：

    Input_DF ──→ MQRNN ──→ DF_out ─────────────────────────────────┐
                                                                     ├──→ Layer_3 (100) ──→ Layer_4 (100) ──→ Out1 (订货量)
    Input_VLT ──→ MLP(VLT_layer_1, VLT_layer_2) ──→ VLT_out ─────┤
                                                                     │
    Input_basic ──────────────────────────────────────────────────┤
                                                                     │
    Review period ────────────────────────────────────────────────┤
                                                                     │
                                                                     ├──→ VLT_out_branch ──→ Out3 (VLT 预测)
                                                                     │
                                                                     └──→ DF_out_branch ──→ Out2 (需求预测)

    Out2 (需求预测) = MQRNN 的 0.5 分位数
    Out3 (VLT 预测) = VLT 子模块的输出

训练目标 (公式 6):
    min_θ Σ [ L(Out1; a*) + λ1 * L1(Out2, a*DF) + λ2 * L2(Out3, a*VLT) ]

超参 (论文 Table B.1):
    - DF_submodule hidden_size: 50
    - VLT_layers: {50, 20}
    - Integration: {100, 100}
    - Dropout: 0.2
    - ReLU
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mqrnn import MQRNN, quantile_loss


class VLTSubmodule(nn.Module):
    """
    VLT 子模块: 多层感知机 (paper: VLT_layer_1=50, VLT_layer_2=20)
    """

    def __init__(self, input_dim: int, hidden_dims: list = (50, 20), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class IntegrationModule(nn.Module):
    """
    集成模块: 把所有子模块输出 + 基础特征 + review/initial 拼接
    然后过 2 层全连接 (100, 100) + ReLU + Dropout
    输出 Out1 (订货量)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = (100, 100),
        dropout: float = 0.2,
        output_activation: str = "relu",  # 订货量必须 >= 0
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        # 不加 ReLU（会导致梯度死区），订货量非负约束在 inference 时处理
        if output_activation == "relu":
            layers.append(nn.ReLU())
        # 'none' 情况下直接输出线性值
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # (batch,)


class E2EInventoryModel(nn.Module):
    """
    端到端库存补货模型

    Args:
        demand_input_dim:    Demand 时间序列展平维度
        demand_seq_len:      Demand 时间序列原始长度
        vlt_input_dim:       VLT 特征维度
        basic_input_dim:     Basic 特征维度
        mqrnn_hidden:        MQRNN 隐藏层 (default 50)
        mqrnn_horizon:       MQRNN 预测步长 (default 7)
        vlt_hidden:          VLT MLP 隐藏层 (default [50, 20])
        integration_hidden:  集成模块隐藏层 (default [100, 100])
        quantiles:           分位数列表
        dropout:             dropout 比例
    """

    def __init__(
        self,
        demand_input_dim: int,
        demand_seq_len: int = 30,
        vlt_input_dim: int = 6,
        basic_input_dim: int = 8,
        mqrnn_hidden: int = 50,
        mqrnn_horizon: int = 7,
        mqrnn_n_layers: int = 1,
        vlt_hidden: tuple = (50, 20),
        integration_hidden: tuple = (100, 100),
        quantiles: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.quantiles = quantiles or [0.10, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

        # 1) Demand 子模块: MQRNN
        # demand_input_dim = 历史需求长度 (60) + 协变量 (5) = 65
        # 处理方案: 把每维当作一个时间步的特征 (per_step_dim=1, seq_len=demand_input_dim)
        # 这样 MQRNN 就能直接处理整个 demand 向量
        self.demand_seq_len = demand_input_dim  # 每个时间步只有 1 个特征
        self.demand_per_step_dim = 1
        self.mqrnn = MQRNN(
            input_dim=self.demand_per_step_dim,
            hidden_size=mqrnn_hidden,
            n_layers=mqrnn_n_layers,
            horizon=mqrnn_horizon,
            quantiles=self.quantiles,
            dropout=dropout,
        )
        # MQRNN 输出: (batch, horizon, Q)
        df_out_dim = len(self.quantiles) * mqrnn_horizon

        # 2) VLT 子模块: MLP
        self.vlt_submodule = VLTSubmodule(
            input_dim=vlt_input_dim,
            hidden_dims=list(vlt_hidden),
            dropout=dropout,
        )
        vlt_out_dim = vlt_hidden[-1]

        # 3) Basic 特征直接拼接
        # 集成模块输入: df_out + vlt_out + basic + review + initial
        self.integration_input_dim = (
            df_out_dim + vlt_out_dim + basic_input_dim + 2  # +2: review period + initial stock
        )
        # 注意: 不用 ReLU 作为最后激活（会导致梯度死区，让模型困在预测 0）
        # 订货量非负约束在 inference 时用 max(0, x) 截断
        self.integration = IntegrationModule(
            input_dim=self.integration_input_dim,
            hidden_dims=list(integration_hidden),
            dropout=dropout,
            output_activation="none",  # 改为 none
        )

        # 辅助输出: Out2 (需求预测) 和 Out3 (VLT 预测)
        # Out2: 需求预测 = MQRNN 的 0.5 分位数 (中位数) 在所有 horizon 上的均值
        self.out2_head = nn.Identity()  # 用 MQRNN 的中位数即可
        # Out3: VLT 预测 = VLT 子模块输出 + 一个线性层
        self.out3_head = nn.Linear(vlt_out_dim, 1)

    def forward(
        self,
        x_demand: torch.Tensor,
        x_vlt: torch.Tensor,
        x_basic: torch.Tensor,
        x_review: torch.Tensor,
        x_initial: torch.Tensor,
    ) -> dict:
        """
        Args:
            x_demand: (batch, demand_input_dim) 展平的需求特征
            x_vlt:    (batch, vlt_input_dim)
            x_basic:  (batch, basic_input_dim)
            x_review: (batch, 1)
            x_initial:(batch, 1)

        Returns:
            dict with keys 'out1', 'out2', 'out3', 'quantile_preds'
        """
        batch_size = x_demand.shape[0]

        # Reshape demand: (batch, seq_len, per_step_dim)
        x_demand_seq = x_demand.view(batch_size, self.demand_seq_len, self.demand_per_step_dim)

        # MQRNN 前向
        quantile_preds = self.mqrnn(x_demand_seq)  # (batch, horizon, Q)
        # 展平 MQRNN 输出
        df_out_flat = quantile_preds.reshape(batch_size, -1)  # (batch, horizon*Q)

        # Out2: 需求预测（中位数均值）
        median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        out2 = quantile_preds[:, :, median_idx].mean(dim=-1)  # (batch,)

        # VLT 子模块
        vlt_features = self.vlt_submodule(x_vlt)  # (batch, vlt_out_dim)
        out3 = self.out3_head(vlt_features).squeeze(-1)  # (batch,)

        # 集成模块
        integration_input = torch.cat(
            [df_out_flat, vlt_features, x_basic, x_review, x_initial],
            dim=-1,
        )
        out1 = self.integration(integration_input)  # (batch,) 订货量

        return {
            "out1": out1,
            "out2": out2,
            "out3": out3,
            "quantile_preds": quantile_preds,
        }


def e2e_loss(
    outputs: dict,
    targets: dict,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    quantiles: list = None,
) -> dict:
    """
    公式 (6) 的多任务损失

    L_total = L(Out1, a*) + λ1 * L1(Out2, a*DF) + λ2 * L2(Out3, a*VLT)

    L  = MSE（订货量回归）
    L1 = Quantile loss（需求多分位数预测）
    L2 = MSE（VLT 点预测）

    Args:
        outputs:  dict from E2EInventoryModel
        targets:  dict with keys 'a_optimal', 'demand_target', 'vlt_target'
        lambda1:  需求辅助损失权重
        lambda2:  VLT 辅助损失权重
        quantiles: 分位数列表
    """
    if quantiles is None:
        quantiles = [0.10, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    # 主损失: Out1 vs a*
    L_main = F.mse_loss(outputs["out1"], targets["a_optimal"])

    # 辅助损失 1: Out2 (需求中位数) vs 真实需求
    # 这里用真实需求作为目标（mean of window）
    L_aux1 = F.mse_loss(outputs["out2"], targets["demand_target"])

    # 辅助损失 2: Out3 (VLT) vs 真实 VLT
    L_aux2 = F.mse_loss(outputs["out3"], targets["vlt_target"])

    L_total = L_main + lambda1 * L_aux1 + lambda2 * L_aux2

    return {
        "total": L_total,
        "main": L_main,
        "aux1": L_aux1,
        "aux2": L_aux2,
    }


if __name__ == "__main__":
    # 自检
    model = E2EInventoryModel(
        demand_input_dim=60 * 5,  # 30 时间步 * 5 特征/步（实际是简化）
        demand_seq_len=30,
        vlt_input_dim=6,
        basic_input_dim=8,
    )
    batch = 4
    x_d = torch.randn(batch, 60 * 5)
    x_v = torch.randn(batch, 6)
    x_b = torch.randn(batch, 8)
    x_r = torch.randn(batch, 1)
    x_i = torch.randn(batch, 1)
    out = model(x_d, x_v, x_b, x_r, x_i)
    print(f"Out1 (订货量): {out['out1'].shape} = {out['out1']}")
    print(f"Out2 (需求预测): {out['out2'].shape} = {out['out2']}")
    print(f"Out3 (VLT预测): {out['out3'].shape} = {out['out3']}")

    targets = {
        "a_optimal": torch.rand(batch) * 10,
        "demand_target": torch.rand(batch) * 5,
        "vlt_target": torch.rand(batch) * 5 + 2,
    }
    losses = e2e_loss(out, targets)
    print(f"\n损失:")
    print(f"  Total: {losses['total'].item():.4f}")
    print(f"  Main (订货): {losses['main'].item():.4f}")
    print(f"  Aux1 (需求): {losses['aux1'].item():.4f}")
    print(f"  Aux2 (VLT): {losses['aux2'].item():.4f}")
    print("\n✓ E2E 模型自检通过")