"""
MQRNN (Multi-Quantile RNN) 子模块
=================================
对应论文 Section 3.2.2: "DF_submodule is designed as a multiquantile RNN (MQRNN)"

参考 Wen, Torkkola, Narayanaswamy (2017) "A multi-horizon quantile recurrent forecaster"

核心思想：
- 输入: 多条时间序列 + 协变量
- 输出: 多个分位数 (paper 中 Q=6: 均值, 10%, 60%, 70%, 80%, 90%, 95%)
- 损失: 多分位数 pinball loss (quantile loss)

结构：
    Input (T_steps, n_features)
        ↓
    LSTM/GRU (global encoder)
        ↓
    RepeatVector (horizon)
        ↓
    LSTM/GRU (local decoder)
        ↓
    TimeDistributed Dense(Q)
        ↓
    Output (horizon, Q)
"""

import torch
import torch.nn as nn
import numpy as np


def quantile_loss(pred: torch.Tensor, target: torch.Tensor, quantiles: list) -> torch.Tensor:
    """
    多分位数 pinball loss (a.k.a. quantile loss)

    L_q(y, ŷ) = q * max(y - ŷ, 0) + (1-q) * max(ŷ - y, 0)

    Args:
        pred:   (batch, Q) 分位数预测
        target: (batch,) 真实值
        quantiles: list of quantile levels, e.g. [0.1, 0.5, 0.9]

    Returns:
        scalar loss
    """
    qs = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device)
    # target: (batch,) -> (batch, Q)
    target = target.unsqueeze(-1).expand_as(pred)
    errors = target - pred
    loss = torch.max(qs * errors, (qs - 1) * errors)
    return loss.mean()


class MQRNN(nn.Module):
    """
    多分位数 RNN 模块

    Args:
        input_dim:     输入特征维度 (时间步 * 特征数 flatten 后, 或直接当 1D)
        hidden_size:   LSTM 隐藏层维度 (paper default 50)
        n_layers:      LSTM 层数
        horizon:       预测的时间步长 (paper 用 review period 长度)
        quantiles:     分位数水平列表
        dropout:       dropout 比例
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 50,
        n_layers: int = 1,
        horizon: int = 7,
        quantiles: list = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.10, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
            # paper 中的 Q=6 + 均值（用 0.5 分位数代表）
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.horizon = horizon
        self.quantiles = quantiles

        # Encoder: 把输入时间序列压缩成一个 context vector
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # Decoder: 从 context vector 解码到 horizon 个时间步
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
        )

        # 输出层: 每个时间步输出 Q 个分位数
        self.output_layer = nn.Linear(hidden_size, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim) 输入时间序列
        Returns:
            (batch, horizon, Q) 分位数预测
        """
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # hidden: (n_layers, batch, hidden_size)

        # Repeat context 到 horizon 个时间步
        context = hidden[-1].unsqueeze(1).repeat(1, self.horizon, 1)  # (batch, horizon, hidden)

        # Decode
        decoded, _ = self.decoder(context)  # (batch, horizon, hidden)

        # Project 到分位数
        out = self.output_layer(decoded)  # (batch, horizon, Q)
        return out

    def predict_mean(self, x: torch.Tensor) -> torch.Tensor:
        """返回中位数（0.5 分位数）作为点预测"""
        out = self.forward(x)
        # 找到 0.5 分位数的索引
        try:
            idx = self.quantiles.index(0.5)
        except ValueError:
            # 如果没有 0.5，取所有分位数的均值
            return out.mean(dim=-1)
        return out[:, :, idx]

    def predict_all_quantiles(self, x: torch.Tensor) -> torch.Tensor:
        """返回所有分位数的预测"""
        return self.forward(x)


if __name__ == "__main__":
    # 自检
    model = MQRNN(input_dim=10, hidden_size=50, horizon=7)
    x = torch.randn(8, 30, 10)  # batch=8, seq_len=30, features=10
    out = model(x)
    print(f"MQRNN 输入: {x.shape}, 输出: {out.shape}")
    print(f"分位数: {model.quantiles}")
    assert out.shape == (8, 7, len(model.quantiles)), f"shape mismatch: {out.shape}"
    print("✓ MQRNN 自检通过")