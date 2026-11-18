"""
训练循环
=========
对应论文 Section 4.2 的离线实验设置。

论文参数 (Table B.1):
- Optimizer: SGD
- Learning rate: 0.001 (默认)
- Momentum: 0.85
- Mini-batch: 64
- Weight init: Gaussian(0, 0.01)
- Activation: ReLU
- Dropout: 0.2

Loss (公式 6):
- L_main: MSE(Out1, a*)
- L_aux1: Quantile loss for MQRNN
- L_aux2: MSE(Out3, vlt_target)
- λ1, λ2: 小正数 (e.g., 0.1)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple
from .e2e_model import E2EInventoryModel, e2e_loss


def train_e2e_model(
    model: E2EInventoryModel,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray] = None,
    n_epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 0.001,
    momentum: float = 0.85,
    lambda1: float = 0.1,
    lambda2: float = 0.1,
    device: str = "cpu",
    verbose: bool = True,
    optimizer_type: str = "adam",  # 'adam' or 'sgd'
) -> Dict:
    """
    训练 E2E 模型

    Args:
        model:       E2EInventoryModel 实例
        train_data:  含 X_demand, X_vlt, X_basic, X_review, X_initial, Y_optimal
        val_data:    验证集（同结构）
        n_epochs:    训练轮数
        batch_size:  小批量大小
        learning_rate:
        momentum:
        lambda1, lambda2:  辅助损失权重
        device:      'cpu' or 'cuda'
        optimizer_type: 'adam' 或 'sgd'

    Returns:
        history: {'train_loss': [...], 'val_loss': [...], ...}
    """
    model = model.to(device)
    if optimizer_type.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    # 构造 DataLoader
    X_d = torch.tensor(train_data["X_demand"], dtype=torch.float32)
    X_v = torch.tensor(train_data["X_vlt"], dtype=torch.float32)
    X_b = torch.tensor(train_data["X_basic"], dtype=torch.float32)
    X_r = torch.tensor(train_data["X_review"], dtype=torch.float32)
    X_i = torch.tensor(train_data["X_initial"], dtype=torch.float32)
    Y_opt = torch.tensor(train_data["Y_optimal"], dtype=torch.float32)

    # Out2 的目标: 真实需求均值（用特征里的 demand 历史均值近似）
    D_target = torch.tensor(
        train_data["X_demand"][:, :30].mean(axis=1),  # 用历史需求均值作为目标
        dtype=torch.float32,
    )
    # Out3 的目标: 用 X_vlt 第 1 列作为 VLT 目标（mean of past VLT）
    V_target = torch.tensor(
        train_data["X_vlt"][:, 0],
        dtype=torch.float32,
    )

    dataset = TensorDataset(X_d, X_v, X_b, X_r, X_i, Y_opt, D_target, V_target)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 验证集
    val_loader = None
    if val_data is not None:
        X_d_v = torch.tensor(val_data["X_demand"], dtype=torch.float32)
        X_v_v = torch.tensor(val_data["X_vlt"], dtype=torch.float32)
        X_b_v = torch.tensor(val_data["X_basic"], dtype=torch.float32)
        X_r_v = torch.tensor(val_data["X_review"], dtype=torch.float32)
        X_i_v = torch.tensor(val_data["X_initial"], dtype=torch.float32)
        Y_opt_v = torch.tensor(val_data["Y_optimal"], dtype=torch.float32)
        D_t_v = torch.tensor(
            val_data["X_demand"][:, :30].mean(axis=1),
            dtype=torch.float32,
        )
        V_t_v = torch.tensor(val_data["X_vlt"][:, 0], dtype=torch.float32)
        val_ds = TensorDataset(X_d_v, X_v_v, X_b_v, X_r_v, X_i_v, Y_opt_v, D_t_v, V_t_v)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [], "train_main": [], "train_aux1": [], "train_aux2": [],
        "val_loss": [], "val_main": [],
    }

    for epoch in range(n_epochs):
        model.train()
        ep_loss = ep_main = ep_a1 = ep_a2 = 0.0
        n_batches = 0
        for batch in loader:
            x_d, x_v, x_b, x_r, x_i, y_opt, d_t, v_t = [b.to(device) for b in batch]

            optimizer.zero_grad()
            outputs = model(x_d, x_v, x_b, x_r, x_i)
            targets = {
                "a_optimal": y_opt,
                "demand_target": d_t,
                "vlt_target": v_t,
            }
            losses = e2e_loss(outputs, targets, lambda1=lambda1, lambda2=lambda2)
            losses["total"].backward()
            # 梯度裁剪防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_loss += losses["total"].item()
            ep_main += losses["main"].item()
            ep_a1 += losses["aux1"].item()
            ep_a2 += losses["aux2"].item()
            n_batches += 1

        history["train_loss"].append(ep_loss / n_batches)
        history["train_main"].append(ep_main / n_batches)
        history["train_aux1"].append(ep_a1 / n_batches)
        history["train_aux2"].append(ep_a2 / n_batches)

        # 验证
        if val_loader is not None:
            model.eval()
            val_loss = val_main = 0.0
            n_v = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_d, x_v, x_b, x_r, x_i, y_opt, d_t, v_t = [b.to(device) for b in batch]
                    outputs = model(x_d, x_v, x_b, x_r, x_i)
                    targets = {
                        "a_optimal": y_opt,
                        "demand_target": d_t,
                        "vlt_target": v_t,
                    }
                    losses = e2e_loss(outputs, targets, lambda1=lambda1, lambda2=lambda2)
                    val_loss += losses["total"].item()
                    val_main += losses["main"].item()
                    n_v += 1
            history["val_loss"].append(val_loss / n_v)
            history["val_main"].append(val_main / n_v)

        if verbose and (epoch + 1) % max(1, n_epochs // 10) == 0:
            msg = f"Epoch {epoch+1:3d}/{n_epochs} | Train Loss: {history['train_loss'][-1]:.4f}"
            if val_loader is not None:
                msg += f" | Val Loss: {history['val_loss'][-1]:.4f}"
            print(msg)

    return history


def predict_e2e(
    model: E2EInventoryModel,
    data: Dict[str, np.ndarray],
    device: str = "cpu",
) -> np.ndarray:
    """用训练好的 E2E 模型预测所有样本的订货量（非负截断）"""
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        X_d = torch.tensor(data["X_demand"], dtype=torch.float32, device=device)
        X_v = torch.tensor(data["X_vlt"], dtype=torch.float32, device=device)
        X_b = torch.tensor(data["X_basic"], dtype=torch.float32, device=device)
        X_r = torch.tensor(data["X_review"], dtype=torch.float32, device=device)
        X_i = torch.tensor(data["X_initial"], dtype=torch.float32, device=device)
        out = model(X_d, X_v, X_b, X_r, X_i)
        # 订货量必须非负
        return torch.clamp(out["out1"], min=0.0).cpu().numpy()


if __name__ == "__main__":
    # 自检: 用小数据训练一个 epoch
    from data_simulator import InventoryDataSimulator
    sim = InventoryDataSimulator(n_sku=20, n_orders_per_sku=4)
    data = sim.generate()

    # 划分 train/val
    n = len(data["Y_optimal"])
    idx = np.random.permutation(n)
    train_idx = idx[:int(n * 0.8)]
    val_idx = idx[int(n * 0.8):]
    train_data = {k: v[train_idx] for k, v in data.items() if isinstance(v, np.ndarray) and k.startswith(("X_", "Y_"))}
    val_data = {k: v[val_idx] for k, v in data.items() if isinstance(v, np.ndarray) and k.startswith(("X_", "Y_"))}

    model = E2EInventoryModel(
        demand_input_dim=data["X_demand"].shape[1],
        demand_seq_len=30,
        vlt_input_dim=data["X_vlt"].shape[1],
        basic_input_dim=data["X_basic"].shape[1],
    )
    history = train_e2e_model(
        model, train_data, val_data,
        n_epochs=5, batch_size=16, verbose=True,
    )
    print("\n✓ 训练流程自检通过")