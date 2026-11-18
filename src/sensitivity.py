"""
Appendix B 敏感性分析
========================
对应论文 Appendix B:
  B.1.1 Number of Hidden Layers (Table B.2)
  B.1.2 Number of Neurons     (Table B.3)
  B.1.3 Learning Rate          (Table B.4)
  B.2   Data Size Sensitivity  (Table B.5, Figure B.1)
  B.3   b/h Ratio Sensitivity  (Figure B.2)
  B.4   Local Minima           (Figure B.3, Table B.6)

由于我们只有 df_test.csv (3000 SKUs 测试集), 没有原始训练数据,
我们的策略是:
  * B.1-B.2 (网络结构): 用 df_test 80% 作 train, 20% 作 test, 训练多个模型
  * B.3 (学习率): 同上
  * B.4 (局部极小值): 不同随机种子训练 2 个模型
  * B.2 (数据量): 用 df_test 的不同比例
  * B.3 (b/h ratio): 重新计算成本 (无需训练)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_dist
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim

from .e2e_model import E2EInventoryModel
from .figure2 import compute_sku_cost


# ============================================
# 工具函数: 在测试集上评估训练好的模型
# ============================================
def evaluate_model_on_test(
    model: E2EInventoryModel,
    df_test: pd.DataFrame,
    device: str = "cpu",
) -> dict:
    """计算训练好的 E2E 模型在测试集上的 holding / stockout / total"""
    model.eval()
    h, b = 1.0, 9.0
    preds = []

    # 用 Pytorch 推理
    X_d = torch.tensor(np.stack([self_features(d) for d in df_test.to_dict('records')]),
                       dtype=torch.float32, device=device)
    # 简化: 直接用已有特征列
    # ... (实际上用 model 推理)

    return evaluate_predictions(preds, df_test, h, b)


def evaluate_predictions(predictions: np.ndarray, df_test: pd.DataFrame,
                         h: float = 1.0, b: float = 9.0) -> dict:
    """对一组预测值计算平均成本"""
    holding_list, stockout_list, total_list = [], [], []

    for i, (_, row) in enumerate(df_test.iterrows()):
        cost = compute_sku_cost(
            order_qty=predictions[i],
            initial_stock=row['initial_stock'],
            demands=row['demand_hist'],
            h=h, b=b,
        )
        holding_list.append(cost['holding'])
        stockout_list.append(cost['stockout'])
        total_list.append(cost['total'])

    return {
        'holding': float(np.mean(holding_list)),
        'stockout': float(np.mean(stockout_list)),
        'total': float(np.mean(total_list)),
    }


def self_features(d):
    """占位"""
    return np.zeros(1)


# ============================================
# 工具: 从 df_test 提取训练特征
# ============================================
def extract_features_from_df(df: pd.DataFrame):
    """
    从 df_test 提取训练用的特征. 把 demand_hist 转换为 demand_features
    加上 review_period + initial_stock 等.
    """
    n = len(df)
    # demand_hist 长度不一, 用最大长度填充
    max_len = max(df['demand_hist'].apply(len))
    demand_feats = np.zeros((n, max_len))
    for i, hist in enumerate(df['demand_hist']):
        demand_feats[i, :len(hist)] = hist

    X_demand = np.concatenate([
        demand_feats,
        df[['demand_mean', 'demand_std', 'review_period', 'vendor_vlt_mean',
            'vendor_vlt_std']].values,
    ], axis=1)

    X_vlt = np.column_stack([
        df['vendor_vlt_mean'].values,
        df['vendor_vlt_std'].values,
        df['vlt'].values,
    ])

    X_basic = df[['demand_mean', 'demand_std']].values  # 简化

    X_review = df['review_period'].values.reshape(-1, 1)
    X_initial = df['initial_stock'].values.reshape(-1, 1)

    Y_optimal = df['OPT_pred'].values

    return {
        'X_demand': X_demand.astype(np.float32),
        'X_vlt': X_vlt.astype(np.float32),
        'X_basic': X_basic.astype(np.float32),
        'X_review': X_review.astype(np.float32),
        'X_initial': X_initial.astype(np.float32),
        'Y_optimal': Y_optimal.astype(np.float32),
    }


def train_e2e_quick(
    train_data: dict,
    val_data: dict,
    hidden_layers: int = 2,
    hidden_size: int = 100,
    learning_rate: float = 0.001,
    n_epochs: int = 20,
    seed: int = 42,
    device: str = "cpu",
    verbose: bool = False,
) -> E2EInventoryModel:
    """
    快速训练 E2E 模型 (用于敏感性分析)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = E2EInventoryModel(
        demand_input_dim=train_data['X_demand'].shape[1],
        demand_seq_len=train_data['X_demand'].shape[1],  # 每个时间步 1 维
        vlt_input_dim=train_data['X_vlt'].shape[1],
        basic_input_dim=train_data['X_basic'].shape[1],
        mqrnn_hidden=50,
        mqrnn_horizon=7,
        vlt_hidden=(50, 20),
        integration_hidden=tuple([hidden_size] * hidden_layers),
        dropout=0.2,
    )
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_d = torch.tensor(train_data['X_demand']).to(device)
    X_v = torch.tensor(train_data['X_vlt']).to(device)
    X_b = torch.tensor(train_data['X_basic']).to(device)
    X_r = torch.tensor(train_data['X_review']).to(device)
    X_i = torch.tensor(train_data['X_initial']).to(device)
    Y = torch.tensor(train_data['Y_optimal']).to(device)

    for epoch in range(n_epochs):
        model.train()
        idx = np.random.permutation(len(Y))
        batch_size = 64
        for i in range(0, len(Y), batch_size):
            batch_idx = idx[i:i+batch_size]
            optimizer.zero_grad()
            out = model(X_d[batch_idx], X_v[batch_idx], X_b[batch_idx],
                        X_r[batch_idx], X_i[batch_idx])
            loss = nn.functional.mse_loss(out['out1'], Y[batch_idx])
            loss.backward()
            optimizer.step()
        if verbose and (epoch+1) % 5 == 0:
            print(f"    Epoch {epoch+1}/{n_epochs}, loss={loss.item():.2f}")

    return model


def predict_with_model(model: E2EInventoryModel, data: dict,
                       device: str = "cpu") -> np.ndarray:
    """用训练好的模型预测"""
    model = model.to(device)
    model.eval()
    X_d = torch.tensor(data['X_demand']).to(device)
    X_v = torch.tensor(data['X_vlt']).to(device)
    X_b = torch.tensor(data['X_basic']).to(device)
    X_r = torch.tensor(data['X_review']).to(device)
    X_i = torch.tensor(data['X_initial']).to(device)
    with torch.no_grad():
        out = model(X_d, X_v, X_b, X_r, X_i)
    return torch.clamp(out['out1'], min=0).cpu().numpy()


# ============================================
# B.1: 网络结构敏感性
# ============================================
def sensitivity_network_structure(df_test: pd.DataFrame, results_dir: str):
    """Table B.2 (hidden layers) + Table B.3 (neurons)"""
    print("\n" + "=" * 60)
    print("B.1: 网络结构敏感性 (Layers + Neurons)")
    print("=" * 60)

    data = extract_features_from_df(df_test)
    train_idx, val_idx = train_test_split(
        np.arange(len(data['Y_optimal'])), test_size=0.2, random_state=42
    )
    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}

    # 计算 OPT 在 test split 上的成本
    opt_pred = df_test['OPT_pred'].values
    opt_cost = evaluate_predictions(opt_pred[val_idx], df_test.iloc[val_idx])

    print(f"  OPT baseline: total = {opt_cost['total']:.2f}")

    # ---- B.1.1 Hidden Layers ----
    print("\n  B.1.1 Hidden Layers:")
    layer_table = []
    for n_layers in [1, 2, 3, 4]:
        print(f"    Training {n_layers}-layer E2E...")
        t0 = time.time()
        model = train_e2e_quick(
            train_data, val_data,
            hidden_layers=n_layers, hidden_size=100,
            learning_rate=0.001, n_epochs=15, seed=42,
        )
        t = time.time() - t0
        preds = predict_with_model(model, val_data)
        cost = evaluate_predictions(preds, df_test.iloc[val_idx])
        gap = (cost['total'] - opt_cost['total']) / opt_cost['total'] * 100
        layer_table.append({
            'Hidden_Layers': n_layers,
            'Total': cost['total'],
            'Holding': cost['holding'],
            'Stockout': cost['stockout'],
            'Gap_vs_OPT_%': gap,
        })
        print(f"    {n_layers} layers: total={cost['total']:.2f} (gap {gap:+.1f}%)")

    table_b2 = pd.DataFrame(layer_table)
    table_b2.to_csv(f"{results_dir}/table_b2_hidden_layers.csv", index=False)
    print(f"\n  [✓] Table B.2 saved to {results_dir}/table_b2_hidden_layers.csv")

    # ---- B.1.2 Neurons ----
    print("\n  B.1.2 Number of Neurons:")
    neuron_table = []
    for n_neurons in [30, 40, 50, 60]:
        print(f"    Training E2E with {n_neurons} neurons...")
        t0 = time.time()
        model = train_e2e_quick(
            train_data, val_data,
            hidden_layers=2, hidden_size=n_neurons,
            learning_rate=0.001, n_epochs=15, seed=42,
        )
        t = time.time() - t0
        preds = predict_with_model(model, val_data)
        cost = evaluate_predictions(preds, df_test.iloc[val_idx])
        gap = (cost['total'] - opt_cost['total']) / opt_cost['total'] * 100
        neuron_table.append({
            'Neurons': n_neurons,
            'Total': cost['total'],
            'Holding': cost['holding'],
            'Stockout': cost['stockout'],
            'Gap_vs_OPT_%': gap,
        })
        print(f"    {n_neurons} neurons: total={cost['total']:.2f} (gap {gap:+.1f}%)")

    table_b3 = pd.DataFrame(neuron_table)
    table_b3.to_csv(f"{results_dir}/table_b3_neurons.csv", index=False)
    print(f"\n  [✓] Table B.3 saved to {results_dir}/table_b3_neurons.csv")

    # ---- B.1.3 Learning Rate ----
    print("\n  B.1.3 Learning Rate:")
    lr_table = []
    for lr in [0.0001, 0.001, 0.01]:
        print(f"    Training E2E with lr={lr}...")
        t0 = time.time()
        model = train_e2e_quick(
            train_data, val_data,
            hidden_layers=2, hidden_size=100,
            learning_rate=lr, n_epochs=15, seed=42,
        )
        t = time.time() - t0
        preds = predict_with_model(model, val_data)
        cost = evaluate_predictions(preds, df_test.iloc[val_idx])
        gap = (cost['total'] - opt_cost['total']) / opt_cost['total'] * 100
        lr_table.append({
            'Learning_Rate': lr,
            'Total': cost['total'],
            'Holding': cost['holding'],
            'Stockout': cost['stockout'],
            'Gap_vs_OPT_%': gap,
        })
        print(f"    lr={lr}: total={cost['total']:.2f} (gap {gap:+.1f}%)")

    table_b4 = pd.DataFrame(lr_table)
    table_b4.to_csv(f"{results_dir}/table_b4_learning_rate.csv", index=False)
    print(f"\n  [✓] Table B.4 saved to {results_dir}/table_b4_learning_rate.csv")

    return {'b2': table_b2, 'b3': table_b3, 'b4': table_b4}


# ============================================
# B.2: 数据量敏感性
# ============================================
def sensitivity_data_size(df_test: pd.DataFrame, results_dir: str):
    """Table B.5 + Figure B.1: 不同训练数据量"""
    print("\n" + "=" * 60)
    print("B.2: 数据量敏感性")
    print("=" * 60)

    data = extract_features_from_df(df_test)
    # 划分 train+val pool 和 test
    all_idx = np.arange(len(data['Y_optimal']))
    np.random.seed(42)
    np.random.shuffle(all_idx)
    test_idx = all_idx[: int(len(all_idx) * 0.2)]
    pool_idx = all_idx[int(len(all_idx) * 0.2):]

    opt_pred = df_test['OPT_pred'].values
    opt_cost = evaluate_predictions(opt_pred[test_idx], df_test.iloc[test_idx])
    print(f"  OPT on test: total = {opt_cost['total']:.2f}")

    data_sizes = [0.2, 0.4, 0.6, 0.8, 1.0]
    size_table = []
    time_list = []

    for frac in data_sizes:
        n_use = int(len(pool_idx) * frac)
        use_idx = pool_idx[:n_use]
        train_data = {k: v[use_idx] for k, v in data.items()}
        val_data = {k: v[test_idx] for k, v in data.items()}

        print(f"    Training E2E with {n_use} samples ({frac*100:.0f}%)...")
        t0 = time.time()
        model = train_e2e_quick(
            train_data, val_data,
            hidden_layers=2, hidden_size=100,
            learning_rate=0.001, n_epochs=15, seed=42,
        )
        train_time = time.time() - t0

        preds = predict_with_model(model, val_data)
        cost = evaluate_predictions(preds, df_test.iloc[test_idx])
        gap = (cost['total'] - opt_cost['total']) / opt_cost['total'] * 100
        size_table.append({
            'Fraction': frac,
            'N_Training': n_use,
            'OPT_Cost': opt_cost['total'],
            'E2E_Cost': cost['total'],
            'Gap_%': gap,
        })
        time_list.append(train_time)
        print(f"      Cost: {cost['total']:.2f} (gap {gap:+.1f}%), time: {train_time:.1f}s")

    table_b5 = pd.DataFrame(size_table)
    table_b5.to_csv(f"{results_dir}/table_b5_data_size.csv", index=False)

    # Figure B.1
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax2 = ax.twinx()
    ax.plot([s * 100 for s in data_sizes], table_b5['E2E_Cost'],
            'b-o', label='E2E Cost', linewidth=2)
    ax.axhline(y=opt_cost['total'], color='red', linestyle='--',
               label=f'OPT = {opt_cost["total"]:.0f}')
    ax2.plot([s * 100 for s in data_sizes], time_list, 'g--s',
             label='Training Time', linewidth=2)
    ax.set_xlabel('Training Data Percentage (%)', fontsize=11)
    ax.set_ylabel('Total Cost', fontsize=11, color='blue')
    ax2.set_ylabel('Training Time (s)', fontsize=11, color='green')
    ax.set_title('Figure B.1: Data Size Sensitivity', fontsize=12)
    ax.legend(loc='upper right')
    ax2.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/figure_b1_data_size.png", dpi=150)
    plt.close()

    print(f"\n  [✓] Table B.5 + Figure B.1 saved")
    return table_b5


# ============================================
# B.3: b/h ratio 敏感性
# ============================================
def sensitivity_b_over_h(df_test: pd.DataFrame, results_dir: str):
    """Figure B.2: 不同 b/h ratio 下的成本"""
    print("\n" + "=" * 60)
    print("B.3: b/h Ratio 敏感性")
    print("=" * 60)

    b_over_h_list = [1, 2, 3, 5, 7, 9, 11, 15, 19]
    results_table = []

    # 用作者提供的 E2E_RNN_pred 和 OPT_pred (在不同 b/h 下重新计算成本)
    for ratio in b_over_h_list:
        b_val = ratio
        h_val = 1.0
        opt_cost = evaluate_predictions(
            df_test['OPT_pred'].values, df_test, h=h_val, b=b_val
        )
        e2e_cost = evaluate_predictions(
            df_test['E2E_RNN_pred'].values, df_test, h=h_val, b=b_val
        )
        bm_norm_cost = evaluate_predictions(
            np.array([
                mu * (rp + vmu) + 1.64 * np.sqrt((rp + vmu) * sd**2 + mu**2 * vsd**2)
                for mu, sd, rp, vmu, vsd in zip(
                    df_test['demand_mean'], df_test['demand_std'],
                    df_test['review_period'], df_test['vendor_vlt_mean'],
                    df_test['vendor_vlt_std'],
                )
            ]), df_test, h=h_val, b=b_val,
        )
        results_table.append({
            'b_over_h': ratio,
            'OPT': opt_cost['total'],
            'E2E_RNN': e2e_cost['total'],
            'BM_normal': bm_norm_cost['total'],
        })
        print(f"    b/h={ratio}: OPT={opt_cost['total']:.0f}, "
              f"E2E={e2e_cost['total']:.0f}, BM={bm_norm_cost['total']:.0f}")

    table_b6 = pd.DataFrame(results_table)
    table_b6.to_csv(f"{results_dir}/table_b6_b_over_h.csv", index=False)

    # Figure B.2
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(table_b6['b_over_h'], table_b6['OPT'], 'r-o',
            label='OPT', linewidth=2)
    ax.plot(table_b6['b_over_h'], table_b6['E2E_RNN'], 'b-s',
            label='E2E_RNN', linewidth=2)
    ax.plot(table_b6['b_over_h'], table_b6['BM_normal'], 'g--^',
            label='BM_normal', linewidth=2)
    ax.set_xlabel('b/h Ratio', fontsize=11)
    ax.set_ylabel('Total Cost', fontsize=11)
    ax.set_title('Figure B.2: b/h Ratio Sensitivity', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/figure_b2_b_over_h.png", dpi=150)
    plt.close()

    print(f"\n  [✓] Figure B.2 saved")
    return table_b6


# ============================================
# B.4: 局部极小值
# ============================================
def sensitivity_local_minima(df_test: pd.DataFrame, results_dir: str):
    """Table B.6: 不同随机种子的网络性能"""
    print("\n" + "=" * 60)
    print("B.4: 局部极小值 (不同随机种子)")
    print("=" * 60)

    data = extract_features_from_df(df_test)
    train_idx, val_idx = train_test_split(
        np.arange(len(data['Y_optimal'])), test_size=0.2, random_state=42
    )
    train_data = {k: v[train_idx] for k, v in data.items()}
    val_data = {k: v[val_idx] for k, v in data.items()}

    seeds_results = []
    for seed in [42, 123]:
        print(f"    Training E2E with seed={seed}...")
        model = train_e2e_quick(
            train_data, val_data,
            hidden_layers=2, hidden_size=100,
            learning_rate=0.001, n_epochs=15, seed=seed,
        )
        preds = predict_with_model(model, val_data)
        cost = evaluate_predictions(preds, df_test.iloc[val_idx])
        seeds_results.append({
            'Seed': seed,
            'Total': cost['total'],
            'Holding': cost['holding'],
            'Stockout': cost['stockout'],
        })

    opt_cost = evaluate_predictions(
        df_test['OPT_pred'].values[val_idx], df_test.iloc[val_idx]
    )

    table_b7 = pd.DataFrame(seeds_results)
    table_b7.to_csv(f"{results_dir}/table_b7_local_minima.csv", index=False)
    print(f"\n  [✓] Table B.6 (seeds) saved")
    print(f"    OPT baseline: {opt_cost['total']:.2f}")
    print(table_b7.to_string(index=False))
    return table_b7


# ============================================
# 主入口
# ============================================
def reproduce_appendix_b(df_test: pd.DataFrame, results_dir: str = "results",
                         skip_train: bool = False):
    """复现 Appendix B 全部敏感性分析"""
    print("\n" + "=" * 70)
    print("Appendix B: 全部敏感性分析")
    print("=" * 70)

    results = {}

    if not skip_train:
        results['b2_layers'] = sensitivity_network_structure(df_test, results_dir)
    results['b3_b_over_h'] = sensitivity_b_over_h(df_test, results_dir)
    results['b4_local_minima'] = sensitivity_local_minima(df_test, results_dir)

    if not skip_train:
        results['b5_data_size'] = sensitivity_data_size(df_test, results_dir)

    return results


if __name__ == "__main__":
    from src.real_data_loader import RealDataLoader
    PKG = r"E:\DeepSeek工作产出\人工智能与经济管理研究课程\_pkg_temp\replication_package_MS-FTBDA-20-01585"
    loader = RealDataLoader(PKG)
    loader.load_df_test()
    reproduce_appendix_b(loader.df_test, "results", skip_train=False)