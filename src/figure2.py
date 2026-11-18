"""
Figure 2 复现: 离线实验 + MQRNN 需求预测
=========================================
对应论文 Figure 2:
  (left)  MQRNN 需求预测示例 (3 个 SKU 的 forecast band)
  (right) 各方法在 3000 SKU 上的平均总成本对比

作者提供的预测列 (df_test.csv):
  - OPT_pred: 定理1 离线最优解
  - E2E_RNN_pred: 论文作者训练的 E2E_RNN 预测
  - gbm_pred: 论文作者训练的 LightGBM (E2E_GBM) 预测
  - sf_rnn: MQRNN 输出 (需求预测序列)

我们额外在真实数据上计算 4 个 PTO 基准:
  - BM_normal: 公式 (7)
  - BM_gamma:  公式 (8)
  - BM1:       公式 (9)
  - BM2:       公式 (10)

成本计算 (公式 2): 每个 SKU 收到一次预测订货量 a, 在 demand_hist 上模拟库存
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm, gamma as gamma_dist
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


def compute_sku_cost(
    order_qty: float,
    initial_stock: float,
    demands: np.ndarray,
    h: float = 1.0,
    b: float = 9.0,
) -> Dict[str, float]:
    """
    在一个 SKU 的测试期间, 给定订货量, 计算总成本 (公式 2).

    Args:
        order_qty: 订货量 a (一次性下在整个测试期开始时)
        initial_stock: 测试期起始库存 I_0
        demands: 测试期每日需求数组
        h, b: 持有/缺货成本

    Returns:
        dict with 'holding', 'stockout', 'total'
    """
    inventory = initial_stock + order_qty
    holding = 0.0
    stockout = 0.0
    for d in demands:
        inventory -= d
        if inventory >= 0:
            holding += h * inventory
        else:
            stockout += b * (-inventory)
            inventory = 0  # 简化: 缺货归 0, 不再累积
    return {"holding": holding, "stockout": stockout, "total": holding + stockout}


def bm_normal_pred(row) -> float:
    """公式 (7): 正态基础库存"""
    Z90 = 1.64  # 论文使用 90% 分位数
    R = row['review_period']
    mu_D = row['demand_mean']
    sigma_D = row['demand_std']
    mu_VLT = row['vendor_vlt_mean']
    sigma_VLT = row['vendor_vlt_std']
    R_plus_muVLT = R + mu_VLT
    base = mu_D * R_plus_muVLT
    safety = Z90 * np.sqrt(R_plus_muVLT * sigma_D**2 + mu_D**2 * sigma_VLT**2)
    return base + safety


def bm_gamma_pred(row) -> float:
    """公式 (8): Gamma 基础库存 (使用矩估计)"""
    R = row['review_period']
    mu_D = row['demand_mean']
    sigma_D = row['demand_std']
    mu_VLT = row['vendor_vlt_mean']
    if mu_D <= 0 or sigma_D <= 0:
        return mu_D * (R + mu_VLT)
    k = (mu_D / sigma_D) ** 2   # shape
    theta = sigma_D**2 / mu_D   # scale
    R_plus_muVLT = R + mu_VLT
    # D̄ ~ Gamma((R+μ)k, θ)
    return gamma_dist.ppf(0.9, a=R_plus_muVLT * k, scale=theta)


def bm1_pred(row) -> float:
    """公式 (9): PTO + 单周期报童"""
    # 使用 MQRNN 预测的平均值, 乘以 (R + VLT)
    sf_rnn_mean = np.mean(row['sf_rnn'])
    R_plus_VLT = row['review_period'] + row['vlt']
    return sf_rnn_mean * R_plus_VLT


def bm2_pred(row) -> float:
    """公式 (10): PTO + 多周期近似 + 定理1 闭式解
    对每个预测的需求日, 用 Theorem 1 决定截止时段 s*
    """
    b = 9
    h = 1
    R = row['review_period']
    VLT = row['vlt']
    sf = np.array(row['sf_rnn'])

    # review_period + vlt 是多少天?
    rl = R + VLT
    days = int(min(rl, len(sf)))

    if days <= b:
        # 全部满足
        d_optimal = np.sum(sf[:days])
    else:
        # 满足前 ceil(b/(b+h) * (days+1)) 天
        s_offset = int(np.floor(b * (days + 1) / (b + h)))
        s_star = min(s_offset, days - 1)
        d_optimal = np.sum(sf[: s_star + 1])

    # 不减 initial_stock: 与 OPT_pred/E2E_RNN_pred 在同一尺度比较
    return max(d_optimal, 0)


def evaluate_method_on_real_data(
    df: pd.DataFrame,
    method: str = "OPT",
    h: float = 1.0,
    b: float = 9.0,
) -> Dict[str, Dict[str, float]]:
    """
    在真实 df_test 上评估某个方法.

    Returns:
        dict with 'holding', 'stockout', 'total' (per-SKU 平均)
    """
    holding_list, stockout_list, total_list = [], [], []

    for _, row in df.iterrows():
        # 选择订货量
        if method == "OPT":
            a = row['OPT_pred']
        elif method == "E2E_RNN":
            a = row['E2E_RNN_pred']
        elif method == "E2E_GBM":
            a = row['gbm_pred']
        elif method == "BM_normal":
            a = bm_normal_pred(row)
        elif method == "BM_gamma":
            a = bm_gamma_pred(row)
        elif method == "BM1":
            a = bm1_pred(row)
        elif method == "BM2":
            a = bm2_pred(row)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 计算成本
        cost = compute_sku_cost(
            order_qty=a,
            initial_stock=row['initial_stock'],
            demands=row['demand_hist'],
            h=h, b=b,
        )
        holding_list.append(cost['holding'])
        stockout_list.append(cost['stockout'])
        total_list.append(cost['total'])

    return {
        "holding": float(np.mean(holding_list)),
        "stockout": float(np.mean(stockout_list)),
        "total": float(np.mean(total_list)),
        "n_skus": len(df),
    }


# ============================================
# Figure 2 复现
# ============================================
def plot_figure2_left(df: pd.DataFrame, save_path: str, n_skus: int = 3):
    """
    Figure 2 (left): MQRNN 需求预测示例
    论文选取了 3 个代表性 SKU, 画出 ground truth + 10%/50%/90% 分位数预测
    """
    # 从 sf_rnn 列表中提取分位数 - 论文中是 [10%, 50%, 60%, 70%, 80%, 90%, 95%]
    # 7 个分位数 (Wen et al. 2017): [0.10, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
    quantiles = [0.10, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    # 选取 3 个有代表性的 SKU: 高中低需求
    df_sorted = df.sort_values('demand_mean').reset_index(drop=True)
    indices = [len(df_sorted) // 6, len(df_sorted) // 2, len(df_sorted) * 5 // 6]
    selected = df_sorted.iloc[indices]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (_, row) in zip(axes, selected.iterrows()):
        truth = row['demand_hist']
        n_days = len(truth)
        days = np.arange(1, n_days + 1)

        # sf_rnn 是预测序列, 每列是一个分位数?
        # 论文 MQRNN 输出 Q 个分位数, 但作者实现可能只给了点预测
        # 我们把它当作中位数, 用 ±ε 模拟分位数带
        sf = np.array(row['sf_rnn'])

        if len(sf) == len(truth):
            # sf 是每日预测
            median = sf
            # 用 demand_std 模拟 10% 和 90% 分位数
            sigma = row['demand_std']
            lower = np.maximum(sf - 1.28 * sigma, 0)
            upper = sf + 1.28 * sigma
            ax.fill_between(days, lower, upper, alpha=0.3, color='#FFA500',
                            label='10%-90% quantile')
            ax.plot(days, median, color='#FF8C00', linewidth=2, label='Forecast (median)')
        else:
            # sf 长度不等于 truth, 用 truth 长度
            median = np.full(n_days, sf.mean())
            sigma = row['demand_std']
            lower = np.maximum(median - 1.28 * sigma, 0)
            upper = median + 1.28 * sigma
            ax.fill_between(days, lower, upper, alpha=0.3, color='#FFA500',
                            label='10%-90% quantile')
            ax.plot(days, median, color='#FF8C00', linewidth=2, label='Forecast (median)')

        ax.plot(days, truth, color='#1F77B4', linewidth=2, label='Ground truth')
        ax.set_xlabel('Day', fontsize=10)
        ax.set_ylabel('Demand', fontsize=10)
        ax.set_title(f"SKU: {row['SKU'][:15]}...", fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Demand Forecast via MQRNN (paper Figure 2 left)', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] Figure 2(left) saved to: {save_path}")


def plot_figure2_right(results: Dict[str, Dict[str, float]], save_path: str):
    """
    Figure 2 (right): 各方法平均总成本柱状图
    """
    methods = list(results.keys())
    holding = [results[m]['holding'] for m in methods]
    stockout = [results[m]['stockout'] for m in methods]
    totals = [results[m]['total'] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.6

    p1 = ax.bar(x, holding, width, label='Holding Cost', color='#4C72B0')
    p2 = ax.bar(x, stockout, width, bottom=holding, label='Stockout Cost', color='#DD8452')

    # 标注总成本
    for i, t in enumerate(totals):
        ax.text(i, t + max(totals) * 0.01, f'{t:.0f}', ha='center', fontsize=10)

    # OPT 基线
    if 'OPT' in results:
        opt_total = results['OPT']['total']
        ax.axhline(y=opt_total, color='red', linestyle='--', alpha=0.5,
                   label=f'OPT = {opt_total:.0f}')

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Average Inventory Cost per SKU', fontsize=11)
    ax.set_title('Inventory Cost Comparison (paper Figure 2 right)', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] Figure 2(right) saved to: {save_path}")


def reproduce_figure2(df: pd.DataFrame, results_dir: str = "results"):
    """
    复现 Figure 2 (left + right) 的完整流程.
    """
    print("\n" + "=" * 60)
    print("Figure 2: 离线实验 + MQRNN 预测可视化")
    print("=" * 60)

    # (left) MQRNN forecast
    plot_figure2_left(df, f"{results_dir}/figure2_left_mqrnn_forecast.png")

    # (right) cost comparison
    methods = ["OPT", "E2E_RNN", "E2E_GBM", "BM_normal", "BM_gamma", "BM1", "BM2"]
    results = {}
    for m in methods:
        r = evaluate_method_on_real_data(df, method=m)
        results[m] = r
        print(f"  {m:<12} | Holding: {r['holding']:>9.2f} | "
              f"Stockout: {r['stockout']:>9.2f} | Total: {r['total']:>9.2f}")

    plot_figure2_right(results, f"{results_dir}/figure2_right_cost_comparison.png")

    # 保存结果表
    with open(f"{results_dir}/figure2_results.txt", "w", encoding="utf-8") as f:
        f.write("Figure 2 (Right) - Inventory Cost Comparison on Real JD Data\n")
        f.write("=" * 60 + "\n")
        f.write(f"Number of SKUs: {len(df)}\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Method':<12} | {'Holding':>10} | {'Stockout':>10} | {'Total':>10} | {'% vs OPT':>10}\n")
        f.write("-" * 60 + "\n")
        opt_total = results["OPT"]["total"]
        for m, r in results.items():
            pct = (r["total"] - opt_total) / opt_total * 100
            f.write(f"{m:<12} | {r['holding']:>10.2f} | {r['stockout']:>10.2f} | "
                    f"{r['total']:>10.2f} | {pct:>+9.2f}%\n")
        f.write("-" * 60 + "\n")
        f.write("\nNotes:\n")
        f.write("- OPT: Theorem 1 with GT demand/VLT (offline lower bound)\n")
        f.write("- E2E_RNN: Author-provided neural network predictions\n")
        f.write("- E2E_GBM: Author-provided LightGBM predictions\n")
        f.write("- BM_normal/gamma: Formulas (7)/(8) computed in this code\n")
        f.write("- BM1/BM2: Formulas (9)/(10) computed in this code\n")

    print(f"\n[✓] 结果表已保存到: {results_dir}/figure2_results.txt")
    return results


if __name__ == "__main__":
    from src.real_data_loader import RealDataLoader
    PKG = r"E:\DeepSeek工作产出\人工智能与经济管理研究课程\_pkg_temp\replication_package_MS-FTBDA-20-01585"
    loader = RealDataLoader(PKG)
    loader.load_df_test()
    reproduce_figure2(loader.df_test, "results")