"""
评估器
======
在完整测试轨迹上比较所有方法（论文 Figure 2(b) 的复现）。

评估流程:
1. 模拟生成测试集（每个 SKU 完整 horizon 上的 GT 需求+VLT）
2. 用训练好的 E2E 模型对每个 (SKU, 订单) 给出预测订货量
3. 用各种 PTO 基准对同样 (SKU, 订单) 给出订货量
4. 把每个方法给出的订货量代入"真实轨迹"，计算总成本
5. 输出: 总成本表 + 论文图 2(b) 风格柱状图
"""

import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'


def compute_total_cost(
    order_quantity: float,
    initial_inventory: float,
    demands_window: np.ndarray,
    h: float = 1.0,
    b: float = 9.0,
) -> float:
    """
    在一个订单区间 [v_m, v_{m+1}) 上计算总成本（公式 2）。

    Args:
        order_quantity: 该订单的订货量 a
        initial_inventory: 订单到达时的库存 I_v
        demands_window: 区间内的每日需求
        h, b: 持有和缺货成本

    Returns:
        total cost for this interval
    """
    cost = 0.0
    inventory = initial_inventory + order_quantity
    for d in demands_window:
        inventory -= d
        if inventory >= 0:
            cost += h * inventory
        else:
            cost += b * (-inventory)
            inventory = 0
    return cost


def evaluate_methods_on_simulations(
    e2e_predictions: np.ndarray,
    data_simulator,
    data: Dict[str, np.ndarray],
    h: float = 1.0,
    b: float = 9.0,
    methods_to_evaluate: List[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    在所有测试样本上评估各种方法。
    每个方法输出 holding, stockout, total 三类成本。

    Args:
        e2e_predictions: E2E 模型对每个样本的订货量预测
        data_simulator: InventoryDataSimulator 实例（用于访问 _simulate_single_sku）
        data: 数据集（包含 X_initial 等特征）
        h, b: 成本参数

    Returns:
        dict[method_name] -> {'holding': x, 'stockout': y, 'total': z}
    """
    if methods_to_evaluate is None:
        methods_to_evaluate = ["OPT", "E2E_RNN", "BM_normal", "BM_gamma", "BM1", "BM2"]

    n_sku = data_simulator.n_sku
    n_orders = data_simulator.n_orders_per_sku
    sku_demand_shape = data_simulator.sku_demand_shape
    sku_demand_scale = data_simulator.sku_demand_scale

    results = {m: {"holding": 0.0, "stockout": 0.0, "total": 0.0} for m in methods_to_evaluate}

    # 按 SKU 重新模拟完整轨迹
    rng = np.random.default_rng(data_simulator.seed + 999)

    # 重跑模拟，得到 GT 轨迹
    for sku_idx in range(n_sku):
        # 完整模拟（确定性，给定 seed）
        rng_sku = np.random.default_rng(data_simulator.seed + sku_idx)
        shape = sku_demand_shape[sku_idx]
        scale = sku_demand_scale[sku_idx]
        vlt_lambda = data_simulator.sku_vlt_lambda[sku_idx]

        demands = rng_sku.gamma(shape=shape, scale=scale, size=data_simulator.t_horizon)
        order_times = np.linspace(
            data_simulator.review_period - 1,
            data_simulator.t_horizon - data_simulator.review_period,
            n_orders,
        ).astype(int)
        vlt_real = rng_sku.poisson(lam=vlt_lambda, size=n_orders) + 1
        arrival_times = np.minimum(order_times + vlt_real, data_simulator.t_horizon)

        # 取该 SKU 对应的样本
        sku_mask = data["SKU_ids"] == sku_idx
        sku_indices = np.where(sku_mask)[0]

        for sample_i, m in zip(sku_indices, range(n_orders)):
            t_m = order_times[m]
            v_m = arrival_times[m]
            next_v_m = arrival_times[m + 1] if m + 1 < n_orders else data_simulator.t_horizon
            window_demands = demands[v_m:next_v_m]

            # 初始库存 (基于 GT)
            if v_m >= 1:
                I_v = -np.sum(demands[: v_m])
            else:
                I_v = 0.0

            # --- OPT (定理 1 + GT) ---
            if "OPT" in methods_to_evaluate:
                a_opt = _opt_decision(I_v, window_demands, h, b)
                results["OPT"] = _accumulate_cost(results["OPT"], a_opt, I_v, window_demands, h, b)

            # --- E2E_RNN ---
            if "E2E_RNN" in methods_to_evaluate:
                a_e2e = e2e_predictions[sample_i]
                results["E2E_RNN"] = _accumulate_cost(results["E2E_RNN"], a_e2e, I_v, window_demands, h, b)

            # --- BM_normal ---
            if "BM_normal" in methods_to_evaluate:
                # 用历史估计 mu, sigma
                hist = demands[: max(1, t_m)]
                mu_D = hist.mean() if len(hist) > 0 else 5.0
                sigma_D = hist.std() if len(hist) > 1 else mu_D * 0.5
                past_vlt = vlt_real[: max(1, m)]
                mu_VLT = past_vlt.mean() if len(past_vlt) > 0 else 3.0
                sigma_VLT = past_vlt.std() if len(past_vlt) > 1 else 1.0
                from .pto_baselines import base_stock_normal
                a_normal = base_stock_normal(mu_D, sigma_D, mu_VLT, sigma_VLT,
                                              data_simulator.review_period, b, h)
                results["BM_normal"] = _accumulate_cost(results["BM_normal"], a_normal, I_v, window_demands, h, b)

            # --- BM_gamma ---
            if "BM_gamma" in methods_to_evaluate:
                hist = demands[: max(1, t_m)]
                mu_D = hist.mean() if len(hist) > 0 else 5.0
                sigma_D = hist.std() if len(hist) > 1 else mu_D * 0.5
                past_vlt = vlt_real[: max(1, m)]
                mu_VLT = past_vlt.mean() if len(past_vlt) > 0 else 3.0
                from .pto_baselines import base_stock_gamma
                a_gamma = base_stock_gamma(mu_D, sigma_D, mu_VLT,
                                            data_simulator.review_period, b, h)
                results["BM_gamma"] = _accumulate_cost(results["BM_gamma"], a_gamma, I_v, window_demands, h, b)

            # --- BM1 (公式 9) ---
            if "BM1" in methods_to_evaluate:
                from scipy.stats import norm
                hist = demands[: max(1, t_m)]
                sigma_D = hist.std() if len(hist) > 1 else 1.0
                window_total_pred = window_demands.sum()  # GT 简化
                z = norm.ppf(b / (b + h))
                a_bm1 = I_v + window_total_pred + z * sigma_D * np.sqrt(len(window_demands))
                a_bm1 = max(a_bm1, 0)
                results["BM1"] = _accumulate_cost(results["BM1"], a_bm1, I_v, window_demands, h, b)

            # --- BM2 (公式 10 + 定理 1) ---
            if "BM2" in methods_to_evaluate:
                from .pto_baselines import bm2_decision
                a_bm2 = bm2_decision(I_v, window_demands, b, h)
                results["BM2"] = _accumulate_cost(results["BM2"], a_bm2, I_v, window_demands, h, b)

    # 计算平均（每 SKU）
    for m in results:
        results[m]["holding"] /= n_sku
        results[m]["stockout"] /= n_sku
        results[m]["total"] /= n_sku

    return results


def _opt_decision(I_v: float, window_demands: np.ndarray, h: float, b: float) -> float:
    """定理 1 的闭式解"""
    from .pto_baselines import bm2_decision
    return bm2_decision(I_v, window_demands, b, h)


def _accumulate_cost(
    cost_dict: Dict[str, float],
    a: float,
    I_v: float,
    window_demands: np.ndarray,
    h: float,
    b: float,
) -> Dict[str, float]:
    """累加一个区间的成本到 cost_dict"""
    holding = 0.0
    stockout = 0.0
    inventory = I_v + a
    for d in window_demands:
        inventory -= d
        if inventory >= 0:
            holding += h * inventory
        else:
            stockout += b * (-inventory)
            inventory = 0
    cost_dict["holding"] += holding
    cost_dict["stockout"] += stockout
    cost_dict["total"] += holding + stockout
    return cost_dict


def plot_results_bar(
    results: Dict[str, Dict[str, float]],
    title: str = "Inventory Cost Comparison",
    save_path: str = None,
):
    """
    复现论文 Figure 2(b): 总成本柱状图
    每个方法一根柱子，分 holding/stockout 两段堆叠。
    """
    methods = list(results.keys())
    holding = [results[m]["holding"] for m in methods]
    stockout = [results[m]["stockout"] for m in methods]
    totals = [results[m]["total"] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.6

    p1 = ax.bar(x, holding, width, label='Holding Cost', color='#4C72B0')
    p2 = ax.bar(x, stockout, width, bottom=holding, label='Stockout Cost', color='#DD8452')

    # 标注总成本
    for i, t in enumerate(totals):
        ax.text(i, t + max(totals) * 0.01, f'{t:.0f}', ha='center', fontsize=9)

    ax.set_xlabel('Method', fontsize=11)
    ax.set_ylabel('Average Cost (per SKU)', fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    plt.close()


def print_results_table(results: Dict[str, Dict[str, float]]):
    """打印对齐的结果表（论文 Table 风格）"""
    print("\n" + "=" * 60)
    print(f"{'Method':<12} | {'Holding':>10} | {'Stockout':>10} | {'Total':>10}")
    print("-" * 60)
    for m, costs in results.items():
        print(f"{m:<12} | {costs['holding']:>10.2f} | {costs['stockout']:>10.2f} | {costs['total']:>10.2f}")
    print("=" * 60)