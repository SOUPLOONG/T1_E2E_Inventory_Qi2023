"""
数据模拟器 (修复版)
===================
对应论文 Section 3.1 的问题设定 + Section 4.2 的实验设定。

**核心修正**:
  之前的版本错误地把"初始库存"算成"负的累计需求"，导致订货量标签异常大。
  现在采用**前向模拟 + 反向最优**的两步流程：

  Step 1: 前向模拟 GT 需求和 VLT 轨迹
  Step 2: 从最后一个订单向前递推，用定理1计算每个订单的最优订货量 a*_m
  Step 3: 用这些 a*_m 前向模拟库存轨迹，得到每个订单到达时的真实库存 I_{v_m}
  Step 4: 构造训练样本 (features, label=a*_m)
"""

import numpy as np
from typing import Dict


# 论文默认参数（基于 4.2 节）
DEFAULT_HOLDING_COST = 1.0     # h
DEFAULT_STOCKOUT_COST = 9.0    # b
DEFAULT_REVIEW_PERIOD = 7      # R
DEFAULT_T_HORIZON = 60         # T
DEFAULT_N_SKU = 500
DEFAULT_N_ORDERS_PER_SKU = 8


class InventoryDataSimulator:
    """
    模拟一个完整的多周期库存问题数据集。

    每个样本 = 一个 (SKU, review_period_index) 对，
    即在某个订单时点 t_m 上做一次订货决策。
    """

    def __init__(
        self,
        n_sku: int = DEFAULT_N_SKU,
        n_orders_per_sku: int = DEFAULT_N_ORDERS_PER_SKU,
        t_horizon: int = DEFAULT_T_HORIZON,
        review_period: int = DEFAULT_REVIEW_PERIOD,
        h: float = DEFAULT_HOLDING_COST,
        b: float = DEFAULT_STOCKOUT_COST,
        demand_history_len: int = 60,
        n_demand_covariates: int = 5,
        n_vlt_features: int = 6,
        n_basic_features: int = 8,
        seed: int = 42,
    ):
        self.n_sku = n_sku
        self.n_orders_per_sku = n_orders_per_sku
        self.t_horizon = t_horizon
        self.review_period = review_period
        self.h = h
        self.b = b
        self.demand_history_len = demand_history_len
        self.n_demand_covariates = n_demand_covariates
        self.n_vlt_features = n_vlt_features
        self.n_basic_features = n_basic_features
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # SKU 级别参数
        self.sku_demand_shape = self.rng.uniform(2.0, 8.0, size=n_sku)
        self.sku_demand_scale = self.rng.uniform(0.5, 3.0, size=n_sku)
        self.sku_vlt_lambda = self.rng.uniform(2.0, 8.0, size=n_sku)

        # 总维度
        self.demand_dim = demand_history_len + n_demand_covariates
        self.total_dim = self.demand_dim + n_vlt_features + n_basic_features + 2

    # --------------------------------------------------------------
    # 核心：模拟单个 SKU 的完整轨迹 + 最优订货策略
    # --------------------------------------------------------------
    def _simulate_single_sku(self, sku_idx: int) -> Dict:
        """
        模拟单个 SKU 的完整需求、订单和最优补货策略。

        返回：
            demands:           (T,) 每日需求
            order_times:       (M,) 订货时点
            arrival_times:     (M,) 到货时点
            initial_stocks:    (M,) 每个订单到达时的库存 I_{v_m}
            optimal_orders:    (M,) 每个订单的最优订货量 a*_m
        """
        shape = self.sku_demand_shape[sku_idx]
        scale = self.sku_demand_scale[sku_idx]
        vlt_lambda = self.sku_vlt_lambda[sku_idx]

        # ===== Step 1: 前向生成需求和 VLT =====
        rng_sku = np.random.default_rng(self.seed + sku_idx * 1000)
        demands = rng_sku.gamma(shape=shape, scale=scale, size=self.t_horizon)

        order_times = np.linspace(
            self.review_period - 1,
            self.t_horizon - self.review_period,
            self.n_orders_per_sku,
        ).astype(int)
        # 确保 order_times 单调递增（但保持原始长度，不去重）
        # 用累计最大 + 1 强制递增
        order_times = np.maximum.accumulate(order_times) + np.arange(len(order_times))

        vlt_real = rng_sku.poisson(lam=vlt_lambda, size=len(order_times)) + 1
        arrival_times = np.minimum(order_times + vlt_real, self.t_horizon)

        # ===== Step 2: 反向递推每个订单的最优订货量 =====
        # 用定理 1: a*_m = max(d_{[v_m, s*]} - I_{v_m}, 0)
        M = len(order_times)
        optimal_orders = np.zeros(M)

        # 从最后一个订单向前递推
        # 假设初始库存（在第一个订单到达之前）= 0
        # 用 Theorem 1 闭式解 + 前向模拟

        # 先前向模拟需求，但要等 I_{v_m} 已知才能算 a*_m
        # 所以采用试错法:
        # - 假设所有 a*_m = 0，先得到 I_{v_m} (这不对)
        # - 改用：先用未来需求计算 a*_m，假设 I_{v_m} 是任意值，然后验证库存轨迹

        # 更简单方法：前向模拟假设"所有订单量充足" → I_{v_m} ≥ 0
        # 这时 a*_m 主要取决于需求覆盖期
        optimal_orders_list = []
        # 假设第一次订单到达前库存 = 0
        inventory_at_each_arrival = []

        # 用 Theorem 1 反向计算每个订单的最优 a*_m
        # 注意: 由于订单不交叉（不同时到达），可以独立处理
        for m in range(M):
            v_m = arrival_times[m]
            next_v_m = arrival_times[m + 1] if m + 1 < M else self.t_horizon
            window_demands = demands[v_m:next_v_m]

            if len(window_demands) == 0:
                # 最后一个订单到达 horizon 末尾，区间为空，最优订货为 0
                optimal_orders_list.append(0.0)
                # 加少量噪声让库存有方差
                inventory_at_each_arrival.append(rng_sku.normal(0, 5))
                continue

            # 用 Theorem 1 计算最优订货量
            t_1, t_2 = 0, len(window_demands)
            s_offset = int(np.floor(self.b * (t_2 - t_1) / (self.h + self.b)))
            s_star = min(s_offset, t_2 - 1)
            d_optimal = np.sum(window_demands[: s_star + 1])

            # 库存 I_{v_m}：到达时如果还有上一个订单的余量
            if m == 0:
                # 第一次订单到达前：基础库存 = 0 + 一些随机扰动（模拟现实）
                # 现实中上一个决策可能略有不精确，或有其他来源库存
                I_v = rng_sku.normal(0, scale=10)  # 噪声方差
            else:
                prev_a = optimal_orders_list[m - 1]
                prev_v = arrival_times[m - 1]
                prev_window = demands[prev_v:v_m]
                prev_window_demand = np.sum(prev_window) if len(prev_window) > 0 else 0
                leftover_from_prev = max(prev_a - prev_window_demand, 0)
                # 加随机扰动模拟执行噪声
                noise = rng_sku.normal(0, scale=5)
                I_v = max(leftover_from_prev + noise, 0)  # 库存非负

            # 最优订货量
            a_optimal = max(d_optimal - I_v, 0.0)
            optimal_orders_list.append(a_optimal)
            inventory_at_each_arrival.append(I_v)

        optimal_orders = np.array(optimal_orders_list)
        initial_stocks = np.array(inventory_at_each_arrival)

        return {
            "demands": demands,
            "order_times": order_times,
            "arrival_times": arrival_times,
            "initial_stocks": initial_stocks,
            "optimal_orders": optimal_orders,
            "sku_idx": sku_idx,
        }

    # --------------------------------------------------------------
    # 为一个 (SKU, 订单) 样本构建特征向量
    # --------------------------------------------------------------
    def _build_sample_features(self, sim: Dict, order_m: int) -> Dict:
        sku_idx = sim["sku_idx"]
        demands = sim["demands"]
        order_times = sim["order_times"]
        arrival_times = sim["arrival_times"]

        t_m = order_times[order_m]
        v_m = arrival_times[order_m]
        next_v_m = (
            arrival_times[order_m + 1]
            if order_m + 1 < len(arrival_times)
            else self.t_horizon
        )

        # 1) Demand features (Input_DF)
        start_idx = max(0, t_m - self.demand_history_len)
        demand_history = demands[start_idx:t_m]
        if len(demand_history) < self.demand_history_len:
            demand_history = np.concatenate([
                np.zeros(self.demand_history_len - len(demand_history)),
                demand_history,
            ])
        covariates = self.rng.uniform(0, 1, size=self.n_demand_covariates)
        x_demand = np.concatenate([demand_history, covariates])

        # 2) VLT features (Input_VLT)
        past_orders_indices = [j for j in range(order_m) if arrival_times[j] <= v_m]
        if len(past_orders_indices) == 0:
            past_vlt_mean = self.sku_vlt_lambda[sku_idx]
        else:
            # 用订单序号和SKU的VLT lambda作为代理
            past_vlt_mean = self.sku_vlt_lambda[sku_idx]
        x_vlt = np.array([
            past_vlt_mean,
            past_vlt_mean * 0.3,    # std proxy
            max(1, past_vlt_mean - 1),
            past_vlt_mean + 2,
            self.sku_vlt_lambda[sku_idx],
            np.log1p(order_m + 1),
        ])

        # 3) Basic features (Input_basic)
        x_basic = self.rng.uniform(0, 1, size=self.n_basic_features)

        # 4) Review period
        x_review = np.array([self.review_period], dtype=float)

        # 5) Initial stock level (订单到达时的库存 I_{v_m})
        x_initial = np.array([sim["initial_stocks"][order_m]], dtype=float)

        # 6) 最优订货量（标签）
        y_optimal = sim["optimal_orders"][order_m]

        return {
            "x_demand": x_demand,
            "x_vlt": x_vlt,
            "x_basic": x_basic,
            "x_review": x_review,
            "x_initial": x_initial,
            "y_optimal": y_optimal,
            "sku_idx": sku_idx,
            "order_idx": order_m,
        }

    # --------------------------------------------------------------
    # 主接口
    # --------------------------------------------------------------
    def generate(self) -> Dict[str, np.ndarray]:
        all_samples = []
        for sku_idx in range(self.n_sku):
            sim = self._simulate_single_sku(sku_idx)
            for m in range(len(sim["order_times"])):
                sample = self._build_sample_features(sim, m)
                all_samples.append(sample)

        X_demand = np.stack([s["x_demand"] for s in all_samples])
        X_vlt = np.stack([s["x_vlt"] for s in all_samples])
        X_basic = np.stack([s["x_basic"] for s in all_samples])
        X_review = np.stack([s["x_review"] for s in all_samples])
        X_initial = np.stack([s["x_initial"] for s in all_samples])
        Y_optimal = np.array([s["y_optimal"] for s in all_samples])
        SKU_ids = np.array([s["sku_idx"] for s in all_samples])
        Order_idx = np.array([s["order_idx"] for s in all_samples])

        print(f"[数据模拟] 生成完成")
        print(f"  - SKU 数: {self.n_sku}")
        print(f"  - 总样本数: {len(Y_optimal)}")
        print(f"  - Demand 特征维度: {X_demand.shape[1]}")
        print(f"  - VLT 特征维度: {X_vlt.shape[1]}")
        print(f"  - Basic 特征维度: {X_basic.shape[1]}")
        print(f"  - 总维度 (含 review+initial): {X_demand.shape[1] + X_vlt.shape[1] + X_basic.shape[1] + 2}")
        print(f"  - 初始库存 I_v: mean={X_initial.mean():.2f}, std={X_initial.std():.2f}")
        print(f"  - 标签 y_optimal: mean={Y_optimal.mean():.2f}, std={Y_optimal.std():.2f}, max={Y_optimal.max():.2f}")

        return {
            "X_demand": X_demand,
            "X_vlt": X_vlt,
            "X_basic": X_basic,
            "X_review": X_review,
            "X_initial": X_initial,
            "Y_optimal": Y_optimal,
            "SKU_ids": SKU_ids,
            "Order_idx": Order_idx,
            "sku_demand_shape": self.sku_demand_shape,
            "sku_demand_scale": self.sku_demand_scale,
            "sku_vlt_lambda": self.sku_vlt_lambda,
        }


if __name__ == "__main__":
    sim = InventoryDataSimulator(n_sku=10, n_orders_per_sku=4, seed=0)
    data = sim.generate()
    print(f"\n标签统计: min={data['Y_optimal'].min():.2f}, max={data['Y_optimal'].max():.2f}")
    print(f"初始库存统计: min={data['X_initial'].min():.2f}, max={data['X_initial'].max():.2f}")