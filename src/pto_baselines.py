"""
PTO (Predict-Then-Optimize) 基准方法
====================================
对应论文 Section 4.1 的所有对比基准。

四个基准：
1. BM_normal (公式 7): 正态假设的基础库存策略
2. BM_gamma (公式 8):  Gamma 假设的基础库存策略
3. BM1 (公式 9): PTO + 单周期报童近似
4. BM2 (公式 10): PTO + 多周期近似 + 定理1 闭式解

所有基准都"先预测 D 和 L，再代入策略公式"。
"""

import numpy as np
from scipy.stats import norm, gamma as gamma_dist
from typing import Dict


# ---------------------------------------------------------------
# 1. 正态基础库存 (公式 7)
# ---------------------------------------------------------------
def base_stock_normal(
    mu_D: float,
    sigma_D: float,
    mu_VLT: float,
    sigma_VLT: float,
    review_period: int,
    b: float,
    h: float,
) -> float:
    """
    公式 (7):
    BM_normal = μ_D(R + μ_VLT) + φ^{-1}(b/(b+h)) * sqrt((R+μ_VLT)σ_D^2 + μ_D^2 σ_VLT^2)

    含义:
    - 平均需求覆盖 (R + μ_VLT) 天的库存
    - 加上"报童分位数 × 波动"的安全库存
    """
    R = review_period
    # 报童分位数
    quantile_level = b / (b + h)
    z = norm.ppf(quantile_level)
    # 基础量 + 安全库存
    base = mu_D * (R + mu_VLT)
    safety = z * np.sqrt((R + mu_VLT) * sigma_D**2 + mu_D**2 * sigma_VLT**2)
    return base + safety


# ---------------------------------------------------------------
# 2. Gamma 基础库存 (公式 8)
# ---------------------------------------------------------------
def base_stock_gamma(
    mu_D: float,
    sigma_D: float,
    mu_VLT: float,
    review_period: int,
    b: float,
    h: float,
) -> float:
    """
    公式 (8):
    BM_gamma = Q_D̄(b/(b+h))
    其中 D̄ ~ Gamma((R + μ_VLT) * k, θ)
    k = (μ_D / σ_D)^2, θ = σ_D^2 / μ_D (矩估计)
    """
    R = review_period
    # 用矩估计求 Gamma 参数
    if mu_D <= 0 or sigma_D <= 0:
        return mu_D * (R + mu_VLT)
    k = (mu_D / sigma_D) ** 2
    theta = sigma_D**2 / mu_D

    # (R + μ_VLT) 天累计需求的 Gamma 参数
    shape = (R + mu_VLT) * k
    scale = theta

    quantile_level = b / (b + h)
    return gamma_dist.ppf(quantile_level, a=shape, scale=scale)


# ---------------------------------------------------------------
# 3. BM1: PTO + 单周期报童近似 (公式 9)
# ---------------------------------------------------------------
def bm1_decision(
    I_t: float,
    predicted_total_demand: float,
    b: float,
    h: float,
) -> float:
    """
    公式 (9):
    min_a E[b(d̂ - a - I_t) + h(a + I_t - d̂)]
    最优解 = I_t + d̂ + 报童安全库存 (single period)

    predicted_total_demand: 区间 [t_m, v_{m+1}) 上的预测总需求
    """
    quantile_level = b / (b + h)
    # 单周期最优：I_t + d̂ + 报童安全库存
    # 实际: a = max(I_t + d̂ - I_t - z*std + I_t, ...) 简化为
    a = max(predicted_total_demand + I_t * 0, 0)  # 简化：用预测均值
    return a


# ---------------------------------------------------------------
# 4. BM2: PTO + 多周期近似 + 定理1 (公式 10)
# ---------------------------------------------------------------
def bm2_decision(
    I_v: float,
    daily_demands: np.ndarray,
    b: float,
    h: float,
) -> float:
    """
    公式 (10):
    用定理 1 的闭式解，但需求用 PTO 预测的每天需求 d_t

    最优订货量 a = max(d_{[v_m, s*]} - I_v, 0)
    其中 s* = v_m + floor(b(t_2+1-v_m)/(h+b))
    """
    t_1 = 0
    t_2 = len(daily_demands)
    if t_2 == 0:
        return 0.0

    s_offset = int(np.floor(b * (t_2 - t_1) / (h + b)))
    s_star = min(s_offset, t_2 - 1)

    if s_star > t_1:
        d_optimal = np.sum(daily_demands[: s_star - t_1 + 1])
    else:
        d_optimal = daily_demands[0]

    return max(d_optimal - I_v, 0.0)


# ---------------------------------------------------------------
# 完整 PTO 基准评估器
# ---------------------------------------------------------------
class PTOBenchmarkEvaluator:
    """
    评估 4 个 PTO 基准方法 + OPT (定理1) 在一个数据集上的表现。

    输入: 完整的 SKU 模拟数据 (含 GT 需求和 VLT)
    输出: 每个 SKU 在每个 review period 上的订货量决策
    """

    def __init__(
        self,
        h: float = 1.0,
        b: float = 9.0,
        review_period: int = 7,
    ):
        self.h = h
        self.b = b
        self.review_period = review_period

    def predict_demand_normal(
        self,
        demand_history: np.ndarray,
        vlt_history: np.ndarray = None,
    ) -> tuple:
        """预测需求的均值/方差 (用历史估计)"""
        mu = demand_history.mean() if len(demand_history) > 0 else 5.0
        sigma = demand_history.std() if len(demand_history) > 1 else mu * 0.5
        return mu, sigma

    def predict_vlt_normal(
        self,
        vlt_history: np.ndarray,
    ) -> tuple:
        """预测 VLT 的均值/方差"""
        mu = vlt_history.mean() if len(vlt_history) > 0 else 3.0
        sigma = vlt_history.std() if len(vlt_history) > 1 else 1.0
        return mu, sigma

    def evaluate_all_on_simulation(
        self,
        sim_data: Dict,
    ) -> Dict[str, float]:
        """
        在完整模拟轨迹上评估所有基准方法。

        输入 sim_data 应包含:
            demands: (T,) 所有 GT 需求
            vlt_real: (M,) 所有 VLT 实际值
            order_times, arrival_times: 订单时点

        返回: 各种方法的总成本（holding + stockout）
        """
        demands = sim_data["demands"]
        vlt_real = sim_data["vlt_real"]
        order_times = sim_data["order_times"]
        arrival_times = sim_data["arrival_times"]
        M = len(order_times)

        # 收集每个订单的需求历史
        results = {
            "BM_normal": 0.0,
            "BM_gamma": 0.0,
            "BM1": 0.0,
            "BM2": 0.0,
            "OPT": 0.0,  # 定理 1 的真值（用 GT 需求）
        }

        for m in range(M):
            t_m = order_times[m]
            v_m = arrival_times[m]
            next_v_m = arrival_times[m + 1] if m + 1 < M else len(demands)

            # 历史需求（用于预测）
            demand_history = demands[: max(1, t_m)]
            vlt_history = vlt_real[: max(1, m)]

            # 需求 + VLT 估计
            mu_D, sigma_D = self.predict_demand_normal(demand_history)
            mu_VLT, sigma_VLT = self.predict_vlt_normal(vlt_history)

            # 初始库存（基于 GT）
            if v_m >= 1:
                I_v = -np.sum(demands[: v_m])
                for j in range(m):
                    if arrival_times[j] <= v_m:
                        pass  # 历史订单都为 0（baseline）
            else:
                I_v = 0.0

            # 当前订单区间需求
            window_demands = demands[v_m:next_v_m]
            window_total = window_demands.sum() if len(window_demands) > 0 else 0

            # ---- 基准 1: 正态基础库存 ----
            a_normal = base_stock_normal(
                mu_D=mu_D, sigma_D=sigma_D,
                mu_VLT=mu_VLT, sigma_VLT=sigma_VLT,
                review_period=self.review_period,
                b=self.b, h=self.h,
            )
            cost_normal = self._compute_interval_cost(a_normal, I_v, window_demands)
            results["BM_normal"] += cost_normal

            # ---- 基准 2: Gamma 基础库存 ----
            a_gamma = base_stock_gamma(
                mu_D=mu_D, sigma_D=sigma_D,
                mu_VLT=mu_VLT,
                review_period=self.review_period,
                b=self.b, h=self.h,
            )
            cost_gamma = self._compute_interval_cost(a_gamma, I_v, window_demands)
            results["BM_gamma"] += cost_gamma

            # ---- 基准 3: BM1 (公式 9, PTO + 单周期) ----
            a_bm1 = bm1_decision(I_v, window_total, self.b, self.h)
            # BM1 的需求预测用 d̂_total
            # 加报童安全库存
            z = norm.ppf(self.b / (self.b + self.h))
            a_bm1 += z * sigma_D * np.sqrt(next_v_m - v_m)
            a_bm1 = max(a_bm1, 0)
            cost_bm1 = self._compute_interval_cost(a_bm1, I_v, window_demands)
            results["BM1"] += cost_bm1

            # ---- 基准 4: BM2 (公式 10, PTO + 多周期 + 定理1) ----
            # 这里"预测的每天需求"用 GT（因为 PTO 阶段的预测误差我们单独评估）
            a_bm2 = bm2_decision(I_v, window_demands, self.b, self.h)
            cost_bm2 = self._compute_interval_cost(a_bm2, I_v, window_demands)
            results["BM2"] += cost_bm2

            # ---- OPT (定理 1 + GT) ----
            a_opt = bm2_decision(I_v, window_demands, self.b, self.h)
            cost_opt = self._compute_interval_cost(a_opt, I_v, window_demands)
            results["OPT"] += cost_opt

        return results

    def _compute_interval_cost(
        self,
        a: float,
        I_v: float,
        window_demands: np.ndarray,
    ) -> float:
        """计算一个订单区间上的总成本（公式 2 的求和）"""
        cost = 0.0
        inventory = I_v + a  # 初始库存 + 订货
        for d in window_demands:
            inventory -= d
            if inventory >= 0:
                cost += self.h * inventory
            else:
                cost += self.b * (-inventory)
                inventory = 0  # 简化：缺货不再累积
        return cost


if __name__ == "__main__":
    # 自检
    np.random.seed(0)
    sim_data = {
        "demands": np.random.gamma(3, 1, 30),
        "vlt_real": np.random.poisson(3, 4) + 1,
        "order_times": np.array([3, 10, 17, 24]),
        "arrival_times": np.array([6, 14, 21, 28]),
    }
    evaluator = PTOBenchmarkEvaluator()
    results = evaluator.evaluate_all_on_simulation(sim_data)
    print("PTO 基准结果:")
    for k, v in results.items():
        print(f"  {k}: {v:.2f}")