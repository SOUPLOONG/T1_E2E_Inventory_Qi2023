"""
Figure 3 + Section 5 复现
=========================
对应论文:
  Figure 3: 倾向得分匹配前后分布 (PSM)
  Table 2:  t-test 结果 (5 个指标)
  Table 3:  线性回归 (公式 11)
  Table 4:  双重差分 (DiD)

数据:
  - 55,333 个候选对照 (control_demand_vlt.csv)
  - 500 个 E2E 处理组 (order_e2e_post_500.csv)
  - 500 个 JD  对照组 (order_jd_post_500.csv)  (已经 PSM 后)
  - 4 个 order_*_pre/post 文件用于 DiD

步骤:
  1. PSM: 用 LogisticRegression 估计倾向得分, 用最近邻匹配选 500 对
  2. 计算每个 SKU-DC 对的 5 个性能指标 (holding, stockout, total, turnover, stockout_ratio)
  3. t-test 比较处理组 vs 对照组 (Table 2)
  4. 线性回归: Outcome = θ0 + θ1*Is-E2E + θ2*Ave-Demand + θ3*Ave-VLT (Table 3)
  5. DiD: 比较实验前后变化 (Table 4)
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')


def compute_performance_metrics(
    inv: np.ndarray,
    demand: np.ndarray,
    h: float = 1.0,
    b: float = 9.0,
) -> dict:
    """
    计算一个 (SKU, DC) 对的 5 个性能指标.

    Args:
        inv: 测试期库存数组 (i-th 元素是周期 i 的期初库存)
        demand: 测试期每日需求
        h, b: 持有/缺货成本

    Returns:
        dict with avg_holding, avg_stockout, avg_total, turnover_rate, stockout_ratio
    """
    holding = 0.0
    stockout = 0.0
    stockout_days = 0
    n_days = len(demand)
    n_stockout = 0  # 缺货天数

    # 假设 inv[0] 是初始库存, inv[i] 是周期 i 末库存
    # 计算每个周期的成本
    for i in range(n_days):
        if i < len(inv):
            current_inv = inv[i]
        else:
            current_inv = 0.0
        if current_inv >= 0:
            holding += h * current_inv
        else:
            stockout += b * (-current_inv)
            stockout_days += 1

    # 周转率 = 平均库存 / 平均需求 (论文 Section 5.1)
    avg_inv = np.mean(np.maximum(inv[:n_days], 0))
    avg_demand = np.mean(demand[:n_days]) if n_days > 0 else 1.0
    turnover_rate = avg_inv / max(avg_demand, 1e-3)

    # 缺货率 = 缺货天数 / 总天数
    stockout_ratio = stockout_days / max(n_days, 1)

    return {
        'holding': holding / n_days if n_days > 0 else 0,
        'stockout': stockout / n_days if n_days > 0 else 0,
        'total': (holding + stockout) / n_days if n_days > 0 else 0,
        'turnover_rate': turnover_rate,
        'stockout_ratio': stockout_ratio,
    }


def compute_metrics_for_group(df: pd.DataFrame, h: float = 1.0, b: float = 9.0) -> pd.DataFrame:
    """为整个数据框计算 5 个性能指标"""
    rows = []
    for _, row in df.iterrows():
        m = compute_performance_metrics(
            inv=np.array(row['test_inv']),
            demand=np.array(row['test_demand']),
            h=h, b=b,
        )
        m['sku_dc_pair_index'] = row['sku_dc_pair_index']
        m['ave_demand'] = row['ave_demand']
        m['ave_inv'] = row['ave_inv']
        # post 文件有 vlt_num, pre 文件只有 vlt
        m['vlt'] = row['vlt_num'] if 'vlt_num' in df.columns else row['vlt']
        rows.append(m)
    return pd.DataFrame(rows)


# ============================================
# Figure 3: 倾向得分匹配
# ============================================
def propensity_score_matching(
    treatment_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    n_match: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    用倾向得分匹配 (PSM) 从候选集中选出对照.

    论文做法 (5.2.1):
      - 协变量: ave_demand, vlt_num
      - 用 LogisticRegression 估计倾向得分
      - 用最近邻匹配选 n_match 个对照
    """
    # 1. 准备数据
    treatment_arr = np.column_stack([
        treatment_df['ave_demand'].values,
        treatment_df['vlt'].values.astype(float),
    ])
    candidates_arr = np.column_stack([
        candidates_df['ave_demand'].values,
        candidates_df['vlt_num'].values.astype(float),
    ])

    # 2. 用 LogisticRegression 估计倾向得分
    # 倾向得分: 选中为处理组的概率
    X = np.vstack([treatment_arr, candidates_arr])
    y = np.concatenate([
        np.ones(len(treatment_arr)),
        np.zeros(len(candidates_arr)),
    ])
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    lr.fit(X, y)
    propensity_scores = lr.predict_proba(X)[:, 1]

    treatment_ps = propensity_scores[: len(treatment_arr)]
    candidate_ps = propensity_scores[len(treatment_arr):]

    # 3. 最近邻匹配: 对每个处理样本找最近的候选
    nbrs = NearestNeighbors(n_neighbors=1).fit(candidate_ps.reshape(-1, 1))

    distances, indices = nbrs.kneighbors(treatment_ps.reshape(-1, 1))

    # 取距离最小的 n_match 个候选作为对照
    sorted_indices = np.argsort(distances.flatten())[:n_match]
    matched_candidate_indices = indices[sorted_indices].flatten()

    matched_control = candidates_df.iloc[matched_candidate_indices].reset_index(drop=True)

    print(f"  - 处理组: {len(treatment_arr)} 个 (SKU, DC) 对")
    print(f"  - 候选对照: {len(candidates_arr)} 个")
    print(f"  - 匹配后对照: {len(matched_control)} 个")

    return matched_control, treatment_ps, candidate_ps


def plot_figure3(
    ps_before_t: np.ndarray,
    ps_before_c: np.ndarray,
    ps_after_t: np.ndarray,
    ps_after_c: np.ndarray,
    save_path: str,
):
    """
    Figure 3: PSM 前后倾向得分分布
    (a) 匹配前: 处理组和候选对照的分布差异
    (b) 匹配后: 处理组和匹配对照的分布对齐
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # (a) Before matching
    axes[0].hist(ps_before_t, bins=50, alpha=0.5, label='Treatment',
                 density=True, color='#4C72B0')
    axes[0].hist(ps_before_c, bins=50, alpha=0.5, label='Candidates (control)',
                 density=True, color='#DD8452')
    # KDE 估计
    from scipy.stats import gaussian_kde
    x_grid = np.linspace(0, 1, 200)
    if len(ps_before_t) > 1:
        kde_t = gaussian_kde(ps_before_t)
        axes[0].plot(x_grid, kde_t(x_grid), color='#1F4E79', linewidth=2)
    if len(ps_before_c) > 1:
        kde_c = gaussian_kde(ps_before_c)
        axes[0].plot(x_grid, kde_c(x_grid), color='#A0392E', linewidth=2)
    axes[0].set_xlabel('Propensity Score', fontsize=11)
    axes[0].set_ylabel('Density', fontsize=11)
    axes[0].set_title('(a) Before Matching', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # (b) After matching
    axes[1].hist(ps_after_t, bins=30, alpha=0.5, label='Treatment',
                 density=True, color='#4C72B0')
    axes[1].hist(ps_after_c, bins=30, alpha=0.5, label='Matched Control',
                 density=True, color='#DD8452')
    if len(ps_after_t) > 1:
        kde_t2 = gaussian_kde(ps_after_t)
        axes[1].plot(x_grid, kde_t2(x_grid), color='#1F4E79', linewidth=2)
    if len(ps_after_c) > 1:
        kde_c2 = gaussian_kde(ps_after_c)
        axes[1].plot(x_grid, kde_c2(x_grid), color='#A0392E', linewidth=2)
    axes[1].set_xlabel('Propensity Score', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].set_title('(b) After Matching', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle('Propensity Score Matching (paper Figure 3)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[图] Figure 3 saved to: {save_path}")


# ============================================
# Section 5.3: t-test, 线性回归, DiD
# ============================================
def t_test_comparison(metrics_treat: pd.DataFrame, metrics_ctrl: pd.DataFrame,
                      metrics_names: list) -> pd.DataFrame:
    """Table 2: 两样本 t-test"""
    rows = []
    for m in metrics_names:
        t_vals = metrics_treat[m].values
        c_vals = metrics_ctrl[m].values
        t_stat, p_val = stats.ttest_ind(t_vals, c_vals)
        rows.append({
            'Metric': m,
            'Treatment': t_vals.mean(),
            'Control': c_vals.mean(),
            'Difference': t_vals.mean() - c_vals.mean(),
            'p-value': p_val,
        })
    return pd.DataFrame(rows)


def linear_regression_adjustment(
    metrics_treat: pd.DataFrame,
    metrics_ctrl: pd.DataFrame,
    metrics_names: list,
) -> dict:
    """Table 3: 公式 (11) 线性回归

    Outcome = θ0 + θ1*Is-E2E + θ2*Ave-Demand + θ3*Ave-VLT
    """
    # 合并处理组和对照组
    metrics_treat = metrics_treat.copy()
    metrics_treat['Is_E2E'] = 1
    metrics_ctrl = metrics_ctrl.copy()
    metrics_ctrl['Is_E2E'] = 0
    combined = pd.concat([metrics_treat, metrics_ctrl], ignore_index=True)

    results = {}
    for m in metrics_names:
        X = combined[['Is_E2E', 'ave_demand', 'vlt']].values
        X = sm.add_constant(X)  # 加常数项
        y = combined[m].values
        model = sm.OLS(y, X).fit()
        results[m] = {
            'theta_0': model.params[0],
            'theta_1_IsE2E': model.params[1],
            'p_IsE2E': model.pvalues[1],
            'theta_2_AveDemand': model.params[2],
            'theta_3_AveVLT': model.params[3],
            'r_squared': model.rsquared,
        }
    return results


def did_estimation(
    metrics_treat_post: pd.DataFrame,
    metrics_treat_pre: pd.DataFrame,
    metrics_ctrl_post: pd.DataFrame,
    metrics_ctrl_pre: pd.DataFrame,
    metrics_names: list,
) -> pd.DataFrame:
    """Table 4: Difference-in-Differences

    DiD = (Post_treat - Pre_treat) - (Post_ctrl - Pre_ctrl)
    """
    rows = []
    for m in metrics_names:
        treat_change = metrics_treat_post[m].mean() - metrics_treat_pre[m].mean()
        ctrl_change = metrics_ctrl_post[m].mean() - metrics_ctrl_pre[m].mean()
        did = treat_change - ctrl_change
        # t-test on DiD estimate
        # 构建 DiD 数组
        treat_did = metrics_treat_post[m].values - metrics_treat_pre[m].values
        ctrl_did = metrics_ctrl_post[m].values - metrics_ctrl_pre[m].values
        t_stat, p_val = stats.ttest_ind(treat_did, ctrl_did)
        rows.append({
            'Metric': m,
            'Treatment_Pre': metrics_treat_pre[m].mean(),
            'Treatment_Post': metrics_treat_post[m].mean(),
            'Treatment_Change': treat_change,
            'Control_Pre': metrics_ctrl_pre[m].mean(),
            'Control_Post': metrics_ctrl_post[m].mean(),
            'Control_Change': ctrl_change,
            'DiD': did,
            'p-value': p_val,
        })
    return pd.DataFrame(rows)


def stars(p: float) -> str:
    """p-value 星标"""
    if p < 0.001:
        return '****'
    elif p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.1:
        return '*'
    else:
        return ''


def save_results_table(df: pd.DataFrame, save_path: str, title: str):
    """保存结果表到 Markdown + TXT"""
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(title + '\n')
        f.write('=' * 80 + '\n\n')
        f.write(df.to_markdown(index=False, floatfmt='.4f'))
        f.write('\n\n')
    print(f"[表] Saved to: {save_path}")


def reproduce_section5(
    e2e_post: pd.DataFrame,
    e2e_pre: pd.DataFrame,
    jd_post: pd.DataFrame,
    jd_pre: pd.DataFrame,
    candidates: pd.DataFrame,
    results_dir: str = "results",
):
    """
    Section 5 完整复现.
    """
    print("\n" + "=" * 60)
    print("Section 5: 实地实验复现")
    print("=" * 60)

    h, b = 1.0, 9.0

    # 1. 计算所有 4 个数据集的 5 个指标
    print("\n[1] 计算性能指标 ...")
    metrics_e2e_post = compute_metrics_for_group(e2e_post, h, b)
    metrics_e2e_pre = compute_metrics_for_group(e2e_pre, h, b)
    metrics_jd_post = compute_metrics_for_group(jd_post, h, b)
    metrics_jd_pre = compute_metrics_for_group(jd_pre, h, b)
    print(f"  E2E post: {len(metrics_e2e_post)} 对")
    print(f"  E2E pre:  {len(metrics_e2e_pre)} 对")
    print(f"  JD  post: {len(metrics_jd_post)} 对")
    print(f"  JD  pre:  {len(metrics_jd_pre)} 对")

    metrics_names = ['holding', 'stockout', 'total', 'turnover_rate', 'stockout_ratio']

    # 2. Figure 3: PSM (用真实数据演示)
    print("\n[2] 倾向得分匹配 (Figure 3) ...")
    matched_control, ps_t, ps_c = propensity_score_matching(
        treatment_df=metrics_e2e_post,
        candidates_df=candidates,
        n_match=500,
    )
    # 计算 matched_control 的倾向得分 (用重新训练 LR)
    treat_arr = np.column_stack([
        metrics_e2e_post['ave_demand'].values,
        metrics_e2e_post['vlt'].values.astype(float),
    ])
    cand_arr = np.column_stack([
        matched_control['ave_demand'].values,
        matched_control['vlt_num'].values.astype(float),
    ])
    # 在 matched_control 中找每个处理的最近邻
    X_ps = np.vstack([treat_arr, cand_arr])
    y_ps = np.concatenate([np.ones(len(treat_arr)), np.zeros(len(cand_arr))])
    lr2 = LogisticRegression(random_state=42, max_iter=1000)
    lr2.fit(X_ps, y_ps)
    ps_after_t = lr2.predict_proba(treat_arr)[:, 1]
    ps_after_c = lr2.predict_proba(cand_arr)[:, 1]

    plot_figure3(ps_t, ps_c, ps_after_t, ps_after_c,
                 f"{results_dir}/figure3_propensity_score.png")
    # 不需要 matched_control 的 5 个 metrics (它只是 PSM 的中间结果)

    # 3. Table 2: 两样本 t-test
    print("\n[3] Table 2: 两样本 t-test ...")
    table2 = t_test_comparison(metrics_e2e_post, metrics_jd_post, metrics_names)
    # 添加星标
    table2['sig'] = table2['p-value'].apply(stars)
    # 添加百分比变化
    table2['Pct_Change'] = (table2['Difference'] / table2['Control'] * 100).round(1)
    print(table2.to_string(index=False))
    save_results_table(table2, f"{results_dir}/table2_ttest.md",
                       "Table 2: Two-Sample t-Test Comparison")

    # 4. Table 3: 线性回归
    print("\n[4] Table 3: 线性回归 (公式 11) ...")
    lr_results = linear_regression_adjustment(
        metrics_e2e_post, metrics_jd_post, metrics_names
    )
    table3 = pd.DataFrame([
        {
            'Metric': m,
            'theta_1 (IsE2E)': v['theta_1_IsE2E'],
            'p_IsE2E': v['p_IsE2E'],
            'sig': stars(v['p_IsE2E']),
            'theta_2 (AveDemand)': v['theta_2_AveDemand'],
            'theta_3 (AveVLT)': v['theta_3_AveVLT'],
            'R^2': v['r_squared'],
        }
        for m, v in lr_results.items()
    ])
    print(table3.to_string(index=False))
    save_results_table(table3, f"{results_dir}/table3_linear_regression.md",
                       "Table 3: Linear Regression (Equation 11)")

    # 5. Table 4: DiD
    print("\n[5] Table 4: Difference-in-Differences ...")
    table4 = did_estimation(
        metrics_e2e_post, metrics_e2e_pre,
        metrics_jd_post, metrics_jd_pre,
        metrics_names,
    )
    table4['sig'] = table4['p-value'].apply(stars)
    print(table4.to_string(index=False))
    save_results_table(table4, f"{results_dir}/table4_did.md",
                       "Table 4: Difference-in-Differences")

    return {
        'table2_ttest': table2,
        'table3_linear_reg': table3,
        'table4_did': table4,
        'metrics_e2e_post': metrics_e2e_post,
        'metrics_e2e_pre': metrics_e2e_pre,
        'metrics_jd_post': metrics_jd_post,
        'metrics_jd_pre': metrics_jd_pre,
    }


if __name__ == "__main__":
    from src.real_data_loader import RealDataLoader
    PKG = r"E:\DeepSeek工作产出\人工智能与经济管理研究课程\_pkg_temp\replication_package_MS-FTBDA-20-01585"
    loader = RealDataLoader(PKG)
    loader.load_all()
    results = reproduce_section5(
        e2e_post=loader.e2e_post,
        e2e_pre=loader.e2e_pre,
        jd_post=loader.jd_post,
        jd_pre=loader.jd_pre,
        candidates=loader.control_candidates,
    )