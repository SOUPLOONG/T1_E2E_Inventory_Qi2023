"""
真实数据加载器
==============
加载作者公开的 JD.com 脱敏数据 (来自 replication package)。

文件清单 (路径: replication_package_MS-FTBDA-20-01585/):
1. df_test.csv                     — 3000 SKUs 离线实验测试集 (Section 4)
2. control_demand_vlt.csv           — 55,333 个 (SKU,DC) 对的候选对照集 (Section 5.2.1)
3. order_e2e_pre_500.csv            — E2E 处理组 500 对, 实验前 (Section 5)
4. order_e2e_post_500.csv           — E2E 处理组 500 对, 实验后
5. order_jd_pre_500.csv             — JD 对照组 500 对, 实验前
6. order_jd_post_500.csv            — JD 对照组 500 对, 实验后
7. E2E-model-data/Data/             — 示例原始数据 (rdc_sales_1320.csv, stock.csv, vlt_2019_0305.csv)
"""

import sys
# 修复 Windows 终端 GBK 编码
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import ast
import os
from typing import Dict, Optional


class RealDataLoader:
    """加载作者公开的 replication package 数据"""

    def __init__(self, pkg_root: str):
        self.pkg_root = pkg_root
        self.df_test: Optional[pd.DataFrame] = None
        self.control_candidates: Optional[pd.DataFrame] = None
        self.e2e_pre: Optional[pd.DataFrame] = None
        self.e2e_post: Optional[pd.DataFrame] = None
        self.jd_pre: Optional[pd.DataFrame] = None
        self.jd_post: Optional[pd.DataFrame] = None

    def load_all(self):
        """加载所有数据"""
        print("=" * 60)
        print("加载作者公开的 JD 数据 ...")
        print("=" * 60)
        self.load_df_test()
        self.load_control_candidates()
        self.load_field_experiment_data()
        print()
        return self

    def load_df_test(self):
        """加载 df_test.csv (3000 SKUs)"""
        path = os.path.join(self.pkg_root, "df_test.csv")
        df = pd.read_csv(path)
        # 解析 list 字符串为 numpy array
        df['sf_rnn'] = df['sf_rnn'].apply(self._parse_list)
        df['demand_hist'] = df['demand_hist'].apply(self._parse_list)
        self.df_test = df
        print(f"[✓] df_test.csv        | {len(df):>5} SKUs | "
              f"列: {list(df.columns)}")
        return df

    def load_control_candidates(self):
        """加载 control_demand_vlt.csv (候选对照)"""
        path = os.path.join(self.pkg_root, "control_demand_vlt.csv")
        df = pd.read_csv(path)
        # 去掉第一列索引
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        df = df.rename(columns={'vlt_num': 'vlt_num'})
        self.control_candidates = df
        print(f"[✓] control_demand_vlt | {len(df):>5} 对   | "
              f"列: {list(df.columns)}")
        return df

    def load_field_experiment_data(self):
        """加载 4 个实地实验数据"""
        for name, attr in [
            ("order_e2e_pre_500.csv",  "e2e_pre"),
            ("order_e2e_post_500.csv", "e2e_post"),
            ("order_jd_pre_500.csv",   "jd_pre"),
            ("order_jd_post_500.csv",  "jd_post"),
        ]:
            path = os.path.join(self.pkg_root, name)
            df = pd.read_csv(path)
            # 解析 list 字符串
            df['test_inv'] = df['test_inv'].apply(self._parse_list)
            df['test_demand'] = df['test_demand'].apply(self._parse_list)
            setattr(self, attr, df)
            print(f"[✓] {name:<25} | {len(df):>5} 对")

    @staticmethod
    def _parse_list(s):
        """解析 '[1.0, 2.0, 3.0]' 形式的字符串为 numpy array"""
        if isinstance(s, (list, np.ndarray)):
            return np.array(s, dtype=float)
        if isinstance(s, str):
            s = s.strip()
            if s.startswith('[') and s.endswith(']'):
                try:
                    # 处理 numpy 风格的 '[1. 2. 3.]'
                    parts = s[1:-1].split()
                    return np.array([float(p.rstrip(',')) for p in parts if p.rstrip(',')], dtype=float)
                except Exception:
                    try:
                        return np.array(ast.literal_eval(s), dtype=float)
                    except Exception:
                        return np.array([], dtype=float)
            # 解析 '5 days' 形式的 VLT
            if 'days' in s:
                try:
                    return float(s.split()[0])
                except Exception:
                    return np.nan
            try:
                return float(s)
            except Exception:
                return np.nan
        return s

    def summary(self):
        """打印加载摘要"""
        print("\n" + "=" * 60)
        print("数据摘要")
        print("=" * 60)
        if self.df_test is not None:
            d = self.df_test
            print(f"df_test:")
            print(f"  - {len(d)} SKUs")
            print(f"  - demand_mean: [{d['demand_mean'].min():.2f}, {d['demand_mean'].max():.2f}]")
            print(f"  - demand_hist 长度: {d['demand_hist'].apply(len).describe()}")
            print(f"  - OPT_pred: [{d['OPT_pred'].min():.0f}, {d['OPT_pred'].max():.0f}]")
            print(f"  - E2E_RNN_pred: [{d['E2E_RNN_pred'].min():.0f}, {d['E2E_RNN_pred'].max():.0f}]")
            print(f"  - gbm_pred: [{d['gbm_pred'].min():.0f}, {d['gbm_pred'].max():.0f}]")
        if self.e2e_post is not None:
            d = self.e2e_post
            print(f"\nfield experiment data:")
            print(f"  - E2E post: {len(d)} 对, ave_inv 均值: {d['ave_inv'].mean():.2f}")
            print(f"  - JD  post: {len(self.jd_post)} 对, ave_inv 均值: {self.jd_post['ave_inv'].mean():.2f}")


if __name__ == "__main__":
    PKG = r"E:\DeepSeek工作产出\人工智能与经济管理研究课程\_pkg_temp\replication_package_MS-FTBDA-20-01585"
    loader = RealDataLoader(PKG)
    loader.load_all()
    loader.summary()