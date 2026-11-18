"""
Qi et al. (2023) 全文复现主入口
================================
基于作者公开的 JD.com 脱敏数据 (replication package)。

完整覆盖论文:
  Section 3.2.1  Theorem 1 (闭式最优标签)        - 通过 df_test.OPT_pred 验证
  Section 3.2.2  E2E 模型架构                  - 已实现
  Section 4.1    4 个 PTO 基准                  - 公式 (7)-(10)
  Section 4.2    离线实验 + Figure 2             - figure2.py
  Section 5.2.1  倾向得分匹配 + Figure 3        - field_experiment.py
  Section 5.3.1  t-test + Table 2               - field_experiment.py
  Section 5.3.2  线性回归 + Table 3              - field_experiment.py
  Section 5.3.3  DiD + Table 4                  - field_experiment.py
  Appendix B.1   网络结构敏感性                  - sensitivity.py
  Appendix B.2   数据量敏感性 + Figure B.1      - sensitivity.py
  Appendix B.3   b/h ratio + Figure B.2         - sensitivity.py
  Appendix B.4   局部极小值 + Figure B.3         - sensitivity.py

运行:
    python run_reproduction.py                 # 全部跑
    python run_reproduction.py --skip_sensitivity   # 跳过 B (更快)
    python run_reproduction.py --skip_train         # B 只跑不需要训练的
"""

import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mpl_cache"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import argparse
import time


def main():
    parser = argparse.ArgumentParser(description="Qi et al. (2023) 全文复现")
    parser.add_argument("--pkg_root", type=str,
                        default=r"E:\DeepSeek工作产出\人工智能与经济管理研究课程\_pkg_temp\replication_package_MS-FTBDA-20-01585",
                        help="replication package 根路径")
    parser.add_argument("--results_dir", type=str, default="results",
                        help="结果输出目录")
    parser.add_argument("--skip_sensitivity", action="store_true",
                        help="跳过 Appendix B 训练部分 (Figure B.3 / Tables B.2-B.5)")
    parser.add_argument("--skip_train", action="store_true",
                        help="敏感性分析只跑不需要训练的 (B.3 b/h)")
    args = parser.parse_args()

    if not os.path.exists(args.pkg_root):
        print(f"[ERROR] 找不到数据目录: {args.pkg_root}")
        print(f"请确认 replication_package_MS-FTBDA-20-01585.zip 已解压")
        return

    print("=" * 70)
    print(" "*15 + "Qi et al. (2023) 全文复现")
    print(" "*12 + "'A Practical End-to-End Inventory Management Model with Deep Learning'")
    print(" "*10 + "Management Science 69(2): 759-773")
    print("=" * 70)
    print(f"数据根目录: {args.pkg_root}")
    print(f"结果目录:   {args.results_dir}")

    os.makedirs(args.results_dir, exist_ok=True)

    # 加载数据
    print("\n>>> [Step 1] 加载作者公开的真实 JD 数据 ...")
    from src.real_data_loader import RealDataLoader
    loader = RealDataLoader(args.pkg_root)
    loader.load_all()

    # ============== Section 4 + Figure 2 ==============
    print("\n" + "=" * 70)
    print(">>> [Step 2] Section 4 离线实验 + Figure 2")
    print("=" * 70)
    from src.figure2 import reproduce_figure2
    results_fig2 = reproduce_figure2(loader.df_test, args.results_dir)

    # ============== Section 5 + Figure 3 + Tables 2-4 ==============
    print("\n" + "=" * 70)
    print(">>> [Step 3] Section 5 实地实验 + Figure 3 + Tables 2-4")
    print("=" * 70)
    from src.field_experiment import reproduce_section5
    results_sec5 = reproduce_section5(
        e2e_post=loader.e2e_post,
        e2e_pre=loader.e2e_pre,
        jd_post=loader.jd_post,
        jd_pre=loader.jd_pre,
        candidates=loader.control_candidates,
        results_dir=args.results_dir,
    )

    # ============== Appendix B 敏感性分析 ==============
    if not args.skip_sensitivity:
        print("\n" + "=" * 70)
        print(">>> [Step 4] Appendix B 全部敏感性分析")
        print("=" * 70)
        from src.sensitivity import reproduce_appendix_b
        results_appx = reproduce_appendix_b(
            loader.df_test,
            args.results_dir,
            skip_train=args.skip_train,
        )
    else:
        results_appx = None
        print("\n[SKIP] Appendix B (按 --skip_sensitivity)")

    # ============== 总结 ==============
    print("\n" + "=" * 70)
    print("复现完成！输出文件清单")
    print("=" * 70)

    expected_files = [
        ("Figure 2(left)",  f"{args.results_dir}/figure2_left_mqrnn_forecast.png"),
        ("Figure 2(right)", f"{args.results_dir}/figure2_right_cost_comparison.png"),
        ("Figure 3",        f"{args.results_dir}/figure3_propensity_score.png"),
        ("Figure B.1",      f"{args.results_dir}/figure_b1_data_size.png"),
        ("Figure B.2",      f"{args.results_dir}/figure_b2_b_over_h.png"),
        ("Figure 2 表",      f"{args.results_dir}/figure2_results.txt"),
        ("Table 2 (t-test)", f"{args.results_dir}/table2_ttest.md"),
        ("Table 3 (LR)",     f"{args.results_dir}/table3_linear_regression.md"),
        ("Table 4 (DiD)",    f"{args.results_dir}/table4_did.md"),
        ("Table B.2 (layers)",  f"{args.results_dir}/table_b2_hidden_layers.csv"),
        ("Table B.3 (neurons)", f"{args.results_dir}/table_b3_neurons.csv"),
        ("Table B.4 (LR)",      f"{args.results_dir}/table_b4_learning_rate.csv"),
        ("Table B.5 (size)",    f"{args.results_dir}/table_b5_data_size.csv"),
        ("Table B.6 (seeds)",   f"{args.results_dir}/table_b7_local_minima.csv"),
    ]

    print(f"\n{'输出物':<22} | 路径")
    print("-" * 70)
    for name, path in expected_files:
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"{name:<22} | {exists} {path}")


if __name__ == "__main__":
    main()