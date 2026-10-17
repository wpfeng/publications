# -*- coding: utf-8 -*-
"""
Monte Cristo (30 d 窗) 与 Ridgecrest (1 d 窗) 双 y 轴对比图 (独立运行版)
======================================================================
并排两个子图, 每个子图保持 MT 图 + b 值双 y 轴绘制:
  (a) Ridgecrest 2019 (Mw 7.1, 含 M6.4 前震): 1 天窗口 / 0.5 天步长
  (b) Monte Cristo 2020 (Mw 6.5): 30 天窗口 / 15 天步长
子图序号 (a)/(b) 置于各轴内左上角; 无图题; 图内文字一律西文.

独立运行: 所有路径相对于本脚本所在目录, 可在任意工作目录下执行:
  python plot_mc_rc_dual_axis_compare.py

目录结构 (随包提供):
  data/MonteCristo_NLL-SSST.afts      Monte Cristo NLL-SSST 重定位目录
  data/ridgecrest_srl-2019309.afts    Ridgecrest 重定位目录 (SRL 2019-0309 补充数据)
  results/MonteCristo_bvalue_30d.csv  MLE b 值时间序列 (30 d 窗 / 15 d 步长)
  results/ridgecrest_bvalue_1d.csv    MLE b 值时间序列 (1 d 窗 / 0.5 d 步长)

依赖: python 3 + numpy + pandas + matplotlib (Anaconda/Miniconda 默认自带)
输出: output/mc_rc_dual_axis_compare.{png,pdf}
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bvalue_analysis import load_afts

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "figure_4")

PANELS = [
    dict(
        afts=os.path.join(BASE, "data", "ridgecrest_srl-2019309.afts"),
        csv=os.path.join(BASE, "results", "ridgecrest_bvalue_1d.csv"),
        wlabel="1 d", ann_mag=5.1, xmin_pad=-2.2, tag="a"),
    dict(
        afts=os.path.join(BASE, "data", "MonteCristo_NLL-SSST.afts"),
        csv=os.path.join(BASE, "results", "MonteCristo_bvalue_30d.csv"),
        wlabel="30 d", ann_mag=5.1, xmin_pad=-8.0, tag="b"),
]


def plot_panel(ax1, cfg):
    """单个子图: MT (左 y 轴) + b 值 (右 y 轴) 双轴叠加"""
    df = load_afts(cfg["afts"])
    mags = df.mag.to_numpy()
    mags = mags[np.isfinite(mags)]
    imax = int(np.argmax(mags))
    t0 = df.time.iloc[imax]
    days = (df.time - t0).dt.total_seconds().to_numpy() / 86400.0
    res = pd.read_csv(cfg["csv"])

    # MT (左轴)
    small = df.mag < 4.0
    ax1.scatter(days[small.to_numpy()], df.mag[small], marker="|", s=4,
                color="0.7", alpha=0.5, lw=0.6, rasterized=True, zorder=1)
    big = df.mag >= 4.0
    ax1.vlines(days[big.to_numpy()], 0, df.mag[big], color="tab:blue",
               lw=0.7, alpha=0.6, zorder=2)
    ax1.scatter(days[big.to_numpy()], df.mag[big], s=10, color="tab:blue",
                zorder=3, label=r"Events M $\geq$ 4")
    ax1.plot([days[imax]], [mags[imax]], "*", ms=18, mfc="gold", mec="k",
             mew=0.8, zorder=5,
             label=f"Mainshock (M{mags[imax]:.1f})")
    ax1.annotate(f"M{mags[imax]:.1f}", xy=(days[imax], mags[imax]),
                 xytext=(6, 4), textcoords="offset points",
                 fontsize=12, fontweight="bold", color="tab:red")
    # 强余震标注震级大小
    ann = df.mag >= cfg["ann_mag"]
    for _, r in df[ann].iterrows():
        if r.name == imax:
            continue
        d_ev = (r.time - t0).total_seconds() / 86400
        if abs(d_ev) < 1.0:
            ax1.annotate(f"M{r.mag:.1f}", xy=(d_ev, r.mag),
                         xytext=(-6, -8), textcoords="offset points",
                         fontsize=10, color="tab:blue", zorder=6,
                         va="top", ha="right")
        else:
            ax1.annotate(f"M{r.mag:.1f}", xy=(d_ev, r.mag),
                         xytext=(5, 8), textcoords="offset points",
                         fontsize=10, color="tab:blue", zorder=6,
                         va="bottom", ha="left")
    ax1.set_xlabel("Days since mainshock")
    ax1.set_ylabel("Magnitude", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0, mags[imax] * 1.15)
    ax1.set_xlim(min(days.min(), cfg["xmin_pad"]), days.max())
    # b 值 (右轴)
    ax2 = ax1.twinx()
    ax2.axvline(0, color="tab:red", ls="--", lw=1.0, alpha=0.7, zorder=2)
    ax2.plot(res.t_center_day, res.b_mle, "-o", color="tab:red",
             ms=3.5, lw=1.3, zorder=4,
             label=f"MLE b ({cfg['wlabel']})")
    ax2.fill_between(res.t_center_day,
                     res.b_mle - res.sigma, res.b_mle + res.sigma,
                     color="tab:red", alpha=0.15, zorder=3,
                     label=r"$\pm 1\sigma$")
    ax2.set_ylabel("b value (MLE)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    b_lo = np.nanmin(res.b_mle - res.sigma)
    b_hi = np.nanmax(res.b_mle + res.sigma)
    ax2.set_ylim(max(0.0, b_lo - 0.25), b_hi + 0.35)
    ax1.grid(alpha=0.3)
    ax1.text(0.03, 0.97, f"({cfg['tag']})", transform=ax1.transAxes,
             ha="left", va="top", fontsize=16, fontweight="bold")
    return ax1, ax2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["xtick.labelsize"] = 14
    plt.rcParams["ytick.labelsize"] = 14

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, cfg in zip((ax_a, ax_b), PANELS):
        ax1, ax2 = plot_panel(ax, cfg)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
                   fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}.{ext}", dpi=300)
    plt.close(fig)
    print("->", OUT + ".png / .pdf")


if __name__ == "__main__":
    main()
