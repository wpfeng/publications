# -*- coding: utf-8 -*-
"""
三联图 (竖排版, bEST 扩展版): Italy / Ridgecrest MT 图 + 六方法 Δb 汇总
==========================================================================
plot_three_panel_mt_bdrop_vertical.py 的 bEST 扩展变体:
  (a) 2016-2017 Italy 目录 MT 图, 标注 M6.08 / M6.06 / M5.26;
  (b) 2019 Ridgecrest 目录 MT 图, 标注 M7.1;
      x 轴均为绝对时间 (YYYYMMDD 刻度格式);
  (c) 跨事件 b 值下降汇总, 6 种方法并排柱状图:
      MLE, MLE-bEST (bEST method 1), KS-Taroni (bEST method 2, 事件依赖 Mc),
      bpos (本项目 MAXC(Δm) 自动 dmc), bpos-bEST (bEST method 3 原版),
      bmorepos (bEST method 4, k=100 / dmax=30 km / delta_m=0.1, 200 次 bootstrap)
      —— 所有方法统一使用局部(各子目录自身) MAXC 完备震级, 不用全局 Mc。

b 值计算全部在本脚本内联完成 (源数据: data/before_after_1d/*.afts),
不依赖任何预先计算的 CSV; 仅 bEST 四种估计器引用本工作区引入的
工具箱源码 best.py (Taroni et al., 2026, SRL, doi:10.1785/0220250175)。

事件命名: Italy M6.08 / Italy M6.06 / Italy M5.26 / Ridgecrest M7.1
无图题; (a)(b)(c) 粗体置图框内; 西文标注; PNG+PDF.
独立运行: 所有路径相对于本脚本所在目录, 可在任意工作目录下执行:
用法: python plot_three_panel_mt_bdrop_vertical_best.py
输出: output/three_panel_mt_bdrop_vertical_best.{png,pdf} (+ _deltab.csv)
数据: data/ 随包提供 (Italy 目录, Ridgecrest 目录, before_after_1d/ 前后 ±1 d 子目录),
      以及 bEST 工具箱源码 best.py (Taroni et al., 2026, SRL).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.size"] = 30  # 3x base font size
# 加粗图框与刻度线: 插入 Word 后子图边框保持清晰
plt.rcParams["axes.linewidth"] = 3.0
plt.rcParams["xtick.major.width"] = 3.0
plt.rcParams["ytick.major.width"] = 3.0

from best import best, best_bmorepos_bootstrap  # bEST 工具箱 (Taroni et al. 2026)

BASE = os.path.dirname(os.path.abspath(__file__))

ITALY = os.path.join(BASE, "data", "Amatrice_CAT5_Mw.afts")
RIDGE = os.path.join(BASE, "data", "ridgecrest_srl-2019309.afts")
BA_DIR = os.path.join(BASE, "data", "before_after_1d")   # ±1 d 前后子目录
OUT_DIR = os.path.join(BASE, "output")
OUT = os.path.join(OUT_DIR, "figure_7")

ITALY_TARGETS = [("2016-08-24", "M6.08"), ("2016-10-30", "M6.06"),
                 ("2017-01-18", "M5.26")]
RIDGE_TARGETS = [("2019-07-06", "M7.1")]

# 事件配置: key = before_after_1d 文件词干, label = 面板 (c) 命名
EVENTS = [
    ("Amatrice",    "Italy M6.08"),
    ("Norcia",      "Italy M6.06"),
    ("M526",        "Italy M5.26"),
    ("Ridgecrest",  "Ridgecrest M7.1"),
]

METHODS = ["MLE", "MLE-bEST", "KS-Taroni", "bpos", "bpos-bEST", "bmorepos"]
COLORS = ["#33658A", "#758E4F", "#F6AE2D", "#2F4858", "#C4622D", "#8A3365"]
LOG10E = np.log10(np.e)

# bEST 公共参数 (与 calc_b_best_compare.py 一致)
MAG_BIN = 0.01
DELTA_M = 0.1
K_BMP = 100
DMAX_KM = 30.0
N_BOOT_BMP = 200


# ---------------------------------------------------------------- 数据读取
def load_afts(path):
    df = pd.read_csv(path, header=None,
                     names=["time", "lat", "lon", "depth", "mag"],
                     dtype=str)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ["lat", "lon", "depth", "mag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["time", "mag"]).sort_values("time") \
             .reset_index(drop=True)


def find_event(df, date_str):
    t0 = pd.Timestamp(date_str, tz="UTC")
    near = df[(df.time - t0).abs() <= pd.Timedelta(days=3)]
    near = near[near.mag >= 5.0]
    i = (near.time - t0).abs().idxmin()
    return df.loc[i]


# ---------------------------------------------------------------- MT 面板
def mt_panel(ax, df, targets, stem_thr, gray="0.65", malpha=0.35):
    ax.scatter(df.time, df.mag, marker="|", s=3, color=gray,
               alpha=malpha, lw=0.5, rasterized=True, zorder=1)
    big = df[df.mag >= stem_thr]
    ax.vlines(big.time, 0, big.mag, color="tab:blue", lw=0.6,
             alpha=0.6, zorder=2)
    ax.scatter(big.time, big.mag, s=8, color="tab:blue", zorder=3)
    ymax = df.mag.max()
    ev_times = []
    for date_str, lab in targets:
        ev = find_event(df, date_str)
        ev_times.append(ev.time)
        ax.axvline(ev.time, color="tab:red", ls="--", lw=0.9, alpha=0.7,
                   zorder=2)
        ax.plot([ev.time], [ev.mag], "*", ms=25, mfc="gold",
                mec="k", mew=0.8, zorder=5)
        ax.annotate(lab, xy=(ev.time, ev.mag),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=28, fontweight="bold", color="tab:red",
                    zorder=6)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("Magnitude")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,
                                                      maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y%m%d"))
    # 保证 x 轴刻度包含所有选定地震的时刻; 自动刻度若离地震时刻
    # 太近(按字号估算的最小间距)则移除, 避免标签重叠
    auto = [round(v, 6) for v in ax.get_xticks()]
    extra = [round(mdates.date2num(t), 6) for t in ev_times]
    ax_w_in = ax.figure.get_figwidth() * 0.85
    label_w_in = 8 * 21 * 0.62 / 72.0  # 8 字符 x 21pt 的近似宽度(英寸)
    min_sep = label_w_in * (ax.get_xlim()[1] - ax.get_xlim()[0]) \
        / ax_w_in * 1.25
    kept = [v for v in auto if all(abs(v - e) >= min_sep for e in extra)]
    ax.set_xticks(sorted(set(kept + extra)))
    for lab in ax.get_xticklabels():
        lab.set_rotation(10)
        lab.set_fontsize(21)
        lab.set_ha("center")
    ax.grid(alpha=0.3, axis="y")


# ---------------------------------------------------------------- 内联 b 值计算
def maxc_mc(mags, bin_width=0.1):
    """最大曲率法 (MAXC) 估计完备震级 Mc"""
    mags = np.asarray(mags)
    if len(mags) < 20:
        return np.nan
    lo = np.floor(mags.min() / bin_width) * bin_width
    hi = np.ceil(mags.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    if len(edges) < 3:
        return np.nan
    counts, edges = np.histogram(mags, bins=edges)
    if counts.max() == 0:
        return np.nan
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers[np.argmax(counts)]


def running_maxc_mc(mags):
    """逐事件完备震级: centered running MAXC (窗口 = min(101, 最大奇数<=n))。
    样本不足时退回全目录 MAXC。KS-Taroni (bEST method 2) 使用。"""
    mags = np.asarray(mags)
    n = len(mags)
    full_mc = maxc_mc(mags)
    if np.isnan(full_mc):
        full_mc = np.floor(mags.min() * 10) / 10
    if n < 21:
        return np.full(n, full_mc)
    w = min(101, n if n % 2 == 1 else n - 1)
    half = w // 2
    mc_vec = np.empty(n)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        mcw = maxc_mc(mags[lo:hi])
        mc_vec[i] = full_mc if np.isnan(mcw) else mcw
    return mc_vec


def mle_b(mags, mc):
    """本项目 MLE (Aki 1965): b = log10(e)/(mean(m|m>=mc) - mc), sigma = b/sqrt(n)"""
    sel = mags[mags >= mc]
    n = len(sel)
    if n < 20 or sel.mean() <= mc:
        return np.nan, np.nan
    b = LOG10E / (sel.mean() - mc)
    return b, b / np.sqrt(n)


def bpos_auto_dmc(mags):
    """本项目 b-positive: dmc 由 MAXC(全部 Δm, bin=0.05) 自动估计并扣除"""
    dm = np.diff(mags)
    dm_pos = dm[dm > 0]
    if len(dm_pos) < 10:
        return np.nan, np.nan
    dmc = maxc_mc(dm, bin_width=0.05)
    if np.isnan(dmc):
        dmc = 0.0
    if dm_pos.mean() <= dmc:
        return np.nan, np.nan
    b = LOG10E / (dm_pos.mean() - dmc)
    return b, b / np.sqrt(len(dm_pos))


def compute_methods(df, mc):
    """对单个子目录, 局部 Mc = mc, 返回 6 方法的 (b, sigma) 字典"""
    mags = df.mag.to_numpy()
    mags = mags[np.isfinite(mags)]
    out = {}

    b, s = mle_b(mags, mc)
    out["MLE"] = (b, s)

    cat = mags.reshape(-1, 1)
    b, n, s = best(cat, 0, 1, mc, None, DELTA_M, MAG_BIN, 0)
    out["MLE-bEST"] = (b, s)

    mc_vec = running_maxc_mc(mags)
    cat2 = np.column_stack([mags, mc_vec])
    b, n, s = best(cat2, 0, 2, mc, 1, DELTA_M, MAG_BIN, 0)
    out["KS-Taroni"] = (b, s)

    b, s = bpos_auto_dmc(mags)
    out["bpos"] = (b, s)

    b, n, s = best(cat, 0, 3, mc, None, DELTA_M, MAG_BIN, 0)
    out["bpos-bEST"] = (b, s)

    cat4 = df[["lat", "lon", "mag"]].to_numpy(dtype=float)
    b, n, s = best_bmorepos_bootstrap(
        cat4, magn_column=2, mc=mc, delta_m=DELTA_M, magn_bin=MAG_BIN,
        k=K_BMP, long_column=1, lat_column=0, dmax=DMAX_KM,
        bootstrap_num=N_BOOT_BMP, random_state=42)
    out["bmorepos"] = (b, s)

    return out


def compute_all_deltab():
    """对 4 事件 × 前后 ±1 d 子目录, 统一用局部 MAXC Mc 计算 6 方法 Δb"""
    rows = []
    for key, label in EVENTS:
        res = {}
        for side in ("before", "after"):
            path = os.path.join(BA_DIR, f"{side}_{key}.afts")
            df = load_afts(path)
            mags = df.mag.to_numpy()
            mags = mags[np.isfinite(mags)]
            mc_local = maxc_mc(mags)          # 局部完备震级 (唯一口径)
            res[side] = (compute_methods(df, mc_local), mc_local, len(mags))
        for meth in METHODS:
            b_pre, s_pre = res["before"][0][meth]
            b_post, s_post = res["after"][0][meth]
            if np.isfinite(b_pre) and np.isfinite(b_post):
                d = b_post - b_pre
                ds = np.hypot(s_pre, s_post) \
                    if np.isfinite(s_pre) and np.isfinite(s_post) else np.nan
            else:
                d, ds = np.nan, np.nan
            rows.append({"label": label, "method": meth,
                         "d_b": d, "d_b_sigma": ds,
                         "mc_local_pre": res["before"][1],
                         "mc_local_post": res["after"][1]})
        print(f"{label:18s} Mc_local pre={res['before'][1]:.2f} "
              f"post={res['after'][1]:.2f} | "
              + " ".join(f"{m}={res['after'][0][m][0]-res['before'][0][m][0]:+.2f}"
                         for m in METHODS))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    italy = load_afts(ITALY)
    ridge = load_afts(RIDGE)
    ddf = compute_all_deltab()
    print(f"Italy: {len(italy)} events | Ridgecrest: {len(ridge)}")

    fig, axes = plt.subplots(3, 1, figsize=(18, 19))

    mt_panel(axes[0], italy, ITALY_TARGETS, stem_thr=4.5)
    mt_panel(axes[1], ridge, RIDGE_TARGETS, stem_thr=4.0,
             gray="0.20", malpha=0.6)

    # ---------------- 面板 (c): 六方法 Δb ----------------
    ax = axes[2]
    labels = [label for _, label in EVENTS]
    x = np.arange(len(labels))
    n_m = len(METHODS)
    w = 0.72 / n_m
    for j, meth in enumerate(METHODS):
        vals, sigs = [], []
        for lab in labels:
            r = ddf[(ddf.label == lab) & (ddf.method == meth)]
            vals.append(r.d_b.iloc[0] if len(r) else np.nan)
            sigs.append(r.d_b_sigma.iloc[0] if len(r) else np.nan)
        vals = np.array(vals, float)
        sigs = np.array(sigs, float)
        pos = x + (j - (n_m - 1) / 2) * w
        plot_vals = np.where(np.isfinite(vals), vals, 0.0)
        ax.bar(pos, plot_vals, w, color=COLORS[j], label=meth, alpha=0.9)
        ax.errorbar(pos, plot_vals,
                    yerr=np.where(np.isfinite(sigs), sigs, 0),
                    fmt="none", ecolor="k", elinewidth=0.9, capsize=2)
        for xi, v in zip(pos, vals):
            if np.isfinite(v):
                # 竖排数值标注: 水平占地小, 避免相邻方法重叠
                if v >= 0:
                    ax.text(xi, v + 0.05, f"{v:+.2f}", ha="center",
                            va="bottom", rotation=90,
                            fontsize=15, color=COLORS[j])
                else:
                    ax.text(xi, v - 0.05, f"{v:+.2f}", ha="center",
                            va="top", rotation=90,
                            fontsize=15, color=COLORS[j])
    ax.axhline(0, color="k", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=22)
    for lab in ax.get_xticklabels():
        lab.set_rotation(0)
        lab.set_ha("center")
    ax.set_ylabel(r"$\Delta b$ = b(post) $-$ b(pre)")
    ax.set_ylim(-1.45, 0.62)
    ax.legend(fontsize=24, ncol=2, loc="lower right")
    ax.grid(alpha=0.3, axis="y")

    for k, ax in enumerate(axes):
        ax.text(0.925, 0.98, f"({chr(97+k)})", transform=ax.transAxes,
                fontsize=39, fontweight="bold", va="top", ha="left")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        try:
            fig.savefig(f"{OUT}.{ext}", dpi=360)
        except PermissionError:
            fig.savefig(f"{OUT}_v2.{ext}", dpi=360)
            print(f"! {OUT}.{ext} 被占用, 另存为 _v2")
    plt.close(fig)

    # Δb 数值落盘备份 (面板 c 的内联计算结果)
    csv_path = OUT + "_deltab.csv"
    ddf.to_csv(csv_path, index=False)
    print("->", OUT + ".png / .pdf")
    print("->", csv_path)


if __name__ == "__main__":
    main()
