# -*- coding: utf-8 -*-
"""
b-value 时间序列分析：最大似然法 (MLE) 与 b-positive 方法
针对 data/ 下每个地震目录中的 *.afts 文件（时间, 纬度, 经度, 深度, 震级）

输出（结果与源文件名相关，保存在 results/<目录名>/ 下）：
  <stem>_bvalue_combined.png   累计地震矩 + MLE b值(多窗口) + b-positive(多窗口)
  <stem>_MLE.png               累计地震矩 + MLE b值（单独绘制）
  <stem>_bpositive.png         累计地震矩 + b-positive b值（单独绘制）
  <stem>_bvalue_timeseries.csv b值时间序列数据（长表）
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_ROOT = "data"
OUT_ROOT = "results"

# ---- 可调参数 ----
TIME_WINDOWS_DAYS = [1, 2, 7]        # 滑动窗口（天），以主震(t=0)为锚，步长 = 窗口/2
MIN_EVENTS_IN_WINDOW = 20            # MLE 窗口内最少事件数
MIN_POS_DIFFS = 10                   # b-positive 最少正差分个数
MIN_EVENTS_EMIT = 10                 # 窗口内事件数低于该值则不输出该窗口
SAVE_MLE_FITS = True                 # 是否为每个 MLE 时间窗口独立保存回归拟合图
N_BOOT = 400                         # Bootstrap 重采样次数
CI_PCT = (2.5, 97.5)                 # 置信区间分位数
MAG_BIN = 0.1                        # MAXC 震级直方图 bin 宽
DM_BIN = 0.05                        # b-positive 差分直方图 bin 宽
LOG10E = np.log10(np.e)              # 0.4343
MC_METHOD = "MAXC"                   # "MAXC" | "EMR" (Woessner & Wiemer 2005)
                                     # 命令行加 --emr 可临时切换为 EMR


# ---------------------------------------------------------------- 数据读取
def load_afts(path):
    """读取 .afts: time, lat, lon, depth, mag"""
    df = pd.read_csv(path, header=None, names=["time", "lat", "lon", "depth", "mag"],
                     dtype=str)
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    for c in ["lat", "lon", "depth", "mag"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["time", "mag"]).sort_values("time").reset_index(drop=True)
    return df


# ---------------------------------------------------------------- Mc 估计
def maxc_mc(mags, bin_width=MAG_BIN):
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


def _phi(z):
    """标准正态 CDF（Abramowitz-Stegun 7.1.26 近似，矢量化，精度 ~1e-7）"""
    z = np.asarray(z, float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(z))
    poly = t * (0.3193815300 + t * (-0.356563782 + t * (1.781477937
              + t * (-1.821255978 + t * 1.330274429))))
    p = 1.0 - poly * np.exp(-z * z / 2.0) / 2.5066282746310002
    return np.where(z >= 0, p, 1.0 - p)


def emr_mc(mags, bin_width=MAG_BIN, min_above=20, min_below=5):
    """EMR (Woessner & Wiemer 2005) 估计 Mc:
    对候选 Mc，用 截断正态(下部) + G-R 幂律(上部) 混合模型拟合整条
    震级频度分布，取对数似然最大的 Mc；样本不足返回 nan"""
    mags = np.asarray(mags)
    n = len(mags)
    if n < 60:
        return np.nan
    m_lo = np.floor(mags.min() / bin_width) * bin_width
    m_hi = np.ceil(mags.max() / bin_width) * bin_width
    edges = np.arange(m_lo, m_hi + bin_width, bin_width)
    if len(edges) < 6:
        return np.nan
    counts, _ = np.histogram(mags, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mmax = mags.max()
    cum_below = np.concatenate(([0], np.cumsum(counts)))
    best_ll, best_mc = -np.inf, np.nan
    for k in range(1, len(centers)):
        n_above = n - cum_below[k]
        n_below = cum_below[k]
        if n_above < min_above or n_below < min_below:
            continue
        mc = centers[k]
        above = mags[mags >= mc]
        b = LOG10E / (above.mean() - mc)
        if not np.isfinite(b) or b <= 0.05:
            continue
        below = mags[mags < mc]
        mu, sig = below.mean(), below.std(ddof=1)
        if not np.isfinite(sig) or sig <= 0:
            continue
        # 上部 G-R 幂律的 bin 概率
        num = 10.0 ** (-b * edges[:-1]) - 10.0 ** (-b * edges[1:])
        den = 10.0 ** (-b * mc) - 10.0 ** (-b * (mmax + bin_width))
        p_above = num / den
        # 下部截断正态的 bin 概率
        z_lo = np.array([(m_lo - mu) / sig])
        z_cut = np.array([(mc - mu) / sig])
        norm = _phi(z_cut)[0] - _phi(z_lo)[0]
        if norm <= 1e-12:
            continue
        p_norm = (_phi((edges[1:] - mu) / sig)
                  - _phi((edges[:-1] - mu) / sig)) / norm
        p = np.empty(len(counts))
        p[:k] = (n_below / n) * p_norm[:k]
        p[k:] = (n_above / n) * p_above[k:]
        p = np.clip(p, 1e-300, None)
        ll = float(np.sum(counts * np.log(p)))
        if ll > best_ll:
            best_ll, best_mc = ll, mc
    return best_mc


def mc_bootstrap_sigma(mags, rng, n_rep=30):
    """EMR Mc 的 bootstrap 不确定度 (1 sigma)"""
    vals = [emr_mc(rng.choice(mags, len(mags), replace=True))
            for _ in range(n_rep)]
    vals = np.array([v for v in vals if np.isfinite(v)])
    return float(vals.std(ddof=1)) if len(vals) >= 10 else np.nan


def window_mc(m_w, global_mc):
    """按 MC_METHOD 估计窗口完备震级，失败回退全目录值"""
    mc = emr_mc(m_w) if MC_METHOD == "EMR" else maxc_mc(m_w)
    return global_mc if np.isnan(mc) else mc


# ---------------------------------------------------------------- 两种方法
def b_mle(mags, mc):
    """Aki (1965) / Utsu (1965) 最大似然 b 值; 返回 (b, sigma, n)"""
    sel = mags[mags >= mc]
    n = len(sel)
    if n < MIN_EVENTS_IN_WINDOW:
        return np.nan, np.nan, n
    mean_m = sel.mean()
    if mean_m <= mc:
        return np.nan, np.nan, n
    b = LOG10E / (mean_m - mc)
    sigma = b / np.sqrt(n)                     # Shi & Bolt (1982)
    return b, sigma, n


def b_positive(mags, dmc=None):
    """van der Elst & Page (2023) b-positive; 返回 (b, sigma, n_pos)"""
    mags = np.asarray(mags)
    if len(mags) < 5:
        return np.nan, np.nan, 0
    dm = np.diff(mags)                          # 时间序列相邻事件震级差
    dm_pos = dm[dm > 0]
    if len(dm_pos) < MIN_POS_DIFFS:
        return np.nan, np.nan, len(dm_pos)
    if dmc is None:
        dmc = maxc_mc(dm, bin_width=DM_BIN)     # 差分完备震级（含正负差分）
    if np.isnan(dmc):
        dmc = 0.0
    mean_pos = dm_pos.mean()
    if mean_pos <= dmc:
        return np.nan, np.nan, len(dm_pos)
    b = LOG10E / (mean_pos - dmc)
    sigma = b / np.sqrt(len(dm_pos))
    return b, sigma, len(dm_pos)


# ---------------------------------------------------------------- 不确定性
def bootstrap_ci_mle(mags, mc, rng):
    """MLE b 值的 bootstrap 95% CI: 对 >= Mc 的事件有放回重采样
    返回 (b_lo, b_hi)，样本不足时返回 (nan, nan)"""
    sel = np.asarray(mags[mags >= mc])
    n = len(sel)
    if n < MIN_EVENTS_IN_WINDOW:
        return np.nan, np.nan
    idx = rng.integers(0, n, (N_BOOT, n))
    with np.errstate(divide="ignore", invalid="ignore"):
        bs = LOG10E / (sel[idx].mean(axis=1) - mc)
    bs = bs[np.isfinite(bs) & (bs > 0)]
    if len(bs) < N_BOOT // 2:
        return np.nan, np.nan
    return tuple(np.percentile(bs, CI_PCT))


def bootstrap_ci_bpos(mags, dmc, rng):
    """b-positive 的 bootstrap 95% CI: 对整段震级差分序列重采样
    （保留正/负差分混合比例），返回 (b_lo, b_hi)"""
    mags = np.asarray(mags)
    dm = np.diff(mags)
    k = len(dm)
    if k < 5:
        return np.nan, np.nan
    idx = rng.integers(0, k, (N_BOOT, k))
    dm_s = dm[idx]                                   # (N_BOOT, k)
    pos = np.where(dm_s > 0, dm_s, np.nan)
    counts = (dm_s > 0).sum(axis=1)
    with np.errstate(invalid="ignore"):
        means = np.nanmean(pos, axis=1)
    valid = (counts >= MIN_POS_DIFFS) & (means > dmc)
    bs = np.full(N_BOOT, np.nan)
    bs[valid] = LOG10E / (means[valid] - dmc)
    bs = bs[np.isfinite(bs) & (bs > 0)]
    if len(bs) < N_BOOT // 2:
        return np.nan, np.nan
    return tuple(np.percentile(bs, CI_PCT))


# ---------------------------------------------------------------- 滑动窗口
def day_windows(days, w):
    """以主震(t=0)为锚点的滑动时间窗口，步长 = w/2 天
    返回 (窗口中心时刻[天], 事件掩码)；事件数过少的窗口跳过"""
    step = w / 2.0
    s = np.floor(days.min() / step) * step
    while s <= days.max():
        m = (days >= s) & (days < s + w)
        if m.sum() >= MIN_EVENTS_EMIT:
            yield s + w / 2.0, m
        s += step


def b_time_series(times, mags, days, global_mc):
    """对给定目录计算所有时间窗口的 b 值时间序列，返回长表 DataFrame"""
    dmc_global = maxc_mc(np.diff(mags), bin_width=DM_BIN)
    rng = np.random.default_rng(42)          # 固定种子，结果可复现
    rows = []
    for w in TIME_WINDOWS_DAYS:
        for t_c, m in day_windows(days, w):
            m_w = mags[m]
            # -- MLE: 窗口内定 Mc (MAXC/EMR)，失败则用全目录 Mc
            mc = window_mc(m_w, global_mc)
            b1, s1, ne1 = b_mle(m_w, mc)
            lo1, hi1 = bootstrap_ci_mle(m_w, mc, rng) if np.isfinite(b1) \
                else (np.nan, np.nan)
            rows.append(dict(method="MLE", window=w, t_days=t_c,
                             n_events=ne1, Mc=mc, b=b1, sigma=s1,
                             b_lo=lo1, b_hi=hi1))
            # -- b-positive
            dmc = dmc_global if not np.isnan(dmc_global) else 0.0
            b2, s2, nd = b_positive(m_w, dmc=dmc)
            lo2, hi2 = bootstrap_ci_bpos(m_w, dmc, rng) if np.isfinite(b2) \
                else (np.nan, np.nan)
            rows.append(dict(method="b-positive", window=w, t_days=t_c,
                             n_events=nd, Mc=dmc, b=b2, sigma=s2,
                             b_lo=lo2, b_hi=hi2))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 绘图
def moment_cumulative(mags):
    m0 = 10.0 ** (1.5 * mags + 9.1)              # N·m (Kanamori)
    return np.cumsum(m0)


def plot_mle_fit(mags, mc, b, sigma, t_c, w, png_path, stem):
    """MLE 回归分析中间图: log10 N(>=M) vs M + 最大似然拟合直线"""
    mags = np.asarray(mags)
    sel = mags[mags >= mc]
    lo = np.floor(mags.min() / MAG_BIN) * MAG_BIN
    hi = np.ceil(mags.max() / MAG_BIN) * MAG_BIN
    edges = np.arange(lo, hi + MAG_BIN, MAG_BIN)
    counts, _ = np.histogram(mags, bins=edges)
    cum = np.cumsum(counts[::-1])[::-1]          # N(>=M) 累积频度
    centers = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    ax.semilogy(centers[cum > 0], cum[cum > 0], ".",
                color="0.4", ms=2.5)
    mm = np.array([mc, sel.max()])
    ax.semilogy(mm, len(sel) * 10.0 ** (b * (mc - mm)),
                "r-", lw=1.2)
    ax.axvline(mc, color="r", ls="--", lw=0.7)
    ax.set_xlabel("M", fontsize=8)
    ax.set_ylabel("N($\\geq$M)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(f"{stem}\nW={w:g} d, t={t_c:+.1f} d | "
                 f"b={b:.2f}\u00b1{sigma:.2f}, Mc={mc:.1f}, n={len(sel)}",
                 fontsize=7)
    fig.tight_layout()
    fig.savefig(png_path, dpi=90)
    plt.close(fig)


def save_mle_fit_figs(res, days, mags, out_stem, outdir):
    """为每个 MLE 窗口生成回归拟合图，返回 {行索引: 相对路径}"""
    fit_map = {}
    root = os.path.join(outdir, f"MLE_fits_{MC_METHOD}")
    eps = 1e-6
    for row in res[res.method == "MLE"].itertuples():
        if pd.isna(row.b):
            continue
        w, t_c = row.window, row.t_days
        s = t_c - w / 2.0
        m = (days >= s - eps) & (days < s + w - eps)
        if m.sum() < 5:
            continue
        wd = os.path.join(root, f"{out_stem}_W{w:g}d")
        os.makedirs(wd, exist_ok=True)
        png = os.path.join(wd, f"t{t_c:+.1f}d.png")
        plot_mle_fit(mags[m], row.Mc, row.b, row.sigma, t_c, w, png,
                     out_stem)
        fit_map[row.Index] = os.path.relpath(png)
    return fit_map


def plot_catalog(df, days, mainshock_mag, res, stem, outdir):
    """三个图：combined / MLE / b-positive"""
    mags = df["mag"].values
    cum_m0 = moment_cumulative(mags)

    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(TIME_WINDOWS_DAYS)))
    cdict = dict(zip(TIME_WINDOWS_DAYS, colors))

    def _top_panel(ax):
        ax.scatter(days, mags, s=3, c="0.6", alpha=0.5, rasterized=True)
        ax.axvline(0, color="r", ls="--", lw=0.8)
        ax.set_ylabel("Magnitude")
        ax.set_ylim(mags.min() - 0.3, mags.max() + 0.6)
        ax2 = ax.twinx()
        ax2.plot(days, cum_m0, color="tab:orange", lw=1.2)
        ax2.set_yscale("log")
        ax2.set_ylabel("Cumulative seismic moment (N·m)", color="tab:orange")
        ax2.tick_params(axis="y", labelcolor="tab:orange")
        ax.set_title(f"{stem}  (mainshock M{mainshock_mag:.1f}, "
                     f"{len(mags)} events)", fontsize=10)

    def _b_panel(ax, method):
        sub = res[res["method"] == method]
        for win in TIME_WINDOWS_DAYS:
            w = sub[sub["window"] == win].sort_values("t_days")
            if len(w) == 0:
                continue
            c = cdict[win]
            ax.plot(w["t_days"], w["b"], color=c, lw=1.2,
                    label=f"W={win:g} d")
            if "b_lo" in w.columns and w["b_lo"].notna().any():
                lo, hi = w["b_lo"], w["b_hi"]      # bootstrap 95% CI
            else:
                lo, hi = w["b"] - w["sigma"], w["b"] + w["sigma"]
            ax.fill_between(w["t_days"], lo, hi, color=c, alpha=0.12,
                            linewidth=0)
        ax.axvline(0, color="r", ls="--", lw=0.8)
        ax.set_ylabel(f"b-value ({method})")
        ax.set_xlabel("Days since mainshock")
        ax.set_ylim(0.1, 2.5)   # 避免少量离群窗口压扁整体可读性（CSV 保留原值）
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    # ---- combined ----
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    _top_panel(axes[0])
    _b_panel(axes[1], "MLE")
    _b_panel(axes[2], "b-positive")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}_bvalue_combined.png"), dpi=150)
    plt.close(fig)

    # ---- MLE 单独 ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    _top_panel(axes[0])
    _b_panel(axes[1], "MLE")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}_MLE.png"), dpi=150)
    plt.close(fig)

    # ---- b-positive 单独 ----
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    _top_panel(axes[0])
    _b_panel(axes[1], "b-positive")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{stem}_bpositive.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 主流程
def main():
    global MC_METHOD
    if "--emr" in sys.argv:
        MC_METHOD = "EMR"
        sys.argv.remove("--emr")
    afts_files = []
    for dirpath, _, filenames in os.walk(DATA_ROOT):
        for fn in sorted(filenames):
            if fn.lower().endswith(".afts"):
                afts_files.append(os.path.join(dirpath, fn))

    print(f"Found {len(afts_files)} .afts files | Mc method = {MC_METHOD}\n")
    only = sys.argv[1].lower() if len(sys.argv) > 1 else None
    for path in afts_files:
        rel_tmp = os.path.relpath(path, DATA_ROOT)
        if only and only not in rel_tmp.lower():
            continue
        rel = os.path.relpath(path, DATA_ROOT)
        stem = os.path.splitext(os.path.basename(path))[0]
        cat_dir = os.path.dirname(rel)
        # 结果目录：results/<目录名>/，文件名带上级子目录前缀以防重名
        prefix = cat_dir.replace(os.sep, "__")
        outdir = os.path.join(OUT_ROOT, cat_dir.split(os.sep)[0])
        os.makedirs(outdir, exist_ok=True)
        out_stem = f"{prefix}__{stem}" if cat_dir else stem
        if MC_METHOD == "EMR":
            out_stem += "_EMR"

        print(f"[{rel}]")
        try:
            df = load_afts(path)
        except Exception as e:
            print(f"    FAILED to read: {e}")
            continue
        if len(df) < 100:
            print(f"    skip: only {len(df)} valid events")
            continue

        # 主震 = 最大震级事件，时间以主震为参考
        imax = df["mag"].idxmax()
        t_main = df.loc[imax, "time"]
        m_main = df.loc[imax, "mag"]
        days = (df["time"] - t_main).dt.total_seconds().values / 86400.0
        print(f"    {len(df)} events | mainshock M{m_main} at {t_main} "
              f"| span {days.min():.1f} to {days.max():.1f} days")

        mags = df["mag"].values
        if MC_METHOD == "EMR":
            global_mc = emr_mc(mags)
            sig = mc_bootstrap_sigma(mags, np.random.default_rng(1))
            print(f"    catalog EMR Mc = {global_mc:.2f} ± {sig:.2f}")
        else:
            global_mc = maxc_mc(mags)
            print(f"    catalog MAXC Mc = {global_mc:.2f}")

        res = b_time_series(df["time"].values, mags, days, global_mc)
        res.insert(0, "file", rel)
        if SAVE_MLE_FITS:
            res["fit_png"] = ""
            fit_map = save_mle_fit_figs(res, days, mags, out_stem, outdir)
            res.loc[list(fit_map), "fit_png"] = list(fit_map.values())
            print(f"    {len(fit_map)} MLE fit figures -> "
                  f"{os.path.join(outdir, 'MLE_fits')}")
        csv_path = os.path.join(outdir, f"{out_stem}_bvalue_timeseries.csv")
        res.to_csv(csv_path, index=False, float_format="%.4f")

        plot_catalog(df, days, m_main, res, out_stem, outdir)
        print(f"    -> {csv_path} (+ 3 figures)\n")

    print("All done.")


if __name__ == "__main__":
    sys.exit(main())
