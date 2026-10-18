# b-value Comparison Figures (Figure 4 & Figure 7)

Reproduction scripts and data for two comparison figures on the temporal evolution
of the Gutenberg–Richter b value and its co-seismic drop across earthquake sequences.

## Contents

| File | Description |
|------|-------------|
| `figure_4_plot.py` | Script reproducing **Figure 4** — dual-axis (magnitude-time + b value) comparison of the Ridgecrest 2019 (Mw 7.1) and Monte Cristo 2020 (Mw 6.5) aftershock sequences. |
| `figure_6_plot.py` | Script reproducing **Figure 7** — three-panel figure: (a) 2016–2017 Italy sequence (Amatrice Mw 6.08, Norcia Mw 6.06, Mw 5.26 events marked); (b) 2019 Ridgecrest sequence (Mw 7.1 marked); (c) cross-event Δb summary with six b-value estimators (MLE, MLE-bEST, KS-Taroni, b-positive, bpos-bEST, bmorepos). |
| `bvalue_analysis.py` | Utilities for reading `.afts` catalogues and b-value estimation (MAXC completeness magnitude, MLE). Only depends on `numpy` / `pandas` / `matplotlib`. |
| `best.py` | Source of the bEST toolbox (Taroni et al., 2026, *SRL*, doi:10.1785/0220250175), used by the bEST-family estimators in Figure 7. Only depends on `numpy`. |
| `data/` | Earthquake catalogues in `.afts` ASCII format (one event per row: time, lat, lon, depth, magnitude). |
| `results/` | Pre-computed MLE b-value time series used by Figure 4 (30-day / 1-day sliding windows). |
| `output/` | Published figures (`figure_4`, `figure_7`, PNG + PDF) and the Figure 7 Δb table (`figure_7_deltab.csv`). |

## Data

- `data/ridgecrest_srl-2019309.afts` — relocated Ridgecrest catalogue (34,091 events,
  2019-07-04 to 2019-07-16; includes the M6.4 foreshock and the M7.1 mainshock).
- `data/MonteCristo_NLL-SSST.afts` — NLL-SSST relocated Monte Cristo catalogue.
- Figure 6 additionally expects `data/Amatrice_CAT5_Mw.afts` (2016–2017 Italy sequence,
  Mw magnitudes) and `data/before_after_1d/*.afts` (±1-day before/after sub-catalogues
  for the four target events).

## Requirements

- Python 3.8+
- `numpy`, `pandas`, `matplotlib` (included in Anaconda/Miniconda by default)

## Usage

All paths are resolved relative to the script location, so the scripts can be run
from any working directory:

```bash
python figure_4_plot.py   # -> output/figure_4.png / .pdf
python figure_7_plot.py    # -> output/figure_7.png / .pdf (+ figure_7_deltab.csv)
```

All b-value computations in Figure 7 are performed inline; no pre-computed CSV is
required (the `results/` CSVs are only used by Figure 4). The bmorepos estimator
uses 200 bootstrap resamples with a fixed `random_state=42`, so results are exactly
reproducible.

## Figure descriptions

**Figure 4** — side-by-side dual-axis panels: left y axis, magnitude-time plot
(stems M ≥ 4, gold star = mainshock); right y axis, MLE b-value time series with
±1σ band (Shi & Bolt, 1982); red dashed line marks the mainshock (t = 0).
- (a) Ridgecrest 2019 (Mw 7.1), 1-day window / 0.5-day step: b drops to ~0.3–0.7
  within hours after the mainshock and recovers to ~0.9 within 2–3 days.
- (b) Monte Cristo 2020 (Mw 6.5), 30-day window / 15-day step: b stays near ~0.8
  with a slow, fluctuating post-mainshock recovery.

**Figure 6** — three vertically stacked panels (panel letters inside the frame,
no titles, western text only):
- (a)/(b) magnitude-time plots of the Italy and Ridgecrest catalogues with absolute
  time axes (YYYYMMDD); target earthquakes are marked with gold stars and dashed
  red lines.
- (c) Δb = b(post) − b(pre) for the four events (Italy M6.08 / M6.06 / M5.26 and
  Ridgecrest M7.1), ±1-day windows, six estimators side by side. All methods use the
  local (per-sub-catalogue) MAXC completeness magnitude; error bars are propagated
  1σ uncertainties.

## Licence

The bEST toolbox (`best.py`) is by Taroni et al. (2026), *Seismological Research
Letters*, doi:10.1785/0220250175 — please cite it if you use the bEST estimators.
Catalogue data belong to their respective sources (see original publications).

bvalue_analysis.py is developed by  Lvjian Wu and Wanpeng Feng (2026), *Earthquake Science, (submitted)
