"""Reproduce Task 1 diagnostics using the immutable Task 0 monthly snapshot."""
import argparse
import hashlib
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = {
    "Full": ("1997-09", "2026-09"),
    "1997–2019": ("1997-09", "2019-12"),
    "2020 onward": ("2020-01", "2026-09"),
}
TARGETS = ["pce_goods_mom_pct", "pce_goods_yoy_pct", "goods_log_mom_annualized"]
HEADLINE = ["pce_mom_pct", "pce_yoy_pct"]
FEATURES = ["gscpi", "wti_usd_per_barrel", "wti_log_mom_pct",
            "unemployment_pct", "unemployment_change_pp"]
LABELS = dict(zip(TARGETS + HEADLINE + FEATURES, [
    "Goods MoM", "Goods YoY", "Goods annualized log MoM", "Headline MoM", "Headline YoY",
    "GSCPI", "WTI level", "WTI log MoM", "Unemployment level", "Unemployment change",
]))
UNITS = dict(zip(LABELS, ["%", "%", "% annualized", "%", "%", "std. dev.",
                         "USD/barrel", "% (log change)", "%", "pp"]))
LABELS["gscpi_change_sd"] = "GSCPI change"
UNITS["gscpi_change_sd"] = "std. dev. difference"
COLORS = ["#096A80", "#B05128", "#70468F", "#3A7758", "#845737"]
HORIZONS = [0, 1, 2, 3, 6, 12]
N_LAGS = 24


def load_panel(path):
    panel = pd.read_csv(path).set_index("month")
    panel.index = pd.PeriodIndex(panel.index, freq="M", name="month")
    if panel.index.has_duplicates or not panel.index.is_monotonic_increasing:
        raise ValueError("Expected ordered and unique observation months")
    calendar = pd.period_range(panel.index[0], panel.index[-1], freq="M", name="month")
    if not panel.index.equals(calendar):
        raise ValueError("Task 0 snapshot must contain a complete monthly calendar")
    return panel


def logged_change(series, scale):
    # Missing or nonpositive levels make both adjacent log changes undefined.
    return scale * np.log(series.where(series > 0)).diff()


def derive(panel):
    result = panel.copy()
    result["goods_log_mom_annualized"] = logged_change(panel["pce_goods_index"], 1200)
    result["gscpi_change_sd"] = panel["gscpi"].diff()
    result["wti_log_mom_pct"] = logged_change(panel["wti_usd_per_barrel"], 100)
    result["unemployment_change_pp"] = panel["unemployment_pct"].diff()
    return result.loc[slice(*WINDOWS["Full"])]


def describe(panel):
    records = []
    for period, (start, end) in WINDOWS.items():
        for field in LABELS:
            series = panel.loc[start:end, field]
            records.append({
                "period": period, "series": field, "n": series.count(),
                "missing": series.isna().sum(), "mean": series.mean(), "sd": series.std(),
                "min": series.min(), "p25": series.quantile(.25), "median": series.median(),
                "p75": series.quantile(.75), "max": series.max(), "skew": series.skew(),
                "excess_kurtosis": series.kurt(),
                "lag1_r": series.corr(series.shift()),
            })
    return pd.DataFrame(records)


def longest_contiguous(series):
    if series.empty or not series.notna().any():
        raise ValueError("A contiguous diagnostic sample is required")
    expected = pd.period_range(series.index[0], series.index[-1], freq="M")
    if not series.index.equals(expected):
        raise ValueError("Diagnostic input must retain the monthly calendar, including gaps")
    observed = series.notna()
    runs = observed.ne(observed.shift()).cumsum()
    return max((part for _, part in series[observed].groupby(runs[observed], sort=False)),
               key=len)


def stationarity(panel):
    records = []
    for period in ("Full", "1997–2019"):
        start, end = WINDOWS[period]
        for name in LABELS:
            s = longest_contiguous(panel.loc[start:end, name])
            a_stat, a_p, a_lags, a_n, a_cv, _ = adfuller(s, maxlag=12, regression="c", autolag="AIC")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", InterpolationWarning)
                k_stat, k_p, k_lags, k_cv = kpss(s, regression="c", nlags="auto")
            bound = ""
            if any(isinstance(item.message, InterpolationWarning) for item in caught):
                bound = ">" if k_p == 0.1 else "<"
            records.append({
                "period": period, "series": name, "start": str(s.index[0]), "end": str(s.index[-1]),
                "n": len(s), "adf_nobs": a_n, "adf_lags": a_lags, "adf_stat": a_stat,
                "adf_p": a_p, "adf_critical_5pct": a_cv["5%"], "kpss_lags": k_lags,
                "kpss_stat": k_stat, "kpss_p": k_p, "kpss_p_bound": bound,
                "kpss_critical_5pct": k_cv["5%"],
            })
    return pd.DataFrame(records)


def lead_correlations(panel):
    records = []
    for period, (start, end) in WINDOWS.items():
        window = panel.loc[start:end]
        for target in TARGETS:
            for h in HORIZONS:
                pair = pd.concat([window["gscpi"], window[target].shift(-h)], axis=1).dropna()
                if len(pair) < 2:
                    raise ValueError("Insufficient observations for lead–lag correlation")
                records.append({
                    "period": period, "series": target, "lead_months": h, "n": len(pair),
                    "origin_start": str(pair.index[0]), "origin_end": str(pair.index[-1]),
                    "target_start": str(pair.index[0] + h), "target_end": str(pair.index[-1] + h),
                    "r": pair.iloc[:, 0].corr(pair.iloc[:, 1]),
                })
    return pd.DataFrame(records)


def serial_correlations(panel):
    records = []
    for name in TARGETS + HEADLINE:
        s = longest_contiguous(panel[name])
        a = acf(s, nlags=N_LAGS, fft=False)
        p = pacf(s, nlags=N_LAGS, method="ywm")
        for lag in range(N_LAGS + 1):
            records.append({"series": name, "lag": lag, "acf": a[lag], "pacf": p[lag],
                            "start": str(s.index[0]), "end": str(s.index[-1]), "n": len(s)})
    return pd.DataFrame(records)


def save(fig, directory, name):
    fig.savefig(directory / f"task1_{name}.png", dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)


def display_window(ax, start, end):
    ax.set_xlim(start, end)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_time(panel, directory):
    fig, axes = plt.subplots(5, 2, figsize=(13, 12), layout="constrained", sharex=True)
    dates = panel.index.to_timestamp()
    for ax, name, color in zip(axes.flat, TARGETS + HEADLINE + FEATURES, COLORS * 2):
        ax.axvspan(pd.Timestamp("2020-01-01"), dates[-1], color="#E9F1F4", zorder=0)
        ax.plot(dates, panel[name], color=color, lw=1.25)
        ax.set_title(LABELS[name], loc="left")
        ax.set_ylabel(UNITS[name])
        display_window(ax, dates[0], dates[-1])
    for ax, field in zip(axes[-1], ["unemployment_pct", "unemployment_change_pp"]):
        missing = panel.index[panel[field].isna()].to_timestamp()
        ax.scatter(missing, [0.07] * len(missing), marker="x", color="#A34035",
                   transform=ax.get_xaxis_transform(), label="Missing observation", zorder=4)
        ax.legend(loc="upper right", frameon=False, fontsize=8)
        ax.set_xlabel("Observation month")
    fig.suptitle("U.S. inflation and candidate drivers | 1997-09 to 2026-09\nShading marks January 2020 onward")
    save(fig, directory, "time_series")


def plot_distributions(panel, directory):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.7), layout="constrained")
    for ax, name, color in zip(axes, TARGETS, COLORS):
        s = panel[name].dropna()
        ax.hist(s, bins=26, facecolor=color, alpha=.83, edgecolor="white")
        ax.axvline(s.median(), color="#263B45", lw=1.5, ls="--", label="Median")
        ax.set_title(LABELS[name])
        ax.set_xlabel(UNITS[name])
        ax.set_ylabel("Months")
        ax.legend(frameon=False)
    fig.suptitle("Distribution of goods inflation targets | 1997-09 to 2026-09")
    save(fig, directory, "target_distributions")


def plot_contemporaneous(panel, directory):
    columns = TARGETS[:2] + HEADLINE + FEATURES
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.8), layout="constrained")
    for ax, (period, (start, end)) in zip(axes, WINDOWS.items()):
        sample = panel.loc[start:end, columns].dropna()
        matrix = sample.corr().to_numpy()
        image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
        ticks = [LABELS[name] for name in columns]
        ax.set_xticks(range(len(ticks)), ticks, rotation=47, ha="right", fontsize=8)
        ax.set_yticks(range(len(ticks)), ticks, fontsize=8)
        ax.grid(False)
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(value) >= .59 else "#182E3A")
        ax.set_title(f"{period} | {start} to {end} | n={len(sample)}")
    fig.colorbar(image, ax=axes, shrink=.71, label="Pearson r")
    fig.suptitle("Same-month correlations | complete cases within each period")
    save(fig, directory, "correlations")


def plot_serial(serial, directory):
    fig, axes = plt.subplots(5, 2, figsize=(11.8, 11), layout="constrained", sharex=True)
    for row, name, color in zip(axes, TARGETS + HEADLINE, COLORS):
        data = serial.loc[serial.series.eq(name) & serial.lag.ne(0)]
        for ax, kind in zip(row, ["acf", "pacf"]):
            ax.axhline(0, color="#536978", lw=.9)
            ax.vlines(data.lag, 0, data[kind], color=color, lw=1.4)
            ax.scatter(data.lag, data[kind], s=13, color=color, zorder=4)
            ax.set_title(f"{LABELS[name]} · {kind.upper()}", loc="left")
            ax.set_ylabel("Correlation")
            ax.set_ylim(min(-.25, serial[["acf", "pacf"]].min().min() - .04), 1.0)
    for ax in axes[-1]:
        ax.set_xticks([1, 6, 12, 18, 24])
        ax.set_xlabel("Lag (months)")
    fig.suptitle("Inflation ACF and PACF | 1997-09 to 2026-09 | n=349 per series")
    save(fig, directory, "acf_pacf")


def plot_leads(leads, directory):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), layout="constrained", sharey=True)
    for ax, (period, (start, end)) in zip(axes, WINDOWS.items()):
        for name, color, marker, line in zip(TARGETS, COLORS, ["o", "s", "x"], ["-", "-", ":"]):
            s = leads.loc[leads.period.eq(period) & leads.series.eq(name)]
            ax.plot(s.lead_months, s.r, color=color, marker=marker, ls=line, label=LABELS[name])
        ax.axhline(0, color="#708794", lw=.8)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("Lead h (months)")
        ax.set_title(f"{period}\n{start} to {end} · n={s.n.min()}–{s.n.max()}")
    axes[0].set_ylabel("Pearson r")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("GSCPI at t versus goods inflation at t+h\nBoth months are inside the displayed window")
    save(fig, directory, "gscpi_leads")


def markdown_table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                       "| " + " | ".join(["---"] * len(headers)) + " |",
                       *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def render_report(path, snapshot, summary, tests, leads, serial, panel):
    desc = summary.set_index(["period", "series"])
    unit = tests.set_index(["period", "series"])
    cross = leads.set_index(["period", "series", "lead_months"])
    full = desc.loc["Full"]
    corr = serial.set_index(["series", "lag"])
    best = leads.loc[leads.period.eq("Full") & leads.series.eq(TARGETS[0])
                     & leads.lead_months.gt(0)].sort_values("r", ascending=False).iloc[0]
    pre = cross.loc[("1997–2019", TARGETS[0], best.lead_months), "r"]
    post = cross.loc[("2020 onward", TARGETS[0], best.lead_months), "r"]
    gaps = panel.index[panel.unemployment_pct.isna()]
    diff_gaps = panel.index[panel.unemployment_change_pp.isna()]
    fmt_months = lambda months: "、".join(map(str, months)) or "无"
    rows = [
        "# Task 1 探索性分析与时间序列诊断",
        "",
        "## 样本与变量",
        "",
        "使用 [Task 0 月度快照](../data/monthly.csv)、[审计报告](audit.md)和[数据字典](data_dictionary.md)。"
        "共同区间是 1997-09 至 2026-09，共 349 个日历月。子时期为 1997-09 至 2019-12（268 个月）"
        "与 2020-01 至 2026-09（81 个月）。计算当期变化时，基期取原快照中的上月，再裁切共同区间。"
        "所有极端值均保留。分期以观测月为准，环比和同比使用的基期可在前一时期。时间轴为观测月，实际发布日及当时可见版本需在预测阶段单独核对。",
        "",
        f"失业率水平缺失月为 {fmt_months(gaps)}，一阶百分点变化缺失月为 {fmt_months(diff_gaps)}。"
        "变化率要求连续两个日历月都有值。WTI 对数变化也要求相邻两个价格均为正，"
        f"本样本 WTI 对数变化缺失 {panel.wti_log_mom_pct.isna().sum()} 个月。GSCPI 月差首月缺失，因为源序列从 1997-09 开始。"
        "保留失业率疫情峰值及其变化，描述统计按列使用可用观测，相关矩阵按各子时期取共同完整行。",
        "",
        "## 描述统计与分期比较",
        "",
        "超额峰度字段为 excess_kurtosis，采用 Fisher 定义，正态分布为 0，以 pandas 的无偏估计计算。",
        "",
        markdown_table(["时期", "序列", "单位", "有效 n", "缺失", "均值", "样本标准差", "最小", "中位数", "最大", "偏度", "Fisher 超额峰度", "一阶 Pearson r"],
            ([period, LABELS[name], UNITS[name], int(r.n), int(r.missing),
              *(f"{r[c]:.3f}" for c in ["mean", "sd", "min", "median", "max", "skew", "excess_kurtosis", "lag1_r"])]
             for (period, name), r in desc.iterrows())),
        "",
        *(f"- {LABELS[name]}在两个时期的均值分别为 {desc.loc[('1997–2019', name), 'mean']:.3f}、"
          f"{desc.loc[('2020 onward', name), 'mean']:.3f}，样本标准差分别为 "
          f"{desc.loc[('1997–2019', name), 'sd']:.3f}、{desc.loc[('2020 onward', name), 'sd']:.3f}（{UNITS[name]}）。"
          for name in [TARGETS[0], TARGETS[1], "pce_mom_pct", "gscpi", "wti_usd_per_barrel", "unemployment_pct"]),
        "",
        "[时间序列图](../figures/task1_time_series.png) · [分期同期相关图](../figures/task1_correlations.png)",
        "",
        "## 商品通胀目标比较",
        "",
        markdown_table(["口径", "公式", "标准差", "偏度", "ACF(1)", "ACF(12)", "PACF(1)"],
            ([LABELS[name], formula, f"{full.loc[name, 'sd']:.3f}", f"{full.loc[name, 'skew']:.3f}",
              f"{corr.loc[(name, 1), 'acf']:.3f}", f"{corr.loc[(name, 12), 'acf']:.3f}",
              f"{corr.loc[(name, 1), 'pacf']:.3f}"]
             for name, formula in zip(TARGETS, ["100 × (P[t] / P[t−1] − 1)",
                                              "100 × (P[t] / P[t−12] − 1)",
                                              "1200 × (ln P[t] − ln P[t−1])"]))),
        "",
        f"MoM 衡量一个月的实际百分比变动，适合把预测期限定义为未来第 h 个月的月度通胀。"
        f"年化对数 MoM 衡量同一段价格变动的年率表达，标准差除以 12 后为 "
        f"{full.loc[TARGETS[2], 'sd'] / 12:.3f}%，可与 MoM 的 {full.loc[TARGETS[0], 'sd']:.3f}% 比较。"
        "年化放大单月波动，经济含义是假设该月对数增速持续一年的年率表达。"
        "YoY 体现过去 12 个月累计价格变动，相邻两期共享 11 个月，基期效应与重叠窗口使序列更持久。"
        "建议主目标采用未年化 MoM，年化对数 MoM 作为变换敏感性口径，YoY 作为不同经济目标的对照。",
        "",
        "ACF 使用以样本均值中心化的固定分母估计，PACF 使用 Yule–Walker MLE，均展示 1–24 阶。"
        "这些图用于观察依赖结构，滞后阶数留待滚动验证。",
        "",
        "[目标分布图](../figures/task1_target_distributions.png) · [通胀 ACF/PACF](../figures/task1_acf_pacf.png)",
        "",
        "## ADF 与 KPSS",
        "",
        "ADF 原假设为存在单位根，设常数项且无确定性趋势，在 0–12 阶内按 AIC 选滞后。"
        "KPSS 原假设为水平平稳，设常数项并自动选带宽。以 5% 为参考，表中同时给出统计量、p 值和滞后阶数。"
        "KPSS 的 `<0.01` 和 `>0.10` 表示查表范围边界。"
        "检验以最长连续有效月段为输入。失业率两种形式的主检验都止于 2025-09，"
        "其后的可用观测仍留在描述统计和图中。疫情前结果用于观察样本敏感性。",
        "",
        markdown_table(["时期", "序列", "连续输入月份", "n", "ADF 统计量", "ADF p", "滞后", "ADF 回归 n",
                        "KPSS 统计量", "KPSS p", "带宽"],
                       ([period, LABELS[name], f"{r.start}–{r.end}", r.n, f"{r.adf_stat:.3f}",
                         f"{r.adf_p:.3g}", r.adf_lags, r.adf_nobs, f"{r.kpss_stat:.3f}",
                         f"`{r.kpss_p_bound}{r.kpss_p:.3g}`", r.kpss_lags]
                        for (period, name), r in unit.iterrows())),
        "",
        "ADF 拒绝且 KPSS 未拒绝支持在指定样本和设定下使用水平平稳近似。"
        "若两者均拒绝或均未拒绝，应结合持久性、样本差异与潜在结构变化判断。"
        "WTI 和失业率的变换诊断如下。",
        "",
        *(f"- {LABELS[name]}的相邻月 Pearson r 为 {full.loc[name, 'lag1_r']:.3f}。"
          f"全样本窗口 ADF p={unit.loc[('Full', name), 'adf_p']:.3g}，"
          f"KPSS p{unit.loc[('Full', name), 'kpss_p_bound'] or '='}{unit.loc[('Full', name), 'kpss_p']:.3g}。"
          for name in FEATURES + ["gscpi_change_sd"]),
        "",
        "WTI 水平高度持续，主检验和疫情前检验均为 ADF 未拒绝、KPSS 拒绝。WTI 月度对数变化的两个检验均支持水平平稳近似，宜优先用 100 × Δln(WTI) 刻画能源价格变化。"
        "失业率水平的 ADF 结论随样本而变，百分点月差的检验结论较一致，但其 2020 年 4 月增加 10.4 个百分点，尾部很重。建议优先考虑 ΔU（百分点），把 U 的水平作为稳健性口径。"
        "GSCPI 保留水平作为首选候选，代表相对历史均值的供应链压力偏离。"
        "GSCPI 水平在主样本与疫情前样本中，ADF 和 KPSS 均拒绝各自原假设。月差的两个检验均支持水平平稳近似。"
        "建议将 GSCPI 月变化纳入稳健性规格，并用滚动验证比较水平与变化的信息。",
        "",
        "## GSCPI 领先商品通胀的相关性",
        "",
        "报告 Pearson corr(GSCPI[t], 通胀[t+h])。h=0 为同期，h>0 表示 GSCPI 观察月在前。"
        "每个子时期要求 t 与 t+h 都在该时期内。未来通胀仅用于计算本节统计量，处理后的表保留各变量在自身观测月的取值。",
        "",
        markdown_table(["时期", "商品通胀口径", "h（月）", "起点月 t", "目标月 t+h", "配对 n", "Pearson r"],
            ([period, LABELS[name], h, f"{r.origin_start}–{r.origin_end}",
              f"{r.target_start}–{r.target_end}", r.n, f"{r.r:.3f}"]
             for (period, name, h), r in cross.iterrows())),
        "",
        f"全区间正向期限中，MoM 的最高同期以外相关出现在 h={int(best.lead_months)}，r={best.r:.3f}。"
        f"同一期限在 1997–2019 年为 {pre:.3f}，2020 年起为 {post:.3f}，表明相关强度随时期而变。"
        "短期关联值得跟进，但共同冲击、GSCPI 与通胀的各自持续性、同比窗口重叠和多期限比较都可推高或改变相关系数。"
        "尤其 h<12 时，未来 YoY 的计量窗口仍包含 t 或更早月份的信息。"
        "这些是探索性描述，因果关系和样本外预测增益均待独立检验。",
        "",
        "[GSCPI 交叉相关图](../figures/task1_gscpi_leads.png)",
        "",
        "## 供下一轮审核的预测设计",
        "",
        "建议先考察 h=1、3 个月，h=6 个月作为扩展。以商品 PCE MoM 的可用历史作 AR 基准，"
        "增量组合先加 GSCPI 水平，再加入 WTI 月度对数变化和失业率百分点变化。"
        "用同一预测起点、可用信息集及评价样本比较 AR、ARX 与 Random Forest。对期限 h，目标是 t+h 当月的 MoM。按实际发布日确认各解释变量的可用月份。"
        "整体 PCE 留作描述性对照。滞后阶数、训练窗口及切分日期由研究设计确认。"
        "采用按时间顺序滚动或扩展窗口验证，多期限训练要保证训练标签在预测起点已经可观测。"
        "失业率缺月及其派生缺月需在确定模型样本时明示，按日历月保留时间索引。"
        "当前数据使用修订后快照，疫情冲击、较短的疫情后样本、持久性、"
        "月度发布日期差异与同比重叠仍可能影响检验和后续预测比较。",
        "",
        "## 复现",
        "",
        f"月度输入 SHA-256 为 `{hashlib.sha256(snapshot.read_bytes()).hexdigest()}`。",
        "从项目目录运行以下命令。CSV 结果位于 [data/processed](../data/processed)。",
        "",
        "```sh", "python3 -m pip install -r code/requirements-task1.txt",
        "python3 code/task1_eda.py", "python3 -m unittest discover -s code -v", "```",
        "",
        "方法文档。[ADF](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html)"
        " · [KPSS](https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html)",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project directory containing Task 0 inputs")
    parser.add_argument("--output", type=Path, help="optional separate project-shaped output directory")
    args = parser.parse_args()
    output = args.output or args.root
    snapshot = args.root / "data" / "monthly.csv"
    panel = derive(load_panel(snapshot))
    summary = describe(panel)
    tests = stationarity(panel)
    leads = lead_correlations(panel)
    serial = serial_correlations(panel)
    processed = output / "data" / "processed"
    figures = output / "figures"
    reports = output / "reports"
    for directory in (processed, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)
    for data, filename, index in [
        (panel, "monthly_task1.csv", True), (summary, "task1_descriptive.csv", False),
        (tests, "task1_stationarity.csv", False), (leads, "task1_gscpi_leads.csv", False),
        (serial, "task1_acf_pacf.csv", False),
    ]:
        data.to_csv(processed / filename, index=index, float_format="%.12g")
    plt.rcParams.update({
        "figure.facecolor": "#F8FBFC", "axes.facecolor": "#FFFFFF", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": "#DBE7EC",
        "text.color": "#16323D", "axes.labelcolor": "#16323D",
        "xtick.color": "#405B66", "ytick.color": "#405B66",
        "figure.titlesize": 15, "font.size": 10,
    })
    plot_time(panel, figures)
    plot_distributions(panel, figures)
    plot_contemporaneous(panel, figures)
    plot_serial(serial, figures)
    plot_leads(leads, figures)
    render_report(reports / "task1_eda.md", snapshot, summary, tests, leads, serial, panel)
    print(f"Task 1: {len(panel)} months, {len(tests)} ADF/KPSS rows, {len(leads)} lead correlations")


if __name__ == "__main__":
    main()
