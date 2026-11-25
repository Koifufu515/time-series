"""Download official monthly series or reproduce Task 0 from the saved snapshots."""
import argparse
import hashlib
from io import BytesIO
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NYFED_PAGE = "https://www.newyorkfed.org/research/policy/gscpi"
NYFED_CSV = (
    "https://www.newyorkfed.org/medialibrary/research/interactives/"
    "data/gscpi/gscpi_interactive_data.csv"
)
FRED = "https://fred.stlouisfed.org"
GSCPI_FILE = "gscpi_interactive_data.csv"
SOURCES = {
    "gscpi": (GSCPI_FILE, NYFED_CSV, NYFED_PAGE, "标准差", "官方发布口径"),
    "pce_goods_index": ("DGDSRG3M086SBEA.csv", None, None, "2017=100", "季调"),
    "pce_index": ("PCEPI.csv", None, None, "2017=100", "季调"),
    "wti_usd_per_barrel": ("MCOILWTICO.csv", None, None, "美元/桶", "未季调"),
    "unemployment_pct": ("UNRATE.csv", None, None, "%", "季调"),
}
NAMES = {
    "gscpi": "GSCPI",
    "pce_goods_index": "PCE goods price index",
    "pce_index": "Headline PCE price index",
    "wti_usd_per_barrel": "WTI",
    "unemployment_pct": "Unemployment",
}
PRICE_COLUMNS = {"pce_goods": "pce_goods_index", "pce": "pce_index"}
HORIZONS = {"mom": 1, "yoy": 12}


def source_urls(filename, download, page):
    if download is not None:
        return download, page
    code = Path(filename).stem
    return f"{FRED}/graph/fredgraph.csv?id={code}", f"{FRED}/series/{code}"


def parse_series(raw_bytes, filename, name):
    frame = pd.read_csv(BytesIO(raw_bytes), na_values=["."], encoding="utf-8-sig")
    vintage = ""
    if filename == GSCPI_FILE:
        vintages = [c for c in frame.columns if c != "Date"]
        latest = max(vintages, key=int)
        vintage = pd.to_datetime(int(latest), unit="D", origin="1899-12-30").date().isoformat()
        dates = pd.to_datetime(frame["Date"], format="%d-%b-%Y")
        values = frame[latest]
    else:
        date_col = "observation_date" if "observation_date" in frame else "DATE"
        dates = pd.to_datetime(frame[date_col], format="%Y-%m-%d")
        values = frame[Path(filename).stem]
    series = pd.Series(
        pd.to_numeric(values).to_numpy(dtype=float),
        index=pd.PeriodIndex(dates, freq="M", name="month"),
        name=name,
    ).sort_index()
    if series.index.hasnans or not np.isfinite(series.dropna()).all() or series.dropna().empty:
        raise ValueError(f"{filename}: invalid dates, non-finite values or no data")
    duplicates = series.index[series.index.duplicated(keep=False)].unique()
    if len(duplicates):
        raise ValueError(f"{filename}: duplicate observation months: {list(map(str, duplicates))}")
    return series, vintage


def load_sources(raw_dir, download):
    series, provenance, snapshots = {}, {}, {}
    for name, (filename, endpoint, page, unit, adjustment) in SOURCES.items():
        url, page = source_urls(filename, endpoint, page)
        if download:
            raw_bytes = subprocess.run(
                ["curl", "--fail", "--silent", "--show-error", "--location",
                 "--max-time", "45", url],
                check=True, stdout=subprocess.PIPE,
            ).stdout
        else:
            raw_bytes = (raw_dir / filename).read_bytes()
        series[name], vintage = parse_series(raw_bytes, filename, name)
        provenance[name] = (filename, url, page, unit, adjustment, vintage,
                            hashlib.sha256(raw_bytes).hexdigest())
        snapshots[filename] = raw_bytes
    return series, provenance, snapshots


def build_panel(series):
    levels = pd.concat(series.values(), axis=1).sort_index()
    panel = levels.reindex(pd.period_range(
        levels.index.min(), levels.index.max(), freq="M", name="month"
    ))
    for prefix, name in PRICE_COLUMNS.items():
        price = panel[name]
        if price.dropna().le(0).any():
            raise ValueError(f"{name}: a PCE index value is nonpositive")
        for suffix, months in HORIZONS.items():
            panel[f"{prefix}_{suffix}_pct"] = 100 * (price / price.shift(months) - 1)
    return panel


def span(index):
    return f"{index.min()} 至 {index.max()}"


def month_list(index):
    return "、".join(map(str, index)) or "无"


def write_audit(path, panel, series, provenance, common, screens):
    rows = [
        "# Task 0 数据审计", "",
        "本轮使用随项目保存的纽约联储与 FRED 官方文件快照。观察月按 YYYY-MM 对齐，自最早来源月份至最晚来源月份保留完整日历。", "",
        "## 数据覆盖与重复检查", "",
        f"交付 CSV 覆盖 {span(panel.index)}，共 {len(panel)} 个月。五个水平序列的覆盖交集为 "
        f"{span(common.index)}，共 {len(common)} 个月，其中 {len(common.dropna(subset=list(series)))} 个月五项齐全。"
        "这个交集用于描述性检查，正式预测样本由研究设计决定。", "",
        "| 序列 | 原文件月份范围 | 观测行 | 有效值 | 显式空值 | 范围内漏行 | 重复月 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in series.items():
        months = pd.period_range(s.index.min(), s.index.max(), freq="M")
        rows.append(f"| {name} | {span(s.index)} | {len(s)} | {s.count()} | "
                    f"{s.isna().sum()} | {len(months.difference(s.index))} | {s.index.duplicated().sum()} |")
    rows += ["", "## 缺失与转换检查", ""]
    for name, s in series.items():
        months = pd.period_range(s.index.min(), s.index.max(), freq="M")
        missing = months.difference(s.dropna().index)
        rows.append(f"- {name} 覆盖范围内缺失月份为 {month_list(missing)}。"
                    f"全表另有 {len(panel) - len(months)} 个月在其原文件覆盖范围之外。")
    if pd.Period("2025-10", "M") in series["unemployment_pct"].index and pd.isna(
        series["unemployment_pct"].loc["2025-10"]
    ):
        rows += ["", "2025 年 10 月联邦政府停摆期间，CPS 家庭调查未采集数据，BLS 没有发布当月估计。"
                 "该月在 CSV 中保留，失业率留空。"
                 " [BLS 对 CPS 的说明](https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm)"]
    rows += ["", "通胀候选列在完整月度日历上计算。基期或当期缺值时，候选值留空。CSV 中空白代表缺失，原始月度数据直接对齐。", "",
             "| 候选列 | 首个有效月份 | 最后有效月份 | 全表空值 | PCE 来源覆盖内空值 |",
             "|---|---|---|---:|---:|"]
    for prefix, name in PRICE_COLUMNS.items():
        for suffix in HORIZONS:
            column = f"{prefix}_{suffix}_pct"
            s = panel[column]
            inside = s.loc[series[name].index.min():series[name].index.max()]
            rows.append(f"| {column} | {s.first_valid_index()} | {s.last_valid_index()} | "
                        f"{s.isna().sum()} | {inside.isna().sum()} |")
    rows += ["", "## 单位与异常值检查", "",
             "两项 PCE 指数须大于 0。失业率的定义域为 0–100%。WTI 以美元/桶计，GSCPI 以标准差计，"
             "两者的正负值均按来源保留。下表在共同区间内按各列可用观测筛查。"
             "以低于 Q1 − 3×IQR 或高于 Q3 + 3×IQR 为阈值，标记后保留原值。", "",
             f"全表失业率超出定义域的月数为 {((panel.unemployment_pct < 0) | (panel.unemployment_pct > 100)).sum()}。",
             "", "| 序列 | 最小值及月份 | 最大值及月份 | 标记数 | 标记月份 |",
             "|---|---|---|---:|---|"]
    for name, s in common.items():
        observed = s.dropna()
        q1, q3 = observed.quantile([0.25, 0.75])
        iqr = q3 - q1
        flags = observed[(observed < q1 - 3 * iqr) | (observed > q3 + 3 * iqr)]
        rows.append(f"| {name} | {observed.min():.4g} ({observed.idxmin()}) | "
                    f"{observed.max():.4g} ({observed.idxmax()}) | {len(flags)} | {month_list(flags.index)} |")
    rows += ["", "## 相关性与预测时点", "",
             "图中使用共同区间内的同期 Pearson 相关系数。两个面板使用同一批完整观测月份，"
             "水平值的趋势和同比窗口的重叠均可能影响系数，图仅供初步描述。", ""]
    for label, sample in screens.items():
        goods = sample.columns[0]
        rows.append(f"- {label}使用 {len(sample)} 个月（{span(sample.index)}）。"
                    f"商品通胀与 GSCPI 的相关系数为 {sample[goods].corr(sample['gscpi']):.3f}。")
    rows += ["", "CSV 中月份是观测期。预测时应另行按预测起点核对各序列发布日期和可用版本。"
             "FRED 文件反映抓取时的历史修订值，GSCPI 取下载矩阵中的最新版本，"
             "原矩阵保留了可获得的历史版本。", "",
             "## 原始文件与校验", "",
             "| 文件 | 官方下载 | GSCPI 版本列标签（月初） | SHA-256 |",
             "|---|---|---|---|"]
    for filename, url, _, _, _, vintage, sha in provenance.values():
        rows.append(f"| {filename} | [下载]({url}) | {vintage or '—'} | `{sha}` |")
    rows += ["", "## 复现", "",
             "在项目目录运行以下命令。默认读取随附快照。安装 curl 后可运行 "
             "`python3 code/prepare_data.py --download` 刷新官方文件，刷新可能纳入后续发布和历史修订。", "",
             "```sh", "python3 -m venv .venv", "source .venv/bin/activate",
             "python3 -m pip install -r code/requirements.txt", "python3 code/prepare_data.py",
             "python3 -m unittest discover -s code -v", "```"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_dictionary(path, provenance):
    rows = ["# 数据字典", "", "| 字段 | 含义与单位 | 来源或公式 |", "|---|---|---|",
            "| month | 观测月份，YYYY-MM | 统一使用日历月 |"]
    descriptions = [
        "全球供应链压力指数，相对历史均值的标准差",
        "美国 PCE 商品链式价格指数",
        "美国整体 PCE 链式价格指数",
        "库欣 WTI 现货月度价格",
        "美国 U-3 失业率",
    ]
    for (name, (_, _, page, unit, adjustment, _, _)), description in zip(provenance.items(), descriptions):
        rows.append(f"| {name} | {description}（{unit}，{adjustment}） | [官方来源]({page}) |")
    for prefix, name in PRICE_COLUMNS.items():
        for suffix, months in HORIZONS.items():
            label = "环比" if months == 1 else "同比"
            rows.append(f"| {prefix}_{suffix}_pct | {NAMES[name]} {label}候选值，%，非年化 | "
                        f"100 × ({name}[t] / {name}[t−{months}] − 1) |")
    rows += ["", "商品 PCE 是拟研究的预测对象。环比、同比均供口径审核。"
             "解释变量保留 GSCPI、WTI 和失业率水平口径。整体 PCE 一并交付供描述性比较。"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def make_plots(common, figures):
    colors = ["#0e6681", "#d17d35", "#5e4789", "#3b7861", "#794034"]
    dates = common.index.to_timestamp()
    fig, axes = plt.subplots(5, 1, figsize=(10, 9), sharex=True, layout="constrained")
    for ax, suffix, label in zip(axes, HORIZONS, ["Monthly inflation", "12-month inflation"]):
        ax.plot(dates, common[f"pce_goods_{suffix}_pct"], label="Goods PCE", color=colors[0])
        ax.plot(dates, common[f"pce_{suffix}_pct"], label="Headline PCE", color=colors[1], alpha=0.8)
        ax.set_ylabel(f"{label}\n(%)")
        ax.axhline(0, color="0.75", lw=0.7)
        ax.legend(loc="upper left", ncols=2, frameon=False)
    for ax, name, unit, color in zip(
        axes[2:], ["gscpi", "wti_usd_per_barrel", "unemployment_pct"],
        ["GSCPI\n(std. dev.)", "WTI\n(USD/barrel)", "Unemployment\n(%)"], colors[2:]
    ):
        ax.plot(dates, common[name], color=color)
        ax.set_ylabel(unit)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlim(dates[0], dates[-1])
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(4))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Observation month")
    fig.suptitle("U.S. inflation and supply-chain indicators")
    fig.savefig(figures / "time_series.png", dpi=170)
    plt.close(fig)

    levels = ["gscpi", "wti_usd_per_barrel", "unemployment_pct"]
    candidates = [[f"pce_goods_{suffix}_pct", f"pce_{suffix}_pct", *levels]
                  for suffix in HORIZONS]
    shared = common.dropna(subset=list(dict.fromkeys(candidates[0] + candidates[1])))
    if len(shared) < 2:
        raise ValueError("Insufficient complete months for correlation plots")
    screens = dict(zip(["环比", "同比"], [shared[cols] for cols in candidates]))
    ticks = ["Goods PCE", "Headline PCE", "GSCPI", "WTI", "Unemployment"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, label, sample in zip(axes, ["Monthly inflation", "12-month inflation"], screens.values()):
        corr = sample.corr()
        image = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
        for i, row in enumerate(corr.to_numpy()):
            for j, value in enumerate(row):
                label_text = f"{value:.2f}" if pd.notna(value) else "—"
                ax.text(j, i, label_text, ha="center", va="center",
                        color="white" if abs(value) >= 0.55 else "#222")
        ax.set_xticks(range(len(ticks)), ticks, rotation=32, ha="right")
        ax.set_yticks(range(len(ticks)), ticks)
        ax.set_title(f"{label} · n={len(sample)}")
    fig.colorbar(image, ax=axes, shrink=0.83, label="Pearson r")
    fig.suptitle(f"Same-month correlations | {shared.index.min()} to {shared.index.max()}")
    fig.savefig(figures / "correlations.png", dpi=170)
    plt.close(fig)
    return screens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="location of the data and outputs")
    parser.add_argument("--download", action="store_true", help="refresh all five official snapshots")
    args = parser.parse_args()
    raw_dir = args.root / "data" / "raw"
    series, provenance, snapshots = load_sources(raw_dir, args.download)
    panel = build_panel(series)
    start = max(s.first_valid_index() for s in series.values())
    end = min(s.last_valid_index() for s in series.values())
    common = panel.loc[start:end]
    figures = args.root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    screens = make_plots(common, figures)
    raw_dir.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(raw_dir.parent / "monthly.csv", float_format="%.12g")
    reports = args.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_dictionary(reports / "data_dictionary.md", provenance)
    write_audit(reports / "audit.md", panel, series, provenance, common, screens)
    if args.download:
        raw_dir.mkdir(parents=True, exist_ok=True)
        for filename, raw_bytes in snapshots.items():
            (raw_dir / filename).write_bytes(raw_bytes)
    print(f"{len(panel)} months exported, {len(common)} in the common window")


if __name__ == "__main__":
    main()
