"""Run Task 2 direct, expanding-window forecasts from the existing project snapshots."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
TARGET = "pce_goods_mom_pct"
LAGS = tuple(f"inflation_lag_{i}" for i in range(6))
CONTROLS = ("wti_log_mom_pct", "unemployment_change_pp")
GSCPI = "gscpi"
HORIZONS = (1, 3, 6)
INITIAL_N = 180
BREAK = pd.Period("2020-01", freq="M")
MODELS = ("AR", "ARX-Base", "ARX-GSCPI", "Random Forest")
PERIODS = ("Full", "Pre-COVID", "Post-COVID")
COMPARISONS = ((MODELS[2], MODELS[1]), (MODELS[3], MODELS[2]), (MODELS[1], MODELS[0]))
COLORS = dict(zip(MODELS, ("#23658A", "#9A491F", "#187059", "#724B98")))
RF_GRID = {"max_depth": [2, 3, None], "min_samples_leaf": [5, 10],
           "max_features": [0.7, 1.0]}
RF_FIXED = {"n_estimators": 300, "random_state": 42, "n_jobs": 1}
SAMPLE = ("1997-09", "2026-09")


def read_months(path):
    frame = pd.read_csv(path).set_index("month")
    frame.index = pd.PeriodIndex(frame.index, freq="M", name="month")
    calendar = pd.period_range(frame.index[0], frame.index[-1], freq="M", name="month")
    if not frame.index.equals(calendar):
        raise ValueError(f"Expected a unique, ordered and complete calendar in {path}")
    return frame


def load_inputs(root):
    monthly = read_months(root / "data" / "monthly.csv")
    task1 = read_months(root / "data" / "processed" / "monthly_task1.csv")
    calendar = pd.period_range(*SAMPLE, freq="M", name="month")
    if not task1.index.equals(calendar):
        raise ValueError("Task 1 common sample has changed")
    for field in (TARGET, GSCPI):
        pd.testing.assert_series_equal(task1[field], monthly.loc[calendar, field], check_exact=True)
    return monthly, task1


def direct_frame(monthly, task1, horizon):
    """Put contemporaneous and lagged features at s and the separate response at s+h."""
    if horizon < 1:
        raise ValueError("Expected a strictly positive forecast horizon")
    features = task1.loc[:, [*CONTROLS, GSCPI]].copy()
    for i, lag in enumerate(LAGS):
        features[lag] = monthly[TARGET].shift(i).reindex(task1.index)
    # Restrict scored outcomes to the common sample, including at the right boundary.
    features["target_month"] = features.index + horizon
    features["actual"] = task1[TARGET].shift(-horizon)
    feature_cols = [*LAGS, *CONTROLS, GSCPI]
    features["predictors_ready"] = np.isfinite(features[feature_cols]).all(axis=1)
    return features


def common_rows(frame):
    return frame.loc[frame.predictors_ready & np.isfinite(frame.actual)].copy()


def training_at(rows, origin):
    return rows.loc[rows.target_month.le(origin)]


def ols_fit(train, columns):
    x = np.column_stack((np.ones(len(train)), train.loc[:, columns].to_numpy()))
    y = train.actual.to_numpy()
    params, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError("Rank-deficient OLS specification")
    residual = y - np.sum(x * params, axis=1)
    rss = np.sum(residual ** 2)
    return params, rss


def select_order(initial):
    scores = []
    for p in range(1, len(LAGS) + 1):
        _, rss = ols_fit(initial, list(LAGS[:p]))
        n = len(initial)
        # Gaussian ML AIC. Count the intercept, p slopes and residual variance.
        aic = n * (np.log(2 * np.pi) + 1 + np.log(rss / n)) + 2 * (p + 2)
        scores.append((p, aic))
    return min(scores, key=lambda item: (item[1], item[0]))[0], scores


def rf_splits(initial, horizon):
    if not initial.index.equals(pd.period_range(initial.index[0], initial.index[-1], freq="M",
                                               name=initial.index.name)):
        raise ValueError("TimeSeriesSplit requires consecutive initial feature months")
    splits = list(TimeSeriesSplit(n_splits=5, gap=horizon).split(initial))
    for train, valid in splits:
        if initial.iloc[train].target_month.max() >= initial.index[valid[0]]:
            raise ValueError("Training labels overlap the validation information month")
    return splits


def select_forest(initial, columns, horizon):
    folds = rf_splits(initial, horizon)
    grid = GridSearchCV(RandomForestRegressor(**RF_FIXED), RF_GRID, cv=folds,
                        scoring="neg_mean_squared_error", refit=False, error_score="raise")
    grid.fit(initial.loc[:, columns], initial.actual)
    results = grid.cv_results_
    scores = list(zip(results["params"], -results["mean_test_score"], results["rank_test_score"]))
    return grid.best_params_, scores, folds


def run_horizon(monthly, task1, horizon):
    frame = direct_frame(monthly, task1, horizon)
    rows = common_rows(frame)
    initial = rows.iloc[:INITIAL_N]
    if len(initial) != INITIAL_N:
        raise ValueError(f"h={horizon}: fewer than {INITIAL_N} complete training rows")
    first_origin = initial.target_month.iloc[-1]
    pd.testing.assert_frame_equal(training_at(rows, first_origin), initial)
    p, aic_grid = select_order(initial)
    ar = list(LAGS[:p])
    base = [*ar, *CONTROLS]
    extended = [*base, GSCPI]
    forest_params, forest_grid, folds = select_forest(initial, extended, horizon)
    forest = RandomForestRegressor(**RF_FIXED, **forest_params)

    test_calendar = frame.loc[first_origin:task1.index[-1] - horizon]
    test = common_rows(test_calendar)
    skipped = test_calendar.index.difference(test.index)
    forecasts = []
    for origin, current in test.iterrows():
        train = training_at(rows, origin)
        if train.target_month.max() > origin:
            raise RuntimeError("Future training response")
        predict = []
        for columns in (ar, base, extended):
            coef, _ = ols_fit(train, columns)
            predict.append(float(np.sum(np.r_[1.0, current[columns].to_numpy(dtype=float)] * coef)))
        forest.fit(train.loc[:, extended], train.actual)
        predict.append(float(forest.predict(test.loc[[origin], extended])[0]))
        for model, value in zip(MODELS, predict):
            forecasts.append({
                "forecast_origin": origin, "target_month": current.target_month,
                "horizon": horizon, "actual": float(current.actual),
                "prediction": value, "model": model, "train_n": len(train),
                "train_last_target_month": train.target_month.iloc[-1],
            })
    config = {
        "h": horizon, "p": p, "aic": aic_grid, "rf": forest_params,
        "grid": forest_grid, "folds": folds, "initial": initial,
        "test": test.index, "skipped": skipped, "features": extended,
    }
    return pd.DataFrame(forecasts), config


def subperiod(months):
    return np.where(months < BREAK, "Pre-COVID", "Post-COVID")


def calculate_metrics(forecasts):
    forecasts = forecasts.assign(error=forecasts.actual - forecasts.prediction,
                                 period=subperiod(forecasts.target_month))
    records = []
    for (h, model), sample in forecasts.groupby(["horizon", "model"], sort=False):
        for period in PERIODS:
            subset = sample if period == "Full" else sample.loc[sample.period.eq(period)]
            e = subset.error.to_numpy()
            records.append({"horizon": h, "period": period, "model": model, "n": len(e),
                            "first_target": subset.target_month.min(),
                            "last_target": subset.target_month.max(),
                            "rmse": np.sqrt(np.mean(e ** 2)), "mae": np.mean(np.abs(e)),
                            "mean_error": np.mean(e)})
    return pd.DataFrame(records)


def dm_statistic(loss_diff, horizon):
    """Calendar-lag Bartlett HAC for observed loss differences on matched dates."""
    if (not loss_diff.index.is_unique or not loss_diff.index.is_monotonic_increasing
            or not np.isfinite(loss_diff).all() or len(loss_diff) <= horizon):
        raise ValueError("Expected ordered finite matched monthly loss differences")
    n = len(loss_diff)
    mean = loss_diff.mean()
    centered = loss_diff - mean
    lrv = np.dot(centered, centered) / n
    for lag in range(1, horizon):
        previous = centered.reindex(centered.index - lag).to_numpy()
        matched = np.isfinite(previous)
        gamma = np.dot(centered.to_numpy()[matched], previous[matched]) / n
        lrv += 2 * (1 - lag / horizon) * gamma
    if lrv <= 0:
        return np.nan, np.nan
    statistic = mean / np.sqrt(lrv / n)
    return statistic, 2 * norm.sf(abs(statistic))


def dm_results(forecasts):
    rows = []
    for h in HORIZONS:
        sample = forecasts.loc[forecasts.horizon.eq(h)].assign(
            sqerror=lambda x: (x.actual - x.prediction) ** 2)
        paired = sample.pivot(index="target_month", columns="model", values="sqerror")
        if paired.isna().any().any():
            raise ValueError("All comparison models must use identical target months")
        for a, b in COMPARISONS:
            diff = paired[a] - paired[b]
            stat, p = dm_statistic(diff, h)
            mean = diff.mean()
            rows.append({"horizon": h, "model_a": a, "model_b": b, "n": len(diff),
                         "first_target": diff.index[0], "last_target": diff.index[-1],
                         "hac_lag": h - 1, "mean_loss_diff_a_minus_b": mean,
                         "dm_stat": stat, "p_value": p,
                         "lower_squared_loss_model": a if mean < 0 else b if mean > 0 else "Tie"})
    return pd.DataFrame(rows)


def table(columns, rows):
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "| " + " | ".join(["---"] * len(columns)) + " |",
                      *["| " + " | ".join(map(str, r)) + " |" for r in rows]])


def write_config(path, configs, forecasts, root):
    lines = [
        "# Task 2 模型参数与样本", "",
        "三种线性模型在各预测起点用 OLS 重新估计。Random Forest 固定 "
        "`n_estimators=300, random_state=42, n_jobs=1`，每月重新拟合。"
        "各期限以相同的 180 行比较 AIC，并沿用选出的 AR 阶数。"
        "AIC 计入截距、p 个斜率和残差方差。RF 以五折 `TimeSeriesSplit(gap=h)` 选择最小的平均验证 MSE。"
        "初始窗的起点月连续，CV 折以起点月判断，折内最后训练标签月严格早于首个验证起点。", "",
    ]
    for c in configs:
        h = c["h"]
        initial = c["initial"]
        f = forecasts.loc[forecasts.horizon.eq(h) & forecasts.model.eq("AR")]
        lines += [
            f"## h={h}", "",
            f"- 初始训练起点为 {initial.index[0]} 至 {initial.index[-1]}，"
            f"标签月份为 {initial.target_month.iloc[0]} 至 {initial.target_month.iloc[-1]}，共 {len(initial)} 行。",
            f"- 首末测试起点为 {c['test'][0]} 至 {c['test'][-1]}，"
            f"目标月为 {f.target_month.min()} 至 {f.target_month.max()}，共 {len(f)} 行。",
            f"- 首末拟合使用 {f.train_n.iloc[0]}、{f.train_n.iloc[-1]} 行。"
            f"评价期跳过的起点月为 {', '.join(map(str, c['skipped'])) or '无'}。",
            f"- AR 固定阶数为 {c['p']}。ARX-GSCPI 与 RF 共用字段 `{', '.join(c['features'])}`。",
            f"- RF 固定调参结果为 `{c['rf']}`。", "",
            table(["p", "初始 AR AIC"], ([p, f"{aic:.6f}"] for p, aic in c["aic"])), "",
            table(["折", "训练样本", "最新训练标签", "验证起点月", "验证样本"],
                  ([i, len(tr), initial.target_month.iloc[tr[-1]],
                    f"{initial.index[va[0]]} 至 {initial.index[va[-1]]}", len(va)]
                   for i, (tr, va) in enumerate(c["folds"], 1))), "",
            table(["max_depth", "min_samples_leaf", "max_features", "验证 MSE", "排名"],
                  ([params["max_depth"], params["min_samples_leaf"], params["max_features"],
                    f"{mse:.6f}", rank] for params, mse, rank in c["grid"])), "",
        ]
    lines += ["## 输入快照", ""]
    for relative in ("data/monthly.csv", "data/processed/monthly_task1.csv"):
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        lines.append(f"- `{relative}` SHA-256 `{digest}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def describe_pair(metrics, dm, a, b):
    pieces = []
    for h in HORIZONS:
        s = metrics.loc[metrics.horizon.eq(h) & metrics.period.eq("Full")].set_index("model")
        test = dm.loc[dm.horizon.eq(h) & dm.model_a.eq(a) & dm.model_b.eq(b)].iloc[0]
        change = 100 * (s.loc[a, "rmse"] / s.loc[b, "rmse"] - 1)
        mae = 100 * (s.loc[a, "mae"] / s.loc[b, "mae"] - 1)
        pieces.append(f"h={h} 的 RMSE 变化 {change:+.1f}%，MAE 变化 {mae:+.1f}%，DM p={test.p_value:.3f}")
    return "。".join(pieces) + "。"


def write_report(path, configs, forecasts, metrics, dm):
    counts = []
    winners = []
    for h in HORIZONS:
        f = forecasts.loc[forecasts.horizon.eq(h) & forecasts.model.eq("AR")]
        c = next(c for c in configs if c["h"] == h)
        counts.append([h, c["p"], f.train_n.iloc[0], f.train_n.iloc[-1],
                       f"{f.forecast_origin.min()} 至 {f.forecast_origin.max()}",
                       f"{f.target_month.min()} 至 {f.target_month.max()}",
                       len(f), sum(f.target_month.lt(BREAK)), sum(f.target_month.ge(BREAK))])
        for period in PERIODS:
            s = metrics.loc[metrics.horizon.eq(h) & metrics.period.eq(period)]
            winners.append([h, period, " → ".join(s.sort_values("rmse").model),
                            " → ".join(s.sort_values("mae").model)])
    period_winners = {h: {part: metrics.loc[metrics.horizon.eq(h) & metrics.period.eq(part)]
                         .sort_values("rmse").model.iloc[0] for part in PERIODS[1:]}
                      for h in HORIZONS}
    by_period = metrics.pivot(index=["horizon", "model"], columns="period", values=["rmse", "mae"])
    both_increase = all((by_period[(error, "Post-COVID")] > by_period[(error, "Pre-COVID")]).all()
                        for error in ("rmse", "mae"))
    subperiod_summary = "。".join(
        f"h={h} 的最低 RMSE 模型由疫情前 {period_winners[h]['Pre-COVID']} "
        f"变为 2020 年起 {period_winners[h]['Post-COVID']}"
        if period_winners[h]["Pre-COVID"] != period_winners[h]["Post-COVID"]
        else f"h={h} 两期的最低 RMSE 模型均为 {period_winners[h]['Pre-COVID']}"
        for h in HORIZONS)
    sections = [
        "# Task 2 商品通胀样本外预测", "",
        "## 预测模型", "",
        r"目标为商品 PCE 月环比 $\pi_t=100(P_t/P_{t-1}-1)$，单位为百分比。"
        "三个期限独立估计，直接预测 t+h 当月的环比。h=1、3 为主结果，h=6 为补充。", "",
        r"- AR：$\hat{\pi}_{t+h}=\hat\alpha_h+\sum_{i=0}^{p_h-1}\hat\phi_{h,i}\pi_{t-i}$。",
        r"- ARX-Base：在 AR 的右侧加 $\hat\beta_{h,1} W_t+\hat\beta_{h,2}\Delta U_t$。",
        r"- ARX-GSCPI：在 ARX-Base 的右侧加 $\hat\beta_{h,3}G_t$。",
        r"- Random Forest：$\hat{\pi}_{t+h}=300^{-1}\sum_{b=1}^{300}T_{h,b}(X_t)$，"
        r"$X_t=(\pi_t,\ldots,\pi_{t-p_h+1},W_t,\Delta U_t,G_t)$，与 ARX-GSCPI 同列同样本。", "",
        r"其中 $W_t=100\Delta\ln(\mathrm{WTI}_t)$，$\Delta U_t$ 为失业率的月度百分点变化，"
        r"$G_t$ 为 GSCPI 水平值。初始 AIC、RF 网格与逐期限配置见 [模型配置](task2_model_config.md)。", "",
        "## 训练与评价样本", "",
        "**这是采用当前修订数据快照的伪样本外预测。** t 是可用信息的观测月份索引。实验设定 t 表示当月各输入及标签已经公布的信息截面。"
        "代码按观测月份约束信息集，未重建历史日历发布日期或首次发布版本。实际执行预测时，部分 t 月数据在 t+1 月才公布，"
        "因此 h=1 的实际预测提前量可能短于一个整月。", "",
        "使用 1997-09 至 2026-09 的共同月度样本。只为构造初期 AR 滞后，从 Task 0 读取共同样本前的已知商品通胀。"
        "四个模型在每个 h 使用同一训练行和评价日期。初始窗用 180 行完整样本，并在首个起点完成 AIC 和 RF 调参。"
        "时间索引始终保留日历月。每个测试起点以 s+h≤t 截断训练标签，使用可用行重新估计；在特征缺失起点跳过全部四模型。"
        "2025-10 和 2025-11 的失业率月差缺失，造成各期限对应的目标月缺口。", "",
        table(["h", "p", "初始训练 n", "末次训练 n", "测试起点", "测试目标月", "全期 n", "疫情前 n", "2020 起 n"], counts), "",
        "## 预测表现", "",
        "误差定义为实际值减预测值，平均误差为正表示平均低估。RMSE、MAE 和平均误差均以百分点计。"
        "按目标月划分子时期，2020-01 起归入 Post-COVID。", "",
        table(["h", "时期", "模型", "n", "RMSE", "MAE", "平均误差"],
              ([r.horizon, r.period, r.model, r.n, f"{r.rmse:.4f}", f"{r.mae:.4f}", f"{r.mean_error:+.4f}"]
               for r in metrics.itertuples(index=False))), "",
        table(["h", "时期", "RMSE 从低到高", "MAE 从低到高"], winners), "",
        "[逐月预测](../figures/task2_forecast_comparison.png) · [全期误差](../figures/task2_model_performance.png)"
        " · [分期误差](../figures/task2_subperiod_comparison.png)", "",
        "## Diebold–Mariano 检验", "",
        "原假设为两个模型的期望平方预测损失相等。差值为模型 A 的平方误差减模型 B 的平方误差，"
        "负值指向 A，正值指向 B。报告全评价期、双侧正态近似 p 值。"
        "Newey–West/Bartlett HAC 使用 h−1 阶。自协方差仅配对恰好相隔指定月数的有效误差，"
        "全都除以匹配样本量 n，保留月历缺口。方差为零时统计量与 p 值留空。", "",
        table(["h", "模型 A", "模型 B", "n", "HAC 阶数", "平均损失差", "DM", "p", "损失较低方向"],
              ([r.horizon, r.model_a, r.model_b, r.n, r.hac_lag, f"{r.mean_loss_diff_a_minus_b:+.5f}",
                f"{r.dm_stat:+.3f}", f"{r.p_value:.4f}", r.lower_squared_loss_model]
               for r in dm.itertuples(index=False))), "",
        "ARX-GSCPI 对 ARX-Base 及 ARX-Base 对 AR 是嵌套比较。常规 DM 的正态近似在此类比较中具有局限，"
        "p 值应连同误差幅度和方向解读。检验未做跨期限或跨比较的多重校正。"
        "h−1 带宽针对多步重叠，对其他来源的持久性不作额外调整。"
        " [Clark 与 West 的嵌套模型检验讨论](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf)", "",
        "## 研究问题", "",
        f"1. **GSCPI 的增量价值。** 以 ARX-Base 为参照，{describe_pair(metrics, dm, 'ARX-GSCPI', 'ARX-Base')}"
        "按未校正的 5% 门槛，仅 1 个月期限有名义显著证据。3 个月加入 GSCPI 后仍未达到纯 AR 基准的精度。",
        f"2. **随机森林与线性 ARX。** 以 ARX-GSCPI 为参照，{describe_pair(metrics, dm, 'Random Forest', 'ARX-GSCPI')}"
        "仅 3 个月期限的两项误差都更低，三个期限的 DM p 值均高于 0.05。",
        f"3. **疫情前后。** {subperiod_summary}。"
        + ("四个模型三个期限的 RMSE 和 MAE 在 2020 年起均高于疫情前。" if both_increase else "")
        + "各期仅使用该期目标月的共同样本，误差变化同时反映预测难度、样本组成与模型表现。"
        "2020 年后样本包含供应链冲击和通胀急变，样本量较小，对排名稳定性应保留判断。", "",
        "## 复现", "",
        "各起点的训练行数及最新训练标签写入 [预测 CSV](../data/processed/task2_forecasts.csv)。"
        "评价明细见 [误差 CSV](../data/processed/task2_metrics.csv) 和 [DM CSV](../data/processed/task2_dm_tests.csv)。", "",
        "```sh", "python3 -m pip install -r code/requirements-task2.txt",
        "python3 code/task2_forecast.py", "python3 -m unittest discover -s code -v", "```",
    ]
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")


def save(fig, directory, name):
    fig.savefig(directory / f"task2_{name}.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def plot_forecasts(forecasts, directory):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True, layout="constrained")
    for ax, h in zip(axes, HORIZONS):
        sample = forecasts.loc[forecasts.horizon.eq(h)]
        values = sample.pivot(index="target_month", columns="model", values="prediction")
        full_calendar = pd.period_range(values.index[0], values.index[-1], freq="M")
        values = values.reindex(full_calendar)
        actual = sample.loc[sample.model.eq("AR")].set_index("target_month").actual
        x = full_calendar.to_timestamp()
        ax.axvspan(BREAK.to_timestamp(), x[-1], color="#EAF1F4", zorder=0)
        ax.plot(x, actual.reindex(full_calendar), label="Actual", lw=1.65, color="#1E303A")
        for model in MODELS:
            ax.plot(x, values[model], label=model, lw=1.15, alpha=.91, color=COLORS[model])
        ax.set_title(f"h={h} months · {x[0]:%Y-%m} to {x[-1]:%Y-%m} · n={len(actual)}", loc="left")
        ax.set_ylabel("Goods PCE MoM (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncols=5, frameon=False, loc="outside lower center", fontsize=9)
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlabel("Target month")
    fig.suptitle("U.S. goods inflation · expanding-window forecasts\nShading marks target months from January 2020")
    save(fig, directory, "forecast_comparison")


def plot_full(metrics, directory):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.5), sharey=True, layout="constrained")
    for ax, measure in zip(axes, ("rmse", "mae")):
        width = .18
        for i, model in enumerate(MODELS):
            sample = metrics.loc[metrics.period.eq("Full") & metrics.model.eq(model)]
            pos = np.arange(len(HORIZONS)) + (i - 1.5) * width
            ax.bar(pos, sample[measure], width, color=COLORS[model], label=model)
        ax.set_title(measure.upper())
        ax.set_xticks(range(len(HORIZONS)), [f"h={h}" for h in HORIZONS])
        ax.set_xlabel("Forecast horizon (months)")
    axes[0].set_ylabel("Forecast error (percentage points)")
    axes[1].legend(frameon=False, fontsize=8, ncols=2)
    fig.suptitle("Full evaluation period · forecast accuracy")
    save(fig, directory, "model_performance")


def plot_periods(metrics, directory):
    fig, axes = plt.subplots(2, 3, figsize=(13.1, 6.3), sharey=True, layout="constrained")
    labels = ["AR", "ARX-Base", "ARX-GSCPI", "RF"]
    for row, period in zip(axes, PERIODS[1:]):
        for ax, h in zip(row, HORIZONS):
            s = (metrics.loc[metrics.horizon.eq(h) & metrics.period.eq(period)]
                 .set_index("model").loc[list(MODELS)])
            pos = np.arange(len(MODELS))
            ax.bar(pos - .18, s.rmse, .35, color="#2B6D88", label="RMSE")
            ax.bar(pos + .18, s.mae, .35, color="#BB673C", label="MAE")
            ax.set_title(f"{period} · h={h} · n={s.n.iloc[0]}", loc="left", fontsize=10)
            ax.set_xticks(pos, labels, fontsize=8)
        row[0].set_ylabel("Error (percentage points)")
    axes[0, 2].legend(frameon=False, ncols=2, fontsize=8)
    fig.suptitle("Forecast accuracy before and from January 2020 · split by target month")
    save(fig, directory, "subperiod_comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="project directory containing Task 0 and 1")
    parser.add_argument("--output", type=Path, help="separate output directory (optional)")
    args = parser.parse_args()
    monthly, task1 = load_inputs(args.root)
    with ProcessPoolExecutor(max_workers=len(HORIZONS)) as executor:
        futures = [executor.submit(run_horizon, monthly, task1, h) for h in HORIZONS]
        results = [job.result() for job in futures]
    forecasts = pd.concat([f for f, _ in results], ignore_index=True)
    configs = [config for _, config in results]
    metrics = calculate_metrics(forecasts)
    dm = dm_results(forecasts)
    output = args.output or args.root
    data, reports, figures = output / "data" / "processed", output / "reports", output / "figures"
    for directory in (data, reports, figures):
        directory.mkdir(parents=True, exist_ok=True)
    for name, frame in (("forecasts", forecasts), ("metrics", metrics), ("dm_tests", dm)):
        frame.to_csv(data / f"task2_{name}.csv", index=False, float_format="%.12g")
    write_config(reports / "task2_model_config.md", configs, forecasts, args.root)
    write_report(reports / "task2_forecast.md", configs, forecasts, metrics, dm)

    plt.rcParams.update({
        "figure.facecolor": "#F8FBFC", "axes.facecolor": "#FFFFFF", "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.spines.bottom": False,
        "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": "#DBE7EC",
        "text.color": "#16323D", "axes.labelcolor": "#16323D", "xtick.color": "#405B66",
        "ytick.color": "#405B66", "figure.titlesize": 15, "font.size": 10,
    })
    plot_forecasts(forecasts, figures)
    plot_full(metrics, figures)
    plot_periods(metrics, figures)
    print(f"Task 2: {len(forecasts)} predictions, {len(metrics)} metric rows, {len(dm)} DM comparisons")


if __name__ == "__main__":
    main()
