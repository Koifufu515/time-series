"""Reproduce Task 3 without writing to Task 0–2 artifacts."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from task2_forecast import (
    BREAK, CONTROLS, GSCPI, HORIZONS, INITIAL_N, LAGS, MODELS, ROOT, TARGET,
    clark_west_loss, clark_west_results, clark_west_statistic, common_rows, direct_frame,
    load_inputs, ols_fit,
    paired_forecasts, read_forecasts, select_order, table, training_at,
)

HEADLINE = "pce_mom_pct"
BASE, LEVEL = MODELS[1:3]
TRANSFORMS = {"ARX-GSCPI-Level": GSCPI, "ARX-GSCPI-Lag1": "gscpi_lag1",
              "ARX-GSCPI-Diff": "gscpi_diff"}


def holm_adjust(probabilities):
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Expected a nonempty vector of probabilities")
    order = np.argsort(p, kind="stable")
    corrected = np.empty_like(p)
    corrected[order] = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return corrected


def holm_table(previous):
    records = []
    for (restricted, unrestricted), group in previous.groupby(
        ["restricted_model", "unrestricted_model"], sort=False
    ):
        group = group.sort_values("horizon")
        if tuple(group.horizon) != HORIZONS:
            raise ValueError("Each Holm family must have one p value per horizon")
        p = group.p_value_one_sided.to_numpy()
        for row, corrected in zip(group.itertuples(index=False), holm_adjust(p)):
            records.append({"restricted_model": restricted, "unrestricted_model": unrestricted,
                            "horizon": row.horizon, "n": row.n, "cw_stat": row.cw_stat,
                            "raw_p": row.p_value_one_sided, "holm_p": corrected,
                            "reject_raw_5pct": row.p_value_one_sided < .05,
                            "reject_holm_5pct": corrected < .05})
    return pd.DataFrame(records)


def cw_row(actual, restricted, unrestricted, horizon, **labels):
    if not actual.index.equals(restricted.index) or not actual.index.equals(unrestricted.index):
        raise ValueError("CW requires identical target dates")
    loss = clark_west_loss(actual, restricted, unrestricted)
    stat, probability = clark_west_statistic(loss, horizon)
    return {**labels, "horizon": horizon, "n": len(actual), "target_start": actual.index[0],
            "target_end": actual.index[-1], "hac_lag": horizon - 1,
            "mean_adjusted_loss_diff": loss.mean(), "cw_stat": stat, "p_value_one_sided": probability}


def subperiod_cw(forecasts):
    records = []
    for h in HORIZONS:
        y, pred = paired_forecasts(forecasts, h)
        for period, mask in [("Pre-COVID", y.index < BREAK), ("Post-COVID", y.index >= BREAK)]:
            records.append(cw_row(y.loc[mask], pred.loc[mask, BASE], pred.loc[mask, LEVEL], h,
                                  period=period, restricted_model=BASE, unrestricted_model=LEVEL))
    return pd.DataFrame(records)


def experiment_rows(monthly, task1, horizon, target, features):
    # Both the history and response must use the chosen inflation measure.
    if target != TARGET:
        monthly = monthly.assign(**{TARGET: monthly[target]})
        task1 = task1.assign(**{TARGET: task1[target]})
    frame = direct_frame(monthly, task1, horizon)
    if any(v in features for v in ("gscpi_lag1", "gscpi_diff")):
        frame["gscpi_lag1"] = monthly[GSCPI].shift(1).reindex(task1.index)
        frame["gscpi_diff"] = frame[GSCPI] - frame["gscpi_lag1"]
        frame["predictors_ready"] &= np.isfinite(frame[list(features)]).all(axis=1)
    return common_rows(frame)


def fit_panel(rows, horizon, end, specs, experiment, window="Expanding"):
    initial = rows.iloc[:INITIAL_N]
    if len(initial) != INITIAL_N:
        raise ValueError("Fewer than 180 complete initial training rows")
    first_origin = initial.target_month.iloc[-1]
    pd.testing.assert_frame_equal(training_at(rows, first_origin), initial)
    order, _ = select_order(initial)
    records = []
    for origin, current in rows.loc[first_origin:end - horizon].iterrows():
        train = training_at(rows, origin)
        if window == "Rolling":
            train = train.iloc[-INITIAL_N:]
        base_columns = [*LAGS[:order], *CONTROLS]
        for model, feature in specs.items():
            columns = base_columns + ([feature] if feature else [])
            beta, _ = ols_fit(train, columns)
            prediction = float(np.sum(np.r_[1., current[columns].to_numpy(dtype=float)] * beta))
            records.append({
                "experiment": experiment, "window": window, "horizon": horizon, "model": model,
                "forecast_origin": origin, "target_month": current.target_month,
                "actual": current.actual, "prediction": prediction, "selected_ar_order": order,
                "train_n": len(train), "train_first_origin": train.index[0],
                "train_last_target_month": train.target_month.iloc[-1],
                "initial_first_origin": initial.index[0], "initial_last_origin": initial.index[-1],
            })
    return pd.DataFrame(records)


def paired_sample(frame, models):
    actual = frame.groupby("target_month").actual
    predicted = frame.pivot(index="target_month", columns="model", values="prediction").sort_index()
    aligned = frame.forecast_origin + int(frame.horizon.iloc[0]) == frame.target_month
    if (set(predicted.columns) != set(models) or not np.isfinite(predicted).all().all()
            or not np.isfinite(frame.actual).all() or not actual.nunique().eq(1).all() or not aligned.all()):
        raise ValueError("Inconsistent matched forecast dates or actuals")
    return actual.first().reindex(predicted.index), predicted


def summarize(forecasts):
    metrics, tests = [], []
    for (name, window, h), sample in forecasts.groupby(["experiment", "window", "horizon"], sort=False):
        models = sample.model.unique()
        actual, pred = paired_sample(sample, models)
        # Match the precision of the frozen forecast CSV used by Task 2.1 inference.
        # Use the full-precision estimates for the original RMSE/MAE calculations.
        cw_actual = actual.map(lambda value: float(f"{value:.12g}"))
        cw_pred = pred.apply(lambda column: column.map(lambda value: float(f"{value:.12g}")))
        base_error = actual - pred[BASE]
        base_rmse, base_mae = np.sqrt(np.mean(base_error ** 2)), np.mean(abs(base_error))
        for model in models:
            error = actual - pred[model]
            rmse, mae = np.sqrt(np.mean(error ** 2)), np.mean(abs(error))
            metrics.append({"experiment": name, "window": window, "horizon": h, "model": model,
                            "n": len(actual), "target_start": actual.index[0], "target_end": actual.index[-1],
                            "ar_order": sample.selected_ar_order.iloc[0], "rmse": rmse, "mae": mae,
                            "rmse_improvement_pct": 100 * (1 - rmse / base_rmse),
                            "mae_improvement_pct": 100 * (1 - mae / base_mae)})
            if model != BASE:
                tests.append(cw_row(cw_actual, cw_pred[BASE], cw_pred[model], h, experiment=name, window=window,
                                    restricted_model=BASE, unrestricted_model=model))
    return pd.DataFrame(metrics), pd.DataFrame(tests)


def verify_expanding_cw(tests, frozen):
    """Compare the task-wide expanding specification at the same forecast precision."""
    fields = ["horizon", "n", "target_start", "target_end", "hac_lag",
              "mean_adjusted_loss_diff", "cw_stat", "p_value_one_sided"]
    expanding = tests.loc[tests.experiment.eq("Training window") & tests.window.eq("Expanding")
                          & tests.restricted_model.eq(BASE) & tests.unrestricted_model.eq(LEVEL), fields]
    baseline = clark_west_results(frozen).rename(columns={"first_target": "target_start",
                                                          "last_target": "target_end"})
    baseline = baseline.loc[baseline.restricted_model.eq(BASE) & baseline.unrestricted_model.eq(LEVEL), fields]
    pd.testing.assert_frame_equal(expanding.reset_index(drop=True), baseline.reset_index(drop=True),
                                  check_exact=True)


def compare_frozen(new, frozen, horizon):
    keys = ["horizon", "model", "forecast_origin", "target_month"]
    a = new.sort_values(keys).set_index(keys)
    b = frozen.loc[frozen.model.isin([BASE, LEVEL]) & frozen.horizon.eq(horizon)]
    b = b.sort_values(keys).set_index(keys)
    pd.testing.assert_index_equal(a.index, b.index)
    for col in ("actual", "prediction"):
        np.testing.assert_allclose(a[col], b[col], rtol=0, atol=1e-11)
    for col in ("train_n", "train_last_target_month"):
        np.testing.assert_array_equal(a[col], b[col])


def run_experiments(monthly, task1, frozen):
    results = []
    end = task1.index[-1]
    for h in HORIZONS:
        fields = {BASE: None, LEVEL: GSCPI}
        rows = experiment_rows(monthly, task1, h, HEADLINE, [GSCPI])
        headline = fit_panel(rows, h, end, fields, "Headline PCE")
        results.append(headline)
        baseline = frozen.loc[frozen.horizon.eq(h) & frozen.model.eq(BASE)]
        np.testing.assert_array_equal(headline.loc[headline.model.eq(BASE), "target_month"], baseline.target_month)
        rows = experiment_rows(monthly, task1, h, TARGET, TRANSFORMS.values())
        results.append(fit_panel(rows, h, end, {BASE: None, **TRANSFORMS}, "GSCPI specification"))
        rows = experiment_rows(monthly, task1, h, TARGET, [GSCPI])
        expanding = fit_panel(rows, h, end, fields, "Training window")
        compare_frozen(expanding, frozen, h)
        rolling = fit_panel(rows, h, end, fields, "Training window", window="Rolling")
        keys = ["model", "forecast_origin", "target_month", "actual", "selected_ar_order"]
        pd.testing.assert_frame_equal(expanding[keys], rolling[keys])
        results.extend([expanding, rolling])
    return pd.concat(results, ignore_index=True)


def brief(path, holm, periods, metrics, tests):
    lines = [
        "# Task 3 稳健性分析与复算说明", "",
        "使用 Task 2.1 冻结预测和 Task 0、1 月度快照。各试验按观测月索引进行伪样本外预测，历史实际发布日期和首次发布值未重建。"
        "模型按期限从共同初始 180 条有效记录以 AIC 比较 p=1…6，训练标签满足 s+h≤t，缺失起点由组内全体模型跳过。", "",
        "CW 沿用 Task 2.1 的右尾正态近似与 h−1 阶 Bartlett HAC。第 k 阶只使用日历上恰好相隔 k 个月的"
        "有效观测对，两端有效时允许中间有缺月，缺失月份保留，各阶协方差统一除以有效配对样本总数 n。"
        "Holm 使用冻结的 Task 2.1 p 值，两个主检验家族各含三个期限。分期检验和三项稳健性试验均报告未经校正的逐项 p 值。", "",
        "## Holm 与时期异质性", "",
        table(["受限", "非受限", "h", "原始 p", "Holm p", "原始拒绝", "校正拒绝"], [
            [r.restricted_model, r.unrestricted_model, r.horizon, f"{r.raw_p:.4f}", f"{r.holm_p:.4f}",
             "是" if r.reject_raw_5pct else "否", "是" if r.reject_holm_5pct else "否"]
            for r in holm.itertuples(index=False)]), "",
        table(["目标期", "h", "n", "调整差", "CW", "单侧 p"], [
            [r.period, r.horizon, r.n, f"{r.mean_adjusted_loss_diff:+.5f}", f"{r.cw_stat:+.3f}",
             f"{r.p_value_one_sided:.4f}"] for r in periods.itertuples(index=False)]), "",
        "## 三项稳健性重估", "",
        "A 使用整体 PCE 自己的历史及初始样本选阶，评价月与主规格相同。"
        "B 从 1997-10 统一开始建模，GSCPI 的滞后值和月差损失的首月留空。"
        "C 固定窗口每次使用最近 180 条完整且标签已可用的训练记录，因缺月可能横跨超过 180 个日历月。"
        "C 扩展窗口逐月重新拟合，日期、训练样本和预测值与 T2 已核对。其全期 CW 统计量、单侧 p 值和平均调整损失差与 Task 2.1 完全一致。", "",
        "相对改善为 100×(1−模型误差/同组基准误差)，正值表示误差变小。新增检验的逐项 p 值用于稳健性描述。", "",
        table(["试验", "窗口", "h", "模型", "n", "p 阶", "RMSE", "MAE", "RMSE 改善%", "MAE 改善%"], [
            [r.experiment, r.window, r.horizon, r.model, r.n, r.ar_order, f"{r.rmse:.4f}", f"{r.mae:.4f}",
             f"{r.rmse_improvement_pct:+.2f}", f"{r.mae_improvement_pct:+.2f}"]
            for r in metrics.itertuples(index=False)]), "",
        table(["试验", "窗口", "h", "非受限模型", "n", "调整差", "CW", "单侧 p"], [
            [r.experiment, r.window, r.horizon, r.unrestricted_model, r.n, f"{r.mean_adjusted_loss_diff:+.5f}",
             f"{r.cw_stat:+.3f}", f"{r.p_value_one_sided:.4f}"] for r in tests.itertuples(index=False)]), "",
        "逐月结果及各次训练边界见 [task3_forecasts.csv](../data/processed/task3_forecasts.csv)。",
        "", "## 复现", "", "在项目根目录执行以下命令。报告排版使用 Typst。",
        "", "~~~sh", "python3 -m pip install -r code/requirements-task2.txt",
        "python3 code/task3_robustness.py", "python3 code/task3_report.py",
        "python3 -m unittest discover -s code -v", "~~~", "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def graphs(task1, metrics, frozen_metrics, figures):
    plt.rcParams.update({"figure.facecolor": "#F8FBFC", "axes.facecolor": "#FFFFFF",
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True,
                         "grid.color": "#DBE7EC", "text.color": "#16323D", "font.size": 11})
    shades = ("#23658A", "#187059", "#9A491F")
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True, layout="constrained")
    dates = task1.index.to_timestamp()
    for ax, name, title, unit, color in zip(axes, [GSCPI, TARGET], ["Global supply-chain pressure", "Goods PCE inflation"],
                                           ["Standard deviations", "Monthly change (%)"], shades):
        ax.axvspan(BREAK.to_timestamp(), dates[-1], color="#EAF1F4")
        ax.axhline(0, color="#78919F", lw=.75)
        ax.plot(dates, task1[name], color=color, lw=1.25)
        ax.set_title(title, loc="left")
        ax.set_ylabel(unit)
        ax.set_xlim(dates[0], dates[-1])
    axes[-1].set_xlabel("Observation month")
    fig.suptitle("GSCPI and U.S. goods PCE inflation | 1997-09 to 2026-09")
    fig.savefig(figures / "task3_gscpi_goods.png", dpi=160)
    plt.close(fig)

    def bars(rows, names, name, title):
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.35), layout="constrained")
        for ax, measure in zip(axes, ["rmse", "mae"]):
            width = .68 / len(rows)
            for i, (row, label, color) in enumerate(zip(rows, names, shades)):
                center = np.arange(3) + (i - (len(rows) - 1) / 2) * width
                ax.bar(center, row[f"{measure}_improvement_pct"], width, label=label, color=color)
            ax.axhline(0, color="#536978", lw=.9)
            ax.set_title(f"{measure.upper()} improvement")
            ax.set_xticks(range(3), [f"h={h}" for h in HORIZONS])
            ax.set_xlabel("Horizon (months)")
        axes[0].set_ylabel("Error reduction versus ARX-Base (%)")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", frameon=False, ncols=3, fontsize=9)
        fig.suptitle(title)
        fig.savefig(figures / f"task3_{name}.png", dpi=160)
        plt.close(fig)

    family = metrics.loc[metrics.experiment.eq("GSCPI specification")]
    rows = [family.loc[family.model.eq(model)].sort_values("horizon") for model in TRANSFORMS]
    bars(rows, ["Level", "Lagged level", "First difference"], "gscpi_specification",
         "Goods inflation | GSCPI transformations on common dates")
    goods = frozen_metrics.loc[frozen_metrics.period.eq("Full")].pivot(index="horizon", columns="model")
    goods = pd.DataFrame({f"{metric}_improvement_pct": 100 * (1 - goods[metric][LEVEL] / goods[metric][BASE])
                          for metric in ["rmse", "mae"]})
    headline = metrics.loc[metrics.experiment.eq("Headline PCE") & metrics.model.eq(LEVEL)].sort_values("horizon")
    bars([goods, headline], ["Goods PCE", "Headline PCE"], "headline_goods",
         "Relative forecast improvement | goods versus headline PCE")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root, destination = args.root, args.output or args.root
    source = root / "data" / "processed"
    monthly, task1 = load_inputs(root)
    frozen = read_forecasts(source / "task2_forecasts.csv")
    frozen_metrics = pd.read_csv(source / "task2_metrics.csv")
    holm = holm_table(pd.read_csv(source / "task2_clark_west_tests.csv"))
    periods = subperiod_cw(frozen)
    forecasts = run_experiments(monthly, task1, frozen)
    metrics, cw = summarize(forecasts)
    verify_expanding_cw(cw, frozen)
    data, figures, reports = [destination / relative for relative in ("data/processed", "figures", "reports")]
    for directory in (data, figures, reports):
        directory.mkdir(parents=True, exist_ok=True)
    outputs = {"holm": holm, "subperiod_cw": periods, "forecasts": forecasts}
    for experiment, stem, name in [
        ("Headline PCE", "headline_pce", "headline_pce_cw"),
        ("GSCPI specification", "gscpi_specification", "gscpi_specification"),
        ("Training window", "window", "window_cw"),
    ]:
        outputs[f"{stem}_metrics"] = metrics.loc[metrics.experiment.eq(experiment)]
        outputs[name] = cw.loc[cw.experiment.eq(experiment)]
    for name, frame in outputs.items():
        frame.to_csv(data / f"task3_{name}.csv", index=False, float_format="%.12g")
    graphs(task1, metrics, frozen_metrics, figures)
    brief(reports / "task3_robustness.md", holm, periods, metrics, cw)
    print(f"Task 3: {len(forecasts)} forecasts, {len(metrics)} metric rows, {len(cw)} robustness CW tests")


if __name__ == "__main__":
    main()
