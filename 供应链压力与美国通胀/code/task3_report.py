"""Build the course report Markdown and its equivalent A4 PDF from the project results."""
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from task2_forecast import HORIZONS, ROOT, table
from task3_robustness import BASE, LEVEL


TEMPLATE = Path(__file__).with_name("task3_report_template.md")
PAGE_BREAK = "<!-- pagebreak -->"
MODEL_NAME = {"Random Forest": "RF", "ARX-GSCPI-Level": "当期水平",
              "ARX-GSCPI-Lag1": "滞后水平", "ARX-GSCPI-Diff": "一阶差分"}
VAR_NAME = {"pce_goods_mom_pct": "商品 PCE 月环比", "pce_mom_pct": "整体 PCE 月环比",
            "gscpi": "GSCPI 水平", "wti_log_mom_pct": "WTI 对数月变化",
            "unemployment_change_pp": "失业率月差", "gscpi_change_sd": "GSCPI 月差"}
MATH = {
    "target": r'pi_t = 100 (frac(P_t, P_(t-1)) - 1)',
    "ar": r'hat(pi)_(t+h)^(A R) = alpha_h + sum_(i=0)^(p_h-1) phi_(h,i) pi_(t-i)',
    "arx_base": r'hat(pi)_(t+h)^B = hat(alpha)_h^B + sum_(i=0)^(p_h-1) hat(phi)_(h,i)^B pi_(t-i) + hat(beta)_(h,1)^B W_t + hat(beta)_(h,2)^B Delta U_t',
    "arx_gscpi": r'hat(pi)_(t+h)^G = hat(alpha)_h^G + sum_(i=0)^(p_h-1) hat(phi)_(h,i)^G pi_(t-i) + hat(beta)_(h,1)^G W_t + hat(beta)_(h,2)^G Delta U_t + hat(beta)_(h,3)^G G_t',
    "rf": r'hat(pi)_(t+h)^(R F) = frac(1,300) sum_(b=1)^300 T_(h,b)(X_t)',
    "error": r'"RMSE" = sqrt(frac(sum_t e_t^2,N)), quad "MAE" = frac(sum_t abs(e_t),N), quad e_t = y_t-hat(y)_t',
    "cw": r'd_t = e_(R,t)^2 - e_(U,t)^2 + (hat(y)_(R,t)-hat(y)_(U,t))^2',
}
MD_MATH = {
    "target": r"\pi_t=100\left(\frac{P_t}{P_{t-1}}-1\right)",
    "ar": r"\hat\pi_{t+h}^{AR}=\alpha_h+\sum_{i=0}^{p_h-1}\phi_{h,i}\pi_{t-i}",
    "arx_base": r"\hat\pi_{t+h}^{B}=\hat\alpha_h^{B}+\sum_{i=0}^{p_h-1}\hat\phi_{h,i}^{B}\pi_{t-i}+\hat\beta_{h,1}^{B}W_t+\hat\beta_{h,2}^{B}\Delta U_t",
    "arx_gscpi": r"\hat\pi_{t+h}^{G}=\hat\alpha_h^{G}+\sum_{i=0}^{p_h-1}\hat\phi_{h,i}^{G}\pi_{t-i}+\hat\beta_{h,1}^{G}W_t+\hat\beta_{h,2}^{G}\Delta U_t+\hat\beta_{h,3}^{G}G_t",
    "rf": r"\hat\pi_{t+h}^{RF}=\frac{1}{300}\sum_{b=1}^{300}T_{h,b}(X_t)",
    "error": r"\mathrm{RMSE}=\sqrt{\frac{\sum_t e_t^2}{N}},\quad \mathrm{MAE}=\frac{\sum_t|e_t|}{N},\quad e_t=y_t-\hat y_t",
    "cw": r"d_t=e_{R,t}^2-e_{U,t}^2+(\hat y_{R,t}-\hat y_{U,t})^2",
}


def report_tables():
    data = ROOT / "data/processed"

    def read(stem):
        return pd.read_csv(data / f"{stem}.csv")

    full = read("task2_metrics")
    overall = full.loc[full.period.eq("Full")]
    main = overall.pivot(index="horizon", columns="model", values=["rmse", "mae"])
    stats = read("task1_descriptive").set_index(["period", "series"])
    sample = read("task2_forecasts")
    leads = read("task1_gscpi_leads")
    leads = leads.loc[leads.series.eq("pce_goods_mom_pct")].pivot(
        index="lead_months", columns="period", values="r")
    stationarity = read("task1_stationarity")
    adf = stationarity.loc[stationarity.period.eq("Full")].set_index("series")
    holm = read("task3_holm")
    dm = read("task2_dm_tests")
    period = read("task3_subperiod_cw")
    headline = read("task3_headline_pce_metrics").set_index(["horizon", "model"])
    headline_cw = read("task3_headline_pce_cw").set_index("horizon")
    specifications = read("task3_gscpi_specification_metrics")
    specs_cw = read("task3_gscpi_specification").set_index(["horizon", "unrestricted_model"])
    windows = read("task3_window_metrics")
    windows_cw = read("task3_window_cw").set_index(["horizon", "window"])

    def p_fmt(p):
        return f"{p:.3g}" if p < .001 else f"{p:.4f}"

    rows = {}
    rows["DESCRIPTIVE"] = table(["变量", "有效月数", "均值", "标准差", "最小", "最大", "超额峰度"], [
        [VAR_NAME[k], int(r.n), *[f"{r[c]:.3f}" for c in ["mean", "sd", "min", "max", "excess_kurtosis"]]]
        for k in list(VAR_NAME)[:5] for r in [stats.loc[("Full", k)]]])
    rows["REGIMES"] = table(["变量", "疫情前均值", "疫情前标准差", "2020 年起均值", "2020 年起标准差"], [
        [VAR_NAME[k], *[f"{stats.loc[(part, k), stat]:.3f}"
                        for part in ("1997–2019", "2020 onward") for stat in ("mean", "sd")]]
        for k in list(VAR_NAME)[:5]])
    rows["LEADS"] = table(["领先期 h", "全期", "2020 年前", "2020 年起"], [
        [h, *[f"{r[c]:.3f}" for c in ("Full", "1997–2019", "2020 onward")]]
        for h, r in leads.iterrows()])
    rows["ADF"] = table(["变量", "连续诊断样本 n", "ADF p", "KPSS p"], [
        [VAR_NAME[k], int(r.n), p_fmt(r.adf_p), (r.kpss_p_bound if pd.notna(r.kpss_p_bound) else "") + p_fmt(r.kpss_p)]
        for k, r in adf.loc[list(VAR_NAME)].iterrows()])
    rows["SAMPLE"] = table(["h", "首个起点", "首个目标", "末个目标", "预测 n", "2020 年前 / 起"], [
        [h, s.forecast_origin.min(), s.target_month.min(), s.target_month.max(), len(s),
         f"{sum(s.target_month < '2020-01')} / {sum(s.target_month >= '2020-01')}"]
        for h in HORIZONS for s in [sample.loc[sample.horizon.eq(h) & sample.model.eq("AR")]]])
    models = ["AR", BASE, LEVEL, "Random Forest"]
    rows["MAIN"] = table(["h", "指标", "AR", "ARX-Base", "ARX-GSCPI", "RF"], [
        [h, error.upper(), *[f"{main.loc[h, (error, m)]:.4f}" for m in models]]
        for h in HORIZONS for error in ("rmse", "mae")])
    rows["HOLM"] = table(["受限 → 非受限", "h", "CW", "原始 p", "Holm p", "5% 前 / 后"], [
        [f"{r.restricted_model} → {r.unrestricted_model}", r.horizon, f"{r.cw_stat:+.3f}",
         f"{r.raw_p:.4f}", f"{r.holm_p:.4f}",
         f"{'拒绝' if r.reject_raw_5pct else '保留'} / {'拒绝' if r.reject_holm_5pct else '保留'}"]
        for r in holm.itertuples(index=False)])
    rows["DM"] = table(["h", "n", "RF − ARX-GSCPI 损失差", "DM", "双侧 p"], [
        [r.horizon, r.n, f"{r.mean_loss_diff_a_minus_b:+.5f}", f"{r.dm_stat:+.3f}", f"{r.p_value:.4f}"]
        for r in dm.itertuples(index=False)])
    rows["SUBPERIOD"] = table(["h", "目标时期", "n", "调整差", "CW", "单侧 p"], [
        [r.horizon, "2020 年前" if r.period == "Pre-COVID" else "2020 年起", r.n,
         f"{r.mean_adjusted_loss_diff:+.5f}", f"{r.cw_stat:+.3f}", f"{r.p_value_one_sided:.4f}"]
        for r in period.itertuples(index=False)])
    rows["HEADLINE"] = table(["h", "整体 RMSE 基准 → 扩展", "整体 MAE 基准 → 扩展",
                              "RMSE 改善 商品 / 整体", "MAE 改善 商品 / 整体", "整体 CW / p"], [
        [h, f"{b.rmse:.4f} → {g.rmse:.4f}", f"{b.mae:.4f} → {g.mae:.4f}",
         f"{100*(1-main.loc[h,('rmse',LEVEL)]/main.loc[h,('rmse',BASE)]):.2f}% / {g.rmse_improvement_pct:.2f}%",
         f"{100*(1-main.loc[h,('mae',LEVEL)]/main.loc[h,('mae',BASE)]):.2f}% / {g.mae_improvement_pct:.2f}%",
         f"{c.cw_stat:.3f} / {c.p_value_one_sided:.4f}"]
        for h in HORIZONS for b, g, c in [(headline.loc[(h, BASE)], headline.loc[(h, LEVEL)], headline_cw.loc[h])]])
    rows["SPECS"] = table(["h", "GSCPI 形式", "n", "RMSE", "MAE", "RMSE 改善", "MAE 改善", "CW p"], [
        [r.horizon, MODEL_NAME.get(r.model, r.model), r.n, f"{r.rmse:.4f}", f"{r.mae:.4f}",
         f"{r.rmse_improvement_pct:+.2f}%", f"{r.mae_improvement_pct:+.2f}%",
         f"{specs_cw.loc[(r.horizon, r.model), 'p_value_one_sided']:.4f}" if r.model != BASE else "—"]
        for r in specifications.itertuples(index=False)])
    rows["WINDOW"] = table(["h", "窗口", "n", "基准 RMSE / MAE", "加入 GSCPI RMSE / MAE", "RMSE 改善", "CW / 单侧 p"], [
        [h, "扩展" if window == "Expanding" else "滚动", int(g.n),
         f"{b.rmse:.4f} / {b.mae:.4f}", f"{g.rmse:.4f} / {g.mae:.4f}",
         f"{g.rmse_improvement_pct:.2f}%", f"{c.cw_stat:.3f} / {c.p_value_one_sided:.4f}"]
        for (h, window), s in windows.groupby(["horizon", "window"], sort=True)
        for b, g, c in [(s.set_index("model").loc[BASE], s.set_index("model").loc[LEVEL],
                         windows_cw.loc[(h, window)])]])
    return rows


def quote(text):
    return json.dumps(text, ensure_ascii=False).replace("\u2028", "\\n")


INLINE = re.compile(r"\*\*(.*?)\*\*|\[([^\]]+)\]\(([^)]+)\)|" + chr(96) + "(.*?)" + chr(96))


def inline(text):
    parts = []
    cursor = 0
    for found in INLINE.finditer(text):
        if cursor < found.start():
            parts.append(f"text({quote(text[cursor:found.start()])})")
        bold, label, link, code = found.groups()
        if bold is not None:
            parts.append(f"strong(text({quote(bold)}))")
        elif label is not None:
            if link.startswith("https://"):
                parts.append(f"link({quote(link)})[#text(fill: rgb(\"#176383\"), {quote(label)})]")
            else:
                parts.append(f"text({quote(label)})")
        else:
            parts.append(f"text({quote(code)})")
        cursor = found.end()
    if cursor < len(text):
        parts.append(f"text({quote(text[cursor:])})")
    return " + ".join(parts) or "[]"


PREAMBLE = r'''
#set document(title: "全球供应链压力与美国商品通胀预测", date: none)
// Based on the user's document-organization-and-formatting/assets/coursework.typ.
#set page(paper: "a4", margin: 2.5cm, footer: context {
  let number = counter(page).get().first()
  if number > 1 { align(center, text(size: 9pt, str(number - 1))) }
})
#set text(font: ("STIX Two Text", "Songti SC"), lang: "zh", size: 11pt)
#set par(justify: true, leading: 0.95em, spacing: 0.85em)
#set table(stroke: none, inset: (x: 6pt, y: 4pt))
#show heading.where(level: 1): set text(size: 19pt)
#show heading.where(level: 2): set text(size: 15pt)
#show heading.where(level: 3): set text(size: 12pt)
#show math.equation: set text(font: "STIX Two Math")
#let caption(t) = block(above: 5pt, below: 5pt, text(size: 9.5pt, t))
#let gridtable(n, header, body) = block(above: 6pt, below: 10pt)[
  #set text(size: 9pt)
  #table(columns: (1fr,) * n, align: (x,y) => if x == 0 {left} else {center},
    table.hline(stroke: 0.8pt),
    table.header(..header.map(item => strong(item))),
    table.hline(stroke: 0.5pt), ..body,
    table.hline(stroke: 0.8pt))
]
'''


def typst_body(markdown):
    lines = markdown.splitlines()
    has_cover = markdown.startswith("# 全球供应链压力")
    result = ["#v(42mm)", "#align(center)[", "#set par(justify: false)"] if has_cover else []
    i = 0
    while i < len(lines):
        text = lines[i]
        i += 1
        if not text:
            continue
        if text == PAGE_BREAK:
            if has_cover:
                result.append("]")
                has_cover = False
            result.append("#pagebreak(weak: true)")
        elif text.startswith("@@MATH:"):
            result.append("$ " + MATH[text[7:-2]] + " $")
        elif text.startswith("| "):
            block = [text]
            while i < len(lines) and lines[i].startswith("| "):
                block.append(lines[i])
                i += 1
            cells = [[inline(c.strip()) for c in row.strip("|").split("|")] for row in [block[0], *block[2:]]]
            headers = ", ".join(cells[0])
            body = ", ".join(cell for row in cells[1:] for cell in row)
            result.append(f"#gridtable({len(cells[0])}, ({headers}), ({body}))")
        elif text.startswith("!["):
            alt, src = re.fullmatch(r"!\[([^\]]+)\]\(([^)]+)\)", text).groups()
            path = ROOT / "reports" / src
            if not path.resolve().is_file():
                raise ValueError(f"Missing figure {path}")
            result.append(f"#caption(strong({inline(alt)}))")
            if i < len(lines) and lines[i].startswith("> "):
                result.append(f"#caption({inline(lines[i][2:])})")
                i += 1
            src = str(path.resolve().relative_to(ROOT))
            result.append(f"#align(center, image({quote('/' + src)}, width: 100%))")
        elif text.startswith("#"):
            head, label = text.split(" ", 1)
            result.append(f"#heading(level: {len(head)}, {inline(label)})")
        else:
            result.append("#" + "(" + inline(text) + ")")
    return "\n\n".join(result) + "\n"


TABLE_TITLES = {
    "DESCRIPTIVE": "表 2　全期描述统计。失业率变化的峰度反映明显的尾部波动。",
    "REGIMES": "表 3　分期描述统计。2020 年起商品通胀和供应链压力的均值更高。",
    "LEADS": "表 4　领先相关系数。一个月领先相关在 2020 年起样本更高。",
    "ADF": "表 5　平稳性检验。GSCPI 水平与一阶差分呈现不同诊断结果。",
    "SAMPLE": "表 6　共同评价样本。长预测期限的有效配对月数较少。",
    "MAIN": "表 7　全期预测误差（百分点）。线性模型在 1、6 个月领先，随机森林在 3 个月领先。",
    "HOLM": "表 8　主规格 Clark–West 检验。GSCPI 的三个期限在 Holm 校正后仍达到 5%。",
    "DM": "表 9　随机森林与线性预测的比较。双侧检验均未达到 5%。",
    "SUBPERIOD": "表 10　分期 Clark–West 检验。逐项增量证据集中在 2020 年起。",
    "HEADLINE": "表 11　替换整体 PCE 目标。商品的相对 RMSE 改善在 3 个月更大。",
    "SPECS": "表 12　GSCPI 变换。当期水平在共同样本上获得最大的误差改善。",
    "WINDOW": "表 13　训练窗口比较。加入 GSCPI 后，两种窗口的 RMSE 均下降。",
}


def report_scalars():
    data = ROOT / "data/processed"
    metrics = pd.read_csv(data / "task2_metrics.csv")
    full = metrics.loc[metrics.period.eq("Full")].set_index(["horizon", "model"])
    holm = pd.read_csv(data / "task3_holm.csv")
    holm = holm.loc[holm.unrestricted_model.eq(LEVEL)].set_index("horizon")
    dm = pd.read_csv(data / "task2_dm_tests.csv").set_index("horizon")
    windows = pd.read_csv(data / "task3_window_metrics.csv")
    rolling = windows.loc[windows.window.eq("Rolling") & windows.model.eq(LEVEL)].set_index("horizon")
    headline = pd.read_csv(data / "task3_headline_pce_cw.csv").set_index("horizon")

    def gain(h, new, benchmark, metric):
        return f"{100 * (1 - full.loc[(h, new), metric] / full.loc[(h, benchmark), metric]):.2f}%"

    return {
        "GOODS_GAINS": "、".join(gain(h, LEVEL, BASE, "rmse") for h in HORIZONS),
        "GSCPI_HOLM": "、".join(f"{holm.loc[h, 'holm_p']:.4f}" for h in HORIZONS),
        "RF_GAIN": gain(3, "Random Forest", LEVEL, "rmse"),
        "RF_MAE_GAIN": gain(3, "Random Forest", LEVEL, "mae"),
        "RF_DM": f"{dm.loc[3, 'p_value']:.4f}",
        "HEADLINE_H3_P": f"{headline.loc[3, 'p_value_one_sided']:.4f}",
        "ROLLING_GAINS": "、".join(f"{rolling.loc[h, 'rmse_improvement_pct']:.2f}%" for h in HORIZONS),
    }


def main():
    markdown = TEMPLATE.read_text(encoding="utf-8")
    for key, value in report_tables().items():
        markdown = markdown.replace(f"@@{key}@@", f"**{TABLE_TITLES[key]}**\n\n{value}")
    for key, value in report_scalars().items():
        markdown = markdown.replace(f"@@{key}@@", value)
    if re.search(r"@@[A-Z_]+@@", markdown):
        raise ValueError("Unresolved report table")
    target = ROOT / "reports"
    body = typst_body(markdown)
    for key, value in MD_MATH.items():
        markdown = markdown.replace(f"@@MATH:{key}@@", "$$\n" + value + "\n$$")
    # A project-root-relative path is retained for each embedded figure.
    subprocess.run(["typst", "compile", "--root", str(ROOT), "-", str(target / "final_report.pdf")],
                   input=(PREAMBLE + body).encode("utf-8"), check=True)
    markdown = re.sub(r"(!\[([^\]]+)\]\([^)]+\))\n(> [^\n]+)",
                      lambda m: f"**{m[2]}**\n\n{m[3]}\n\n{m[1]}", markdown)
    (target / "final_report.md").write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
