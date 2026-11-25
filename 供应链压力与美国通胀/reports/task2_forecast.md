# Task 2 商品通胀样本外预测

## 预测模型

目标为商品 PCE 月环比 $\pi_t=100(P_t/P_{t-1}-1)$，单位为百分比。三个期限独立估计，直接预测 t+h 当月的环比。h=1、3 为主结果，h=6 为补充。

- AR：$\hat{\pi}_{t+h}=\hat\alpha_h+\sum_{i=0}^{p_h-1}\hat\phi_{h,i}\pi_{t-i}$。
- ARX-Base：在 AR 的右侧加 $\hat\beta_{h,1} W_t+\hat\beta_{h,2}\Delta U_t$。
- ARX-GSCPI：在 ARX-Base 的右侧加 $\hat\beta_{h,3}G_t$。
- Random Forest：$\hat{\pi}_{t+h}=300^{-1}\sum_{b=1}^{300}T_{h,b}(X_t)$，$X_t=(\pi_t,\ldots,\pi_{t-p_h+1},W_t,\Delta U_t,G_t)$，与 ARX-GSCPI 同列同样本。

其中 $W_t=100\Delta\ln(\mathrm{WTI}_t)$，$\Delta U_t$ 为失业率的月度百分点变化，$G_t$ 为 GSCPI 水平值。初始 AIC、RF 网格与逐期限配置见 [模型配置](task2_model_config.md)。

## 训练与评价样本

**这是采用当前修订数据快照的伪样本外预测。** t 是可用信息的观测月份索引。实验设定 t 表示当月各输入及标签已经公布的信息截面。代码按观测月份约束信息集，未重建历史日历发布日期或首次发布版本。实际执行预测时，部分 t 月数据在 t+1 月才公布，因此 h=1 的实际预测提前量可能短于一个整月。

使用 1997-09 至 2026-09 的共同月度样本。只为构造初期 AR 滞后，从 Task 0 读取共同样本前的已知商品通胀。四个模型在每个 h 使用同一训练行和评价日期。初始窗用 180 行完整样本，并在首个起点完成 AIC 和 RF 调参。时间索引始终保留日历月。每个测试起点以 s+h≤t 截断训练标签，使用可用行重新估计；在特征缺失起点跳过全部四模型。2025-10 和 2025-11 的失业率月差缺失，造成各期限对应的目标月缺口。

| h | p | 初始训练 n | 末次训练 n | 测试起点 | 测试目标月 | 全期 n | 疫情前 n | 2020 起 n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 2 | 180 | 345 | 2012-09 至 2026-08 | 2012-10 至 2026-09 | 166 | 87 | 79 |
| 3 | 1 | 180 | 341 | 2012-11 至 2026-06 | 2013-02 至 2026-09 | 162 | 83 | 79 |
| 6 | 1 | 180 | 337 | 2013-02 至 2026-03 | 2013-08 至 2026-09 | 156 | 77 | 79 |

## 预测表现

误差定义为实际值减预测值，平均误差为正表示平均低估。RMSE、MAE 和平均误差均以百分点计。按目标月划分子时期，2020-01 起归入 Post-COVID。

| h | 时期 | 模型 | n | RMSE | MAE | 平均误差 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Full | AR | 166 | 0.4138 | 0.3088 | +0.0146 |
| 1 | Pre-COVID | AR | 87 | 0.3563 | 0.2693 | -0.0774 |
| 1 | Post-COVID | AR | 79 | 0.4691 | 0.3523 | +0.1159 |
| 1 | Full | ARX-Base | 166 | 0.4186 | 0.3159 | +0.0317 |
| 1 | Pre-COVID | ARX-Base | 87 | 0.3212 | 0.2446 | -0.0621 |
| 1 | Post-COVID | ARX-Base | 79 | 0.5045 | 0.3945 | +0.1350 |
| 1 | Full | ARX-GSCPI | 166 | 0.3867 | 0.2869 | -0.0223 |
| 1 | Pre-COVID | ARX-GSCPI | 87 | 0.3211 | 0.2406 | -0.0824 |
| 1 | Post-COVID | ARX-GSCPI | 79 | 0.4480 | 0.3379 | +0.0439 |
| 1 | Full | Random Forest | 166 | 0.4129 | 0.3060 | +0.0383 |
| 1 | Pre-COVID | Random Forest | 87 | 0.3353 | 0.2508 | -0.0723 |
| 1 | Post-COVID | Random Forest | 79 | 0.4842 | 0.3669 | +0.1600 |
| 3 | Full | AR | 162 | 0.4640 | 0.3388 | +0.0265 |
| 3 | Pre-COVID | AR | 83 | 0.3804 | 0.2758 | -0.1124 |
| 3 | Post-COVID | AR | 79 | 0.5380 | 0.4051 | +0.1725 |
| 3 | Full | ARX-Base | 162 | 0.5326 | 0.3806 | +0.0362 |
| 3 | Pre-COVID | ARX-Base | 83 | 0.3994 | 0.3014 | -0.1285 |
| 3 | Post-COVID | ARX-Base | 79 | 0.6435 | 0.4639 | +0.2092 |
| 3 | Full | ARX-GSCPI | 162 | 0.4969 | 0.3626 | -0.0123 |
| 3 | Pre-COVID | ARX-GSCPI | 83 | 0.3999 | 0.3030 | -0.1350 |
| 3 | Post-COVID | ARX-GSCPI | 79 | 0.5817 | 0.4251 | +0.1167 |
| 3 | Full | Random Forest | 162 | 0.4456 | 0.3351 | +0.0108 |
| 3 | Pre-COVID | Random Forest | 83 | 0.3815 | 0.2794 | -0.1064 |
| 3 | Post-COVID | Random Forest | 79 | 0.5043 | 0.3937 | +0.1340 |
| 6 | Full | AR | 156 | 0.4617 | 0.3364 | +0.0280 |
| 6 | Pre-COVID | AR | 77 | 0.3689 | 0.2690 | -0.1115 |
| 6 | Post-COVID | AR | 79 | 0.5369 | 0.4022 | +0.1639 |
| 6 | Full | ARX-Base | 156 | 0.4640 | 0.3365 | +0.0244 |
| 6 | Pre-COVID | ARX-Base | 77 | 0.3720 | 0.2705 | -0.1194 |
| 6 | Post-COVID | ARX-Base | 79 | 0.5388 | 0.4008 | +0.1646 |
| 6 | Full | ARX-GSCPI | 156 | 0.4337 | 0.3317 | -0.0674 |
| 6 | Pre-COVID | ARX-GSCPI | 77 | 0.3692 | 0.2770 | -0.1521 |
| 6 | Post-COVID | ARX-GSCPI | 79 | 0.4883 | 0.3850 | +0.0151 |
| 6 | Full | Random Forest | 156 | 0.4604 | 0.3524 | -0.0204 |
| 6 | Pre-COVID | Random Forest | 77 | 0.3686 | 0.2787 | -0.1306 |
| 6 | Post-COVID | Random Forest | 79 | 0.5350 | 0.4242 | +0.0871 |

| h | 时期 | RMSE 从低到高 | MAE 从低到高 |
| --- | --- | --- | --- |
| 1 | Full | ARX-GSCPI → Random Forest → AR → ARX-Base | ARX-GSCPI → Random Forest → AR → ARX-Base |
| 1 | Pre-COVID | ARX-GSCPI → ARX-Base → Random Forest → AR | ARX-GSCPI → ARX-Base → Random Forest → AR |
| 1 | Post-COVID | ARX-GSCPI → AR → Random Forest → ARX-Base | ARX-GSCPI → AR → Random Forest → ARX-Base |
| 3 | Full | Random Forest → AR → ARX-GSCPI → ARX-Base | Random Forest → AR → ARX-GSCPI → ARX-Base |
| 3 | Pre-COVID | AR → Random Forest → ARX-Base → ARX-GSCPI | AR → Random Forest → ARX-Base → ARX-GSCPI |
| 3 | Post-COVID | Random Forest → AR → ARX-GSCPI → ARX-Base | Random Forest → AR → ARX-GSCPI → ARX-Base |
| 6 | Full | ARX-GSCPI → Random Forest → AR → ARX-Base | ARX-GSCPI → AR → ARX-Base → Random Forest |
| 6 | Pre-COVID | Random Forest → AR → ARX-GSCPI → ARX-Base | AR → ARX-Base → ARX-GSCPI → Random Forest |
| 6 | Post-COVID | ARX-GSCPI → Random Forest → AR → ARX-Base | ARX-GSCPI → ARX-Base → AR → Random Forest |

[逐月预测](../figures/task2_forecast_comparison.png) · [全期误差](../figures/task2_model_performance.png) · [分期误差](../figures/task2_subperiod_comparison.png)

## Diebold–Mariano 检验

原假设为两个模型的期望平方预测损失相等。差值为模型 A 的平方误差减模型 B 的平方误差，负值指向 A，正值指向 B。报告全评价期、双侧正态近似 p 值。Newey–West/Bartlett HAC 使用 h−1 阶。自协方差仅配对恰好相隔指定月数的有效误差，全都除以匹配样本量 n，保留月历缺口。方差为零时统计量与 p 值留空。

| h | 模型 A | 模型 B | n | HAC 阶数 | 平均损失差 | DM | p | 损失较低方向 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | ARX-GSCPI | ARX-Base | 166 | 0 | -0.02564 | -2.027 | 0.0426 | ARX-GSCPI |
| 1 | Random Forest | ARX-GSCPI | 166 | 0 | +0.02091 | +1.366 | 0.1719 | ARX-GSCPI |
| 1 | ARX-Base | AR | 166 | 0 | +0.00398 | +0.300 | 0.7642 | AR |
| 3 | ARX-GSCPI | ARX-Base | 162 | 2 | -0.03672 | -1.775 | 0.0760 | ARX-GSCPI |
| 3 | Random Forest | ARX-GSCPI | 162 | 2 | -0.04832 | -1.033 | 0.3018 | Random Forest |
| 3 | ARX-Base | AR | 162 | 2 | +0.06833 | +1.432 | 0.1521 | AR |
| 6 | ARX-GSCPI | ARX-Base | 156 | 5 | -0.02726 | -0.940 | 0.3470 | ARX-GSCPI |
| 6 | Random Forest | ARX-GSCPI | 156 | 5 | +0.02394 | +1.221 | 0.2219 | ARX-GSCPI |
| 6 | ARX-Base | AR | 156 | 5 | +0.00217 | +1.704 | 0.0885 | AR |

ARX-GSCPI 对 ARX-Base 及 ARX-Base 对 AR 是嵌套比较。常规 DM 的正态近似在此类比较中具有局限，p 值应连同误差幅度和方向解读。检验未做跨期限或跨比较的多重校正。h−1 带宽针对多步重叠，对其他来源的持久性不作额外调整。 [Clark 与 West 的嵌套模型检验讨论](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf)

## 研究问题

1. **GSCPI 的增量价值。** 以 ARX-Base 为参照，h=1 的 RMSE 变化 -7.6%，MAE 变化 -9.2%，DM p=0.043。h=3 的 RMSE 变化 -6.7%，MAE 变化 -4.7%，DM p=0.076。h=6 的 RMSE 变化 -6.5%，MAE 变化 -1.4%，DM p=0.347。按未校正的 5% 门槛，仅 1 个月期限有名义显著证据。3 个月加入 GSCPI 后仍未达到纯 AR 基准的精度。
2. **随机森林与线性 ARX。** 以 ARX-GSCPI 为参照，h=1 的 RMSE 变化 +6.8%，MAE 变化 +6.7%，DM p=0.172。h=3 的 RMSE 变化 -10.3%，MAE 变化 -7.6%，DM p=0.302。h=6 的 RMSE 变化 +6.2%，MAE 变化 +6.2%，DM p=0.222。仅 3 个月期限的两项误差都更低，三个期限的 DM p 值均高于 0.05。
3. **疫情前后。** h=1 两期的最低 RMSE 模型均为 ARX-GSCPI。h=3 的最低 RMSE 模型由疫情前 AR 变为 2020 年起 Random Forest。h=6 的最低 RMSE 模型由疫情前 Random Forest 变为 2020 年起 ARX-GSCPI。四个模型三个期限的 RMSE 和 MAE 在 2020 年起均高于疫情前。各期仅使用该期目标月的共同样本，误差变化同时反映预测难度、样本组成与模型表现。2020 年后样本包含供应链冲击和通胀急变，样本量较小，对排名稳定性应保留判断。

## 复现

各起点的训练行数及最新训练标签写入 [预测 CSV](../data/processed/task2_forecasts.csv)。评价明细见 [误差 CSV](../data/processed/task2_metrics.csv) 和 [DM CSV](../data/processed/task2_dm_tests.csv)。

```sh
python3 -m pip install -r code/requirements-task2.txt
python3 code/task2_forecast.py
python3 -m unittest discover -s code -v
```
