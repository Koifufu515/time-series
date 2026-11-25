# Task 2 商品通胀样本外预测

## 预测模型

目标为商品 PCE 月环比 $\pi_t=100(P_t/P_{t-1}-1)$，单位为百分比。三个期限独立估计，直接预测 t+h 当月的环比。h=1、3 为主结果，h=6 为补充。

- AR：$\hat{\pi}_{t+h}=\hat\alpha_h+\sum_{i=0}^{p_h-1}\hat\phi_{h,i}\pi_{t-i}$。
- ARX-Base：在 AR 的右侧加 $\hat\beta_{h,1} W_t+\hat\beta_{h,2}\Delta U_t$。
- ARX-GSCPI：在 ARX-Base 的右侧加 $\hat\beta_{h,3}G_t$。
- Random Forest：$\hat{\pi}_{t+h}=300^{-1}\sum_{b=1}^{300}T_{h,b}(X_t)$，$X_t=(\pi_t,\ldots,\pi_{t-p_h+1},W_t,\Delta U_t,G_t)$，与 ARX-GSCPI 同列同样本。

其中 $W_t=100\Delta\ln(\mathrm{WTI}_t)$，$\Delta U_t$ 为失业率的月度百分点变化，$G_t$ 为 GSCPI 水平值。初始 AIC、RF 网格与逐期限配置见 [模型配置](task2_model_config.md)。

## 训练与评价样本

**实验定义为 Observation-indexed Pseudo Out-of-Sample Forecasting（按观测月索引的伪样本外预测）。** 输入是修订后的月度数据快照。t 是月度观测索引，实际预测发布日期另需确定。h 是目标月相对起点月的观测期差。预测信息集依据观测月构造，历史实际发布日期和首次发布版本尚未完整重建，因此本实验的结果只适用于这一观测索引设定，真实历史实时预测能力仍待验证。

使用 1997-09 至 2026-09 的共同月度样本。只为构造初期 AR 滞后，从 Task 0 读取共同样本前的已知商品通胀。四个模型在每个 h 使用同一训练行和评价日期。初始窗用 180 行完整样本，并在首个起点完成 AIC 和 RF 调参。时间索引始终保留日历月。每个测试起点以 s+h≤t 截断训练标签，使用可用行重新估计；在特征缺失起点跳过全部四模型。2025-10 和 2025-11 的失业率月差缺失，造成各期限对应的目标月缺口。

| h | 初始训练 n | 末次训练 n | 测试起点 | 测试目标月 | 全期 n | 疫情前 n | 2020 起 n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 180 | 345 | 2012-09 至 2026-08 | 2012-10 至 2026-09 | 166 | 87 | 79 |
| 3 | 180 | 341 | 2012-11 至 2026-06 | 2013-02 至 2026-09 | 162 | 83 | 79 |
| 6 | 180 | 337 | 2013-02 至 2026-03 | 2013-08 至 2026-09 | 156 | 77 | 79 |

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

## 统计推断

两种检验均使用全评价期的实际配对月份。Newey–West/Bartlett HAC 滞后阶数为 h−1。自协方差只计入恰好相隔指定月数的已配对观测，分母为有效配对总数 n。月历缺口保留，长期方差为零时统计量与 p 值留空。

### Clark–West 嵌套模型检验

调整后差值为 $d_t=e_{R,t}^2-e_{U,t}^2+(\hat y_{R,t}-\hat y_{U,t})^2$。检验 $H_0:E(d_t)\leq 0$ 对 $H_1:E(d_t)>0$，使用右尾正态近似 p 值。正的统计量指向非受限模型。调整项针对额外参数估计带来的预测噪声。 [Clark 与 West](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf)

| h | 受限模型 R | 非受限模型 U | n | HAC 阶数 | 平均调整损失差 | CW | 单侧 p | 样本调整损失方向 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | AR | ARX-Base | 166 | 0 | +0.05014 | +3.069 | 0.0011 | ARX-Base |
| 1 | ARX-Base | ARX-GSCPI | 166 | 0 | +0.04227 | +2.989 | 0.0014 | ARX-GSCPI |
| 3 | AR | ARX-Base | 162 | 2 | -0.01602 | -1.791 | 0.9633 | AR |
| 3 | ARX-Base | ARX-GSCPI | 162 | 2 | +0.05637 | +2.202 | 0.0138 | ARX-GSCPI |
| 6 | AR | ARX-Base | 156 | 5 | -0.00163 | -1.198 | 0.8846 | AR |
| 6 | ARX-Base | ARX-GSCPI | 156 | 5 | +0.07171 | +1.836 | 0.0332 | ARX-GSCPI |

### Diebold–Mariano 非嵌套比较

比较 Random Forest（A）与 ARX-GSCPI（B）。原假设为两者的期望平方误差相等。损失差 A−B 为负时，随机森林的样本平方误差更低。使用双侧正态近似。

| h | 模型 A | 模型 B | n | HAC 阶数 | 平均损失差 | DM | 双侧 p | 损失较低方向 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | Random Forest | ARX-GSCPI | 166 | 0 | +0.02091 | +1.366 | 0.1719 | ARX-GSCPI |
| 3 | Random Forest | ARX-GSCPI | 162 | 2 | -0.04832 | -1.033 | 0.3018 | Random Forest |
| 6 | Random Forest | ARX-GSCPI | 156 | 5 | +0.02394 | +1.221 | 0.2219 | ARX-GSCPI |

p 值为逐项结果，未做跨期限或跨模型的多重校正。h−1 带宽针对多步重叠。

## 研究问题

1. **GSCPI 的增量预测信息。** 相对 ARX-Base，h=1 的 RMSE 变化 -7.6%，MAE 变化 -9.2%，CW p=0.0014。h=3 的 RMSE 变化 -6.7%，MAE 变化 -4.7%，CW p=0.0138。h=6 的 RMSE 变化 -6.5%，MAE 变化 -1.4%，CW p=0.0332。三个期限的原始误差均下降，CW 在逐项 5% 水平下均拒绝原假设，6 个月的证据较弱。结果与供应链压力包含商品定价相关信息相符，经济因果机制仍需独立识别。
2. **Random Forest 与 ARX-GSCPI。** h=1 的 RMSE 变化 +6.8%，MAE 变化 +6.7%，DM p=0.1719。h=3 的 RMSE 变化 -10.3%，MAE 变化 -7.6%，DM p=0.3018。h=6 的 RMSE 变化 +6.2%，MAE 变化 +6.2%，DM p=0.2219。3 个月的原始误差较低，三个期限的双侧 DM 均未达到 5% 水平。
3. **WTI 与失业率的联合增量。** 相对 AR，h=1 的 RMSE 变化 +1.2%，MAE 变化 +2.3%，CW p=0.0011。h=3 的 RMSE 变化 +14.8%，MAE 变化 +12.3%，CW p=0.9633。h=6 的 RMSE 变化 +0.5%，MAE 变化 +0.0%，CW p=0.8846。仅 1 个月的 CW 支持这组控制变量含有增量信息，但三个期限的原始 RMSE 和 MAE 均上升。该比较检验两项变量的联合贡献，各变量的单独贡献仍待识别。
4. **最优模型随期限变化。** 全期 RMSE 与 MAE 在 1 和 6 个月均由 ARX-GSCPI 领先，3 个月均由随机森林领先。这些排名描述误差大小，显著性依据各自的配对检验。
5. **疫情前后。** h=1 两期的最低 RMSE 模型均为 ARX-GSCPI。h=3 的最低 RMSE 模型由疫情前 AR 变为 2020 年起 Random Forest。h=6 的最低 RMSE 模型由疫情前 Random Forest 变为 2020 年起 ARX-GSCPI。四个模型三个期限的 RMSE 和 MAE 在 2020 年起均高于疫情前。6 个月疫情前的最低 MAE 为 AR，2020 年起为 ARX-GSCPI。这些是按目标月分期的描述性结果，每个期限 2020 年起均有 79 个配对月。

## 复现

各起点的训练行数及最新训练标签写入 [预测 CSV](../data/processed/task2_forecasts.csv)。评价明细见 [误差 CSV](../data/processed/task2_metrics.csv)、[CW CSV](../data/processed/task2_clark_west_tests.csv)和 [DM CSV](../data/processed/task2_dm_tests.csv)。下列推断命令直接读取既有预测，用同一组样本复算检验并生成报告。

```sh
python3 -m pip install -r code/requirements-task2.txt
python3 code/task2_forecast.py --inference-only
python3 -m unittest discover -s code -v
```
