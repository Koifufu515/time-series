# Task 3 稳健性分析与复算说明

使用 Task 2.1 冻结预测和 Task 0、1 月度快照。各试验按观测月索引进行伪样本外预测，历史实际发布日期和首次发布值未重建。模型按期限从共同初始 180 条有效记录以 AIC 比较 p=1…6，训练标签满足 s+h≤t，缺失起点由组内全体模型跳过。

新增 CW 使用右尾正态近似和 h−1 阶 Bartlett HAC。仅配对同一连续日期区段内恰好相隔对应月数的观测，各阶协方差以总配对样本数为分母。Holm 直接使用冻结的 Task 2.1 p 值，两个家族各含三个期限。

## Holm 与时期异质性

| 受限 | 非受限 | h | 原始 p | Holm p | 原始拒绝 | 校正拒绝 |
| --- | --- | --- | --- | --- | --- | --- |
| AR | ARX-Base | 1 | 0.0011 | 0.0032 | 是 | 是 |
| AR | ARX-Base | 3 | 0.9633 | 1.0000 | 否 | 否 |
| AR | ARX-Base | 6 | 0.8846 | 1.0000 | 否 | 否 |
| ARX-Base | ARX-GSCPI | 1 | 0.0014 | 0.0042 | 是 | 是 |
| ARX-Base | ARX-GSCPI | 3 | 0.0138 | 0.0276 | 是 | 是 |
| ARX-Base | ARX-GSCPI | 6 | 0.0332 | 0.0332 | 是 | 是 |

| 目标期 | h | n | 调整差 | CW | 单侧 p |
| --- | --- | --- | --- | --- | --- |
| Pre-COVID | 1 | 87 | +0.00331 | +0.876 | 0.1906 |
| Post-COVID | 1 | 79 | +0.08518 | +2.973 | 0.0015 |
| Pre-COVID | 3 | 83 | -0.00020 | -0.221 | 0.5876 |
| Post-COVID | 3 | 79 | +0.11581 | +2.319 | 0.0102 |
| Pre-COVID | 6 | 77 | +0.01024 | +1.427 | 0.0768 |
| Post-COVID | 6 | 79 | +0.13163 | +1.812 | 0.0350 |

## 三项稳健性重估

A 使用整体 PCE 自己的历史及初始样本选阶，评价月与主规格相同。B 从 1997-10 统一开始建模，GSCPI 的滞后值和月差损失的首月留空。C 固定窗口每次使用最近 180 条完整且标签已可用的训练记录，因缺月可能横跨超过 180 个日历月。C 扩展窗口逐月重新拟合，日期、训练样本和预测值与 T2 已核对。

相对改善为 100×(1−模型误差/同组基准误差)，正值表示误差变小。新增检验的逐项 p 值用于稳健性描述。

| 试验 | 窗口 | h | 模型 | n | p 阶 | RMSE | MAE | RMSE 改善% | MAE 改善% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Headline PCE | Expanding | 1 | ARX-Base | 166 | 1 | 0.1767 | 0.1317 | +0.00 | +0.00 |
| Headline PCE | Expanding | 1 | ARX-GSCPI | 166 | 1 | 0.1622 | 0.1218 | +8.18 | +7.54 |
| GSCPI specification | Expanding | 1 | ARX-Base | 165 | 2 | 0.4194 | 0.3163 | +0.00 | +0.00 |
| GSCPI specification | Expanding | 1 | ARX-GSCPI-Level | 165 | 2 | 0.3874 | 0.2870 | +7.65 | +9.26 |
| GSCPI specification | Expanding | 1 | ARX-GSCPI-Lag1 | 165 | 2 | 0.3930 | 0.2945 | +6.30 | +6.89 |
| GSCPI specification | Expanding | 1 | ARX-GSCPI-Diff | 165 | 2 | 0.4198 | 0.3163 | -0.10 | +0.02 |
| Training window | Expanding | 1 | ARX-Base | 166 | 2 | 0.4186 | 0.3159 | +0.00 | +0.00 |
| Training window | Expanding | 1 | ARX-GSCPI | 166 | 2 | 0.3867 | 0.2869 | +7.61 | +9.18 |
| Training window | Rolling | 1 | ARX-Base | 166 | 2 | 0.4403 | 0.3219 | +0.00 | +0.00 |
| Training window | Rolling | 1 | ARX-GSCPI | 166 | 2 | 0.4101 | 0.2931 | +6.85 | +8.95 |
| Headline PCE | Expanding | 3 | ARX-Base | 162 | 1 | 0.2593 | 0.1661 | +0.00 | +0.00 |
| Headline PCE | Expanding | 3 | ARX-GSCPI | 162 | 1 | 0.2495 | 0.1598 | +3.78 | +3.77 |
| GSCPI specification | Expanding | 3 | ARX-Base | 161 | 1 | 0.5320 | 0.3798 | +0.00 | +0.00 |
| GSCPI specification | Expanding | 3 | ARX-GSCPI-Level | 161 | 1 | 0.4964 | 0.3618 | +6.69 | +4.75 |
| GSCPI specification | Expanding | 3 | ARX-GSCPI-Lag1 | 161 | 1 | 0.5154 | 0.3708 | +3.11 | +2.39 |
| GSCPI specification | Expanding | 3 | ARX-GSCPI-Diff | 161 | 1 | 0.5324 | 0.3794 | -0.08 | +0.11 |
| Training window | Expanding | 3 | ARX-Base | 162 | 1 | 0.5326 | 0.3806 | +0.00 | +0.00 |
| Training window | Expanding | 3 | ARX-GSCPI | 162 | 1 | 0.4969 | 0.3626 | +6.70 | +4.75 |
| Training window | Rolling | 3 | ARX-Base | 162 | 1 | 0.5303 | 0.3816 | +0.00 | +0.00 |
| Training window | Rolling | 3 | ARX-GSCPI | 162 | 1 | 0.4988 | 0.3617 | +5.94 | +5.21 |
| Headline PCE | Expanding | 6 | ARX-Base | 156 | 1 | 0.2065 | 0.1444 | +0.00 | +0.00 |
| Headline PCE | Expanding | 6 | ARX-GSCPI | 156 | 1 | 0.1868 | 0.1365 | +9.56 | +5.45 |
| GSCPI specification | Expanding | 6 | ARX-Base | 155 | 1 | 0.4657 | 0.3392 | +0.00 | +0.00 |
| GSCPI specification | Expanding | 6 | ARX-GSCPI-Level | 155 | 1 | 0.4360 | 0.3343 | +6.38 | +1.43 |
| GSCPI specification | Expanding | 6 | ARX-GSCPI-Lag1 | 155 | 1 | 0.4450 | 0.3404 | +4.44 | -0.36 |
| GSCPI specification | Expanding | 6 | ARX-GSCPI-Diff | 155 | 1 | 0.4650 | 0.3384 | +0.14 | +0.22 |
| Training window | Expanding | 6 | ARX-Base | 156 | 1 | 0.4640 | 0.3365 | +0.00 | +0.00 |
| Training window | Expanding | 6 | ARX-GSCPI | 156 | 1 | 0.4337 | 0.3317 | +6.54 | +1.42 |
| Training window | Rolling | 6 | ARX-Base | 156 | 1 | 0.5045 | 0.3626 | +0.00 | +0.00 |
| Training window | Rolling | 6 | ARX-GSCPI | 156 | 1 | 0.4801 | 0.3549 | +4.84 | +2.11 |

| 试验 | 窗口 | h | 非受限模型 | n | 调整差 | CW | 单侧 p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Headline PCE | Expanding | 1 | ARX-GSCPI | 166 | +0.00684 | +3.258 | 0.0006 |
| GSCPI specification | Expanding | 1 | ARX-GSCPI-Level | 165 | +0.04257 | +2.997 | 0.0014 |
| GSCPI specification | Expanding | 1 | ARX-GSCPI-Lag1 | 165 | +0.03443 | +2.638 | 0.0042 |
| GSCPI specification | Expanding | 1 | ARX-GSCPI-Diff | 165 | +0.00065 | +0.220 | 0.4128 |
| Training window | Expanding | 1 | ARX-GSCPI | 166 | +0.04227 | +2.989 | 0.0014 |
| Training window | Rolling | 1 | ARX-GSCPI | 166 | +0.04164 | +3.241 | 0.0006 |
| Headline PCE | Expanding | 3 | ARX-GSCPI | 162 | +0.00822 | +1.567 | 0.0585 |
| GSCPI specification | Expanding | 3 | ARX-GSCPI-Level | 161 | +0.05623 | +2.195 | 0.0141 |
| GSCPI specification | Expanding | 3 | ARX-GSCPI-Lag1 | 161 | +0.03257 | +1.525 | 0.0636 |
| GSCPI specification | Expanding | 3 | ARX-GSCPI-Diff | 161 | +0.00208 | +0.471 | 0.3187 |
| Training window | Expanding | 3 | ARX-GSCPI | 162 | +0.05637 | +2.202 | 0.0138 |
| Training window | Rolling | 3 | ARX-GSCPI | 162 | +0.05681 | +2.086 | 0.0185 |
| Headline PCE | Expanding | 6 | ARX-GSCPI | 156 | +0.01246 | +1.738 | 0.0411 |
| GSCPI specification | Expanding | 6 | ARX-GSCPI-Level | 155 | +0.07065 | +1.822 | 0.0343 |
| GSCPI specification | Expanding | 6 | ARX-GSCPI-Lag1 | 155 | +0.05582 | +1.726 | 0.0422 |
| GSCPI specification | Expanding | 6 | ARX-GSCPI-Diff | 155 | +0.00144 | +0.695 | 0.2434 |
| Training window | Expanding | 6 | ARX-GSCPI | 156 | +0.07171 | +1.837 | 0.0331 |
| Training window | Rolling | 6 | ARX-GSCPI | 156 | +0.07340 | +1.709 | 0.0437 |

逐月结果及各次训练边界见 [task3_forecasts.csv](../data/processed/task3_forecasts.csv)。

## 复现

在项目根目录执行以下命令。报告排版使用 Typst。

~~~sh
python3 -m pip install -r code/requirements-task2.txt
python3 code/task3_robustness.py
python3 code/task3_report.py
python3 -m unittest discover -s code -v
~~~
