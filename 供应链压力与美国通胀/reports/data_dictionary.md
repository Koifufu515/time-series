# 数据字典

| 字段 | 含义与单位 | 来源或公式 |
|---|---|---|
| month | 观测月份，YYYY-MM | 统一使用日历月 |
| gscpi | 全球供应链压力指数，相对历史均值的标准差（标准差，官方发布口径） | [官方来源](https://www.newyorkfed.org/research/policy/gscpi) |
| pce_goods_index | 美国 PCE 商品链式价格指数（2017=100，季调） | [官方来源](https://fred.stlouisfed.org/series/DGDSRG3M086SBEA) |
| pce_index | 美国整体 PCE 链式价格指数（2017=100，季调） | [官方来源](https://fred.stlouisfed.org/series/PCEPI) |
| wti_usd_per_barrel | 库欣 WTI 现货月度价格（美元/桶，未季调） | [官方来源](https://fred.stlouisfed.org/series/MCOILWTICO) |
| unemployment_pct | 美国 U-3 失业率（%，季调） | [官方来源](https://fred.stlouisfed.org/series/UNRATE) |
| pce_goods_mom_pct | PCE goods price index 环比候选值，%，非年化 | 100 × (pce_goods_index[t] / pce_goods_index[t−1] − 1) |
| pce_goods_yoy_pct | PCE goods price index 同比候选值，%，非年化 | 100 × (pce_goods_index[t] / pce_goods_index[t−12] − 1) |
| pce_mom_pct | Headline PCE price index 环比候选值，%，非年化 | 100 × (pce_index[t] / pce_index[t−1] − 1) |
| pce_yoy_pct | Headline PCE price index 同比候选值，%，非年化 | 100 × (pce_index[t] / pce_index[t−12] − 1) |

商品 PCE 是拟研究的预测对象。环比、同比均供口径审核。解释变量保留 GSCPI、WTI 和失业率水平口径。整体 PCE 一并交付供描述性比较。
