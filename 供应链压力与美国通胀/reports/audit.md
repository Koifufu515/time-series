# Task 0 数据审计

本轮使用随项目保存的纽约联储与 FRED 官方文件快照。观察月按 YYYY-MM 对齐，自最早来源月份至最晚来源月份保留完整日历。

## 数据覆盖与重复检查

交付 CSV 覆盖 1948-01 至 2026-10，共 946 个月。五个水平序列的覆盖交集为 1997-09 至 2026-09，共 349 个月，其中 348 个月五项齐全。这个交集用于描述性检查，正式预测样本由研究设计决定。

| 序列 | 原文件月份范围 | 观测行 | 有效值 | 显式空值 | 范围内漏行 | 重复月 |
|---|---|---:|---:|---:|---:|---:|
| gscpi | 1997-09 至 2026-10 | 350 | 350 | 0 | 0 | 0 |
| pce_goods_index | 1959-01 至 2026-09 | 813 | 813 | 0 | 0 | 0 |
| pce_index | 1959-01 至 2026-09 | 813 | 813 | 0 | 0 | 0 |
| wti_usd_per_barrel | 1986-01 至 2026-10 | 490 | 490 | 0 | 0 | 0 |
| unemployment_pct | 1948-01 至 2026-10 | 946 | 945 | 1 | 0 | 0 |

## 缺失与转换检查

- gscpi 覆盖范围内缺失月份为 无。全表另有 596 个月在其原文件覆盖范围之外。
- pce_goods_index 覆盖范围内缺失月份为 无。全表另有 133 个月在其原文件覆盖范围之外。
- pce_index 覆盖范围内缺失月份为 无。全表另有 133 个月在其原文件覆盖范围之外。
- wti_usd_per_barrel 覆盖范围内缺失月份为 无。全表另有 456 个月在其原文件覆盖范围之外。
- unemployment_pct 覆盖范围内缺失月份为 2025-10。全表另有 0 个月在其原文件覆盖范围之外。

2025 年 10 月联邦政府停摆期间，CPS 家庭调查未采集数据，BLS 没有发布当月估计。该月在 CSV 中保留，失业率留空。 [BLS 对 CPS 的说明](https://www.bls.gov/cps/methods/2025-federal-government-shutdown-impact-cps.htm)

通胀候选列在完整月度日历上计算。基期或当期缺值时，候选值留空。CSV 中空白代表缺失，原始月度数据直接对齐。

| 候选列 | 首个有效月份 | 最后有效月份 | 全表空值 | PCE 来源覆盖内空值 |
|---|---|---|---:|---:|
| pce_goods_mom_pct | 1959-02 | 2026-09 | 134 | 1 |
| pce_goods_yoy_pct | 1960-01 | 2026-09 | 145 | 12 |
| pce_mom_pct | 1959-02 | 2026-09 | 134 | 1 |
| pce_yoy_pct | 1960-01 | 2026-09 | 145 | 12 |

## 单位与异常值检查

两项 PCE 指数须大于 0。失业率的定义域为 0–100%。WTI 以美元/桶计，GSCPI 以标准差计，两者的正负值均按来源保留。下表在共同区间内按各列可用观测筛查。以低于 Q1 − 3×IQR 或高于 Q3 + 3×IQR 为阈值，标记后保留原值。

全表失业率超出定义域的月数为 0。

| 序列 | 最小值及月份 | 最大值及月份 | 标记数 | 标记月份 |
|---|---|---|---:|---|
| gscpi | -1.563 (2023-05) | 4.426 (2021-12) | 16 | 2020-04、2020-07、2021-04、2021-05、2021-06、2021-07、2021-08、2021-09、2021-10、2021-11、2021-12、2022-01、2022-02、2022-03、2022-04、2022-05 |
| pce_goods_index | 91.27 (1999-03) | 120.7 (2026-09) | 0 | 无 |
| pce_index | 70.62 (1997-09) | 132.3 (2026-09) | 0 | 无 |
| wti_usd_per_barrel | 11.35 (1998-12) | 133.9 (2008-06) | 0 | 无 |
| unemployment_pct | 3.4 (2023-04) | 14.8 (2020-04) | 2 | 2020-04、2020-05 |
| pce_goods_mom_pct | -3.459 (2008-11) | 1.954 (2005-09) | 3 | 2005-09、2008-11、2008-12 |
| pce_goods_yoy_pct | -4.538 (2009-07) | 10.75 (2022-03) | 5 | 2022-03、2022-04、2022-05、2022-06、2022-07 |
| pce_mom_pct | -1.18 (2008-11) | 0.9651 (2005-09) | 2 | 2008-10、2008-11 |
| pce_yoy_pct | -1.466 (2009-07) | 7.218 (2022-06) | 6 | 2022-03、2022-05、2022-06、2022-07、2022-08、2022-09 |

## 相关性与预测时点

图中使用共同区间内的同期 Pearson 相关系数。两个面板使用同一批完整观测月份，水平值的趋势和同比窗口的重叠均可能影响系数，图仅供初步描述。

- 环比使用 348 个月（1997-09 至 2026-09）。商品通胀与 GSCPI 的相关系数为 0.334。
- 同比使用 348 个月（1997-09 至 2026-09）。商品通胀与 GSCPI 的相关系数为 0.572。

CSV 中月份是观测期。预测时应另行按预测起点核对各序列发布日期和可用版本。FRED 文件反映抓取时的历史修订值，GSCPI 取下载矩阵中的最新版本，原矩阵保留了可获得的历史版本。

## 原始文件与校验

| 文件 | 官方下载 | GSCPI 版本列标签（月初） | SHA-256 |
|---|---|---|---|
| gscpi_interactive_data.csv | [下载](https://www.newyorkfed.org/medialibrary/research/interactives/data/gscpi/gscpi_interactive_data.csv) | 2026-11-01 | `ef5109319ee888d704a5512ec4790b929c38d07069085138765f1229125f4328` |
| DGDSRG3M086SBEA.csv | [下载](https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGDSRG3M086SBEA) | — | `6a66f0a5839a776f3f151daecfebb840ef98d49fc39e30f35bfff2c6e69337cf` |
| PCEPI.csv | [下载](https://fred.stlouisfed.org/graph/fredgraph.csv?id=PCEPI) | — | `beca75e07e79a0c3293dd068703c966f5fe1adb181d0989fad36877e012f11a2` |
| MCOILWTICO.csv | [下载](https://fred.stlouisfed.org/graph/fredgraph.csv?id=MCOILWTICO) | — | `fee4f983b2e949061982535e4d7a399602ddfdcbd12cb272770364f82fb5f3ab` |
| UNRATE.csv | [下载](https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE) | — | `fb669461abc34e00324b940be6bad1c7f89117652f28587196a7b8bf4a9f56a1` |

## 复现

在项目目录运行以下命令。默认读取随附快照。安装 curl 后可运行 `python3 code/prepare_data.py --download` 刷新官方文件，刷新可能纳入后续发布和历史修订。

```sh
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r code/requirements.txt
python3 code/prepare_data.py
python3 -m unittest discover -s code -v
```
