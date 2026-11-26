# Global Supply Chain Pressure and U.S. Inflation Forecasting

时间序列课程项目。用 1997-09 至 2026-09 的月度数据检验 GSCPI 是否改善美国商品 PCE 环比的 1、3、6 个月预测，并比较 AR、两种 ARX 与随机森林。

- 相对 ARX-Base，加入 GSCPI 后的全期 RMSE 分别下降 7.61%、6.70%、6.54%，Holm 校正的 Clark–West p 值分别为 0.0042、0.0276、0.0332。
- 随机森林在 3 个月原始误差排名第一，三个期限的 DM 检验均未达 5%。
- GSCPI 水平的增量在滚动窗保留，月差证据较弱。2020 年起的分期证据更集中。本研究采用修订快照与观测月索引。

## 文件索引

- [课程原件](2026Project.pdf)
- [最终报告 PDF](供应链压力与美国通胀/reports/final_report.pdf) · [Markdown](供应链压力与美国通胀/reports/final_report.md)
- [课程要求核对](供应链压力与美国通胀/reports/final_checklist.md) · [汇报交接](供应链压力与美国通胀/reports/presentation_brief.md) · [测试记录](供应链压力与美国通胀/reports/final_test_results.md)
- 项目目录下，code 存模型、报告生成器和测试，data/raw 存官方快照，data/processed 存预测和统计表，figures 存图表，参考文献存论文。

## 复现

验收环境为 Python 3.10.11、Typst 0.15.1、Songti SC、STIX Two Text 与 STIX Two Math。依赖文件固定 Python 包版本。以下命令从仓库根目录开始，使用项目内的快照：

~~~sh
cd '供应链压力与美国通胀'
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r code/requirements-task2.txt
python3 code/prepare_data.py
python3 code/task1_eda.py
python3 code/task2_forecast.py
python3 code/task2_forecast.py --inference-only
python3 code/task3_robustness.py
python3 code/task3_report.py
python3 -m unittest discover -s code -v
~~~

只生成论文时，安装依赖后运行报告生成器即可，它从当前冻结的 CSV 提取数据。跨平台复现的舍入和字体渲染可能有细微差异。

## 课程提交

补齐封面、成员实际贡献及 AI 使用记录中的人工核验部分。由小组确认版次后生成 ZIP，并在现场汇报前上传课程网站。对已提交的交付版本可执行：

~~~sh
git archive --format=zip --output=../time-series-course-submission.zip HEAD
~~~
