# 最终验收与测试记录

## 环境

- Python 3.10.11
- numpy 2.2.6
- pandas 2.3.3
- scipy 1.15.3
- statsmodels 0.14.6
- scikit-learn 1.7.2
- matplotlib 3.10.8
- typst 0.15.1 (unknown commit)

## 验收结果

- 执行 python3 -m unittest discover -s code -v，33 项通过，0 项失败。
- 从基础快照在独立目录重算稳健性预测、统计表和图，逐字节匹配最终版。主规格推断与稳健性扩展窗口的 CW 对账通过。
- 相比指定基线，所有数据 CSV、图表、参考文献 PDF 与课程原件保持字节一致。
- Markdown 与 PDF 可确定性复建；核对了 14 张表、272 个小数或科学记数值和 6 张核心图。
- A4 PDF 共 20 页，封面独立，页码自摘要开始；确认中文宋体、英文字体和数学字体嵌入，并完成逐页渲染检查。

## 原始测试输出

~~~text
test_all_candidate_rates (test_prepare_data.AuditTests) ... ok
test_calendar_and_source_values (test_prepare_data.AuditTests) ... ok
test_calendar_gap_does_not_become_one_month_change (test_prepare_data.AuditTests) ... ok
test_duplicate_month_stops_processing (test_prepare_data.AuditTests) ... ok
test_calendar_and_preservation (test_task1_eda.Task1Tests) ... ok
test_cross_correlation_direction_counts_and_windows (test_task1_eda.Task1Tests) ... ok
test_diagnostics_record_inputs_and_lags (test_task1_eda.Task1Tests) ... ok
test_fisher_excess_kurtosis_matches_independent_estimator (test_task1_eda.Task1Tests) ... ok
test_missing_does_not_bridge_the_gap (test_task1_eda.Task1Tests) ... ok
test_report_fingerprints_task0_snapshot (test_task1_eda.Task1Tests) ... ok
test_transforms_use_only_current_and_past_calendar_values (test_task1_eda.Task1Tests) ... ok
test_aic_and_cv_use_initial_rows_with_calendar_gap (test_task2_forecast.ForecastTests) ... ok
test_clark_west_direction_and_one_sided_tail (test_task2_forecast.ForecastTests) ... ok
test_clark_west_hac_reaches_h_minus_one_for_each_horizon (test_task2_forecast.ForecastTests) ... ok
test_clark_west_hac_uses_calendar_pairs_across_gaps (test_task2_forecast.ForecastTests) ... ok
test_clark_west_squared_forecast_difference_adjustment (test_task2_forecast.ForecastTests) ... ok
test_delivered_forecasts_have_common_dates_and_traceable_training (test_task2_forecast.ForecastTests) ... ok
test_dm_uses_calendar_lags_and_squared_error_direction (test_task2_forecast.ForecastTests) ... ok
test_forecast_direction_and_history (test_task2_forecast.ForecastTests) ... ok
test_future_mutation_cannot_change_fit_or_origin_features (test_task2_forecast.ForecastTests) ... ok
test_pairing_rejects_misaligned_forecasts (test_task2_forecast.ForecastTests) ... ok
test_target_month_controls_period_and_metric_sign (test_task2_forecast.ForecastTests) ... ok
test_training_cutoff_and_missing_months (test_task2_forecast.ForecastTests) ... ok
test_cw_pairs_by_calendar_lag_across_missing_month (test_task3_robustness.RobustnessTests) ... ok
test_expanding_inference_matches_task2_1_exactly (test_task3_robustness.RobustnessTests) ... ok
test_headline_target_changes_both_history_and_response (test_task3_robustness.RobustnessTests) ... ok
test_holm_is_separate_by_family (test_task3_robustness.RobustnessTests) ... ok
test_holm_matches_statsmodels_with_unsorted_ties (test_task3_robustness.RobustnessTests) ... ok
test_report_renders_tables_from_results (test_task3_robustness.RobustnessTests) ... ok
test_reported_metrics_and_cw_recalculate_from_saved_predictions (test_task3_robustness.RobustnessTests) ... ok
test_stored_forecasts_share_dates_and_respect_training_windows (test_task3_robustness.RobustnessTests) ... ok
test_subperiod_pairs_use_target_month (test_task3_robustness.RobustnessTests) ... ok
test_transformed_gscpi_drops_first_month_and_uses_only_past (test_task3_robustness.RobustnessTests) ... ok

----------------------------------------------------------------------
Ran 33 tests in 0.745s

OK
~~~
