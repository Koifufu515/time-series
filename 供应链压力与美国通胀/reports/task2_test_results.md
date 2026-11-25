# Task 2.1 测试与复现验收

## 运行环境

- Python 3.10.11
- numpy 2.2.6
- pandas 2.3.3
- scipy 1.15.3
- statsmodels 0.14.6
- scikit-learn 1.7.2
- matplotlib 3.10.8
- joblib 1.5.3
- threadpoolctl 3.6.0

## 数据与复算核验

- 冻结预测 CSV 的 SHA-256 为 e12c4d1f3aa5e617165e4e7f51e75bad1043c33bc2838abb930e82d6a92a1a43。
- 从冻结预测 CSV 复跑推断，Clark–West、DM 和报告三个文件逐字节一致。
- 用标准库按实际配对月份独立复算六项 Clark–West 和三项 DM，统计量、p 值及损失差与 CSV 一致。

## 单元测试

### T1

命令为 `python3 -m unittest discover -s code -p test_task1_eda.py -v`。共 7 项，7 项通过，0 项失败。

```text
test_calendar_and_preservation (test_task1_eda.Task1Tests) ... ok
test_cross_correlation_direction_counts_and_windows (test_task1_eda.Task1Tests) ... ok
test_diagnostics_record_inputs_and_lags (test_task1_eda.Task1Tests) ... ok
test_fisher_excess_kurtosis_matches_independent_estimator (test_task1_eda.Task1Tests) ... ok
test_missing_does_not_bridge_the_gap (test_task1_eda.Task1Tests) ... ok
test_report_fingerprints_task0_snapshot (test_task1_eda.Task1Tests) ... ok
test_transforms_use_only_current_and_past_calendar_values (test_task1_eda.Task1Tests) ... ok

----------------------------------------------------------------------
Ran 7 tests in 0.093s

OK
```

### T2

命令为 `python3 -m unittest discover -s code -p test_task2_forecast.py -v`。共 12 项，12 项通过，0 项失败。

```text
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

----------------------------------------------------------------------
Ran 12 tests in 0.275s

OK
```

### 完整测试

命令为 `python3 -m unittest discover -s code -v`。共 23 项，23 项通过，0 项失败。

```text
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

----------------------------------------------------------------------
Ran 23 tests in 0.381s

OK
```
