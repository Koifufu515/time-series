import unittest

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from task2_forecast import (
    BREAK, CONTROLS, GSCPI, HORIZONS, INITIAL_N, LAGS, MODELS, ROOT, TARGET,
    calculate_metrics, common_rows, direct_frame, dm_results, dm_statistic,
    load_inputs, ols_fit, rf_splits, select_order, subperiod, training_at,
)


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.monthly, cls.task1 = load_inputs(ROOT)

    def test_forecast_direction_and_history(self):
        for h in HORIZONS:
            frame = direct_frame(self.monthly, self.task1, h)
            self.assertTrue(frame.index.equals(self.task1.index))
            self.assertEqual(frame.target_month.tolist(), (frame.index + h).tolist())
            first = frame.index[0]
            for i, lag in enumerate(LAGS):
                self.assertEqual(frame.loc[first, lag], self.monthly.loc[first - i, TARGET])
            self.assertEqual(frame.loc[first, "actual"], self.task1.loc[first + h, TARGET])
            self.assertTrue(frame.actual.iloc[-h:].isna().all())
            self.assertTrue(frame.predictors_ready.iloc[0])

    def test_training_cutoff_and_missing_months(self):
        missing = {pd.Period(m, freq="M") for m in ("2025-10", "2025-11")}
        for h in HORIZONS:
            frame = direct_frame(self.monthly, self.task1, h)
            rows = common_rows(frame)
            self.assertEqual(set(frame.index[~frame.predictors_ready]), missing)
            initial = rows.iloc[:INITIAL_N]
            origin = initial.target_month.iloc[-1]
            self.assertEqual(str(initial.index[-1]), "2012-08")
            pd.testing.assert_frame_equal(training_at(rows, origin), initial)
            for origin in pd.period_range(origin, self.task1.index[-1] - h, freq="M"):
                train = training_at(rows, origin)
                self.assertTrue((train.target_month <= origin).all())
                self.assertTrue((train.index <= origin - h).all())
                self.assertFalse(bool(set(train.index) & missing))

    def test_future_mutation_cannot_change_fit_or_origin_features(self):
        origin = pd.Period("2020-08", freq="M")
        future_monthly = self.monthly.copy()
        future_task1 = self.task1.copy()
        future_monthly.loc[origin + 1:, TARGET] = 1_000
        future_task1.loc[origin + 1:, [TARGET, *CONTROLS, GSCPI]] = -1_000
        features = [*LAGS, *CONTROLS, GSCPI]
        for h in HORIZONS:
            original = direct_frame(self.monthly, self.task1, h)
            edited = direct_frame(future_monthly, future_task1, h)
            train = training_at(common_rows(original), origin)
            edited_train = training_at(common_rows(edited), origin)
            pd.testing.assert_frame_equal(train, edited_train)
            pd.testing.assert_series_equal(original.loc[origin, features], edited.loc[origin, features])
            self.assertNotEqual(original.loc[origin, "actual"], edited.loc[origin, "actual"])
            np.testing.assert_array_equal(ols_fit(train, features)[0], ols_fit(edited_train, features)[0])
            a = RandomForestRegressor(n_estimators=5, random_state=42).fit(train[features], train.actual)
            b = RandomForestRegressor(n_estimators=5, random_state=42).fit(edited_train[features],
                                                                          edited_train.actual)
            np.testing.assert_array_equal(a.predict(original.loc[[origin], features]),
                                          b.predict(edited.loc[[origin], features]))

    def test_aic_and_cv_use_initial_rows_with_calendar_gap(self):
        for h in HORIZONS:
            rows = common_rows(direct_frame(self.monthly, self.task1, h))
            initial = rows.iloc[:INITIAL_N]
            p, scores = select_order(initial)
            self.assertEqual([order for order, _ in scores], list(range(1, 7)))
            self.assertEqual(p, min(scores, key=lambda entry: entry[1])[0])
            folds = rf_splits(initial, h)
            self.assertEqual(len(folds), 5)
            for train, valid in folds:
                self.assertEqual(valid[0] - train[-1] - 1, h)
                self.assertLess(initial.target_month.iloc[train[-1]], initial.index[valid[0]])
                self.assertLess(valid[-1], INITIAL_N)

    def test_dm_uses_calendar_lags_and_squared_error_direction(self):
        months = pd.PeriodIndex(["2020-01", "2020-02", "2020-04", "2020-05"], freq="M")
        loss = pd.Series([0.0, 1.0, 3.0, 4.0], index=months)
        # Centered differences are -2, -1, +1, +2. Lag 1 products are 2 and 2.
        expected = 2 / np.sqrt((2.5 + 2 * 0.5 * 1.0) / 4)
        stat, p = dm_statistic(loss, 2)
        self.assertAlmostEqual(stat, expected)
        self.assertAlmostEqual(dm_statistic(-loss, 2)[0], -stat)
        self.assertAlmostEqual(dm_statistic(-loss, 2)[1], p)
        self.assertFalse(np.isfinite(dm_statistic(pd.Series(0.0, index=months), 2)[0]))

    def test_target_month_controls_period_and_metric_sign(self):
        months = pd.PeriodIndex(["2019-12", "2020-01"], freq="M")
        self.assertEqual(list(subperiod(months)), ["Pre-COVID", "Post-COVID"])
        frame = pd.DataFrame({"horizon": 1, "model": "AR", "target_month": months,
                              "actual": [2.0, 1.0], "prediction": [1.0, 3.0]})
        result = calculate_metrics(frame).set_index("period")
        self.assertEqual(result.loc["Full", "mean_error"], -0.5)
        self.assertEqual(result.loc["Pre-COVID", "rmse"], 1.0)
        self.assertEqual(result.loc["Post-COVID", "mae"], 2.0)

    def test_delivered_forecasts_have_common_dates_and_traceable_training(self):
        data = ROOT / "data" / "processed"
        f = pd.read_csv(data / "task2_forecasts.csv")
        f["forecast_origin"] = pd.PeriodIndex(f.forecast_origin, freq="M")
        f["target_month"] = pd.PeriodIndex(f.target_month, freq="M")
        f["train_last_target_month"] = pd.PeriodIndex(f.train_last_target_month, freq="M")
        self.assertFalse(f.duplicated(["forecast_origin", "horizon", "model"]).any())
        self.assertTrue(np.isfinite(f[["actual", "prediction"]]).all().all())
        for h in HORIZONS:
            part = f.loc[f.horizon.eq(h)]
            self.assertEqual(set(part.model), set(MODELS))
            groups = part.groupby("forecast_origin")
            self.assertTrue(groups.model.nunique().eq(len(MODELS)).all())
            self.assertTrue(groups.actual.nunique().eq(1).all())
            self.assertTrue(groups.train_n.nunique().eq(1).all())
            self.assertTrue((part.forecast_origin + h == part.target_month).all())
            self.assertTrue((part.train_last_target_month <= part.forecast_origin).all())
            rows = common_rows(direct_frame(self.monthly, self.task1, h))
            for origin, current in groups:
                eligible = training_at(rows, origin)
                self.assertEqual(current.train_n.iloc[0], len(eligible))
                self.assertEqual(current.train_last_target_month.iloc[0], eligible.target_month.iloc[-1])
                self.assertEqual(current.actual.iloc[0], rows.loc[origin, "actual"])
            self.assertEqual(part.train_n.min(), INITIAL_N)
            self.assertEqual(groups.ngroups, {1: 166, 3: 162, 6: 156}[h])
        expected = pd.read_csv(data / "task2_metrics.csv")
        actual = calculate_metrics(f)
        np.testing.assert_allclose(actual[["rmse", "mae", "mean_error"]],
                                   expected[["rmse", "mae", "mean_error"]], atol=1e-10)
        expected = pd.read_csv(data / "task2_dm_tests.csv")
        actual = dm_results(f)
        np.testing.assert_allclose(actual[["dm_stat", "p_value"]],
                                   expected[["dm_stat", "p_value"]], atol=1e-10)
        self.assertEqual(str(BREAK), "2020-01")


if __name__ == "__main__":
    unittest.main()
