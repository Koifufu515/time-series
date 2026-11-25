import unittest

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from task2_forecast import GSCPI, HORIZONS, LAGS, ROOT, TARGET, load_inputs, read_forecasts, training_at
from task3_report import report_tables, typst_body
from task3_robustness import (BASE, HEADLINE, LEVEL, TRANSFORMS, experiment_rows, holm_adjust,
                              holm_table, segmented_cw, subperiod_cw, summarize)


class RobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = ROOT / "data/processed"
        cls.monthly, cls.task1 = load_inputs(ROOT)
        cls.frozen = read_forecasts(cls.data / "task2_forecasts.csv")
        cls.new = read_forecasts(cls.data / "task3_forecasts.csv")

    def test_holm_matches_statsmodels_with_unsorted_ties(self):
        p = [.063, .05, .003, .05, .91]
        np.testing.assert_allclose(holm_adjust(p), multipletests(p, method="holm")[1])
        with self.assertRaises(ValueError):
            holm_adjust([.02, np.nan])

    def test_holm_is_separate_by_family(self):
        old = pd.read_csv(self.data / "task2_clark_west_tests.csv")
        new = holm_table(old)
        for _, family in new.groupby("restricted_model"):
            self.assertEqual(list(family.horizon), list(HORIZONS))
            np.testing.assert_allclose(family.holm_p, multipletests(family.raw_p, method="holm")[1])
        saved = pd.read_csv(self.data / "task3_holm.csv")
        np.testing.assert_allclose(new.holm_p, saved.holm_p)

    def test_cw_omits_pairs_spanning_a_missing_calendar_month(self):
        idx = pd.PeriodIndex(["2020-01", "2020-02", "2020-04", "2020-05"], freq="M")
        loss = pd.Series([0., 1., 3., 4.], index=idx)
        # Lag one contributes 2+2. The nominal lag-two Feb–Apr pair crosses March.
        lrv = 2.5 + 2 * (2/3) * (4/4)
        self.assertAlmostEqual(segmented_cw(loss, 3)[0], 2 / np.sqrt(lrv / 4))

    def test_subperiod_pairs_use_target_month(self):
        cw = subperiod_cw(self.frozen)
        self.assertEqual(list(cw.n), [87, 79, 83, 79, 77, 79])
        self.assertTrue(cw.loc[cw.period.eq("Pre-COVID"), "target_end"].eq(pd.Period("2019-12", "M")).all())
        self.assertTrue(cw.loc[cw.period.eq("Post-COVID"), "target_start"].eq(pd.Period("2020-01", "M")).all())

    def test_headline_target_changes_both_history_and_response(self):
        rows = experiment_rows(self.monthly, self.task1, 3, HEADLINE, [GSCPI])
        t = rows.index[0]
        self.assertEqual(rows.loc[t, "actual"], self.task1.loc[t + 3, HEADLINE])
        for i, field in enumerate(LAGS):
            self.assertEqual(rows.loc[t, field], self.monthly.loc[t - i, HEADLINE])

    def test_transformed_gscpi_drops_first_month_and_uses_only_past(self):
        rows = experiment_rows(self.monthly, self.task1, 1, TARGET, TRANSFORMS.values())
        t = rows.index[0]
        self.assertEqual(str(t), "1997-10")
        self.assertEqual(rows.loc[t, "gscpi_lag1"], self.monthly.loc[t - 1, GSCPI])
        self.assertEqual(rows.loc[t, "gscpi_diff"], self.monthly.loc[t, GSCPI] - self.monthly.loc[t - 1, GSCPI])
        altered = self.monthly.copy()
        altered.loc[t + 1:, GSCPI] = 1000
        changed = experiment_rows(altered, self.task1, 1, TARGET, TRANSFORMS.values())
        pd.testing.assert_series_equal(rows.loc[t], changed.loc[t])

    def test_stored_forecasts_share_dates_and_respect_training_windows(self):
        self.assertEqual(len(self.new), 4828)
        for (experiment, window, h), part in self.new.groupby(["experiment", "window", "horizon"]):
            self.assertTrue((part.forecast_origin + h == part.target_month).all())
            self.assertFalse(part.duplicated(["forecast_origin", "model"]).any())
            expected_n = 4 if experiment == "GSCPI specification" else 2
            self.assertTrue(part.groupby("forecast_origin").model.nunique().eq(expected_n).all())
            self.assertTrue(part.groupby("forecast_origin").train_n.nunique().eq(1).all())
            self.assertEqual(part.forecast_origin.nunique(), {1:166, 3:162, 6:156}[h]
                             - (experiment == "GSCPI specification"))
            fields = TRANSFORMS.values() if experiment == "GSCPI specification" else [GSCPI]
            target = HEADLINE if experiment == "Headline PCE" else TARGET
            rows = experiment_rows(self.monthly, self.task1, h, target, fields)
            for row in part.loc[part.model.eq(BASE)].itertuples():
                train = training_at(rows, row.forecast_origin)
                if window == "Rolling":
                    train = train.iloc[-180:]
                    self.assertEqual(len(train), 180)
                self.assertEqual(row.train_first_origin, str(train.index[0]))
                self.assertEqual(row.train_n, len(train))
                self.assertEqual(row.train_last_target_month, train.target_month.iloc[-1])

    def test_report_renders_tables_from_results(self):
        tables = report_tables()
        holm = pd.read_csv(self.data / "task3_holm.csv")
        for r in holm.itertuples(index=False):
            self.assertIn(f"| {r.raw_p:.4f} | {r.holm_p:.4f} |", tables["HOLM"])
        body = typst_body("## 测试\n\n" + tables["HOLM"] + "\n\n<!-- pagebreak -->")
        self.assertIn("#gridtable(6,", body)
        self.assertIn("#pagebreak(weak: true)", body)

    def test_reported_metrics_and_cw_recalculate_from_saved_predictions(self):
        metrics, tests = summarize(self.new)
        for name, stem, cw_name in [
            ("Headline PCE", "headline_pce", "headline_pce_cw"),
            ("GSCPI specification", "gscpi_specification", "gscpi_specification"),
            ("Training window", "window", "window_cw"),
        ]:
            for frame, path, keys in [
                (metrics, f"{stem}_metrics", ["rmse", "mae", "rmse_improvement_pct", "mae_improvement_pct"]),
                (tests, cw_name, ["cw_stat", "p_value_one_sided", "mean_adjusted_loss_diff"]),
            ]:
                saved = pd.read_csv(self.data / f"task3_{path}.csv")
                np.testing.assert_allclose(frame.loc[frame.experiment.eq(name), keys], saved[keys], atol=1e-9)


if __name__ == "__main__":
    unittest.main()
