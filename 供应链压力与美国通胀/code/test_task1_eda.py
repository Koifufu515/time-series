import hashlib
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from task1_eda import (HORIZONS, TARGETS, WINDOWS, derive, describe, lead_correlations, load_panel,
                      logged_change, longest_contiguous, serial_correlations, stationarity)


ROOT = Path(__file__).resolve().parents[1]


class Task1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = ROOT / "data" / "monthly.csv"
        cls.raw = load_panel(cls.source)
        cls.panel = derive(cls.raw)

    def test_calendar_and_preservation(self):
        index = pd.period_range("1997-09", "2026-09", freq="M", name="month")
        pd.testing.assert_index_equal(self.panel.index, index)
        pd.testing.assert_frame_equal(self.panel[self.raw.columns], self.raw.loc[index])
        self.assertEqual(self.panel.unemployment_pct.idxmax(), pd.Period("2020-04", "M"))
        self.assertEqual(self.panel.unemployment_pct.max(), 14.8)

    def test_missing_does_not_bridge_the_gap(self):
        self.assertEqual(list(self.panel.index[self.panel.unemployment_pct.isna()].astype(str)),
                         ["2025-10"])
        self.assertEqual(list(self.panel.index[self.panel.unemployment_change_pp.isna()].astype(str)),
                         ["2025-10", "2025-11"])
        self.assertAlmostEqual(self.panel.loc["2025-12", "unemployment_change_pp"],
                               self.raw.loc["2025-12", "unemployment_pct"] -
                               self.raw.loc["2025-11", "unemployment_pct"])
        run = longest_contiguous(self.panel.unemployment_pct)
        self.assertEqual((str(run.index[-1]), len(run)), ("2025-09", 337))
        with self.assertRaisesRegex(ValueError, "monthly calendar"):
            longest_contiguous(self.panel.unemployment_pct.dropna())

    def test_transforms_use_only_current_and_past_calendar_values(self):
        prefix = derive(self.raw.loc[:"2019-12"])
        pd.testing.assert_frame_equal(prefix, self.panel.loc[:"2019-12"])
        first = self.panel.iloc[0]
        for level, change, scale in [
            ("pce_goods_index", "goods_log_mom_annualized", 1200),
            ("wti_usd_per_barrel", "wti_log_mom_pct", 100),
        ]:
            base = self.raw.loc["1997-08", level]
            self.assertAlmostEqual(first[change], scale * np.log(first[level] / base))
        values = pd.Series([10., 0., 20., 40., np.nan, 50.])
        self.assertEqual(list(logged_change(values, 100).dropna().index), [3])

    def test_cross_correlation_direction_counts_and_windows(self):
        table = lead_correlations(self.panel)
        records = table.loc[table.period.eq("Full") & table.series.eq(TARGETS[0])].set_index("lead_months")
        self.assertEqual(records.index.tolist(), HORIZONS)
        for h, row in records.iterrows():
            x = self.panel.gscpi.to_numpy()[:len(self.panel) - h]
            y = self.panel[TARGETS[0]].to_numpy()[h:]
            self.assertEqual(row.n, len(self.panel) - h)
            self.assertAlmostEqual(row.r, np.corrcoef(x, y)[0, 1])
        pre = table.loc[table.period.eq("1997–2019")]
        self.assertEqual(set(pre.target_end), {"2019-12"})

    def test_diagnostics_record_inputs_and_lags(self):
        result = stationarity(self.panel).set_index(["period", "series"])
        unemp = result.loc[("Full", "unemployment_change_pp")]
        self.assertEqual((unemp["end"], unemp["n"]), ("2025-09", 337))
        self.assertTrue(result.adf_p.between(0, 1).all())
        self.assertTrue(result.kpss_p.between(.01, .10).all())
        self.assertTrue(result.adf_lags.between(0, 12).all())
        self.assertTrue(result.adf_stat.notna().all() and result.kpss_stat.notna().all())
        curves = serial_correlations(self.panel)
        self.assertEqual(len(curves), 5 * 25)
        self.assertTrue((curves.loc[curves.lag.eq(0), ["acf", "pacf"]] == 1).all().all())

    def test_fisher_excess_kurtosis_matches_independent_estimator(self):
        columns = describe(self.panel)
        delivered = pd.read_csv(ROOT / "data" / "processed" / "task1_descriptive.csv")
        self.assertEqual(len(columns), len(delivered))
        self.assertNotIn("excess_kurtosis_fisher", delivered.columns)
        np.testing.assert_allclose(columns.excess_kurtosis, delivered.excess_kurtosis, rtol=1e-10)
        for row in delivered.itertuples(index=False):
            start, end = WINDOWS[row.period]
            values = self.panel.loc[start:end, row.series].dropna().to_numpy()
            self.assertAlmostEqual(row.excess_kurtosis, kurtosis(values, fisher=True, bias=False),
                                   places=9)

    def test_report_fingerprints_task0_snapshot(self):
        report = ROOT / "reports" / "task1_eda.md"
        self.assertIn(hashlib.sha256(self.source.read_bytes()).hexdigest(),
                      report.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
