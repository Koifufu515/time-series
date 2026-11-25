"""Checks against the delivered official snapshots using the standard CSV reader."""
import csv
from datetime import datetime
from pathlib import Path
import unittest

from prepare_data import build_panel, parse_series


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = read_csv(ROOT / "data" / "monthly.csv")

    def test_calendar_and_source_values(self):
        by_month = {r["month"]: r for r in self.panel}
        start = datetime.strptime(self.panel[0]["month"], "%Y-%m")
        expected = [
            f"{year:04d}-{month + 1:02d}"
            for year, month in (divmod(start.year * 12 + start.month - 1 + i, 12)
                                for i in range(len(self.panel)))
        ]
        self.assertEqual(list(by_month), expected)
        self.assertEqual(len(by_month), len(self.panel))
        for code, field in {
            "DGDSRG3M086SBEA": "pce_goods_index",
            "PCEPI": "pce_index",
            "MCOILWTICO": "wti_usd_per_barrel",
            "UNRATE": "unemployment_pct",
        }.items():
            records = read_csv(RAW / f"{code}.csv")
            values = {(r.get("observation_date") or r["DATE"])[:7]: r[code] for r in records}
            for month, record in by_month.items():
                expected_value = values.get(month, "")
                if expected_value in ("", "."):
                    self.assertEqual(record[field], "", (month, field))
                else:
                    self.assertAlmostEqual(float(record[field]), float(expected_value), places=8)
        records = read_csv(RAW / "gscpi_interactive_data.csv")
        latest = max((key for key in records[0] if key != "Date"), key=int)
        values = {datetime.strptime(r["Date"], "%d-%b-%Y").strftime("%Y-%m"): r[latest]
                  for r in records}
        for month, record in by_month.items():
            if values.get(month, "") in ("", "."):
                self.assertEqual(record["gscpi"], "", month)
            else:
                self.assertAlmostEqual(float(record["gscpi"]), float(values[month]), places=8)

    def test_all_candidate_rates(self):
        for i, row in enumerate(self.panel):
            for prefix in ("pce_goods", "pce"):
                field = f"{prefix}_index"
                for suffix, lag in (("mom", 1), ("yoy", 12)):
                    actual = row[f"{prefix}_{suffix}_pct"]
                    prior = self.panel[i - lag][field] if i >= lag else ""
                    if prior and row[field]:
                        expected = 100 * (float(row[field]) / float(prior) - 1)
                        self.assertAlmostEqual(float(actual), expected, places=8)
                    else:
                        self.assertEqual(actual, "", (row["month"], prefix, suffix))

    def test_calendar_gap_does_not_become_one_month_change(self):
        raw = (RAW / "PCEPI.csv").read_bytes().splitlines()
        with_gap = b"\n".join([raw[0], raw[1], raw[3]]) + b"\n"
        s, _ = parse_series(with_gap, "PCEPI.csv", "pce_index")
        panel = build_panel({"goods": s.rename("pce_goods_index"), "headline": s})
        self.assertTrue(panel["pce_index"].isna().loc["1959-02"])
        self.assertTrue(panel["pce_mom_pct"].isna().loc["1959-03"])

    def test_duplicate_month_stops_processing(self):
        raw = (RAW / "UNRATE.csv").read_bytes().splitlines()
        duplicate = b"\n".join([raw[0], raw[1], raw[1]]) + b"\n"
        with self.assertRaisesRegex(ValueError, "duplicate observation months"):
            parse_series(duplicate, "UNRATE.csv", "unemployment_pct")


if __name__ == "__main__":
    unittest.main()
