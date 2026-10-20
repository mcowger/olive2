"""Pure-math tests for Olive Insights aggregation (services/insights/query.py); no DB involved."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest

from olive.services.insights.query import (
    DEFAULTS,
    baseline,
    bucket_key,
    classify_trend,
    rank_outliers,
    zscores,
)


def _ms(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> int:
    return int(datetime(year, month, day, hour, minute, second, tzinfo=UTC).timestamp() * 1000)


class TestBucketKey:
    def test_month_boundaries_are_utc(self) -> None:
        assert bucket_key(_ms(2026, 1, 31, 23, 59, 59), "month") == "2026-01"
        assert bucket_key(_ms(2026, 2, 1), "month") == "2026-02"
        # One millisecond before midnight UTC is still the previous month.
        assert bucket_key(_ms(2026, 2, 1) - 1, "month") == "2026-01"

    def test_week_uses_iso_weeks_starting_monday(self) -> None:
        # 2026-01-05 is a Monday; the Sunday before belongs to the previous ISO week.
        assert bucket_key(_ms(2026, 1, 5), "week") == "2026-W02"
        assert bucket_key(_ms(2026, 1, 5) - 1, "week") == "2026-W01"
        assert bucket_key(_ms(2026, 1, 11, 23, 59, 59), "week") == "2026-W02"

    def test_week_year_boundary_uses_iso_year(self) -> None:
        assert bucket_key(_ms(2025, 12, 29), "week") == "2026-W01"
        assert bucket_key(_ms(2027, 1, 1), "week") == "2026-W53"

    def test_default_is_month_and_unknown_bucket_rejected(self) -> None:
        assert bucket_key(_ms(2026, 3, 15)) == "2026-03"
        with pytest.raises(ValueError):
            bucket_key(0, "day")


class TestBaseline:
    def test_mean_and_population_std(self) -> None:
        mean, std = baseline([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        assert mean == pytest.approx(5.0)
        assert std == pytest.approx(2.0)

    def test_empty_and_single(self) -> None:
        assert baseline([]) == (0.0, 0.0)
        assert baseline([3.0]) == (3.0, 0.0)


class TestZscores:
    def test_values(self) -> None:
        zs = zscores([1.0, 3.0])
        assert zs == [pytest.approx(-1.0), pytest.approx(1.0)]

    def test_constant_series_is_all_zero(self) -> None:
        assert zscores([0.4, 0.4, 0.4]) == [0.0, 0.0, 0.0]

    def test_empty(self) -> None:
        assert zscores([]) == []


class TestRankOutliers:
    def test_below_min_samples_yields_no_outliers(self) -> None:
        assert rank_outliers([0.0, 0.0, 10.0], z=1.0, min_samples=4) == []

    def test_flagged_exactly_at_z(self) -> None:
        # n=2: both values sit exactly one population std from the mean.
        hits = rank_outliers([1.0, 3.0], z=1.0, min_samples=2)
        assert {i for i, _ in hits} == {0, 1}
        assert all(math.isclose(abs(score), 1.0) for _, score in hits)

    def test_not_flagged_just_below_z(self) -> None:
        assert rank_outliers([1.0, 3.0], z=1.0001, min_samples=2) == []

    def test_ranked_by_absolute_z_with_direction_in_sign(self) -> None:
        values = [0.5] * 8 + [0.9, 0.0]
        hits = rank_outliers(values, z=1.0, min_samples=5)
        assert hits[0][0] == 9 and hits[0][1] < 0  # 0.0 is the biggest deviation, below the mean
        assert hits[1][0] == 8 and hits[1][1] > 0  # then the 0.9 spike, above it
        assert abs(hits[0][1]) > abs(hits[1][1])

    def test_constant_and_empty(self) -> None:
        assert rank_outliers([0.5] * 6, z=0.1, min_samples=3) == []
        assert rank_outliers([], z=1.0, min_samples=1) == []


class TestClassifyTrend:
    def test_warming(self) -> None:
        assert classify_trend([0.1, 0.2, 0.3, 0.4, 0.5]) == "warming"

    def test_cooling(self) -> None:
        assert classify_trend([0.5, 0.4, 0.3, 0.2, 0.1]) == "cooling"

    def test_stable(self) -> None:
        assert classify_trend([0.5, 0.5, 0.51, 0.5, 0.49]) == "stable"

    def test_volatile(self) -> None:
        assert classify_trend([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]) == "volatile"

    def test_insufficient(self) -> None:
        assert classify_trend([]) == "insufficient"
        assert classify_trend([0.1, 0.9]) == "insufficient"
        assert classify_trend([0.1, 0.9, 0.5], min_samples=4) == "insufficient"

    def test_thresholds_are_parameters_not_constants(self) -> None:
        series = [0.5, 0.52, 0.54, 0.56, 0.58]
        assert classify_trend(series) == "stable"
        assert classify_trend(series, slope_threshold=0.01) == "warming"
        assert classify_trend([0.1, 0.2, 0.3, 0.4, 0.5], volatility=0.0) == "warming"

    def test_defaults_expose_config_keys(self) -> None:
        for key in ("trendSlope", "volatility", "minSamples", "outlierZ"):
            assert key in DEFAULTS
