"""Aggregation and query layer for Olive Insights (spec §5, §7).

Reads the per-meeting rows written by the extraction stage (``meeting_speaker_signals`` and
``meeting_moments``) and rolls them up at query time: per-dimension series by week or month,
per-person baselines and outliers, and me<->attendee dyad trajectories. Everything here is plain
SQL (``sqlalchemy.text``) plus pure-Python math; there is no ORM, ML, or LLM. Every number the
service returns carries the ``meetingIds`` and ``startMs`` needed to drill into the call.

Thresholds (``outlierZ``, ``trendSlope``, ``volatility``, ``minSamples``, ...) are tuning data,
not logic: they live in ``DEFAULTS`` below and are overridden per key by
``<configDir>/insights/config.json`` (via ``load_insights_config``) and then by the ``options``
passed to the service.

Week and month buckets are computed in Python from UTC epoch-ms (ISO weeks, Monday start) rather
than with SQLite ``strftime``, whose ``%W`` is not ISO and would disagree with ``bucket_key``.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

from sqlalchemy import text
from sqlalchemy.engine import Engine

from ...db import DbHandle, get_db
from ...settings import resolve_paths
from .prompts import load_insights_config

Bucket = Literal["month", "week"]
Trend = Literal["warming", "cooling", "stable", "volatile", "insufficient"]

ME_STATE_KEY = "insights.me"
_EPS = 1e-9

# Local fallbacks; every key can be overridden in ``<configDir>/insights/config.json``.
DEFAULTS: dict[str, Any] = {
    # |z| at or above which a value counts as an outlier against the person's own baseline.
    "outlierZ": 2.0,
    # Minimum samples before a baseline, outlier, or trend claim is made.
    "minSamples": 3,
    # Least-squares slope (per bucket step) at or above which a series is warming/cooling.
    "trendSlope": 0.05,
    # Residual std around the fitted line above which a series is volatile.
    "volatility": 0.25,
    # Signal/moment rows with identity confidence below this are 'unattributed'.
    "minIdentityConfidence": 0.5,
    # Cap on the overview's outlier feed.
    "topOutliers": 10,
    # Severity assumed for a good/bad moment that carries none.
    "defaultMomentSeverity": 0.5,
    # Dimensions that make up a dyad's tone (warmth minus hostility).
    "warmthDimension": "warmth",
    "hostilityDimension": "hostility",
}


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def bucket_key(ms: int, bucket: str = "month") -> str:
    """UTC bucket label for epoch-ms: ``YYYY-MM`` (month) or ISO ``YYYY-Www`` (week)."""
    if bucket not in ("month", "week"):
        raise ValueError(f"unknown bucket: {bucket!r}")
    dt = datetime.fromtimestamp(ms / 1000, tz=UTC)
    if bucket == "month":
        return f"{dt.year:04d}-{dt.month:02d}"
    iso_year, iso_week, _ = dt.isocalendar()
    return f"{iso_year:04d}-W{iso_week:02d}"


def baseline(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, population std)``; ``(0.0, 0.0)`` for no values."""
    if not values:
        return 0.0, 0.0
    mean = math.fsum(values) / len(values)
    var = math.fsum((v - mean) ** 2 for v in values) / len(values)
    return mean, math.sqrt(var)


def zscores(values: Sequence[float]) -> list[float]:
    """Z-score of each value against the series' own baseline; all zero when it has no spread."""
    mean, std = baseline(values)
    if std <= _EPS:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def rank_outliers(values: Sequence[float], z: float, min_samples: int) -> list[tuple[int, float]]:
    """``(index, zscore)`` for values with ``|zscore| >= z``, largest deviation first.

    Empty when there are fewer than ``min_samples`` values. The sign of the score gives the direction.
    """
    if len(values) < min_samples or not values:
        return []
    hits = [(i, s) for i, s in enumerate(zscores(values)) if abs(s) >= z - _EPS and abs(s) > _EPS]
    return sorted(hits, key=lambda h: (-abs(h[1]), h[0]))


def _slope_and_residual_std(series: Sequence[float]) -> tuple[float, float]:
    n = len(series)
    xs = range(n)
    mx = (n - 1) / 2
    my = math.fsum(series) / n
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    slope = math.fsum((x - mx) * (y - my) for x, y in zip(xs, series, strict=True)) / sxx if sxx else 0.0
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, series, strict=True)]
    return slope, math.sqrt(math.fsum(r * r for r in resid) / n)


def classify_trend(
    series: Sequence[float],
    *,
    slope_threshold: float | None = None,
    volatility: float | None = None,
    min_samples: int | None = None,
) -> Trend:
    """Label a chronological series from its least-squares slope and the scatter around that line."""
    slope_threshold = DEFAULTS["trendSlope"] if slope_threshold is None else slope_threshold
    volatility = DEFAULTS["volatility"] if volatility is None else volatility
    min_samples = DEFAULTS["minSamples"] if min_samples is None else min_samples
    if len(series) < max(min_samples, 2):
        return "insufficient"
    slope, resid_std = _slope_and_residual_std(series)
    if resid_std > volatility + _EPS:
        return "volatile"
    if slope >= slope_threshold:
        return "warming"
    if slope <= -slope_threshold:
        return "cooling"
    return "stable"


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_ms(value: int | float | str | None) -> int | None:
    """Accept epoch-ms or an ISO-8601 date/datetime (UTC when naive)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    s = str(value).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def _parse_dimensions(raw: Any) -> dict[str, float]:
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        str(k): float(v)
        for k, v in data.items()
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    }


def _round(value: float) -> float:
    return round(value, 6)


def _scope_sql(
    frm: int | None, to: int | None, tag: str | None, *, alias: str = "m"
) -> tuple[str, dict[str, Any]]:
    clauses: list[str] = []
    params: dict[str, Any] = {}
    if frm is not None:
        clauses.append(f"{alias}.start_time >= :frm")
        params["frm"] = frm
    if to is not None:
        clauses.append(f"{alias}.start_time <= :to")
        params["to"] = to
    if tag:
        clauses.append(f"EXISTS (SELECT 1 FROM json_each({alias}.tags) WHERE json_each.value = :tag)")
        params["tag"] = tag
    return (" AND " + " AND ".join(clauses)) if clauses else "", params


class _Sample:
    """One (meeting, speaker) signal row, already parsed."""

    __slots__ = ("confidence", "dims", "meeting_id", "source", "speaker_id", "start_ms", "title")

    def __init__(self, row: Mapping[str, Any]) -> None:
        self.meeting_id: str = row["meeting_id"]
        self.start_ms: int = int(row["start_time"])
        self.title: str = row["title"]
        self.speaker_id: str | None = row["speaker_id"]
        self.source: str | None = row["identity_source"]
        self.confidence: float | None = row["identity_confidence"]
        self.dims: dict[str, float] = _parse_dimensions(row["dimensions"])


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class InsightsQueryService:
    """Query-time rollups over signals and moments; read-only except for the ``me`` setting."""

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        db: Engine | DbHandle | None = None,
        config_dir: str | None = None,
    ) -> None:
        if isinstance(db, DbHandle):
            self.db: Engine = db.engine
        else:
            self.db = db or get_db()
        cfg: dict[str, Any] = dict(DEFAULTS)
        cfg.update(load_insights_config(config_dir or resolve_paths().config_dir))
        cfg.update(options or {})
        self.config = cfg

    # -- me ---------------------------------------------------------------

    def get_me(self) -> dict[str, Any]:
        with self.db.connect() as conn:
            raw = conn.execute(text("SELECT value FROM sync_state WHERE key = :k"), {"k": ME_STATE_KEY}).scalar()
        data: Any = None
        if raw:
            try:
                data = json.loads(raw)
            except ValueError:
                data = None
        if not isinstance(data, dict):
            return {"speakerId": None, "aliases": []}
        aliases = data.get("aliases")
        return {
            "speakerId": data.get("speakerId"),
            "aliases": [str(a) for a in aliases] if isinstance(aliases, list) else [],
        }

    def set_me(self, speaker_id: str | None, aliases: Iterable[str] = ()) -> dict[str, Any]:
        me = {"speakerId": speaker_id, "aliases": [str(a) for a in aliases]}
        with self.db.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO sync_state (key, value) VALUES (:k, :v) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value"
                ),
                {"k": ME_STATE_KEY, "v": json.dumps(me)},
            )
        return me

    # -- shared loading ---------------------------------------------------

    def _is_attributed(self, sample: _Sample) -> bool:
        if not sample.speaker_id:
            return False
        if sample.confidence is not None and sample.confidence < self.config["minIdentityConfidence"]:
            return False
        return True

    def _load_samples(
        self, frm: int | None, to: int | None, tag: str | None, speaker_id: str | None = None
    ) -> list[_Sample]:
        scope, params = _scope_sql(frm, to, tag)
        sql = (
            "SELECT s.meeting_id, m.start_time, m.title, s.speaker_id, s.identity_source, "
            "s.identity_confidence, s.dimensions "
            "FROM meeting_speaker_signals s JOIN meetings m ON m.id = s.meeting_id "
            "WHERE 1 = 1" + scope
        )
        if speaker_id is not None:
            sql += " AND s.speaker_id = :sid"
            params["sid"] = speaker_id
        sql += " ORDER BY m.start_time, s.meeting_id, s.id"
        with self.db.connect() as conn:
            return [_Sample(r) for r in conn.execute(text(sql), params).mappings().all()]

    def _load_moments(
        self, frm: int | None, to: int | None, tag: str | None, include_unverified: bool = False
    ) -> list[dict[str, Any]]:
        """Moments in scope. Ungrounded (``verified = 0``) moments are excluded unless asked for."""
        scope, params = _scope_sql(frm, to, tag)
        if not include_unverified:
            scope += " AND mo.verified = 1"
        sql = (
            "SELECT mo.id, mo.meeting_id, m.start_time AS meeting_start, mo.speaker_id, mo.speaker_label, "
            "mo.toward_speaker_id, mo.toward_label, mo.start_ms, mo.end_ms, mo.event_type, mo.valence, "
            "mo.framework, mo.dimension, mo.severity, mo.quote, mo.rationale, mo.guidance, "
            "mo.identity_source, mo.identity_confidence, mo.verified "
            "FROM meeting_moments mo JOIN meetings m ON m.id = mo.meeting_id "
            "WHERE 1 = 1" + scope + " ORDER BY mo.start_ms, mo.id"
        )
        with self.db.connect() as conn:
            return [dict(r) for r in conn.execute(text(sql), params).mappings().all()]

    def _moment_attributed(self, moment: Mapping[str, Any]) -> bool:
        conf = moment["identity_confidence"]
        return bool(moment["speaker_id"]) and (conf is None or conf >= self.config["minIdentityConfidence"])

    @staticmethod
    def _moment_row(m: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "id": m["id"],
            "meetingId": m["meeting_id"],
            "startMs": int(m["start_ms"]),
            "endMs": m["end_ms"],
            "meetingStartMs": int(m["meeting_start"]),
            "speakerId": m["speaker_id"],
            "speakerLabel": m["speaker_label"],
            "towardSpeakerId": m["toward_speaker_id"],
            "towardLabel": m["toward_label"],
            "eventType": m["event_type"],
            "valence": m["valence"],
            "framework": m["framework"],
            "dimension": m["dimension"],
            "severity": m["severity"],
            "quote": m["quote"],
            "rationale": m["rationale"],
            "guidance": m["guidance"],
            "identitySource": m["identity_source"],
            "verified": bool(m["verified"]),
        }

    def _coverage(self, frm: int | None, to: int | None, tag: str | None) -> dict[str, int]:
        scope, params = _scope_sql(frm, to, tag)
        with self.db.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM meetings m WHERE 1 = 1" + scope), params).scalar_one()
            analyzed = conn.execute(
                text(
                    "SELECT COUNT(*) FROM meetings m WHERE 1 = 1" + scope
                    + " AND EXISTS (SELECT 1 FROM meeting_speaker_signals s WHERE s.meeting_id = m.id)"
                ),
                params,
            ).scalar_one()
        return {"analyzedMeetings": int(analyzed), "totalMeetings": int(total)}

    # -- series building --------------------------------------------------

    def _series(self, points: Iterable[tuple[str, int, float]], bucket: str) -> list[dict[str, Any]]:
        """``(meetingId, startMs, value)`` triples -> chronological bucket points."""
        grouped: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
        for mid, start, value in points:
            grouped[bucket_key(start, bucket)].append((mid, start, value))
        out = []
        for key in sorted(grouped):
            items = grouped[key]
            ids = sorted({m for m, _, _ in items}, key=lambda m: min(s for mm, s, _ in items if mm == m))
            out.append(
                {
                    "bucket": key,
                    "value": _round(_mean([v for _, _, v in items])),
                    "n": len(items),
                    "meetingIds": ids,
                    "startMs": min(s for _, s, _ in items),
                }
            )
        return out

    def _dimension_series(self, samples: Sequence[_Sample], bucket: str) -> dict[str, list[dict[str, Any]]]:
        by_dim: dict[str, list[tuple[str, int, float]]] = defaultdict(list)
        for s in samples:
            for dim, value in s.dims.items():
                by_dim[dim].append((s.meeting_id, s.start_ms, value))
        return {dim: self._series(pts, bucket) for dim, pts in sorted(by_dim.items())}

    def _per_meeting(self, samples: Sequence[_Sample]) -> dict[str, dict[str, tuple[int, float]]]:
        """dimension -> meetingId -> (startMs, mean value) so a meeting counts once per dimension."""
        acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        starts: dict[str, int] = {}
        for s in samples:
            starts[s.meeting_id] = s.start_ms
            for dim, value in s.dims.items():
                acc[dim][s.meeting_id].append(value)
        return {
            dim: {mid: (starts[mid], _mean(vals)) for mid, vals in per.items()}
            for dim, per in acc.items()
        }

    def _meeting_outliers(
        self, samples: Sequence[_Sample], speaker_id: str
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
        z, min_n = self.config["outlierZ"], self.config["minSamples"]
        outliers: list[dict[str, Any]] = []
        baselines: dict[str, dict[str, Any]] = {}
        for dim, per in sorted(self._per_meeting(samples).items()):
            ordered = sorted(per.items(), key=lambda kv: (kv[1][0], kv[0]))
            values = [v for _, (_, v) in ordered]
            mean, std = baseline(values)
            baselines[dim] = {"mean": _round(mean), "std": _round(std), "n": len(values)}
            for idx, score in rank_outliers(values, z, min_n):
                mid, (start, value) = ordered[idx]
                outliers.append(
                    {
                        "speakerId": speaker_id,
                        "meetingId": mid,
                        "startMs": start,
                        "dimension": dim,
                        "value": _round(value),
                        "z": _round(score),
                        "direction": "above" if score > 0 else "below",
                        "baselineMean": _round(mean),
                        "baselineStd": _round(std),
                    }
                )
        outliers.sort(key=lambda o: (-abs(o["z"]), o["startMs"]))
        return outliers, baselines

    def _moment_counts(self, moments: Iterable[Mapping[str, Any]]) -> dict[str, int]:
        counts = {"good": 0, "bad": 0}
        for m in moments:
            if m["valence"] in counts:
                counts[m["valence"]] += 1
        return counts

    # -- overview ---------------------------------------------------------

    def overview(
        self,
        frm: int | str | None = None,
        to: int | str | None = None,
        speaker_id: str | None = None,
        tag: str | None = None,
        bucket: str = "month",
        include_unverified: bool = False,
    ) -> dict[str, Any]:
        """Corpus-level series by dimension, top outliers, good/bad counts, and coverage."""
        bucket_key(0, bucket)  # validate early
        frm_ms, to_ms = _to_ms(frm), _to_ms(to)
        samples = self._load_samples(frm_ms, to_ms, tag, speaker_id)
        attributed = [s for s in samples if self._is_attributed(s)]
        # A named person is a per-person claim, so only trusted rows count; the whole-corpus view keeps
        # every row (an unidentified voice still shaped the call's tone) but reports how many were unattributed.
        used = attributed if speaker_id is not None else samples
        unattributed = [s for s in samples if not self._is_attributed(s)]

        outliers: list[dict[str, Any]] = []
        by_speaker: dict[str, list[_Sample]] = defaultdict(list)
        for s in attributed:
            assert s.speaker_id is not None
            by_speaker[s.speaker_id].append(s)
        for sid, rows in sorted(by_speaker.items()):
            found, _ = self._meeting_outliers(rows, sid)
            outliers.extend(found)
        outliers.sort(key=lambda o: (-abs(o["z"]), o["startMs"]))

        moments = self._load_moments(frm_ms, to_ms, tag, include_unverified)
        if speaker_id is not None:
            moments = [m for m in moments if m["speaker_id"] == speaker_id and self._moment_attributed(m)]

        return {
            "from": frm_ms,
            "to": to_ms,
            "speakerId": speaker_id,
            "tag": tag,
            "bucket": bucket,
            "series": self._dimension_series(used, bucket),
            "outliers": outliers[: int(self.config["topOutliers"])],
            "moments": self._moment_counts(moments),
            "coverage": self._coverage(frm_ms, to_ms, tag),
            "unattributed": {"rows": len(unattributed), "meetingIds": sorted({s.meeting_id for s in unattributed})},
        }

    # -- person -----------------------------------------------------------

    def person(
        self,
        speaker_id: str,
        frm: int | str | None = None,
        to: int | str | None = None,
        bucket: str = "month",
        include_unverified: bool = False,
    ) -> dict[str, Any]:
        """One speaker over time: series, own-baseline outliers (meetings and weeks), linked meetings."""
        bucket_key(0, bucket)
        frm_ms, to_ms = _to_ms(frm), _to_ms(to)
        rows = self._load_samples(frm_ms, to_ms, None, speaker_id)
        samples = [s for s in rows if self._is_attributed(s)]
        low_conf = [s for s in rows if not self._is_attributed(s)]

        with self.db.connect() as conn:
            name = conn.execute(text("SELECT name FROM speakers WHERE id = :id"), {"id": speaker_id}).scalar()

        outlier_meetings, baselines = self._meeting_outliers(samples, speaker_id)

        outlier_weeks: list[dict[str, Any]] = []
        z, min_n = self.config["outlierZ"], self.config["minSamples"]
        for dim, points in sorted(self._dimension_series(samples, "week").items()):
            for idx, score in rank_outliers([p["value"] for p in points], z, min_n):
                p = points[idx]
                outlier_weeks.append(
                    {
                        "dimension": dim,
                        "bucket": p["bucket"],
                        "value": p["value"],
                        "z": _round(score),
                        "direction": "above" if score > 0 else "below",
                        "n": p["n"],
                        "meetingIds": p["meetingIds"],
                        "startMs": p["startMs"],
                    }
                )
        outlier_weeks.sort(key=lambda o: (-abs(o["z"]), o["startMs"]))

        meetings: dict[str, dict[str, Any]] = {}
        for s in samples:
            meetings.setdefault(s.meeting_id, {"meetingId": s.meeting_id, "startMs": s.start_ms, "title": s.title})

        sources = [s.source for s in samples]
        total = len(samples)
        share = (lambda src: _round(sum(1 for x in sources if x == src) / total)) if total else (lambda _s: 0.0)
        named, voice = share("name"), share("voiceprint")
        identity = {
            "rows": total,
            "nameShare": named,
            "voiceprintShare": voice,
            "otherShare": _round(max(0.0, 1.0 - named - voice)) if total else 0.0,
        }

        moments = [
            m
            for m in self._load_moments(frm_ms, to_ms, None, include_unverified)
            if m["speaker_id"] == speaker_id
        ]
        moments = [m for m in moments if self._moment_attributed(m)]

        return {
            "speakerId": speaker_id,
            "name": name,
            "from": frm_ms,
            "to": to_ms,
            "bucket": bucket,
            "series": self._dimension_series(samples, bucket),
            "baseline": baselines,
            "outlierMeetings": outlier_meetings,
            "outlierWeeks": outlier_weeks,
            "meetings": sorted(meetings.values(), key=lambda m: (m["startMs"], m["meetingId"])),
            "moments": self._moment_counts(moments),
            "identityConfidence": identity,
            "unattributed": {"rows": len(low_conf), "meetingIds": sorted({s.meeting_id for s in low_conf})},
        }

    # -- meeting ----------------------------------------------------------

    def meeting(self, meeting_id: str) -> dict[str, Any] | None:
        """A single meeting's speaker signals and moments, or ``None`` when the meeting is unknown."""
        with self.db.connect() as conn:
            head = conn.execute(
                text("SELECT id, title, start_time FROM meetings WHERE id = :id"), {"id": meeting_id}
            ).mappings().first()
            if head is None:
                return None
            sig_rows = conn.execute(
                text(
                    "SELECT speaker_id, speaker_label, identity_source, identity_confidence, dimensions, talk_ms, "
                    "turn_count, interrupt_count, stance FROM meeting_speaker_signals "
                    "WHERE meeting_id = :id ORDER BY speaker_label, id"
                ),
                {"id": meeting_id},
            ).mappings().all()
            mom_rows = conn.execute(
                text(
                    "SELECT mo.id, mo.meeting_id, m.start_time AS meeting_start, mo.speaker_id, mo.speaker_label, "
                    "mo.toward_speaker_id, mo.toward_label, mo.start_ms, mo.end_ms, mo.event_type, mo.valence, "
                    "mo.framework, mo.dimension, mo.severity, mo.quote, mo.rationale, mo.guidance, "
                    "mo.identity_source, mo.identity_confidence, mo.verified "
                    "FROM meeting_moments mo JOIN meetings m ON m.id = mo.meeting_id "
                    "WHERE mo.meeting_id = :id ORDER BY mo.start_ms, mo.id"
                ),
                {"id": meeting_id},
            ).mappings().all()

        speakers = []
        for r in sig_rows:
            conf = r["identity_confidence"]
            attributed = bool(r["speaker_id"]) and (conf is None or conf >= self.config["minIdentityConfidence"])
            speakers.append(
                {
                    "meetingId": meeting_id,
                    "startMs": int(head["start_time"]),
                    "speakerId": r["speaker_id"],
                    "speakerLabel": r["speaker_label"],
                    "identitySource": r["identity_source"],
                    "identityConfidence": conf,
                    "attributed": attributed,
                    "dimensions": _parse_dimensions(r["dimensions"]),
                    "talkMs": r["talk_ms"],
                    "turnCount": r["turn_count"],
                    "interruptCount": r["interrupt_count"],
                    "stance": r["stance"],
                }
            )
        return {
            "meetingId": meeting_id,
            "title": head["title"],
            "startMs": int(head["start_time"]),
            "speakers": speakers,
            "moments": [self._moment_row(dict(m)) for m in mom_rows],
        }

    # -- dyads ------------------------------------------------------------

    def _moment_score(self, moments: Sequence[Mapping[str, Any]]) -> float | None:
        default = self.config["defaultMomentSeverity"]
        scores = []
        for m in moments:
            sev = m["severity"] if m["severity"] is not None else default
            if m["valence"] == "good":
                scores.append(float(sev))
            elif m["valence"] == "bad":
                scores.append(-float(sev))
        return _mean(scores) if scores else None

    def dyads(
        self,
        me_id: str | None = None,
        frm: int | str | None = None,
        to: int | str | None = None,
        bucket: str = "month",
        include_unverified: bool = False,
    ) -> dict[str, Any]:
        """Per ordered pair (me->X and X->me): bucketed tone series, trend, and supporting meetings.

        A point's value is warmth minus hostility from the sender's signal rows in shared meetings,
        averaged with the net valence of the sender's moments aimed at the receiver.
        """
        bucket_key(0, bucket)
        me_id = me_id or self.get_me()["speakerId"]
        frm_ms, to_ms = _to_ms(frm), _to_ms(to)
        samples = self._load_samples(frm_ms, to_ms, None)
        unattributed = [s for s in samples if not self._is_attributed(s)]
        result: dict[str, Any] = {
            "meId": me_id,
            "bucket": bucket,
            "dyads": [],
            "unattributed": {"rows": len(unattributed), "meetingIds": sorted({s.meeting_id for s in unattributed})},
        }
        if not me_id:
            return result

        rows: dict[tuple[str, str], list[_Sample]] = defaultdict(list)  # (meeting, speaker) -> samples
        starts: dict[str, int] = {}
        for s in samples:
            if self._is_attributed(s):
                assert s.speaker_id is not None
                rows[(s.meeting_id, s.speaker_id)].append(s)
                starts[s.meeting_id] = s.start_ms
        present: dict[str, set[str]] = defaultdict(set)  # meeting -> attributed speakers
        for mid, sid in rows:
            present[mid].add(sid)

        moments = [
            m
            for m in self._load_moments(frm_ms, to_ms, None, include_unverified)
            if self._moment_attributed(m)
        ]
        by_edge: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)  # (from, to, meeting)
        for m in moments:
            if m["toward_speaker_id"]:
                by_edge[(m["speaker_id"], m["toward_speaker_id"], m["meeting_id"])].append(m)
                starts.setdefault(m["meeting_id"], int(m["meeting_start"]))

        others = {sid for sids in present.values() if me_id in sids for sid in sids if sid != me_id}
        others |= {a for (a, b, _) in by_edge if b == me_id} | {b for (a, b, _) in by_edge if a == me_id}
        others.discard(me_id)

        warm_dim, host_dim = self.config["warmthDimension"], self.config["hostilityDimension"]
        names = self._speaker_names({me_id, *others})
        dyads = []
        for other in sorted(others):
            for sender, receiver, direction in ((me_id, other, "meToThem"), (other, me_id, "themToMe")):
                meeting_ids = {
                    mid for mid, sids in present.items() if sender in sids and receiver in sids
                } | {mid for (a, b, mid) in by_edge if a == sender and b == receiver}
                points: list[tuple[str, int, float]] = []
                warmth: dict[str, list[tuple[str, int, float]]] = {"w": [], "h": []}
                good = bad = 0
                per_bucket_moments: dict[str, dict[str, int]] = defaultdict(lambda: {"good": 0, "bad": 0})
                for mid in sorted(meeting_ids, key=lambda m: (starts[m], m)):
                    parts: list[float] = []
                    sig = rows.get((mid, sender), [])
                    w = [x.dims[warm_dim] for x in sig if warm_dim in x.dims]
                    h = [x.dims[host_dim] for x in sig if host_dim in x.dims]
                    if w or h:
                        parts.append((_mean(w) if w else 0.0) - (_mean(h) if h else 0.0))
                    if w:
                        warmth["w"].append((mid, starts[mid], _mean(w)))
                    if h:
                        warmth["h"].append((mid, starts[mid], _mean(h)))
                    edge = by_edge.get((sender, receiver, mid), [])
                    ms = self._moment_score(edge)
                    if ms is not None:
                        parts.append(ms)
                    key = bucket_key(starts[mid], bucket)
                    for m in edge:
                        if m["valence"] in ("good", "bad"):
                            per_bucket_moments[key][m["valence"]] += 1
                    if parts:
                        points.append((mid, starts[mid], _mean(parts)))
                if not points:
                    continue
                series = self._series(points, bucket)
                w_by = {p["bucket"]: p["value"] for p in self._series(warmth["w"], bucket)}
                h_by = {p["bucket"]: p["value"] for p in self._series(warmth["h"], bucket)}
                for p in series:
                    p["tone"] = p["value"]
                    p["warmth"] = w_by.get(p["bucket"])
                    p["hostility"] = h_by.get(p["bucket"])
                    p["moments"] = dict(per_bucket_moments.get(p["bucket"], {"good": 0, "bad": 0}))
                    good += p["moments"]["good"]
                    bad += p["moments"]["bad"]
                dyads.append(
                    {
                        "fromSpeakerId": sender,
                        "toSpeakerId": receiver,
                        "fromName": names.get(sender),
                        "toName": names.get(receiver),
                        "otherSpeakerId": other,
                        "direction": direction,
                        "series": series,
                        "trend": classify_trend(
                            [p["value"] for p in series],
                            slope_threshold=self.config["trendSlope"],
                            volatility=self.config["volatility"],
                            min_samples=self.config["minSamples"],
                        ),
                        "n": len(points),
                        "moments": {"good": good, "bad": bad},
                        "meetingIds": [m for m, _, _ in sorted(points, key=lambda p: (p[1], p[0]))],
                        "startMs": min(p["startMs"] for p in series),
                    }
                )
        result["dyads"] = dyads
        return result

    def _speaker_names(self, ids: Iterable[str]) -> dict[str, str]:
        ids = list(ids)
        if not ids:
            return {}
        params = {f"i{n}": v for n, v in enumerate(ids)}
        sql = f"SELECT id, name FROM speakers WHERE id IN ({', '.join(':' + k for k in params)})"
        with self.db.connect() as conn:
            return {r["id"]: r["name"] for r in conn.execute(text(sql), params).mappings().all()}


__all__ = [
    "DEFAULTS",
    "InsightsQueryService",
    "baseline",
    "bucket_key",
    "classify_trend",
    "rank_outliers",
    "zscores",
]
