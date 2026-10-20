"""Tests for the Olive Insights aggregation/query service (services/insights/query.py)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import text

from olive.db import create_db
from olive.services.insights.query import InsightsQueryService

ME = "spk-me"
ALICE = "spk-alice"
BOB = "spk-bob"


def _ms(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, tzinfo=UTC).timestamp() * 1000)


class Seeder:
    def __init__(self, db: Any) -> None:
        self.db = db
        self._n = 0

    def _id(self, prefix: str) -> str:
        self._n += 1
        return f"{prefix}-{self._n}"

    def speaker(self, speaker_id: str, name: str) -> None:
        with self.db.engine.begin() as conn:
            conn.execute(
                text("INSERT OR IGNORE INTO speakers (id, name, created_at) VALUES (:id, :n, 0)"),
                {"id": speaker_id, "n": name},
            )

    def meeting(self, meeting_id: str, start: int, tags: tuple[str, ...] = (), title: str = "") -> None:
        with self.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meetings (id, title, start_time, end_time, source, status, tags, created_at, updated_at) "
                    "VALUES (:id, :t, :s, :e, 'upload', 'ready', :tags, 0, 0)"
                ),
                {"id": meeting_id, "t": title or meeting_id, "s": start, "e": start + 60_000, "tags": json.dumps(list(tags))},
            )

    def signal(
        self,
        meeting_id: str,
        speaker_id: str | None,
        dims: dict[str, float],
        *,
        source: str | None = "voiceprint",
        confidence: float | None = 0.9,
        label: str | None = None,
    ) -> None:
        with self.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meeting_speaker_signals (id, meeting_id, speaker_id, speaker_label, identity_source, "
                    "identity_confidence, dimensions, talk_ms, turn_count, interrupt_count, created_at) "
                    "VALUES (:id, :m, :s, :l, :src, :c, :d, 1000, 4, 1, 0)"
                ),
                {
                    "id": self._id("sig"),
                    "m": meeting_id,
                    "s": speaker_id,
                    "l": label or speaker_id or "Speaker 9",
                    "src": source,
                    "c": confidence,
                    "d": json.dumps(dims),
                },
            )

    def moment(
        self,
        meeting_id: str,
        start: int,
        speaker_id: str | None,
        toward: str | None,
        valence: str,
        *,
        severity: float = 0.8,
        dimension: str = "hostility",
    ) -> None:
        with self.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO meeting_moments (id, meeting_id, speaker_id, toward_speaker_id, start_ms, end_ms, "
                    "event_type, valence, dimension, severity, quote, created_at) "
                    "VALUES (:id, :m, :s, :t, :st, :st, 'evt', :v, :dim, :sev, 'q', 0)"
                ),
                {
                    "id": self._id("mom"),
                    "m": meeting_id,
                    "s": speaker_id,
                    "t": toward,
                    "st": start,
                    "v": valence,
                    "dim": dimension,
                    "sev": severity,
                },
            )


@pytest.fixture
def env(tmp_path: Path) -> tuple[Any, Seeder, InsightsQueryService]:
    db = create_db(":memory:")
    seed = Seeder(db)
    for sid, name in ((ME, "Me"), (ALICE, "Alice"), (BOB, "Bob")):
        seed.speaker(sid, name)
    svc = InsightsQueryService(db=db, config_dir=str(tmp_path / "cfg"))
    return db, seed, svc


def _assert_drillable(point: dict[str, Any]) -> None:
    assert point["meetingIds"], point
    assert isinstance(point["startMs"], int)


def test_me_round_trip(env: Any) -> None:
    _, _, svc = env
    assert svc.get_me() == {"speakerId": None, "aliases": []}
    svc.set_me(ME, ["Phil", "Phil McNeely"])
    assert svc.get_me() == {"speakerId": ME, "aliases": ["Phil", "Phil McNeely"]}
    svc.set_me(ALICE, [])
    assert svc.get_me() == {"speakerId": ALICE, "aliases": []}


def test_me_persisted_in_sync_state(env: Any) -> None:
    db, _, svc = env
    svc.set_me(ME, ["Phil"])
    with db.engine.connect() as conn:
        raw = conn.execute(text("SELECT value FROM sync_state WHERE key = 'insights.me'")).scalar_one()
    assert json.loads(raw) == {"speakerId": ME, "aliases": ["Phil"]}


def test_overview_series_points_carry_meeting_ids_and_start_ms(env: Any) -> None:
    _, seed, svc = env
    seed.meeting("m1", _ms(2026, 1, 10))
    seed.meeting("m2", _ms(2026, 1, 20))
    seed.meeting("m3", _ms(2026, 2, 5))
    seed.signal("m1", ME, {"hostility": 0.2, "warmth": 0.6})
    seed.signal("m2", ME, {"hostility": 0.4, "warmth": 0.4})
    seed.signal("m3", ME, {"hostility": 0.1})
    out = svc.overview(bucket="month")
    host = out["series"]["hostility"]
    assert [p["bucket"] for p in host] == ["2026-01", "2026-02"]
    assert host[0]["value"] == pytest.approx(0.3)
    assert host[0]["n"] == 2
    assert sorted(host[0]["meetingIds"]) == ["m1", "m2"]
    assert host[0]["startMs"] == _ms(2026, 1, 10)
    for dim_points in out["series"].values():
        for p in dim_points:
            _assert_drillable(p)
    assert json.dumps(out)  # JSON-serializable


def test_overview_week_bucket(env: Any) -> None:
    _, seed, svc = env
    seed.meeting("m1", _ms(2026, 1, 5))
    seed.meeting("m2", _ms(2026, 1, 12))
    seed.signal("m1", ME, {"stress": 0.1})
    seed.signal("m2", ME, {"stress": 0.9})
    out = svc.overview(bucket="week")
    assert [p["bucket"] for p in out["series"]["stress"]] == ["2026-W02", "2026-W03"]


def test_overview_filters_by_date_tag_and_speaker(env: Any) -> None:
    _, seed, svc = env
    seed.meeting("jan", _ms(2026, 1, 10), tags=("1:1",))
    seed.meeting("feb", _ms(2026, 2, 10), tags=("1:1", "jpr"))
    seed.meeting("mar", _ms(2026, 3, 10), tags=("jpr",))
    for mid in ("jan", "feb", "mar"):
        seed.signal(mid, ME, {"hostility": 0.5})
        seed.signal(mid, ALICE, {"hostility": 0.1})

    by_tag = svc.overview(tag="jpr")
    ids = {m for p in by_tag["series"]["hostility"] for m in p["meetingIds"]}
    assert ids == {"feb", "mar"}

    by_date = svc.overview(frm=_ms(2026, 2, 1), to=_ms(2026, 2, 28))
    ids = {m for p in by_date["series"]["hostility"] for m in p["meetingIds"]}
    assert ids == {"feb"}

    by_speaker = svc.overview(speaker_id=ALICE)
    vals = [p["value"] for p in by_speaker["series"]["hostility"]]
    assert vals and all(v == pytest.approx(0.1) for v in vals)


def test_overview_coverage_and_moment_counts(env: Any) -> None:
    _, seed, svc = env
    seed.meeting("a", _ms(2026, 1, 10))
    seed.meeting("b", _ms(2026, 1, 11))
    seed.meeting("c", _ms(2026, 1, 12))
    seed.signal("a", ME, {"hostility": 0.2})
    seed.signal("b", ME, {"hostility": 0.2})
    seed.moment("a", _ms(2026, 1, 10), ME, ALICE, "good")
    seed.moment("a", _ms(2026, 1, 10), ME, ALICE, "bad")
    seed.moment("b", _ms(2026, 1, 11), ALICE, ME, "bad")
    out = svc.overview()
    assert out["coverage"] == {"analyzedMeetings": 2, "totalMeetings": 3}
    assert out["moments"] == {"good": 1, "bad": 2}


def test_overview_top_outliers_reference_meetings(env: Any) -> None:
    _, seed, svc = env
    for i in range(1, 9):
        seed.meeting(f"m{i}", _ms(2026, 1, i))
        seed.signal(f"m{i}", ME, {"hostility": 0.9 if i == 8 else 0.1})
    out = svc.overview()
    assert out["outliers"], out
    top = out["outliers"][0]
    assert top["meetingId"] == "m8"
    assert top["speakerId"] == ME
    assert top["dimension"] == "hostility"
    assert top["z"] > 0
    assert top["startMs"] == _ms(2026, 1, 8)


def test_person_series_outliers_and_links(env: Any) -> None:
    _, seed, svc = env
    for i in range(1, 9):
        seed.meeting(f"m{i}", _ms(2026, 1, i))
        seed.signal(f"m{i}", ALICE, {"hostility": 0.95 if i == 5 else 0.1, "warmth": 0.5})
    # Another speaker must not leak into Alice's view.
    seed.signal("m1", BOB, {"hostility": 0.7})
    out = svc.person(ALICE)
    assert out["speakerId"] == ALICE
    assert out["name"] == "Alice"
    for p in out["series"]["hostility"]:
        _assert_drillable(p)
    assert all(m["meetingId"] and isinstance(m["startMs"], int) for m in out["meetings"])
    assert len(out["meetings"]) == 8
    flagged = [o for o in out["outlierMeetings"] if o["dimension"] == "hostility"]
    assert [o["meetingId"] for o in flagged] == ["m5"]
    assert flagged[0]["startMs"] == _ms(2026, 1, 5)
    assert flagged[0]["z"] >= svc.config["outlierZ"]
    assert out["baseline"]["hostility"]["n"] == 8


def test_person_below_min_samples_has_no_outliers(env: Any) -> None:
    _, seed, svc = env
    for i, v in enumerate((0.1, 0.9), start=1):
        seed.meeting(f"m{i}", _ms(2026, 1, i))
        seed.signal(f"m{i}", ALICE, {"hostility": v})
    out = svc.person(ALICE)
    assert out["outlierMeetings"] == []
    assert out["outlierWeeks"] == []


def test_person_outlier_weeks(env: Any) -> None:
    _, seed, svc = env
    # Eight consecutive Mondays; week 5 is hot.
    for i in range(8):
        day = _ms(2026, 1, 5) + i * 7 * 86_400_000
        seed.meeting(f"w{i}", day)
        seed.signal(f"w{i}", ALICE, {"hostility": 0.95 if i == 4 else 0.1})
    out = svc.person(ALICE)
    weeks = [o for o in out["outlierWeeks"] if o["dimension"] == "hostility"]
    assert len(weeks) == 1
    assert weeks[0]["meetingIds"] == ["w4"]
    assert weeks[0]["bucket"] == "2026-W06"
    assert isinstance(weeks[0]["startMs"], int)


def test_person_identity_confidence_mixed_sources(env: Any) -> None:
    _, seed, svc = env
    for i, src in enumerate(("name", "name", "voiceprint", "voiceprint"), start=1):
        seed.meeting(f"m{i}", _ms(2026, 1, i))
        seed.signal(f"m{i}", ALICE, {"warmth": 0.5}, source=src)
    out = svc.person(ALICE)["identityConfidence"]
    assert out["rows"] == 4
    assert out["nameShare"] == pytest.approx(0.5)
    assert out["voiceprintShare"] == pytest.approx(0.5)


def test_unattributed_rows_excluded_from_per_person_claims(env: Any) -> None:
    _, seed, svc = env
    seed.meeting("m1", _ms(2026, 1, 1))
    seed.meeting("m2", _ms(2026, 1, 2))
    seed.meeting("m3", _ms(2026, 1, 3))
    seed.signal("m1", ALICE, {"hostility": 0.2})
    seed.signal("m2", None, {"hostility": 0.99}, source=None, confidence=None, label="Speaker 3")
    seed.signal("m3", ALICE, {"hostility": 0.99}, source="voiceprint", confidence=0.05)
    out = svc.person(ALICE)
    assert [m["meetingId"] for m in out["meetings"]] == ["m1"]
    assert out["series"]["hostility"][0]["value"] == pytest.approx(0.2)
    assert out["unattributed"]["rows"] == 1  # the low-confidence row claimed for Alice
    overview = svc.overview()
    assert overview["unattributed"]["rows"] == 2
    detail = svc.overview(speaker_id=ALICE)
    assert {m for p in detail["series"]["hostility"] for m in p["meetingIds"]} == {"m1"}


def test_meeting_returns_signals_and_moments_with_identity_source(env: Any) -> None:
    _, seed, svc = env
    seed.meeting("m1", _ms(2026, 1, 1), title="Standup")
    seed.signal("m1", ME, {"warmth": 0.4}, source="voiceprint")
    seed.signal("m1", ALICE, {"warmth": 0.8}, source="name")
    seed.moment("m1", _ms(2026, 1, 1) + 5000, ME, ALICE, "bad")
    out = svc.meeting("m1")
    assert out is not None
    assert out["meetingId"] == "m1"
    assert out["startMs"] == _ms(2026, 1, 1)
    assert out["title"] == "Standup"
    by = {s["speakerId"]: s for s in out["speakers"]}
    assert by[ALICE]["identitySource"] == "name"
    assert by[ME]["dimensions"] == {"warmth": 0.4}
    assert out["moments"][0]["meetingId"] == "m1"
    assert out["moments"][0]["startMs"] == _ms(2026, 1, 1) + 5000
    assert out["moments"][0]["valence"] == "bad"
    assert svc.meeting("nope") is None


def _seed_dyads(seed: Seeder) -> None:
    """Alice warms toward me over six months; Bob cools."""
    for i in range(6):
        mid = f"d{i}"
        seed.meeting(mid, _ms(2026, 1 + i, 10))
        seed.signal(mid, ME, {"warmth": 0.5, "hostility": 0.1})
        seed.signal(mid, ALICE, {"warmth": 0.2 + 0.12 * i, "hostility": 0.3 - 0.05 * i})
        seed.signal(mid, BOB, {"warmth": 0.8 - 0.12 * i, "hostility": 0.1 + 0.05 * i})


def test_dyads_find_warming_and_cooling_pairs_with_n(env: Any) -> None:
    _, seed, svc = env
    svc.set_me(ME, [])
    _seed_dyads(seed)
    out = svc.dyads()
    assert out["meId"] == ME
    by = {(d["fromSpeakerId"], d["toSpeakerId"]): d for d in out["dyads"]}
    assert by[(ALICE, ME)]["trend"] == "warming"
    assert by[(BOB, ME)]["trend"] == "cooling"
    for d in out["dyads"]:
        assert d["series"] and all(p["n"] >= 1 for p in d["series"])
        for p in d["series"]:
            _assert_drillable(p)
        assert d["meetingIds"]
    assert {d["direction"] for d in out["dyads"]} == {"meToThem", "themToMe"}
    assert json.dumps(out)


def test_dyads_use_moments_toward_speaker(env: Any) -> None:
    _, seed, svc = env
    for i in range(5):
        mid = f"x{i}"
        seed.meeting(mid, _ms(2026, 1 + i, 3))
        seed.signal(mid, ME, {})  # shared presence, no dimension scores
        seed.signal(mid, BOB, {})
        # Bob's moments toward me turn from praise to insults.
        valence, severity = [("good", 0.8), ("good", 0.4), ("bad", 0.2), ("bad", 0.6), ("bad", 1.0)][i]
        seed.moment(mid, _ms(2026, 1 + i, 3), BOB, ME, valence, severity=severity)
    out = svc.dyads(me_id=ME)
    d = next(x for x in out["dyads"] if x["fromSpeakerId"] == BOB and x["toSpeakerId"] == ME)
    assert d["trend"] == "cooling"
    assert d["series"][0]["moments"]["good"] == 1
    assert d["series"][-1]["moments"]["bad"] == 1


def test_dyads_without_me_returns_empty(env: Any) -> None:
    _, _, svc = env
    out = svc.dyads()
    assert out["meId"] is None
    assert out["dyads"] == []


def test_dyads_exclude_unattributed_rows(env: Any) -> None:
    _, seed, svc = env
    _seed_dyads(seed)
    seed.meeting("u", _ms(2026, 7, 1))
    seed.signal("u", ME, {"warmth": 0.5})
    seed.signal("u", None, {"warmth": 0.0}, source=None, confidence=None)
    out = svc.dyads(me_id=ME)
    assert out["unattributed"]["rows"] == 1
    assert all(d["fromSpeakerId"] is not None and d["toSpeakerId"] is not None for d in out["dyads"])


def test_config_override_changes_outlier_threshold(tmp_path: Path) -> None:
    db = create_db(":memory:")
    seed = Seeder(db)
    seed.speaker(ALICE, "Alice")
    for i in range(1, 9):
        seed.meeting(f"m{i}", _ms(2026, 1, i))
        seed.signal(f"m{i}", ALICE, {"hostility": 0.95 if i == 5 else 0.1})
    cfg = tmp_path / "cfg" / "insights"
    cfg.mkdir(parents=True)
    (cfg / "config.json").write_text(json.dumps({"outlierZ": 50.0}))
    svc = InsightsQueryService(db=db, config_dir=str(tmp_path / "cfg"))
    assert svc.config["outlierZ"] == 50.0
    assert svc.person(ALICE)["outlierMeetings"] == []


def test_options_override_config(tmp_path: Path) -> None:
    db = create_db(":memory:")
    svc = InsightsQueryService({"minSamples": 7}, db=db, config_dir=str(tmp_path / "cfg"))
    assert svc.config["minSamples"] == 7
