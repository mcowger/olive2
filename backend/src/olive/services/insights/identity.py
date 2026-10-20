"""Resolve transcript speaker labels to canonical ``speakers.id`` values.

A transcript carries a display label per segment ("Alice", "Speaker 1"). For insights to
aggregate per person across meetings, named labels are mapped to ``speakers.id`` via this
meeting's ``meeting_speakers`` rows (case-insensitive name), then via ``speaker_aliases``.
A label whose name a human confirmed for this meeting (``meeting_speaker_name_proposals`` status
``confirmed``) resolves to the speaker linked under that name, keeping any stronger voiceprint or
manual identity already on the link. Generic placeholder labels with no confirmation are not
people and stay unresolved (``speaker_id`` NULL) with the label retained, so unknown speakers are
never silently merged.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection

from ...speaker_labels import is_generic_speaker_label

IDENTITY_SOURCES = ("name", "voiceprint", "manual")


@dataclass(frozen=True)
class ResolvedIdentity:
    """Outcome of resolving one label. ``speaker_id`` is None when unresolved."""

    label: str
    speaker_id: str | None = None
    identity_source: str | None = None
    identity_confidence: float | None = None


def normalize_alias(label: str | None) -> str:
    """Case-folded, whitespace-collapsed form used for name and alias comparison."""
    return " ".join((label or "").split()).casefold()


def _source_or_default(value: str | None) -> str:
    return value if value in IDENTITY_SOURCES else "name"


def resolve_labels(conn: Connection, meeting_id: str, labels: Iterable[str]) -> dict[str, ResolvedIdentity]:
    """Map each label (keyed exactly as passed in) to a :class:`ResolvedIdentity`."""
    meeting_rows = (
        conn.execute(
            text(
                """
                SELECT s.id AS speaker_id, s.name AS name,
                       ms.identity_source AS identity_source,
                       ms.identity_confidence AS identity_confidence
                FROM meeting_speakers ms
                INNER JOIN speakers s ON s.id = ms.speaker_id
                WHERE ms.meeting_id = :mid
                ORDER BY s.id
                """
            ),
            {"mid": meeting_id},
        )
        .mappings()
        .all()
    )
    # A name shared by two of this meeting's speakers is ambiguous: leave it unresolved rather
    # than guess which person a label means.
    by_name: dict[str, dict] = {}
    ambiguous_names: set[str] = set()
    by_speaker: dict[str, dict] = {}
    for row in meeting_rows:
        key = normalize_alias(row["name"])
        by_speaker.setdefault(row["speaker_id"], dict(row))
        if key in by_name and by_name[key]["speaker_id"] != row["speaker_id"]:
            ambiguous_names.add(key)
        by_name.setdefault(key, dict(row))

    alias_rows = (
        conn.execute(
            text(
                """
                SELECT alias_norm, speaker_id, confirmed
                FROM speaker_aliases
                WHERE speaker_id IS NOT NULL
                ORDER BY confirmed DESC, created_at, id
                """
            )
        )
        .mappings()
        .all()
    )
    alias_hits: dict[str, list[dict]] = {}
    for row in alias_rows:
        alias_hits.setdefault(row["alias_norm"], []).append(dict(row))
    by_alias: dict[str, dict] = {}
    for key, hits in alias_hits.items():
        distinct = {h["speaker_id"] for h in hits}
        if len(distinct) > 1:
            # Several people carry this alias; only a single one of them present in this meeting
            # disambiguates it.
            in_meeting = [h for h in hits if h["speaker_id"] in by_speaker]
            if len({h["speaker_id"] for h in in_meeting}) != 1:
                continue
            hits = in_meeting
        by_alias[key] = hits[0]

    def _from_alias_hit(label: str, hit: dict) -> ResolvedIdentity:
        # A speaker already linked to this meeting keeps the provenance on that link (a voiceprint
        # or manual match must not decay to a bare name match because the label hit an alias).
        linked = by_speaker.get(hit["speaker_id"])
        if linked is not None and linked["identity_source"] in IDENTITY_SOURCES:
            return ResolvedIdentity(
                label=label,
                speaker_id=hit["speaker_id"],
                identity_source=linked["identity_source"],
                identity_confidence=linked["identity_confidence"],
            )
        confirmed = bool(hit["confirmed"])
        return ResolvedIdentity(
            label=label,
            speaker_id=hit["speaker_id"],
            identity_source="manual" if confirmed else "name",
            identity_confidence=1.0 if confirmed else None,
        )

    # Labels a human confirmed for this meeting (e.g. "Speaker 2" -> "Vineet") resolve through
    # the confirmed name, to the speaker already linked to the meeting under that name.
    confirmed_names: dict[str, str] = {}
    for row in (
        conn.execute(
            text(
                """
                SELECT speaker_label, proposed_name
                FROM meeting_speaker_name_proposals
                WHERE meeting_id = :mid AND status = 'confirmed' AND proposed_name IS NOT NULL
                """
            ),
            {"mid": meeting_id},
        )
        .mappings()
        .all()
    ):
        confirmed_names.setdefault(
            normalize_alias(row["speaker_label"]), normalize_alias(row["proposed_name"])
        )

    def _from_confirmed(label: str, norm: str) -> ResolvedIdentity | None:
        name = confirmed_names.get(norm)
        if not name:
            return None
        if name in ambiguous_names:
            return None
        if name in by_name:
            hit = by_name[name]
            return ResolvedIdentity(
                label=label,
                speaker_id=hit["speaker_id"],
                identity_source=_source_or_default(hit["identity_source"]),
                identity_confidence=hit["identity_confidence"],
            )
        if name in by_alias:
            return _from_alias_hit(label, by_alias[name])
        return None

    resolved: dict[str, ResolvedIdentity] = {}
    for label in labels:
        if label in resolved:
            continue
        norm = normalize_alias(label)
        if is_generic_speaker_label(label):
            resolved[label] = _from_confirmed(label, norm) or ResolvedIdentity(label=label)
        elif norm in ambiguous_names:
            resolved[label] = ResolvedIdentity(label=label)
        elif norm in by_name:
            hit = by_name[norm]
            resolved[label] = ResolvedIdentity(
                label=label,
                speaker_id=hit["speaker_id"],
                identity_source=_source_or_default(hit["identity_source"]),
                identity_confidence=hit["identity_confidence"],
            )
        elif norm in by_alias:
            resolved[label] = _from_alias_hit(label, by_alias[norm])
        else:
            resolved[label] = _from_confirmed(label, norm) or ResolvedIdentity(label=label)
    return resolved


__all__ = ["IDENTITY_SOURCES", "ResolvedIdentity", "normalize_alias", "resolve_labels"]
