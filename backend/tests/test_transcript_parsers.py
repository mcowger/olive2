"""Pure unit tests for the transcript-import parsers (Zoom VTT, SRT, speaker text)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from olive.services.transcript_parsers import (
    ParsedImport,
    detect_format,
    infer_start_time,
    parse_speaker_text,
    parse_srt,
    parse_transcript,
    parse_vtt,
)

ZOOM_VTT = """WEBVTT

1
00:00:01.000 --> 00:00:04.500
Alice Smith: Hello everyone, thanks for joining.

2
00:00:04.500 --> 00:00:06.000
Alice Smith: Let's get started.

3
00:00:06.500 --> 00:00:09.250
Bob Jones: Sounds good.

4
01:02:03.004 --> 01:02:05.000
Alice Smith: One hour in.
"""


def _tuples(parsed: ParsedImport):
    return [(s.speaker, s.startMs, s.endMs, s.text) for s in parsed.transcript.segments]


def test_zoom_vtt_exact_times_and_merging() -> None:
    parsed = parse_vtt(ZOOM_VTT)
    assert parsed.format == "vtt"
    assert parsed.timingSynthetic is False
    assert _tuples(parsed) == [
        (
            "Alice Smith",
            1000,
            6000,
            "Hello everyone, thanks for joining. Let's get started.",
        ),
        ("Bob Jones", 6500, 9250, "Sounds good."),
        ("Alice Smith", 3723004, 3725000, "One hour in."),
    ]
    assert parsed.transcript.durationMs == 3725000


def test_zoom_subtype_from_cue_style_and_filename() -> None:
    assert parse_vtt(ZOOM_VTT).subtype == "zoom"
    plain = "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nA: hi\n"
    assert parse_vtt(plain).subtype is None
    assert parse_transcript(plain, filename="zoom_meeting.vtt").subtype == "zoom"
    assert parse_transcript(plain, filename="GMT20240315-150000_Rec.vtt").subtype == "zoom"


def test_voice_tags() -> None:
    vtt = (
        "WEBVTT\n\n"
        "00:00:00.000 --> 00:00:02.000\n<v Carol>Hi there</v>\n\n"
        "00:00:02.000 --> 00:00:04.000\n<v.loud Dave Q>Hello <i>back</i></v>\n"
    )
    parsed = parse_vtt(vtt)
    assert _tuples(parsed) == [
        ("Carol", 0, 2000, "Hi there"),
        ("Dave Q", 2000, 4000, "Hello back"),
    ]


def test_vtt_without_hours() -> None:
    vtt = "WEBVTT\n\n01:05.250 --> 01:08.000\nEve: Short form.\n"
    assert _tuples(parse_vtt(vtt)) == [("Eve", 65250, 68000, "Short form.")]


def test_multiline_cue_joined_with_space() -> None:
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:05.000\nFrank: first line\nsecond line\n"
    assert _tuples(parse_vtt(vtt)) == [("Frank", 0, 5000, "first line second line")]


def test_no_speaker_prefix_is_unknown_and_merges() -> None:
    vtt = "WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nsome words\n\n00:00:02.000 --> 00:00:03.000\nmore words\n"
    assert _tuples(parse_vtt(vtt)) == [("Unknown", 0, 3000, "some words more words")]


def test_note_style_region_blocks_ignored() -> None:
    vtt = (
        "WEBVTT - exported\n\n"
        "NOTE this is a comment\nspanning lines\n\n"
        "STYLE\n::cue { color: red }\n\n"
        "REGION\nid:r1\n\n"
        "00:00:00.000 --> 00:00:01.000 align:start position:0%\nGina: kept\n"
    )
    assert _tuples(parse_vtt(vtt)) == [("Gina", 0, 1000, "kept")]


def test_cue_settings_and_ids_tolerated() -> None:
    vtt = "WEBVTT\n\nintro-cue\n00:00:00.000 --> 00:00:01.000 line:90%\nHal: ok\n"
    assert _tuples(parse_vtt(vtt)) == [("Hal", 0, 1000, "ok")]


def test_srt_comma_milliseconds_and_prefix() -> None:
    srt = (
        "1\n00:00:01,200 --> 00:00:03,400\nIvy: Good morning.\n\n"
        "2\n00:00:03,500 --> 00:00:05,000\nno prefix here\n\n"
        "3\n00:00:05,000 --> 00:00:06,000\nJack: Bye\n"
    )
    parsed = parse_srt(srt)
    assert parsed.format == "srt"
    assert _tuples(parsed) == [
        ("Ivy", 1200, 3400, "Good morning."),
        ("Unknown", 3500, 5000, "no prefix here"),
        ("Jack", 5000, 6000, "Bye"),
    ]


def test_text_without_timestamps_is_synthetic() -> None:
    text = "Alice: one two three four\nBob: five six\nAlice: seven\n"
    parsed = parse_speaker_text(text)
    assert parsed.format == "text"
    assert parsed.timingSynthetic is True
    assert any("synthetic" in w.lower() for w in parsed.warnings)
    segs = parsed.transcript.segments
    assert [s.speaker for s in segs] == ["Alice", "Bob", "Alice"]
    assert segs[0].startMs == 0
    for prev, nxt in zip(segs, segs[1:], strict=False):
        assert nxt.startMs == prev.endMs
    # longer text gets a longer synthetic duration
    assert (segs[0].endMs - segs[0].startMs) > (segs[2].endMs - segs[2].startMs)


def test_text_bracket_timestamp_prefix() -> None:
    text = "[00:00:05] Alice: Hello\n[00:00:12] Bob: Hi back\n"
    parsed = parse_speaker_text(text)
    assert parsed.timingSynthetic is False
    assert parsed.warnings == []
    assert _tuples(parsed)[0][:3] == ("Alice", 5000, 12000)
    assert _tuples(parsed)[1][:2] == ("Bob", 12000)
    assert parsed.transcript.segments[1].endMs > 12000


def test_text_parenthesized_timestamp() -> None:
    text = "Alice (00:01:00): Hello\nBob (01:02:03): Hi\n"
    parsed = parse_speaker_text(text)
    assert _tuples(parsed)[0][:3] == ("Alice", 60000, 3723000)
    assert _tuples(parsed)[1][:2] == ("Bob", 3723000)
    assert parsed.timingSynthetic is False


def test_text_zoom_bracket_name_then_text_line() -> None:
    text = "[Alice Smith] 00:00:03\nHello there.\nSecond line.\n\n[Bob Jones] 00:00:09\nHi!\n"
    parsed = parse_speaker_text(text)
    assert parsed.subtype == "zoom"
    assert _tuples(parsed)[0] == ("Alice Smith", 3000, 9000, "Hello there. Second line.")
    assert _tuples(parsed)[1][:2] == ("Bob Jones", 9000)


def test_text_merges_same_speaker_and_continuations() -> None:
    text = "Alice: first\nAlice: second\ncontinued\nBob: x\n"
    parsed = parse_speaker_text(text)
    assert [(s.speaker, s.text) for s in parsed.transcript.segments] == [
        ("Alice", "first second continued"),
        ("Bob", "x"),
    ]


def test_detect_format() -> None:
    assert detect_format("WEBVTT\n\n", None) == "vtt"
    assert detect_format("﻿WEBVTT\n", None) == "vtt"
    assert detect_format("1\n00:00:01,000 --> 00:00:02,000\nhi\n", None) == "srt"
    assert detect_format("Alice: hi\n", None) == "text"
    assert detect_format("Alice: hi\n", "x.vtt") == "vtt"
    assert detect_format("Alice: hi\n", "x.SRT") == "srt"
    assert detect_format("Alice: hi\n", "x.txt") == "text"
    assert detect_format("Alice: hi\n", "") == "text"


def test_infer_start_time_zoom_filename() -> None:
    expected = int(datetime(2024, 3, 15, 15, 0, 0, tzinfo=UTC).timestamp() * 1000)
    assert infer_start_time("GMT20240315-150000_Recording.transcript.vtt", "") == expected


def test_infer_start_time_iso_filename_and_header() -> None:
    expected = int(datetime(2024, 3, 15, 9, 30, 0, tzinfo=UTC).timestamp() * 1000)
    assert infer_start_time("standup 2024-03-15T09:30:00.txt", "") == expected
    date_only = int(datetime(2024, 3, 15, tzinfo=UTC).timestamp() * 1000)
    assert infer_start_time("notes_2024-03-15.txt", "") == date_only
    assert infer_start_time(None, "Meeting started 2024-03-15 09:30:00\nAlice: hi") == expected
    assert infer_start_time(None, "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nA: hi") is None
    assert infer_start_time("notes.txt", "Alice: hi") is None
    assert infer_start_time("x_2024-13-45.txt", "") is None


def test_parse_transcript_sets_started_at_ms() -> None:
    parsed = parse_transcript(ZOOM_VTT, filename="GMT20240315-150000_Recording.transcript.vtt")
    assert parsed.startedAtMs == int(datetime(2024, 3, 15, 15, 0, 0, tzinfo=UTC).timestamp() * 1000)
    assert parse_transcript(ZOOM_VTT).startedAtMs is None


def test_parse_transcript_dispatch_and_fmt_override() -> None:
    assert parse_transcript(ZOOM_VTT).format == "vtt"
    assert parse_transcript("1\n00:00:01,000 --> 00:00:02,000\nA: hi\n").format == "srt"
    assert parse_transcript("Alice: hi\n").format == "text"
    forced = parse_transcript("Alice: hi\n", fmt="text", filename="x.vtt")
    assert forced.format == "text"


@pytest.mark.parametrize(
    "content",
    ["", "   \n\n", "WEBVTT\n\n", "WEBVTT\n\nNOTE only a note\n", "!!! ??? ###", "just a prose line"],
)
def test_empty_or_garbage_raises(content: str) -> None:
    with pytest.raises(ValueError, match="No speech segments found"):
        parse_transcript(content)


def test_bom_and_crlf() -> None:
    content = ZOOM_VTT.replace("\n", "\r\n")
    parsed = parse_transcript("﻿" + content, filename="a.vtt")
    assert _tuples(parsed) == _tuples(parse_vtt(ZOOM_VTT))
    text = "﻿[00:00:05] Alice: Hello\r\n[00:00:12] Bob: Hi\r\n"
    assert _tuples(parse_transcript(text))[0][:3] == ("Alice", 5000, 12000)


def test_transcript_shape_matches_schema() -> None:
    from olive.schemas import Transcript

    parsed = parse_vtt(ZOOM_VTT)
    assert isinstance(parsed.transcript, Transcript)
    dumped = parsed.transcript.model_dump(by_alias=True, exclude_none=True)
    assert dumped["segments"][0]["speaker"] == "Alice Smith"
    assert dumped["durationMs"] == 3725000
